"""Koopa — Koopman-based time series forecasting.

Paper: https://arxiv.org/pdf/2305.18803.pdf
Pipeline: "large" (forward takes x_enc, x_mark_enc, x_dec, x_mark_dec).
"""

import math
import torch
import torch.nn as nn

from .base import BaseModel


# ── Internal components ─────────────────────────────────────────


class FourierFilter(nn.Module):
    """Fourier Filter: split into time-variant and time-invariant terms."""
    def __init__(self, mask_spectrum):
        super().__init__()
        self.mask_spectrum = mask_spectrum

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf * mask, dim=1)
        x_inv = x - x_var
        return x_var, x_inv


class MLP(nn.Module):
    """Multilayer perceptron for encoding/decoding."""
    def __init__(self, f_in, f_out, hidden_dim=128, hidden_layers=2,
                 dropout=0.05, activation='tanh'):
        super().__init__()
        act = nn.ReLU() if activation == 'relu' else nn.Tanh()
        layers = [nn.Linear(f_in, hidden_dim), act, nn.Dropout(dropout)]
        for _ in range(hidden_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class KPLayer(nn.Module):
    """Koopman layer — DMD-based one-step transition."""
    def __init__(self):
        super().__init__()
        self.K = None

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, L, E = z.shape
        x, y = z[:, :-1], z[:, 1:]
        self.K = torch.linalg.lstsq(x, y).solution
        if torch.isnan(self.K).any():
            self.K = torch.eye(E, device=z.device).unsqueeze(0).repeat(B, 1, 1)
        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred
        return z_pred

    def forward(self, z, pred_len=1):
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)
        preds = [z_pred]
        for _ in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            preds.append(z_pred)
        return z_rec, torch.cat(preds, dim=1)


class KPLayerApprox(nn.Module):
    """Koopman layer with multistep K approximation."""
    def __init__(self):
        super().__init__()
        self.K = None
        self.K_step = None

    def forward(self, z, pred_len=1):
        B, L, E = z.shape
        x, y = z[:, :-1], z[:, 1:]
        self.K = torch.linalg.lstsq(x, y).solution
        if torch.isnan(self.K).any():
            self.K = torch.eye(E, device=z.device).unsqueeze(0).repeat(B, 1, 1)
        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)

        if pred_len <= L:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                self.K_step = torch.eye(E, device=z.device).unsqueeze(0).repeat(B, 1, 1)
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, L)
            if torch.isnan(self.K_step).any():
                self.K_step = torch.eye(E, device=z.device).unsqueeze(0).repeat(B, 1, 1)
            temp, all_pred = z, []
            for _ in range(math.ceil(pred_len / L)):
                temp = torch.bmm(temp, self.K_step)
                all_pred.append(temp)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]
        return z_rec, z_pred


class TimeVarKP(nn.Module):
    """Koopman predictor for time-variant term (local variations)."""
    def __init__(self, enc_in=8, input_len=96, pred_len=96, seg_len=24,
                 dynamic_dim=128, encoder=None, decoder=None, multistep=False):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder = encoder
        self.decoder = decoder
        self.freq = math.ceil(self.input_len / self.seg_len)
        self.step = math.ceil(self.pred_len / self.seg_len)
        self.padding_len = self.seg_len * self.freq - self.input_len
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer()

    def forward(self, x):
        B, L, C = x.shape
        res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)
        res = res.chunk(self.freq, dim=1)
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)
        res = self.encoder(res)
        x_rec, x_pred = self.dynamics(res, self.step)
        x_rec = self.decoder(x_rec).reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, :self.input_len, :]
        x_pred = self.decoder(x_pred).reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, :self.pred_len, :]
        return x_rec, x_pred


class TimeInvKP(nn.Module):
    """Koopman predictor for time-invariant term (learnable operator)."""
    def __init__(self, input_len=96, pred_len=96, dynamic_dim=128,
                 encoder=None, decoder=None):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder
        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init)
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

    def forward(self, x):
        res = x.transpose(1, 2)
        res = self.encoder(res)
        res = self.K(res)
        res = self.decoder(res)
        return res.transpose(1, 2)


# ── Wrapper ─────────────────────────────────────────────────────


class KoopaWrapper(BaseModel):
    """Koopa wrapper for the DeepLearning Labs framework."""

    pipeline = "large"
    uses_internal_normalization = True  # Series stationarization in forecast()

    def __init__(self, input_dim, output_dim,
                 seq_len=96, pred_len=1,
                 dynamic_dim=128, hidden_dim=64, hidden_layers=2,
                 num_blocks=3, multistep=False,
                 **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len
        self.input_len = seq_len
        self.seg_len = pred_len
        self.num_blocks = num_blocks
        self.dynamic_dim = dynamic_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.multistep = multistep
        # Empty mask spectrum → no spectrum filtering (full spectrum preserved)
        self.mask_spectrum = []

        self.disentanglement = FourierFilter(self.mask_spectrum)

        # Shared encoders/decoders
        self.time_inv_encoder = MLP(f_in=self.input_len, f_out=self.dynamic_dim,
                                    activation='relu', hidden_dim=self.hidden_dim,
                                    hidden_layers=self.hidden_layers)
        self.time_inv_decoder = MLP(f_in=self.dynamic_dim, f_out=self.pred_len,
                                    activation='relu', hidden_dim=self.hidden_dim,
                                    hidden_layers=self.hidden_layers)
        self.time_inv_kps = nn.ModuleList([
            TimeInvKP(input_len=self.input_len, pred_len=self.pred_len,
                      dynamic_dim=self.dynamic_dim,
                      encoder=self.time_inv_encoder, decoder=self.time_inv_decoder)
            for _ in range(self.num_blocks)])

        self.time_var_encoder = MLP(f_in=self.seg_len * self.input_dim,
                                    f_out=self.dynamic_dim, activation='tanh',
                                    hidden_dim=self.hidden_dim,
                                    hidden_layers=self.hidden_layers)
        self.time_var_decoder = MLP(f_in=self.dynamic_dim,
                                    f_out=self.seg_len * self.input_dim,
                                    activation='tanh',
                                    hidden_dim=self.hidden_dim,
                                    hidden_layers=self.hidden_layers)
        self.time_var_kps = nn.ModuleList([
            TimeVarKP(enc_in=self.input_dim, input_len=self.input_len,
                      pred_len=self.pred_len, seg_len=self.seg_len,
                      dynamic_dim=self.dynamic_dim,
                      encoder=self.time_var_encoder, decoder=self.time_var_decoder,
                      multistep=self.multistep)
            for _ in range(self.num_blocks)])

        # Project enc_in → 1
        self.output_proj = nn.Linear(self.input_dim, output_dim)

    def forecast(self, x_enc):
        """Koopman forecasting with series stationarization (internal norm)."""
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        residual = x_enc
        forecast = None
        for i in range(self.num_blocks):
            tv_input, ti_input = self.disentanglement(residual)
            ti_output = self.time_inv_kps[i](ti_input)
            tv_backcast, tv_output = self.time_var_kps[i](tv_input)
            residual = residual - tv_backcast
            out = ti_output + tv_output
            forecast = out if forecast is None else forecast + out

        return forecast * std_enc + mean_enc

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc)
        dec_out = dec_out[:, -self.pred_len:, :]  # (B, pred_len, enc_in)
        dec_out = self.output_proj(dec_out)       # (B, pred_len, output_dim)
        return dec_out
