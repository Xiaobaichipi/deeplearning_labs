"""FiLM model — large-pipeline wrapper.

FiLM (Frequency-enhanced Legendre Memory) projects input onto Legendre
polynomial basis via HiPPO-LegT, applies SpectralConv1d (FFT) in the
frequency domain, and reconstructs the output.  Uses internal instance
normalization (like Non-stationary Transformer), so **external**
normalization is bypassed (``uses_internal_normalization = True``).
"""

import torch
import torch.nn as nn

from .base import BaseModel
from .film_layers import HiPPO_LegT, SpectralConv1d


def _parse_int_list(value, default):
    """Parse an int list from a string (comma-separated) or list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return default


# ===================================================================
#  Raw FiLM model (adapted from time_series_models_labs)
# ===================================================================

class _RawFiLM(nn.Module):
    """FiLM model — forecast task only.

    Performs internal instance normalisation (RevIN-style), HiPPO-LegT
    projection, SpectralConv1d frequency processing, and Legendre
    reconstruction.  Output shape is ``(batch, pred_len, enc_in)``.
    """

    def __init__(self, enc_in, seq_len, pred_len,
                 window_size=None, multiscale=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        ws_list = _parse_int_list(window_size, [256])
        ms_list = _parse_int_list(multiscale, [1, 2, 4])
        self.window_size = ws_list
        self.multiscale = ms_list

        # Affine transform (learnable scale/bias)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, enc_in))

        # HiPPO-LegT + SpectralConv1d per combination
        self.legts = nn.ModuleList(
            [HiPPO_LegT(N=n, dt=1. / pred_len / i)
             for n in ws_list for i in ms_list])
        self.spec_conv_1 = nn.ModuleList(
            [SpectralConv1d(in_channels=n, out_channels=n,
                            seq_len=min(pred_len, seq_len), ratio=0.5)
             for n in ws_list for _ in ms_list])

        # MLP fusion across multi-scale outputs
        self.mlp = nn.Linear(len(ms_list) * len(ws_list), 1)

    def forecast(self, x_enc):
        """Forecast.

        Parameters
        ----------
        x_enc : (batch, seq_len, enc_in)

        Returns
        -------
        (batch, pred_len, enc_in)
        """
        # Internal normalisation (Non-stationary Transformer style)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        x_enc = x_enc * self.affine_weight + self.affine_bias

        # Multi-scale processing
        x_decs = []
        n_combs = len(self.multiscale) * len(self.window_size)
        for i in range(n_combs):
            scale = self.multiscale[i % len(self.multiscale)]
            x_in_len = scale * self.pred_len
            x_in = x_enc[:, -x_in_len:]                         # (batch, x_in_len, enc_in)

            legt = self.legts[i]
            # HiPPO projection: (batch, enc_in, x_in_len) → (x_in_len, batch, enc_in, N)
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])
            # SpectralConv1d
            out1 = self.spec_conv_1[i](x_in_c)                  # (batch, enc_in, N, freq)

            # Extract relevant timestep
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]

            # Legendre reconstruction to pred_len
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T  # (batch, enc_in, pred_len)
            x_decs.append(x_dec)

        # MLP fusion: stack multi-scale outputs → fuse to 1
        x_dec = torch.stack(x_decs, dim=-1)                      # (batch, enc_in, pred_len, n_combs)
        x_dec = self.mlp(x_dec).squeeze(-1)                      # (batch, enc_in, pred_len)
        x_dec = x_dec.permute(0, 2, 1)                           # (batch, pred_len, enc_in)

        # Reverse internal normalisation
        x_dec = x_dec - self.affine_bias
        x_dec = x_dec / (self.affine_weight + 1e-10)
        x_dec = x_dec * stdev
        x_dec = x_dec + means
        return x_dec

    def forward(self, x_enc):
        return self.forecast(x_enc)


# ===================================================================
#  FilmWrapper — public-facing class registered in MODEL_REGISTRY
# ===================================================================

class FilmWrapper(BaseModel):
    """FiLM wrapper compatible with the two-pipeline system.

    Pipeline: "large" — 4-argument forward (only ``x_enc`` is used).
    Uses internal instance normalization so external normalization is
    bypassed (``uses_internal_normalization = True``).

    Parameters exposed in the frontend:
    - window_size : comma-separated int list (default ``[256]``)
    - multiscale  : comma-separated int list (default ``[1,2,4]``)
    - dropout     : float (not used by FiLM core, available for future)
    """

    pipeline = "large"
    uses_internal_normalization = True

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        seq_len = kwargs.get("seq_len", 96)
        pred_len = output_dim

        self._model = _RawFiLM(
            enc_in=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            window_size=kwargs.get("window_size", [256]),
            multiscale=kwargs.get("multiscale", [1, 2, 4]),
        )
        self.output_proj = nn.Linear(input_dim, 1)

    def forward(self, batch_x, batch_x_mark, batch_dec_inp, batch_y_mark):
        """Forward pass.

        Parameters
        ----------
        batch_x : (batch, seq_len, enc_in)
        batch_x_mark : (batch, seq_len, n_time_features) — **ignored**
        batch_dec_inp : (batch, label_len+pred_len, dec_in) — **ignored**
        batch_y_mark : (batch, label_len+pred_len, n_time_features) — **ignored**

        Returns
        -------
        torch.Tensor (batch, pred_len, 1)
        """
        out = self._model(batch_x)          # (batch, pred_len, enc_in)
        out = self.output_proj(out)         # (batch, pred_len, 1)
        return out
