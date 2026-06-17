"""FEDformer model — large-pipeline wrapper.

FEDformer replaces the standard self-attention with frequency-domain
attention (Fourier or Wavelets).  The Encoder/Decoder stack is shared
with Autoformer, only the inner attention blocks differ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from .base import BaseModel
from .autoformer_layers.Embed import TokenEmbedding, PositionalEmbedding
from .fedformer_layers import (
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    my_Layernorm, series_decomp,
    AutoCorrelationLayer,
    FourierBlock, FourierCrossAttention,
    MultiWaveletTransform, MultiWaveletCross,
)


# ===================================================================
#  Custom DataEmbedding — value + position + temporal (linear)
# ===================================================================

class _DataEmbedding(nn.Module):
    """DataEmbedding with position encoding and n_time_features linear projection."""

    def __init__(self, c_in, d_model, n_time_features, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = nn.Linear(n_time_features, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        if x_mark is not None:
            x = x + self.temporal_embedding(x_mark)
        return self.dropout(x)


# ===================================================================
#  Raw FEDformer model (adapted from time_series_models_labs)
# ===================================================================

class _RawFEDformer(nn.Module):
    """FEDformer model — forecast task only."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.decomp = series_decomp(configs.moving_avg)

        self.enc_embedding = _DataEmbedding(
            configs.enc_in, configs.d_model, configs.n_time_features, configs.dropout)
        self.dec_embedding = _DataEmbedding(
            configs.dec_in, configs.d_model, configs.n_time_features, configs.dropout)

        # Attention blocks
        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model, out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=configs.modes, ich=configs.d_model,
                base='legendre', activation='tanh')
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model, out_channels=configs.d_model,
                n_heads=configs.n_heads, seq_len=self.seq_len,
                modes=configs.modes, mode_select_method=configs.mode_select)
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model, out_channels=configs.d_model,
                n_heads=configs.n_heads, seq_len=self.seq_len // 2 + self.pred_len,
                modes=configs.modes, mode_select_method=configs.mode_select)
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model, out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=configs.modes, mode_select_method=configs.mode_select,
                num_heads=configs.n_heads)

        # Encoder
        self.encoder = Encoder([
            EncoderLayer(
                AutoCorrelationLayer(encoder_self_att, configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for _ in range(configs.e_layers)
        ], norm_layer=my_Layernorm(configs.d_model))

        # Decoder
        self.decoder = Decoder([
            DecoderLayer(
                AutoCorrelationLayer(decoder_self_att, configs.d_model, configs.n_heads),
                AutoCorrelationLayer(decoder_cross_att, configs.d_model, configs.n_heads),
                configs.d_model, configs.c_out, configs.d_ff,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for _ in range(configs.d_layers)
        ], norm_layer=my_Layernorm(configs.d_model),
           projection=nn.Linear(configs.d_model, configs.c_out, bias=True))
        self.trend_proj = nn.Linear(configs.enc_in, configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        if self.label_len > 0:
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        else:
            trend_init = mean
            seasonal_init = zeros
        trend_init = self.trend_proj(trend_init)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        return trend_part + seasonal_part

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]


# ===================================================================
#  FEDformerWrapper — public-facing class registered in MODEL_REGISTRY
# ===================================================================

class FEDformerWrapper(BaseModel):
    """FEDformer wrapper compatible with the two-pipeline system.

    Pipeline: "large" — 4-argument forward.
    Supports two frequency-domain attention variants:
    - Fourier: FFT-based linear transform
    - Wavelets: multi-wavelet decomposition
    """

    pipeline = "large"

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        seq_len = kwargs.get("seq_len", 96)
        label_len = kwargs.get("label_len", seq_len // 2)
        pred_len = output_dim
        n_time_features = kwargs.get("n_time_features", 4)

        configs = SimpleNamespace(
            enc_in=input_dim,
            dec_in=input_dim,
            c_out=1,
            n_time_features=n_time_features,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            d_model=kwargs.get("d_model", 256),
            n_heads=kwargs.get("n_heads", 8),
            e_layers=kwargs.get("e_layers", 3),
            d_layers=kwargs.get("d_layers", 3),
            d_ff=kwargs.get("d_ff", 32),
            moving_avg=kwargs.get("moving_avg", 25),
            dropout=kwargs.get("dropout", 0.1),
            activation=kwargs.get("activation", "gelu"),
            version=kwargs.get("version", "Fourier"),
            mode_select=kwargs.get("mode_select", "random"),
            modes=kwargs.get("modes", 32),
        )
        self._configs = configs
        self._model = _RawFEDformer(configs)

    def forward(self, batch_x, batch_x_mark, batch_dec_inp, batch_y_mark):
        """Forward pass.

        Parameters
        ----------
        batch_x : (batch, seq_len, enc_in)
        batch_x_mark : (batch, seq_len, n_time_features)
        batch_dec_inp : (batch, label_len+pred_len, dec_in)
        batch_y_mark : (batch, label_len+pred_len, n_time_features)

        Returns
        -------
        torch.Tensor (batch, pred_len, 1)
        """
        return self._model(batch_x, batch_x_mark, batch_dec_inp, batch_y_mark)
