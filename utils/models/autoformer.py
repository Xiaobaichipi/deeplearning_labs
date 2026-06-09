"""Autoformer model — large-pipeline wrapper.

Wraps the original Autoformer implementation from
``time_series_models_labs/models/Autoformer.py`` as a ``BaseModel`` subclass
with ``pipeline = "large"``.

The configs object needed by Autoformer.Model is constructed internally from
the (input_dim, output_dim, **kwargs) pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from types import SimpleNamespace

from .base import BaseModel

# ---------------------------------------------------------------------------
#  Autoformer dependency layers (copied from time_series_models_labs/layers/)
# ---------------------------------------------------------------------------
from .autoformer_layers.Embed import DataEmbedding_wo_pos as _OrigDataEmbeddingWoPos, TokenEmbedding
from .autoformer_layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .autoformer_layers.Autoformer_EncDec import (
    Encoder, Decoder, EncoderLayer, DecoderLayer,
    my_Layernorm, series_decomp,
)


# ---------------------------------------------------------------------------
#  Custom DataEmbeddingWoPos — uses actual n_time_features instead of
#  hardcoded freq → d_inp mapping from the reference.
# ---------------------------------------------------------------------------

class _DataEmbeddingWoPos(nn.Module):
    """DataEmbedding without positional encoding — n_time_features aware."""

    def __init__(self, c_in, d_model, n_time_features, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = nn.Linear(n_time_features, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# ===================================================================
#  Raw Autoformer model (adapted from time_series_models_labs)
# ===================================================================

class _RawAutoformer(nn.Module):
    """Autoformer model — forecast task only.

    Adapted from ``time_series_models_labs/models/Autoformer.py``.
    Supports only ``long_term_forecast`` task.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding (n_time_features-aware, no hardcoded freq mapping)
        self.enc_embedding = _DataEmbeddingWoPos(
            configs.enc_in, configs.d_model, configs.n_time_features, configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor,
                                        attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )
        # Project trend_init from enc_in → c_out so it matches the decoder's
        # residual_trend dimensions (the original reference has enc_in == c_out
        # in all experiments, so broadcasting papers over the mismatch).
        self.trend_proj = nn.Linear(configs.enc_in, configs.c_out)

        # Decoder
        self.dec_embedding = _DataEmbeddingWoPos(
            configs.dec_in, configs.d_model, configs.n_time_features, configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor,
                                        attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor,
                                        attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # encoder input decomp
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]],
                            device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        trend_init = self.trend_proj(trend_init)   # (batch, L+pl, enc_in) → (batch, L+pl, c_out)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out[:, -self.pred_len:, :]   # (batch, pred_len, c_out)


# ===================================================================
#  AutoformerWrapper — public-facing class registered in MODEL_REGISTRY
# ===================================================================

class AutoformerWrapper(BaseModel):
    """Autoformer wrapper compatible with the two-pipeline system.

    Pipeline: "large" — requires 4-argument forward:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec) → (batch, pred_len, 1)

    The encoder input ``x_enc`` should contain all value features (including
    the target column).  The output predicts only the target (c_out=1).

    Typical usage from the large training loop::

        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # outputs: (batch, pred_len, 1)
        loss = criterion(outputs.squeeze(-1), batch_y)
    """

    pipeline = "large"

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        seq_len = kwargs.get("seq_len", 96)
        label_len = kwargs.get("label_len", seq_len // 2)
        pred_len = output_dim
        n_time_features = kwargs.get("n_time_features", 4)

        # Build configs object for the underlying Autoformer
        configs = SimpleNamespace(
            # dimensions
            enc_in=input_dim,
            dec_in=input_dim,
            c_out=1,
            n_time_features=n_time_features,
            # sequence lengths
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            # architecture
            d_model=kwargs.get("d_model", 256),
            n_heads=kwargs.get("n_heads", 8),
            e_layers=kwargs.get("e_layers", 3),
            d_layers=kwargs.get("d_layers", 3),
            d_ff=kwargs.get("d_ff", 32),
            moving_avg=kwargs.get("moving_avg", 25),
            factor=kwargs.get("factor", 3),
            dropout=kwargs.get("dropout", 0.1),
            activation=kwargs.get("activation", "gelu"),
        )
        self._configs = configs
        self._model = _RawAutoformer(configs)

    def forward(self, batch_x, batch_x_mark, batch_dec_inp, batch_y_mark):
        """Forward pass.

        Parameters
        ----------
        batch_x : torch.Tensor (batch, seq_len, enc_in)
            Value features (including target column) for the encoder window.
        batch_x_mark : torch.Tensor (batch, seq_len, n_time_features)
            Time features for the encoder window.
        batch_dec_inp : torch.Tensor (batch, label_len+pred_len, dec_in)
            Decoder input — values don't matter for forecast (decomp uses
            the encoder output), but shape must be correct.
        batch_y_mark : torch.Tensor (batch, label_len+pred_len, n_time_features)
            Time features for the decoder window.

        Returns
        -------
        torch.Tensor (batch, pred_len, 1)
            Target predictions.
        """
        return self._model(batch_x, batch_x_mark, batch_dec_inp, batch_y_mark)
