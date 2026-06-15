"""Informer model — large-pipeline wrapper.

Based on the original implementation from
``time_series_models_labs/models/Informer.py``, adapted to the two-pipeline
system with ``pipeline = "large"``.

The key difference from Autoformer: Informer uses ProbSparse self-attention
(``ProbAttention``) and optional convolutional distillation (``ConvLayer``)
between encoder layers.
"""

import torch
import torch.nn as nn
from types import SimpleNamespace

from .base import BaseModel

from .shared_layers.Embed import DataEmbedding
from .shared_layers.Transformer_EncDec import (
    Encoder, EncoderLayer, Decoder, DecoderLayer, ConvLayer,
)
from .informer_layers.SelfAttention_Family import ProbAttention, AttentionLayer


class _RawInformer(nn.Module):
    """Raw Informer model — forecast only.

    Adapted from ``time_series_models_labs/models/Informer.py``.
    Only supports the ``long_term_forecast`` task.
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding (n_time_features-aware)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.n_time_features, configs.dropout)
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.n_time_features, configs.dropout)

        # Encoder with ProbSparse attention + optional distillation
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            [
                ConvLayer(configs.d_model)
                for _ in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # Decoder with ProbSparse self-attention + cross-attention
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out[:, -self.pred_len:, :]  # (batch, pred_len, c_out)


class InformerWrapper(BaseModel):
    """Informer wrapper compatible with the two-pipeline system.

    Pipeline: "large" — requires 4-argument forward:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec) → (batch, pred_len, 1)

    Uses ProbSparse self-attention (O(L log L)) and optional convolutional
    distillation between encoder layers.
    """

    pipeline = "large"

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        seq_len = kwargs.get("seq_len", 96)
        label_len = kwargs.get("label_len", seq_len // 2)
        pred_len = output_dim
        n_time_features = kwargs.get("n_time_features", 4)

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
            factor=kwargs.get("factor", 3),
            dropout=kwargs.get("dropout", 0.1),
            activation=kwargs.get("activation", "gelu"),
            # Informer-specific
            distil=kwargs.get("distil", True),
            output_attention=kwargs.get("output_attention", False),
        )
        self._configs = configs
        self._model = _RawInformer(configs)

    def forward(self, batch_x, batch_x_mark, batch_dec_inp, batch_y_mark):
        """Forward pass.

        Parameters
        ----------
        batch_x : torch.Tensor (batch, seq_len, enc_in)
            Value features (including target column) for the encoder window.
        batch_x_mark : torch.Tensor (batch, seq_len, n_time_features)
            Time features for the encoder window.
        batch_dec_inp : torch.Tensor (batch, label_len+pred_len, dec_in)
            Decoder input — values don't matter for forecast (decoder uses
            cross-attention with encoder output), but shape must be correct.
        batch_y_mark : torch.Tensor (batch, label_len+pred_len, n_time_features)
            Time features for the decoder window.

        Returns
        -------
        torch.Tensor (batch, pred_len, 1)
            Target predictions.
        """
        return self._model(batch_x, batch_x_mark, batch_dec_inp, batch_y_mark)
