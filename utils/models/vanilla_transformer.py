"""Vanilla Transformer — full Encoder-Decoder with DataEmbedding.

Pipeline: "large" — requires 4-argument forward:
    forward(x_enc, x_mark_enc, x_dec, x_mark_dec) → (batch, pred_len, 1)
"""

import torch.nn as nn
from types import SimpleNamespace

from .base import BaseModel
from .shared_layers import (
    DataEmbedding,
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    FullAttention, AttentionLayer,
)


class _RawVanillaTransformer(nn.Module):
    """Vanilla Transformer core — Encoder + Decoder + DataEmbedding.

    Config attributes used:
        enc_in, dec_in, c_out, n_time_features (dimensions)
        seq_len, label_len, pred_len (sequence lengths)
        d_model, n_heads, e_layers, d_layers, d_ff (architecture)
        dropout, activation (regularisation)
    """

    def __init__(self, configs):
        super().__init__()
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.n_time_features, configs.dropout,
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.n_time_features, configs.dropout,
        )

        # Encoder stack
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # Decoder stack
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
        )

    def forward(self, batch_x, batch_x_mark, batch_dec_inp, batch_y_mark):
        # Encoder: (batch, seq_len, enc_in) → (batch, seq_len, d_model)
        enc_out = self.enc_embedding(batch_x, batch_x_mark)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Decoder: (batch, label_len+pred_len, dec_in) → (batch, label_len+pred_len, c_out)
        dec_out = self.dec_embedding(batch_dec_inp, batch_y_mark)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out[:, -self.pred_len:, :]  # (batch, pred_len, c_out)


class VanillaTransformerWrapper(BaseModel):
    """Vanilla Transformer wrapper compatible with the two-pipeline system.

    Pipeline: "large" — requires 4-argument forward:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec) → (batch, pred_len, 1)
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
            dropout=kwargs.get("dropout", 0.1),
            activation=kwargs.get("activation", "gelu"),
        )
        self._configs = configs
        self._model = _RawVanillaTransformer(configs)

    def forward(self, batch_x, batch_x_mark, batch_dec_inp, batch_y_mark):
        return self._model(batch_x, batch_x_mark, batch_dec_inp, batch_y_mark)
