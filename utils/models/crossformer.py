"""Crossformer model — large-pipeline wrapper.

Based on ``time_series_models_labs/models/Crossformer.py`` with
``pipeline = "large"`` and a final linear projection from enc_in → 1
so the output shape matches the single-target convention.

Crossformer paper: https://openreview.net/pdf?id=vSVLM2j9eie
"""

import torch
import torch.nn as nn
from math import ceil
from einops import rearrange, repeat
from types import SimpleNamespace

from .base import BaseModel
from .crossformer_layers import (
    PatchEmbedding, scale_block, Encoder, Decoder, DecoderLayer,
    TwoStageAttentionLayer, AttentionLayer, FullAttention,
)


class _RawCrossformer(nn.Module):
    """Raw Crossformer model — forecast only.

    Adapted from ``time_series_models_labs/models/Crossformer.py``.
    Only supports the ``long_term_forecast`` task.
    Output shape: (batch, pred_len, enc_in) — multivariate.
    """

    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.win_size = configs.win_size

        # Segment accounting — pad to multiples of seg_len
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            configs.d_model, self.seg_len, self.seg_len,
            self.pad_in_len - configs.seq_len, 0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder — stack of scale_blocks with hierarchical merging
        self.encoder = Encoder([
            scale_block(
                configs,
                1 if l == 0 else self.win_size,
                configs.d_model, configs.n_heads, configs.d_ff,
                1, configs.dropout,
                self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l),
                configs.factor,
                activation=configs.activation,
            )
            for l in range(configs.e_layers)
        ])

        # Decoder — learned position embedding + DecoderLayers
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model))
        self.decoder = Decoder([
            DecoderLayer(
                TwoStageAttentionLayer(
                    configs, (self.pad_out_len // self.seg_len), configs.factor,
                    configs.d_model, configs.n_heads, configs.d_ff, configs.dropout,
                    activation=configs.activation,
                ),
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads,
                ),
                self.seg_len, configs.d_model, configs.d_ff, dropout=configs.dropout,
                activation=configs.activation,
            )
            for _ in range(configs.e_layers + 1)
        ])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Forecast — ignores x_dec / x_mark_dec (uses learned decoder embedding)."""
        # Embedding: (batch, seq_len, enc_in) → (batch, enc_in, seq_len) → patch → TSA format
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        # Encoder → list of multi-scale representations
        enc_out, attns = self.encoder(x_enc)

        # Decoder
        dec_in = repeat(
            self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        # dec_out: (batch, pad_out_len, enc_in) → trim to pred_len
        return dec_out[:, :self.pred_len, :]


class CrossformerWrapper(BaseModel):
    """Crossformer wrapper compatible with the two-pipeline system.

    Pipeline: "large" — 4-argument forward:
        forward(x_enc, x_mark_enc, x_dec, x_mark_dec) → (batch, pred_len, 1)

    Crossformer natively predicts all variables (enc_in channels).  A final
    linear projection ``nn.Linear(enc_in, 1)`` maps the output down to the
    single-target convention used by ``LargePipelineStrategy``.
    """

    pipeline = "large"

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        seq_len = kwargs.get("seq_len", 96)
        pred_len = output_dim
        n_time_features = kwargs.get("n_time_features", 4)

        configs = SimpleNamespace(
            # dimensions
            enc_in=input_dim,
            c_out=1,
            n_time_features=n_time_features,
            # sequence lengths
            seq_len=seq_len,
            pred_len=pred_len,
            # architecture
            d_model=kwargs.get("d_model", 256),
            n_heads=kwargs.get("n_heads", 8),
            e_layers=kwargs.get("e_layers", 3),
            d_ff=kwargs.get("d_ff", 32),
            factor=kwargs.get("factor", 3),
            dropout=kwargs.get("dropout", 0.1),
            activation=kwargs.get("activation", "gelu"),
            # Crossformer-specific
            seg_len=kwargs.get("seg_len", 12),
            win_size=kwargs.get("win_size", 2),
        )
        self._configs = configs
        self._model = _RawCrossformer(configs)

        # Project multivariate output (enc_in) → single target (1)
        self.output_proj = nn.Linear(input_dim, 1)

    def forward(self, batch_x, batch_x_mark, batch_dec_inp, batch_y_mark):
        """Forward pass.

        Parameters
        ----------
        batch_x : (batch, seq_len, enc_in) — value features.
        batch_x_mark : (batch, seq_len, n_time_features) — time features (unused).
        batch_dec_inp : (batch, label_len+pred_len, enc_in) — unused by Crossformer.
        batch_y_mark : (batch, label_len+pred_len, n_time_features) — unused.

        Returns
        -------
        torch.Tensor (batch, pred_len, 1) — target predictions.
        """
        output = self._model(batch_x, batch_x_mark, batch_dec_inp, batch_y_mark)
        # output: (batch, pred_len, enc_in) → (batch, pred_len, 1)
        return self.output_proj(output)
