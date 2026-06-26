"""PatchTST — Patch Time Series Transformer (Encoder-only).

Paper: https://arxiv.org/pdf/2211.14730.pdf
Pipeline: "small" (forward takes only x_enc).
"""

import torch
import torch.nn as nn
from .base import BaseModel
from .patchtst_layers import PatchEmbedding, FlattenHead
from .shared_layers.Transformer_EncDec import Encoder, EncoderLayer
from .shared_layers.SelfAttention_Family import FullAttention, AttentionLayer


class Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.transpose(*self.dims)


class PatchTSTWrapper(BaseModel):
    """PatchTST wrapper — Encoder-only, patch-based time series model."""

    pipeline = "small"
    uses_internal_normalization = True

    def __init__(self, input_dim, output_dim,
                 d_model=128, n_heads=16, e_layers=3, d_ff=256,
                 patch_len=16, stride=8,
                 dropout=0.2, activation="gelu",
                 seq_len=96, pred_len=1, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len
        padding = stride

        # Patch embedding
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)

        # Encoder (with BatchNorm1d, original PatchTST style)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout),
                                   d_model, n_heads),
                    d_model, d_ff, dropout, activation,
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)),
        )

        # Prediction head
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(input_dim, self.head_nf, pred_len, head_dropout=dropout)

        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Series stationarization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        # Patch + Embed → (B*C, patch_num, d_model)
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)

        # Encoder
        enc_out, _ = self.encoder(enc_out)

        # Reshape → (B, C, d_model, patch_num) → head → (B, pred_len, C)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out).permute(0, 2, 1)

        # Denormalize
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Project to output dim
        dec_out = self.output_proj(dec_out)
        return dec_out
