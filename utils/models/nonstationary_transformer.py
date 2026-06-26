"""Nonstationary Transformer — de-stationary attention with learned tau/delta.

Pipeline: "large".  Uses internal normalization (series stationarization).
"""

import torch
import torch.nn as nn
from .base import BaseModel
from .shared_layers.Embed import DataEmbedding
from .shared_layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from .shared_layers.SelfAttention_Family import AttentionLayer
from .nonstationary_layers import DSAttention


class Projector(nn.Module):
    """MLP that learns de-stationary factors tau and delta."""
    def __init__(self, enc_in, seq_len, hidden_dims=None, hidden_layers=2, output_dim=1, kernel_size=3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        padding = 1
        self.series_conv = nn.Conv1d(
            seq_len, 1, kernel_size=kernel_size,
            padding=padding, padding_mode='circular', bias=False)
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        B = x.shape[0]
        x = self.series_conv(x)           # (B, 1, E)
        x = torch.cat([x, stats], dim=1)  # (B, 2, E)
        x = x.view(B, -1)                 # (B, 2E)
        return self.backbone(x)           # (B, output_dim)


class NonstationaryTransformerWrapper(BaseModel):
    """Nonstationary Transformer wrapper."""

    pipeline = "large"
    uses_internal_normalization = True

    def __init__(self, input_dim, output_dim,
                 d_model=256, n_heads=8, e_layers=3, d_layers=3, d_ff=32,
                 dropout=0.1, activation="gelu",
                 seq_len=96, label_len=0, pred_len=1,
                 p_hidden_dims="128,128", p_hidden_layers=2,
                 n_time_features=4, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len
        self.label_len = label_len
        self.seq_len = seq_len

        p_dims = [int(d) for d in p_hidden_dims.split(",")]

        # Encoder
        self.enc_embedding = DataEmbedding(input_dim, d_model, n_time_features, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, attention_dropout=dropout),
                        d_model, n_heads),
                    d_model, d_ff, dropout, activation,
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Decoder
        self.dec_embedding = DataEmbedding(input_dim, d_model, n_time_features, dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(DSAttention(True, attention_dropout=dropout), d_model, n_heads),
                    AttentionLayer(DSAttention(False, attention_dropout=dropout), d_model, n_heads),
                    d_model, d_ff, dropout, activation,
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, output_dim, bias=True),
        )

        # De-stationary factor learners
        self.tau_learner = Projector(
            enc_in=input_dim, seq_len=seq_len,
            hidden_dims=p_dims, hidden_layers=p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(
            enc_in=input_dim, seq_len=seq_len,
            hidden_dims=p_dims, hidden_layers=p_hidden_layers, output_dim=seq_len)

        self.output_proj = nn.Linear(input_dim, output_dim)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()

        # Series stationarization
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # Learn de-stationary factors
        tau = self.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        # Prepare decoder input — guard against label_len=0 (-0 == 0 in Python)
        if self.label_len > 0:
            x_dec_new = torch.cat([
                x_enc[:, -self.label_len:, :],
                torch.zeros_like(x_dec[:, -self.pred_len:, :]),
            ], dim=1)
        else:
            x_dec_new = torch.zeros_like(x_dec[:, -self.pred_len:, :])

        # Encoder-Decoder with de-stationary attention
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # Align x_mark_dec length with x_dec_new (handles label_len mismatches)
        dec_mark = x_mark_dec[:, :x_dec_new.shape[1], :]
        dec_out = self.dec_embedding(x_dec_new, dec_mark)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                               tau=tau, delta=delta)

        # Denormalize
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        out = out[:, -self.pred_len:, :]    # (B, pred_len, enc_in)
        # Ensure last dim matches output_proj in_features (handles broadcast from denorm)
        if out.size(-1) != self.output_proj.in_features:
            out = out[:, :, :self.output_proj.in_features]
        out = self.output_proj(out)         # (B, pred_len, output_dim)
        return out
