"""LightTS — Light Time Series model with interval/continuous sampling.

Paper: https://arxiv.org/abs/2207.01186
Pipeline: "large" (forward takes x_enc, x_mark_enc, x_dec, x_mark_dec).
"""

import torch
import torch.nn as nn

from .base import BaseModel


class IEBlock(nn.Module):
    """Interval / continuous sampling block."""
    def __init__(self, input_dim, hid_dim, output_dim, num_node):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim // 4),
        )
        self.channel_proj = nn.Linear(num_node, num_node)
        nn.init.eye_(self.channel_proj.weight)
        self.output_proj = nn.Linear(hid_dim // 4, output_dim)

    def forward(self, x):
        # x: (B, chunk_size, num_chunks) after reshape
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class LightTSWrapper(BaseModel):
    """LightTS wrapper for the DeepLearning Labs framework."""

    pipeline = "large"

    def __init__(self, input_dim, output_dim,
                 seq_len=96, pred_len=1,
                 d_model=128, chunk_size=24, dropout=0.1,
                 **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.chunk_size = min(pred_len, seq_len, chunk_size)
        if self.seq_len % self.chunk_size != 0:
            self.seq_len += self.chunk_size - self.seq_len % self.chunk_size
        self.num_chunks = self.seq_len // self.chunk_size
        self.d_model = d_model

        # Layer stack
        self.layer_1 = IEBlock(
            input_dim=self.chunk_size, hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4, num_node=self.num_chunks,
        )
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = IEBlock(
            input_dim=self.chunk_size, hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4, num_node=self.num_chunks,
        )
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = IEBlock(
            input_dim=self.d_model // 2, hid_dim=self.d_model // 2,
            output_dim=self.pred_len, num_node=self.input_dim,
        )

        self.ar = nn.Linear(self.seq_len, self.pred_len)
        self.output_proj = nn.Linear(self.input_dim, output_dim)

    def encoder(self, x):
        B, T, N = x.shape
        if T < self.seq_len:
            x = torch.cat([x, torch.zeros(B, self.seq_len - T, N, device=x.device)], dim=1)

        highway = self.ar(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, pred_len, N)

        # Continuous sampling: (B, num_chunks, chunk_size, N)
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1).reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(-1)  # (B*N, chunk_size)

        # Interval sampling
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2).reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(-1)

        x3 = torch.cat([x1, x2], dim=-1)  # (B*N, chunk_size*2 = d_model//2)
        x3 = x3.reshape(B, N, -1).permute(0, 2, 1)  # (B, d_model//2, N)

        out = self.layer_3(x3)  # (B, pred_len, N)
        return out + highway

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.encoder(x_enc)
        dec_out = dec_out[:, -self.pred_len:, :]  # (B, pred_len, N)
        dec_out = self.output_proj(dec_out)       # (B, pred_len, output_dim)
        return dec_out
