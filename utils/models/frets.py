"""
FreTS — Frequency-enhanced Time Series model.

Paper: https://arxiv.org/pdf/2311.06184.pdf
Pipeline: "large" (forward takes x_enc, x_mark_enc, x_dec, x_mark_dec)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class FreTSWrapper(BaseModel):
    """FreTS wrapper for the DeepLearning Labs framework."""

    pipeline = "large"
    uses_internal_normalization = False

    def __init__(self, input_dim, output_dim,
                 seq_len=96, pred_len=1, channel_independence=0,
                 embed_size=128, hidden_size=256,
                 **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.feature_size = input_dim
        self.seq_len = seq_len
        # channel_independence: 0 = enable channel MLP, 1 = disable
        self.channel_independence = str(int(channel_independence))
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

        # Project output_dim → 1 for compatibility with LargePipelineStrategy
        self.output_proj = nn.Linear(self.feature_size, output_dim)

    # ── Internal FreTS methods ─────────────────────────────────────

    def tokenEmb(self, x):
        """Dimension extension: [B, T, N] → [B, N, T, D]"""
        x = x.permute(0, 2, 1).unsqueeze(3)
        return x * self.embeddings

    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        """Frequency-domain MLP with real/imaginary weights."""
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size], device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - torch.einsum('bijd,dd->bijd', x.imag, i) + rb)
        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + torch.einsum('bijd,dd->bijd', x.real, i) + ib)

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        return torch.view_as_complex(y)

    def MLP_temporal(self, x, B, N, L):
        """Frequency temporal learner — FFT along time dimension."""
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        return torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")

    def MLP_channel(self, x, B, N, L):
        """Frequency channel learner — FFT along channel dimension."""
        x = x.permute(0, 2, 1, 3)  # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        return x.permute(0, 2, 1, 3)  # [B, N, T, D]

    def forecast(self, x_enc):
        B, T, N = x_enc.shape
        x = self.tokenEmb(x_enc)  # [B, N, T, D]
        bias = x
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)  # [B, pred_len, N]
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc)
        # Select last pred_len and project to output dimension
        dec_out = dec_out[:, -self.pred_len:, :]  # [B, pred_len, enc_in]
        dec_out = self.output_proj(dec_out)       # [B, pred_len, output_dim]
        return dec_out  # LargePipelineStrategy.format_output squeezes last dim
