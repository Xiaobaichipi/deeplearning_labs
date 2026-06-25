"""Mamba — State Space Model for time series forecasting.

Paper: https://arxiv.org/abs/2312.00752
Pipeline: "large" (forward takes x_enc, x_mark_enc, x_dec, x_mark_dec).
Depends on ``mamba-ssm``: pip install mamba-ssm causal-conv1d
"""

import torch
import torch.nn as nn

from .base import BaseModel
from .shared_layers.Embed import DataEmbedding


# Lazy import with friendly error
_MAMBA_AVAILABLE = False
_MAMBA_ERR = None
try:
    from mamba_ssm import Mamba as _MambaBlock
    _MAMBA_AVAILABLE = True
except ImportError as e:
    _MAMBA_ERR = str(e)


class MambaWrapper(BaseModel):
    """Mamba wrapper for the DeepLearning Labs framework."""

    pipeline = "large"
    uses_internal_normalization = True

    def __init__(self, input_dim, output_dim,
                 d_model=256, d_state=16, d_conv=4, expand=2,
                 dropout=0.1, pred_len=1, n_time_features=4,
                 **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len

        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "Mamba requires ``mamba-ssm``.\n"
                "  pip install mamba-ssm causal-conv1d\n"
                f"  (underlying error: {_MAMBA_ERR})"
            )

        self.embedding = DataEmbedding(
            c_in=input_dim, d_model=d_model,
            n_time_features=n_time_features, dropout=dropout,
        )

        self.mamba = _MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.out_layer = nn.Linear(d_model, output_dim, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x = self.out_layer(x)

        return x * std_enc + mean_enc

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out = self.forecast(x_enc, x_mark_enc)
        return out[:, -self.pred_len:, :]
