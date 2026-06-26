"""MICN — Multi-scale Isometric Convolution Network for time series.

Paper: https://openreview.net/pdf?id=zt53IDUR1U
Pipeline: "large" (forward takes x_enc, x_mark_enc, x_dec, x_mark_dec).
"""

import torch
import torch.nn as nn

from .base import BaseModel
from .shared_layers.Embed import DataEmbedding
from .micn_layers import series_decomp_multi, SeasonalPrediction


class MICNWrapper(BaseModel):
    """MICN wrapper for the DeepLearning Labs framework."""

    pipeline = "large"

    def __init__(self, input_dim, output_dim,
                 d_model=256, n_heads=8, d_layers=1, dropout=0.1,
                 conv_kernel="12,16",
                 seq_len=96, pred_len=1, n_time_features=4,
                 **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len
        self.seq_len = seq_len

        # Parse conv_kernel from comma-separated string
        conv_kernel_list = [int(k.strip()) for k in conv_kernel.split(",")]

        # Auto-compute decomp_kernel and isometric_kernel from conv_kernel
        decomp_kernel = []
        isometric_kernel = []
        for k in conv_kernel_list:
            if k % 2 == 0:
                decomp_kernel.append(k + 1)
                isometric_kernel.append((seq_len + pred_len + k) // k)
            else:
                decomp_kernel.append(k)
                isometric_kernel.append((seq_len + pred_len + k - 1) // k)

        # Multi-scale decomposition
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # Decoder embedding (seasonal init + x_mark_dec)
        self.dec_embedding = DataEmbedding(
            input_dim, d_model, n_time_features, dropout,
        )

        # Seasonal prediction with stacked MIC layers
        self.conv_trans = SeasonalPrediction(
            embedding_size=d_model, dropout=dropout,
            d_layers=d_layers, decomp_kernel=decomp_kernel,
            conv_kernel=conv_kernel_list,
            isometric_kernel=isometric_kernel,
            c_out=input_dim,  # seasonal output in enc_in space
        )

        # Trend regression: seq_len → pred_len
        self.regression = nn.Linear(seq_len, pred_len)
        self.regression.weight = nn.Parameter(
            (1 / pred_len) * torch.ones([pred_len, seq_len]),
            requires_grad=True)

        # Final projection: enc_in → output_dim
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Multi-scale decomposition: seasonal + trend
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        # Trend: (B, seq_len, enc_in) → (B, pred_len, enc_in)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # Prepare decoder input: last seq_len of seasonal + zeros for pred
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]],
                            device=x_enc.device)
        seasonal_init_dec = torch.cat([
            seasonal_init_enc[:, -self.seq_len:, :], zeros], dim=1)

        # Build x_mark_dec to match seasonal_init_dec length (seq_len + pred_len)
        # x_mark_enc has seq_len timesteps; x_mark_dec has label_len + pred_len
        dec_len = self.seq_len + self.pred_len
        pad = dec_len - self.seq_len  # = pred_len
        mark_zeros = torch.zeros(
            x_mark_enc.shape[0], pad, x_mark_enc.shape[2], device=x_mark_enc.device)
        dec_mark = torch.cat([x_mark_enc[:, -self.seq_len:, :], mark_zeros], dim=1)

        # Embedding → MIC layers
        dec_out = self.dec_embedding(seasonal_init_dec, dec_mark)
        dec_out = self.conv_trans(dec_out)

        # Combine seasonal output + trend, slice to pred_len
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        dec_out = self.output_proj(dec_out)  # (B, pred_len, output_dim)
        return dec_out
