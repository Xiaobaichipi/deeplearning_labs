"""Series decomposition blocks — extracted from Autoformer layers for MICN."""

import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block."""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class series_decomp_multi(nn.Module):
    """Multiple Series decomposition block from FEDformer."""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(k) for k in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, trend = func(x)
            res.append(sea.unsqueeze(0))
            moving_mean.append(trend.unsqueeze(0))
        sea = torch.mean(torch.cat(res, dim=0), dim=0)
        trend = torch.mean(torch.cat(moving_mean, dim=0), dim=0)
        return sea, trend
