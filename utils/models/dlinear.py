"""DLinear model — small-pipeline wrapper.

Series decomposition + two linear layers (seasonal + trend).
Paper: https://arxiv.org/pdf/2205.13504.pdf

Pipeline: "small" — forward(x) → (batch, pred_len).
Unlike Autoformer/Informer/Crossformer, DLinear does NOT use time features
or decoder input, so it fits the small pipeline naturally.
"""

import torch
import torch.nn as nn
from types import SimpleNamespace

from .base import BaseModel


# ---------------------------------------------------------------------------
#  Series decomposition (from Autoformer — copied inline to avoid a
#  dependency on autoformer_layers for 30 lines of code)
# ---------------------------------------------------------------------------


class _MovingAvg(nn.Module):
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


class _SeriesDecomp(nn.Module):
    """Series decomposition block — seasonal + trend."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ===================================================================
#  Raw DLinear model
# ===================================================================


class _RawDLinear(nn.Module):
    """DLinear — series decomposition + two Linear layers.

    Adapted from ``time_series_models_labs/models/DLinear.py``.
    Output shape: (batch, pred_len, enc_in) — multivariate.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.individual = configs.individual

        self.decomp = _SeriesDecomp(configs.moving_avg)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.enc_in):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # Weight initialisation from the paper — start with mean prediction.
        self._init_weights()

    def _init_weights(self):
        if self.individual:
            for i in range(self.enc_in):
                nn.init.constant_(self.Linear_Seasonal[i].weight, 1.0 / self.seq_len)
                nn.init.constant_(self.Linear_Trend[i].weight, 1.0 / self.seq_len)
        else:
            nn.init.constant_(self.Linear_Seasonal.weight, 1.0 / self.seq_len)
            nn.init.constant_(self.Linear_Trend.weight, 1.0 / self.seq_len)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor (batch, seq_len, enc_in)
            Value features (including target column).

        Returns
        -------
        torch.Tensor (batch, pred_len, enc_in)
            Forecast per channel (multivariate).
        """
        # Series decomposition
        seasonal_init, trend_init = self.decomp(x)          # (batch, seq_len, enc_in)

        # Permute so Linear operates on the time dimension
        seasonal_init = seasonal_init.permute(0, 2, 1)      # (batch, enc_in, seq_len)
        trend_init = trend_init.permute(0, 2, 1)            # (batch, enc_in, seq_len)

        if self.individual:
            seasonal_output = torch.zeros(
                seasonal_init.size(0), seasonal_init.size(1), self.pred_len,
                dtype=seasonal_init.dtype, device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                trend_init.size(0), trend_init.size(1), self.pred_len,
                dtype=trend_init.dtype, device=trend_init.device,
            )
            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)   # (batch, enc_in, pred_len)
            trend_output = self.Linear_Trend(trend_init)            # (batch, enc_in, pred_len)

        x = seasonal_output + trend_output                          # (batch, enc_in, pred_len)
        return x.permute(0, 2, 1)                                   # (batch, pred_len, enc_in)


# ===================================================================
#  DLinearWrapper — public-facing class registered in MODEL_REGISTRY
# ===================================================================


class DLinearWrapper(BaseModel):
    """DLinear wrapper compatible with the two-pipeline system.

    Pipeline: "small" — forward(x) → (batch, pred_len).

    DLinear does NOT rely on time features, so it uses ``SmallPipelineStrategy``
    which feeds the model ``(batch, seq_len, enc_in)`` directly — no need for
    ``PipelineData`` or 4-argument forward.

    A final linear projection ``nn.Linear(enc_in, 1)`` maps the multivariate
    output ``(batch, pred_len, enc_in)`` down to the single-target convention.
    """

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        seq_len = kwargs.get("seq_len", 96)

        configs = SimpleNamespace(
            enc_in=input_dim,
            seq_len=seq_len,
            pred_len=output_dim,
            moving_avg=kwargs.get("moving_avg", 25),
            individual=kwargs.get("individual", False),
        )
        self._configs = configs
        self._model = _RawDLinear(configs)

        # Project multivariate output (enc_in) → single target (1)
        self.output_proj = nn.Linear(input_dim, 1)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor (batch, seq_len, enc_in)
            Value features (including target column).

        Returns
        -------
        torch.Tensor (batch, pred_len)
            Target predictions.
        """
        output = self._model(x)             # (batch, pred_len, enc_in)
        output = self.output_proj(output)   # (batch, pred_len, 1)
        return output.squeeze(-1)           # (batch, pred_len)
