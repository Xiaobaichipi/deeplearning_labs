"""1D Convolutional Neural Network (CNN) model."""

from torch import nn
from .base import BaseModel


class CNN1DModel(BaseModel):
    """1D CNN for tabular data. Treats features as a 1D signal."""

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        hidden_channels = kwargs.get("hidden_channels", 64)
        kernel_size = kwargs.get("kernel_size", 3)
        dropout = kwargs.get("dropout", 0.2)

        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.conv(x).squeeze(-1)
        x = self.fc(x)
        return x
