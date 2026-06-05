"""LSTM model using nn.LSTM."""

from torch import nn
from .base import BaseModel


class LSTMModel(BaseModel):
    """Long Short-Term Memory network for capturing long-range dependencies."""

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        hidden_size = kwargs.get("hidden_size", 64)
        num_layers = kwargs.get("num_layers", 2)
        bidirectional = kwargs.get("bidirectional", False)
        dropout = kwargs.get("dropout", 0.2)

        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_directions, output_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
