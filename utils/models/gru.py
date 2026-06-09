"""GRU model using nn.GRU."""

from torch import nn
from .base import BaseModel


class GRUModel(BaseModel):
    """Gated Recurrent Unit — lighter alternative to LSTM with similar performance."""

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        hidden_size = kwargs.get("hidden_size", 64)
        num_layers = kwargs.get("num_layers", 2)
        bidirectional = kwargs.get("bidirectional", False)
        dropout = kwargs.get("dropout", 0.2)

        self.num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_directions, output_dim),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
