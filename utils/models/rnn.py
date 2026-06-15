"""Vanilla RNN model using nn.RNN."""

from torch import nn
from .base import BaseModel


class RNNModel(BaseModel):
    """Vanilla RNN with tanh activation for sequential feature processing."""

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        hidden_size = kwargs.get("hidden_size", 64)
        num_layers = kwargs.get("num_layers", 2)
        bidirectional = kwargs.get("bidirectional", False)
        dropout = kwargs.get("dropout", 0.2)

        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.RNN(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity="tanh",
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_directions, output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
