"""Multi-Layer Perceptron (MLP) model."""

from torch import nn
from .base import BaseModel


class MLPModel(BaseModel):
    """Fully-connected feedforward network with configurable hidden layers."""

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        hidden_layers = kwargs.get("hidden_layers", [128, 64, 32])
        dropout = kwargs.get("dropout", 0.2)

        layers = []
        prev = input_dim
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            batch, seq_len, nf = x.shape
            x = x.view(batch * seq_len, nf)   # (B*S, nf)
            x = self.net(x)                    # (B*S, output_dim)
            x = x.view(batch, seq_len, -1)     # (B, S, output_dim)
            x = x.mean(dim=1)                  # (B, output_dim)
        else:
            x = self.net(x)
        return x
