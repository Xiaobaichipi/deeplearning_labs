"""Transformer Encoder model for tabular data."""

from torch import nn
from .base import BaseModel


class TransformerTabularModel(BaseModel):
    """Transformer Encoder — self-attention mechanism for tabular data.

    Projects input features to a latent dimension, applies positional encoding,
    passes through TransformerEncoder layers, and outputs predictions.
    """

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        d_model = kwargs.get("d_model", 64)
        nhead = kwargs.get("nhead", 4)
        num_layers = kwargs.get("num_layers", 2)
        dim_feedforward = kwargs.get("dim_feedforward", 256)
        dropout = kwargs.get("dropout", 0.1)

        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)          # (batch, seq_len=1, d_model)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)            # (batch, d_model)
        x = self.norm(x)
        x = self.fc(x)
        return x
