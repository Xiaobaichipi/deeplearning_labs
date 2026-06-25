import torch.nn as nn
from .base import BaseModel
from .itransformer_layers import (
    Encoder, EncoderLayer,
    FullAttention, AttentionLayer,
    DataEmbedding_inverted,
)


class iTransformerWrapper(BaseModel):
    """iTransformer — inverted Transformer treating variates as tokens.

    Pipeline: "large".  Encoder-only.  Decoder args ignored.
    """
    pipeline = "large"

    def __init__(self, input_dim, output_dim,
                 d_model=256, n_heads=8, e_layers=3, d_ff=32,
                 dropout=0.1, activation="gelu",
                 seq_len=96, pred_len=1, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self.pred_len = pred_len

        # Inverted embedding: (B, T, N) → (B, N, d_model)
        self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout)

        # Encoder stack — FullAttention across variate dimension
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout),
                        d_model, n_heads,
                    ),
                    d_model, d_ff, dropout, activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Project d_model → pred_len for each variate
        self.projection = nn.Linear(d_model, pred_len)
        # Final projection: enc_in → output_dim
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: (batch, seq_len, enc_in)
        # Embed: ignore x_mark to avoid variate-dim concatenation
        x = self.embedding(x_enc, None)

        # Encoder — processes across variates
        x, _ = self.encoder(x)  # (batch, enc_in, d_model)

        # Project d_model → pred_len per variate
        x = self.projection(x)  # (batch, enc_in, pred_len)

        # Permute → (batch, pred_len, enc_in), then project → (batch, pred_len, 1)
        x = x.permute(0, 2, 1)  # (batch, pred_len, enc_in)
        x = self.output_proj(x)  # (batch, pred_len, output_dim)
        return x
