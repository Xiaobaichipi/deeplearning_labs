"""ETSformer model — large-pipeline wrapper.

Based on the original implementation from
``time_series_models_labs/models/ETSformer.py``, adapted to the two-pipeline
system with ``pipeline = "large"``.

Key innovation: exponential smoothing attention + Fourier seasonal
decomposition, replacing the standard self-attention with growth/seasonal/level
decomposition.
"""

import torch
import torch.nn as nn
from types import SimpleNamespace

from .base import BaseModel
from .etsformer_layers import (
    DataEmbedding, Encoder, EncoderLayer, Decoder, DecoderLayer, Transform,
)


class _RawETSformer(nn.Module):
    """Raw ETSformer model — forecast only.

    Adapted from ``time_series_models_labs/models/ETSformer.py``.
    Only supports the ``long_term_forecast`` task.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Embedding (n_time_features-aware, with positional encoding)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.n_time_features, configs.dropout)

        # Encoder (d_layers == e_layers is enforced by the wrapper)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    d_model=configs.d_model,
                    nhead=configs.n_heads,
                    c_out=configs.c_out,
                    seq_len=configs.seq_len,
                    pred_len=configs.pred_len,
                    k=configs.top_k,
                    dim_feedforward=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ]
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    d_model=configs.d_model,
                    nhead=configs.n_heads,
                    c_out=configs.c_out,
                    pred_len=configs.pred_len,
                    dropout=configs.dropout,
                )
                for _ in range(configs.e_layers)  # d_layers == e_layers
            ]
        )

        # Data augmentation (training only)
        self.transform = Transform(sigma=0.2)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        with torch.no_grad():
            if self.training:
                x_enc = self.transform.transform(x_enc)

        res = self.enc_embedding(x_enc, x_mark_enc)
        level, growths, seasons = self.encoder(res, x_enc, attn_mask=None)
        growth, season = self.decoder(growths, seasons)
        preds = level[:, -1:] + growth + season
        return preds

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # (batch, pred_len, c_out)


class ETSformerWrapper(BaseModel):
    """ETSformer wrapper compatible with the two-pipeline system.

    Pipeline: "large" — requires 4-argument forward::

        forward(x_enc, x_mark_enc, x_dec, x_mark_dec) → (batch, pred_len, 1)

    Internally uses ``c_out == input_dim`` (ETSformer architecture requires
    ``c_out`` to match the level dimension), then projects to ``output_dim=1``
    via a linear output projection layer.

    Training-time data augmentation (jitter + scale + shift) is applied
    inside ``_RawETSformer.forecast()`` when ``self.training`` is ``True``.
    """

    pipeline = "large"

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)

        seq_len = kwargs.get("seq_len", 96)
        pred_len = output_dim
        n_time_features = kwargs.get("n_time_features", 4)
        e_layers = kwargs.get("e_layers", 2)

        # ETSformer requires c_out == enc_in internally (LevelLayer.view)
        c_out = input_dim

        configs = SimpleNamespace(
            # dimensions
            enc_in=input_dim,
            dec_in=input_dim,
            c_out=c_out,
            n_time_features=n_time_features,
            # sequence lengths
            seq_len=seq_len,
            pred_len=pred_len,
            # architecture
            d_model=kwargs.get("d_model", 256),
            n_heads=kwargs.get("n_heads", 8),
            e_layers=e_layers,
            d_layers=e_layers,  # enforced: d_layers == e_layers
            d_ff=kwargs.get("d_ff", 32),
            top_k=kwargs.get("top_k", 5),
            dropout=kwargs.get("dropout", 0.1),
            activation=kwargs.get("activation", "sigmoid"),
        )
        self._configs = configs
        self._model = _RawETSformer(configs)

        # Project from input_dim (internal c_out) → 1 (target column)
        self.output_proj = nn.Linear(input_dim, 1)

    def forward(self, batch_x, batch_x_mark, batch_dec_inp, batch_y_mark):
        """Forward pass.

        Parameters
        ----------
        batch_x : torch.Tensor (batch, seq_len, enc_in)
            Value features (including target column) for the encoder window.
        batch_x_mark : torch.Tensor (batch, seq_len, n_time_features)
            Time features for the encoder window.
        batch_dec_inp : torch.Tensor (batch, label_len+pred_len, dec_in)
            Decoder input — required by large pipeline signature, not used
            by ETSformer (it generates its own decoder input from encoder).
        batch_y_mark : torch.Tensor (batch, label_len+pred_len, n_time_features)
            Time features for the decoder window — not used by ETSformer.

        Returns
        -------
        torch.Tensor (batch, pred_len, 1)
            Target predictions.
        """
        out = self._model(batch_x, batch_x_mark, batch_dec_inp, batch_y_mark)
        # out: (batch, pred_len, input_dim) → (batch, pred_len, 1)
        return self.output_proj(out)
