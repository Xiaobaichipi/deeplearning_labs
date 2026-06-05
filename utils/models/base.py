"""Base model class for all deep learning models."""

import torch.nn as nn


class BaseModel(nn.Module):
    """Abstract base class for all models in DeepLearning Labs.

    Subclasses must implement:
        __init__(self, input_dim, output_dim, **kwargs)
        forward(self, x)

    The registry in models/__init__.py maps string keys to model classes.
    Each model extracts its own parameters from the kwargs dict, falling back
    to sensible defaults for any missing keys.
    """

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")
