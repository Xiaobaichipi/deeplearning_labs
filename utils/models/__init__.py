"""Model registry — add new models here and in MODEL_REGISTRY."""

from .base import BaseModel
from .mlp import MLPModel
from .cnn import CNN1DModel
from .rnn import RNNModel
from .lstm import LSTMModel
from .gru import GRUModel
from .transformer import TransformerTabularModel

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
# To add a new model:
#   1. Create a new file in utils/models/ (e.g., attention_rnn.py)
#   2. Define a class inheriting from BaseModel
#   3. Import it above and add an entry below.
#
#   Key:     string used in the frontend dropdown and API request
#   Value:   the model class
#   Params:  dict describing the hyperparameters the model needs
#            (used by the frontend config UI and the backend param builder)
#
# Each model class receives input_dim, output_dim, and a **kwargs dict.
# Extract your parameters from kwargs with sensible defaults.
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "mlp": {
        "class": MLPModel,
        "name": "MLP (Fully Connected)",
        "params": {
            "hidden_layers": {"type": "string", "default": "128,64,32", "label": "Hidden layers (comma-separated)"},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout"},
        },
    },
    "cnn": {
        "class": CNN1DModel,
        "name": "CNN (1D Convolutional)",
        "params": {
            "hidden_channels": {"type": "int", "default": 64, "label": "Hidden channels"},
            "kernel_size": {"type": "int", "default": 3, "label": "Kernel size"},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout"},
        },
    },
    "rnn": {
        "class": RNNModel,
        "name": "RNN (Vanilla RNN)",
        "params": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size"},
            "num_layers": {"type": "int", "default": 2, "label": "Number of layers"},
            "bidirectional": {"type": "bool", "default": False, "label": "Bidirectional"},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout"},
        },
    },
    "lstm": {
        "class": LSTMModel,
        "name": "LSTM (Long Short-Term Memory)",
        "params": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size"},
            "num_layers": {"type": "int", "default": 2, "label": "Number of layers"},
            "bidirectional": {"type": "bool", "default": False, "label": "Bidirectional"},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout"},
        },
    },
    "gru": {
        "class": GRUModel,
        "name": "GRU (Gated Recurrent Unit)",
        "params": {
            "hidden_size": {"type": "int", "default": 64, "label": "Hidden size"},
            "num_layers": {"type": "int", "default": 2, "label": "Number of layers"},
            "bidirectional": {"type": "bool", "default": False, "label": "Bidirectional"},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout"},
        },
    },
    "transformer": {
        "class": TransformerTabularModel,
        "name": "Transformer (Encoder)",
        "params": {
            "d_model": {"type": "int", "default": 64, "label": "Model dimension (d_model)"},
            "nhead": {"type": "int", "default": 4, "label": "Attention heads (nhead)"},
            "num_layers": {"type": "int", "default": 2, "label": "Encoder layers"},
            "dim_feedforward": {"type": "int", "default": 256, "label": "Feedforward dimension"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
        },
    },
}


def get_model_class(model_type):
    """Return the model class for the given type string."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]["class"]


def get_model_names():
    """Return {key: display_name} for all registered models."""
    return {k: v["name"] for k, v in MODEL_REGISTRY.items()}


def get_model_params(model_type):
    """Return the param schema dict for a given model type."""
    if model_type not in MODEL_REGISTRY:
        return {}
    return MODEL_REGISTRY[model_type]["params"]


__all__ = [
    "BaseModel",
    "MLPModel",
    "CNN1DModel",
    "RNNModel",
    "LSTMModel",
    "GRUModel",
    "TransformerTabularModel",
    "MODEL_REGISTRY",
    "get_model_class",
    "get_model_names",
    "get_model_params",
]
