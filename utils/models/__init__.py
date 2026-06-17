"""Model registry — add new models here and in MODEL_REGISTRY."""

from .base import BaseModel
from .mlp import MLPModel
from .cnn import CNN1DModel
from .rnn import RNNModel
from .lstm import LSTMModel
from .gru import GRUModel
from .transformer import TransformerTabularModel
from .autoformer import AutoformerWrapper
from .informer import InformerWrapper
from .crossformer import CrossformerWrapper
from .dlinear import DLinearWrapper
from .etsformer import ETSformerWrapper

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
        "pipeline": "small",
        "params": {
            "hidden_layers": {"type": "string", "default": "128,64,32", "label": "Hidden layers (comma-separated)"},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout"},
        },
    },
    "cnn": {
        "class": CNN1DModel,
        "name": "CNN (1D Convolutional)",
        "pipeline": "small",
        "params": {
            "hidden_channels": {"type": "int", "default": 64, "label": "Hidden channels"},
            "kernel_size": {"type": "int", "default": 3, "label": "Kernel size"},
            "dropout": {"type": "float", "default": 0.2, "label": "Dropout"},
        },
    },
    "rnn": {
        "class": RNNModel,
        "name": "RNN (Vanilla RNN)",
        "pipeline": "small",
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
        "pipeline": "small",
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
        "pipeline": "small",
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
        "pipeline": "small",
        "params": {
            "d_model": {"type": "int", "default": 64, "label": "Model dimension (d_model)"},
            "nhead": {"type": "int", "default": 4, "label": "Attention heads (nhead)"},
            "num_layers": {"type": "int", "default": 2, "label": "Encoder layers"},
            "dim_feedforward": {"type": "int", "default": 256, "label": "Feedforward dimension"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
        },
    },
    "etsformer": {
        "class": ETSformerWrapper,
        "name": "ETSformer (Exp Smoothing Transformer)",
        "pipeline": "large",
        "params": {
            "d_model": {"type": "int", "default": 256, "label": "Model dimension (d_model)"},
            "n_heads": {"type": "int", "default": 8, "label": "Attention heads"},
            "e_layers": {"type": "int", "default": 2, "label": "Encoder layers (= Decoder layers)"},
            "d_ff": {"type": "int", "default": 32, "label": "Feedforward dimension (d_ff)"},
            "top_k": {"type": "int", "default": 5, "label": "Top-k Fourier frequencies"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
            "activation": {"type": "string", "default": "sigmoid", "label": "Activation (sigmoid/gelu/relu)"},
        },
    },
    "dlinear": {
        "class": DLinearWrapper,
        "name": "DLinear (Decomposition Linear)",
        "pipeline": "small",
        "params": {
            "moving_avg": {"type": "int", "default": 25, "label": "Moving average kernel"},
            "individual": {"type": "bool", "default": False, "label": "Individual channels"},
        },
    },
    "crossformer": {
        "class": CrossformerWrapper,
        "name": "Crossformer (Two-Stage Attention)",
        "pipeline": "large",
        "params": {
            "d_model": {"type": "int", "default": 256, "label": "Model dimension (d_model)"},
            "n_heads": {"type": "int", "default": 8, "label": "Attention heads"},
            "e_layers": {"type": "int", "default": 3, "label": "Encoder layers"},
            "d_ff": {"type": "int", "default": 32, "label": "Feedforward dimension (d_ff)"},
            "factor": {"type": "int", "default": 3, "label": "Attention factor (top-k)"},
            "seg_len": {"type": "int", "default": 12, "label": "Segment length (seg_len)"},
            "win_size": {"type": "int", "default": 2, "label": "Merge window (win_size)"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
            "activation": {"type": "string", "default": "gelu", "label": "Activation (relu/gelu)"},
        },
    },
    "autoformer": {
        "class": AutoformerWrapper,
        "name": "Autoformer (Long-term Forecast)",
        "pipeline": "large",
        "params": {
            "d_model": {"type": "int", "default": 256, "label": "Model dimension (d_model)"},
            "n_heads": {"type": "int", "default": 8, "label": "Attention heads"},
            "e_layers": {"type": "int", "default": 3, "label": "Encoder layers"},
            "d_layers": {"type": "int", "default": 3, "label": "Decoder layers"},
            "d_ff": {"type": "int", "default": 32, "label": "Feedforward dimension (d_ff)"},
            "moving_avg": {"type": "int", "default": 25, "label": "Moving average kernel"},
            "factor": {"type": "int", "default": 3, "label": "Attention factor (top-k)"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
            "activation": {"type": "string", "default": "gelu", "label": "Activation (relu/gelu)"},
        },
    },
    "informer": {
        "class": InformerWrapper,
        "name": "Informer (ProbSparse Attention)",
        "pipeline": "large",
        "params": {
            "d_model": {"type": "int", "default": 256, "label": "Model dimension (d_model)"},
            "n_heads": {"type": "int", "default": 8, "label": "Attention heads"},
            "e_layers": {"type": "int", "default": 3, "label": "Encoder layers"},
            "d_layers": {"type": "int", "default": 3, "label": "Decoder layers"},
            "d_ff": {"type": "int", "default": 32, "label": "Feedforward dimension (d_ff)"},
            "factor": {"type": "int", "default": 3, "label": "Attention factor (top-k)"},
            "distil": {"type": "bool", "default": True, "label": "Distillation (ConvLayer)"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
            "activation": {"type": "string", "default": "gelu", "label": "Activation (relu/gelu)"},
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


def get_model_pipeline(model_type):
    """Return the pipeline type ('small' or 'large') for a given model type."""
    if model_type not in MODEL_REGISTRY:
        return "small"
    return MODEL_REGISTRY[model_type].get("pipeline", "small")


def get_large_model_types():
    """Return list of model type strings whose pipeline is 'large'."""
    return [k for k, v in MODEL_REGISTRY.items() if v.get("pipeline") == "large"]


__all__ = [
    "BaseModel",
    "MLPModel",
    "CNN1DModel",
    "RNNModel",
    "LSTMModel",
    "GRUModel",
    "TransformerTabularModel",
    "AutoformerWrapper",
    "InformerWrapper",
    "CrossformerWrapper",
    "DLinearWrapper",
    "ETSformerWrapper",
    "MODEL_REGISTRY",
    "get_model_class",
    "get_model_names",
    "get_model_params",
    "get_model_pipeline",
    "get_large_model_types",
]
