"""Model registry — add new models here and in MODEL_REGISTRY."""

from .base import BaseModel
from .mlp import MLPModel
from .cnn import CNN1DModel
from .rnn import RNNModel
from .lstm import LSTMModel
from .gru import GRUModel
from .transformer import TransformerTabularModel
from .vanilla_transformer import VanillaTransformerWrapper
from .autoformer import AutoformerWrapper
from .informer import InformerWrapper
from .crossformer import CrossformerWrapper
from .dlinear import DLinearWrapper
from .etsformer import ETSformerWrapper
from .fedformer import FEDformerWrapper
from .film import FilmWrapper
from .frets import FreTSWrapper
from .itransformer import iTransformerWrapper
from .koopa import KoopaWrapper
from .lightts import LightTSWrapper
from .mamba_model import MambaWrapper
from .classical_ml import (
    RandomForestRegressorWrapper,
    RandomForestClassifierWrapper,
    XGBRegressorWrapper,
    XGBClassifierWrapper,
    LGBMRegressorWrapper,
    LGBMClassifierWrapper,
    DecisionTreeRegressorWrapper,
    DecisionTreeClassifierWrapper,
)

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
        "name": "Transformer (Tabular)",
        "pipeline": "small",
        "params": {
            "d_model": {"type": "int", "default": 64, "label": "Model dimension (d_model)"},
            "nhead": {"type": "int", "default": 4, "label": "Attention heads (nhead)"},
            "num_layers": {"type": "int", "default": 2, "label": "Encoder layers"},
            "dim_feedforward": {"type": "int", "default": 256, "label": "Feedforward dimension"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
        },
    },
    "vanilla_transformer": {
        "class": VanillaTransformerWrapper,
        "name": "Vanilla Transformer",
        "pipeline": "large",
        "params": {
            "d_model": {"type": "int", "default": 256, "label": "Model dimension (d_model)"},
            "n_heads": {"type": "int", "default": 8, "label": "Attention heads"},
            "e_layers": {"type": "int", "default": 3, "label": "Encoder layers"},
            "d_layers": {"type": "int", "default": 3, "label": "Decoder layers"},
            "d_ff": {"type": "int", "default": 32, "label": "Feedforward dimension"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
            "activation": {"type": "string", "default": "gelu", "label": "Activation (gelu/relu)"},
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
    "fedformer": {
        "class": FEDformerWrapper,
        "name": "FEDformer (Frequency Enhanced Decomp Transformer)",
        "pipeline": "large",
        "params": {
            "d_model": {"type": "int", "default": 256, "label": "Model dimension (d_model)"},
            "n_heads": {"type": "int", "default": 8, "label": "Attention heads"},
            "e_layers": {"type": "int", "default": 3, "label": "Encoder layers"},
            "d_layers": {"type": "int", "default": 3, "label": "Decoder layers"},
            "d_ff": {"type": "int", "default": 32, "label": "Feedforward dimension (d_ff)"},
            "moving_avg": {"type": "int", "default": 25, "label": "Moving average kernel"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
            "activation": {"type": "string", "default": "gelu", "label": "Activation (relu/gelu)"},
            "version": {"type": "string", "default": "Fourier", "label": "Frequency domain (Fourier/Wavelets)"},
            "mode_select": {"type": "string", "default": "random", "label": "Mode selection (random/low)"},
            "modes": {"type": "int", "default": 32, "label": "Number of frequency modes"},
        },
    },
    "film": {
        "class": FilmWrapper,
        "name": "FiLM (Frequency-enhanced Legendre Memory)",
        "pipeline": "large",
        "params": {
            "window_size": {"type": "string", "default": "256", "label": "HiPPO window sizes (comma-separated)"},
            "multiscale": {"type": "string", "default": "1,2,4", "label": "Multi-scale factors (comma-separated)"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
        },
    },
    "frets": {
        "class": FreTSWrapper,
        "name": "FreTS (Frequency-enhanced Time Series)",
        "pipeline": "large",
        "params": {
            "channel_independence": {"type": "str", "default": "0", "label": "Channel independence (0=enable, 1=disable)"},
            "embed_size": {"type": "int", "default": 128, "label": "Embedding size"},
            "hidden_size": {"type": "int", "default": 256, "label": "Hidden size"},
        },
    },
    "koopa": {
        "class": KoopaWrapper,
        "name": "Koopa (Koopman Forecasting)",
        "pipeline": "large",
        "params": {
            "dynamic_dim": {"type": "int", "default": 128, "label": "Koopman embedding dimension"},
            "hidden_dim": {"type": "int", "default": 64, "label": "MLP hidden dimension"},
            "hidden_layers": {"type": "int", "default": 2, "label": "MLP hidden layers"},
            "num_blocks": {"type": "int", "default": 3, "label": "Number of Koopa blocks"},
            "multistep": {"type": "bool", "default": False, "label": "Multistep K approximation"},
        },
    },
    "itransformer": {
        "class": iTransformerWrapper,
        "name": "iTransformer (Inverted Transformer)",
        "pipeline": "large",
        "params": {
            "d_model": {"type": "int", "default": 256, "label": "Model dimension"},
            "n_heads": {"type": "int", "default": 8, "label": "Attention heads"},
            "e_layers": {"type": "int", "default": 3, "label": "Encoder layers"},
            "d_ff": {"type": "int", "default": 32, "label": "Feedforward dimension"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
            "activation": {"type": "str", "default": "gelu", "label": "Activation"},
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
    # ── Classical ML (sklearn backend) ────────────────────────────────────
    "random_forest_regressor": {
        "class": RandomForestRegressorWrapper,
        "name": "Random Forest (Regression)",
        "pipeline": "small",
        "uses_sklearn_backend": True,
        "params": {
            "n_estimators": {"type": "int", "default": 100, "label": "Number of trees"},
            "max_depth": {"type": "int_or_none", "default": None, "label": "Max depth (None for unlimited)"},
            "min_samples_split": {"type": "int", "default": 2, "label": "Min samples split"},
            "min_samples_leaf": {"type": "int", "default": 1, "label": "Min samples leaf"},
        },
    },
    "random_forest_classifier": {
        "class": RandomForestClassifierWrapper,
        "name": "Random Forest (Classification)",
        "pipeline": "small",
        "uses_sklearn_backend": True,
        "params": {
            "n_estimators": {"type": "int", "default": 100, "label": "Number of trees"},
            "max_depth": {"type": "int_or_none", "default": None, "label": "Max depth (None for unlimited)"},
            "min_samples_split": {"type": "int", "default": 2, "label": "Min samples split"},
            "min_samples_leaf": {"type": "int", "default": 1, "label": "Min samples leaf"},
        },
    },
    "xgboost_regressor": {
        "class": XGBRegressorWrapper,
        "name": "XGBoost (Regression)",
        "pipeline": "small",
        "uses_sklearn_backend": True,
        "params": {
            "n_estimators": {"type": "int", "default": 100, "label": "Number of trees"},
            "max_depth": {"type": "int_or_none", "default": None, "label": "Max depth (None for unlimited)"},
            "min_samples_split": {"type": "int", "default": 2, "label": "Min samples split"},
            "min_samples_leaf": {"type": "int", "default": 1, "label": "Min samples leaf"},
        },
    },
    "xgboost_classifier": {
        "class": XGBClassifierWrapper,
        "name": "XGBoost (Classification)",
        "pipeline": "small",
        "uses_sklearn_backend": True,
        "params": {
            "n_estimators": {"type": "int", "default": 100, "label": "Number of trees"},
            "max_depth": {"type": "int_or_none", "default": None, "label": "Max depth (None for unlimited)"},
            "min_samples_split": {"type": "int", "default": 2, "label": "Min samples split"},
            "min_samples_leaf": {"type": "int", "default": 1, "label": "Min samples leaf"},
        },
    },
    "lightgbm_regressor": {
        "class": LGBMRegressorWrapper,
        "name": "LightGBM (Regression)",
        "pipeline": "small",
        "uses_sklearn_backend": True,
        "params": {
            "n_estimators": {"type": "int", "default": 100, "label": "Number of trees"},
            "max_depth": {"type": "int_or_none", "default": None, "label": "Max depth (None for unlimited)"},
            "min_samples_split": {"type": "int", "default": 2, "label": "Min samples split"},
            "min_samples_leaf": {"type": "int", "default": 1, "label": "Min samples leaf"},
        },
    },
    "lightts": {
        "class": LightTSWrapper,
        "name": "LightTS (Light Time Series)",
        "pipeline": "large",
        "params": {
            "d_model": {"type": "int", "default": 128, "label": "Model dimension"},
            "chunk_size": {"type": "int", "default": 24, "label": "Chunk size"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
        },
    },
    "mamba": {
        "class": MambaWrapper,
        "name": "Mamba (State Space Model)",
        "pipeline": "large",
        "params": {
            "d_model": {"type": "int", "default": 256, "label": "Model dimension"},
            "d_state": {"type": "int", "default": 16, "label": "State dimension"},
            "d_conv": {"type": "int", "default": 4, "label": "Convolution kernel"},
            "expand": {"type": "int", "default": 2, "label": "Expansion factor"},
            "dropout": {"type": "float", "default": 0.1, "label": "Dropout"},
        },
    },
    "lightgbm_classifier": {
        "class": LGBMClassifierWrapper,
        "name": "LightGBM (Classification)",
        "pipeline": "small",
        "uses_sklearn_backend": True,
        "params": {
            "n_estimators": {"type": "int", "default": 100, "label": "Number of trees"},
            "max_depth": {"type": "int_or_none", "default": None, "label": "Max depth (None for unlimited)"},
            "min_samples_split": {"type": "int", "default": 2, "label": "Min samples split"},
            "min_samples_leaf": {"type": "int", "default": 1, "label": "Min samples leaf"},
        },
    },
    "decision_tree_regressor": {
        "class": DecisionTreeRegressorWrapper,
        "name": "Decision Tree (Regression)",
        "pipeline": "small",
        "uses_sklearn_backend": True,
        "params": {
            "max_depth": {"type": "int_or_none", "default": None, "label": "Max depth (None for unlimited)"},
            "min_samples_split": {"type": "int", "default": 2, "label": "Min samples split"},
            "min_samples_leaf": {"type": "int", "default": 1, "label": "Min samples leaf"},
        },
    },
    "decision_tree_classifier": {
        "class": DecisionTreeClassifierWrapper,
        "name": "Decision Tree (Classification)",
        "pipeline": "small",
        "uses_sklearn_backend": True,
        "params": {
            "max_depth": {"type": "int_or_none", "default": None, "label": "Max depth (None for unlimited)"},
            "min_samples_split": {"type": "int", "default": 2, "label": "Min samples split"},
            "min_samples_leaf": {"type": "int", "default": 1, "label": "Min samples leaf"},
        },
    },
}


# ── Auto-register canvas-generated models on startup ──────────────
try:
    from utils.canvas_generator import register_all_generated
    register_all_generated()
except Exception:
    pass


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


def uses_sklearn_backend(model_type):
    """Return True if the model type uses sklearn backend (not PyTorch)."""
    entry = MODEL_REGISTRY.get(model_type, {})
    return entry.get("uses_sklearn_backend", False)


__all__ = [
    "BaseModel",
    "MLPModel",
    "CNN1DModel",
    "RNNModel",
    "LSTMModel",
    "GRUModel",
    "TransformerTabularModel",
    "VanillaTransformerWrapper",
    "AutoformerWrapper",
    "InformerWrapper",
    "CrossformerWrapper",
    "DLinearWrapper",
    "ETSformerWrapper",
    "FEDformerWrapper",
    "FilmWrapper",
    "RandomForestRegressorWrapper",
    "RandomForestClassifierWrapper",
    "XGBRegressorWrapper",
    "XGBClassifierWrapper",
    "LGBMRegressorWrapper",
    "LGBMClassifierWrapper",
    "DecisionTreeRegressorWrapper",
    "DecisionTreeClassifierWrapper",
    "MODEL_REGISTRY",
    "get_model_class",
    "get_model_names",
    "get_model_params",
    "get_model_pipeline",
    "get_large_model_types",
    "uses_sklearn_backend",
]
