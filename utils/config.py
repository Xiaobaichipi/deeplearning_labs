"""Centralized default configuration for training hyperparameters.

All training defaults live here instead of being scattered across
route handlers, model registries, and templates.
"""

import torch


def get_available_devices():
    """Return a list of available device strings for the frontend dropdown."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append(f"cuda:{i}  ({name})")
        if torch.cuda.device_count() > 1:
            devices.append("all  (DataParallel multi-GPU)")
    return devices


def parse_device(device_str: str) -> str | list[str]:
    """Parse user-selected device string into a device identifier.

    Returns a single device string (e.g. 'cpu', 'cuda:0') or a list of
    device strings for DataParallel multi-GPU.
    """
    s = device_str.strip()
    if s.startswith("all"):
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    # Strip trailing parenthetical info like "cuda:0  (NVIDIA XXX)"
    s = s.split("  (")[0].strip()
    return s


TRAINING = {
    "test_size": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "patience": 10,
    "dropout": 0.2,
    "normalization": "none",
}

MODEL = {
    "mlp": {"hidden_layers": "128,64,32"},
    "cnn": {"hidden_channels": 64, "kernel_size": 3},
    "rnn": {"hidden_size": 64, "num_layers": 2},
    "lstm": {"hidden_size": 64, "num_layers": 2},
    "gru": {"hidden_size": 64, "num_layers": 2},
    "transformer": {"d_model": 64, "nhead": 4, "dim_feedforward": 256, "num_layers": 2},
}

CV = {
    "default_folds": 5,
    "max_epochs_per_fold": 20,
    "fold_val_split": 0.2,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
