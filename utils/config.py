"""Centralized default configuration for training hyperparameters.

All training defaults live here instead of being scattered across
route handlers, model registries, and templates.
"""

import subprocess
import re
import torch


def _detect_gpus_via_nvidia_smi():
    """Fallback: detect GPUs via nvidia-smi when torch.cuda is unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(", ", 1)
            if len(parts) == 2:
                gpus.append((int(parts[0]), parts[1]))
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return []


def get_available_devices():
    """Return a list of available device strings for the frontend dropdown."""
    cudnn_note = ""
    if not torch.backends.cudnn.enabled:
        cudnn_note = " (cuDNN disabled)"
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append(f"cuda:{i}  ({name}{cudnn_note})")
        if torch.cuda.device_count() > 1:
            devices.append(f"all  (DataParallel multi-GPU{cudnn_note})")
    else:
        gpus = _detect_gpus_via_nvidia_smi()
        if gpus:
            for idx, name in gpus:
                devices.append(f"cuda:{idx}  ({name}) (driver incompatible)")
            if len(gpus) > 1:
                devices.append("all  (DataParallel multi-GPU) (driver incompatible)")
    return devices


def parse_device(device_str: str) -> str | list[str]:
    """Parse user-selected device string into a device identifier.

    Returns a single device string (e.g. 'cpu', 'cuda:0') or a list of
    device strings for DataParallel multi-GPU.
    """
    s = device_str.strip()
    if s.startswith("all"):
        return [f"cuda:{i}" for i in range(_gpu_count())]
    # Strip trailing parenthetical info like "cuda:0  (NVIDIA XXX)"
    s = s.split("  (")[0].strip()
    return s


def _gpu_count() -> int:
    """Return the number of GPUs, using torch or nvidia-smi fallback."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return len(_detect_gpus_via_nvidia_smi())


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
    "autoformer": {"d_model": 256, "n_heads": 8, "e_layers": 3, "d_layers": 3, "d_ff": 32,
                   "moving_avg": 25, "factor": 3, "dropout": 0.1, "activation": "gelu"},
    "informer": {"d_model": 256, "n_heads": 8, "e_layers": 3, "d_layers": 3, "d_ff": 32,
                 "factor": 3, "distil": True, "dropout": 0.1, "activation": "gelu"},
    "crossformer": {"d_model": 256, "n_heads": 8, "e_layers": 3, "d_ff": 32,
                    "factor": 3, "seg_len": 12, "win_size": 2, "dropout": 0.1, "activation": "gelu"},
    "etsformer": {"d_model": 256, "n_heads": 8, "e_layers": 2, "d_ff": 32,
                  "top_k": 5, "dropout": 0.1, "activation": "sigmoid"},
    "dlinear": {"moving_avg": 25, "individual": False},
    "fedformer": {"d_model": 256, "n_heads": 8, "e_layers": 3, "d_layers": 3, "d_ff": 32,
                  "moving_avg": 25, "dropout": 0.1, "activation": "gelu",
                  "version": "Fourier", "mode_select": "random", "modes": 32},
    "film": {"window_size": "256", "multiscale": "1,2,4", "dropout": 0.1},
}

TIME_SERIES = {
    "seq_len": 10,
    "pred_len": 1,
    "label_len": 0,
}

CV = {
    "default_folds": 5,
    "max_epochs_per_fold": 20,
    "fold_val_split": 0.2,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── cuDNN compatibility check ─────────────────────────────────────────────

# Some environments have a mismatched cuDNN library (e.g. cuDNN 9.x installed
# alongside PyTorch compiled for cuDNN 8.x), which causes cryptic
# CUDNN_STATUS_NOT_INITIALIZED errors during training with RNN / LSTM / GRU /
# CNN models.  We detect this at startup and disable cuDNN globally so that
# PyTorch falls back to its native implementations.
if torch.cuda.is_available():
    try:
        if not getattr(torch.backends, "cudnn", None) or not torch.backends.cudnn.is_available():
            pass
    except Exception:
        torch.backends.cudnn.enabled = False
    else:
        # Runtime probe: RNNs trigger cuDNN flatten_parameters() which is
        # the first operation that breaks under a version mismatch.
        try:
            _gru_probe = torch.nn.GRU(1, 1, 1, batch_first=True).cuda()
            _gru_probe(torch.randn(1, 2, 1).cuda())
            del _gru_probe
        except Exception:
            import sys
            print(
                "WARNING: cuDNN is incompatible with this PyTorch build "
                "(version mismatch). Disabling cuDNN. Models will use native "
                "PyTorch implementations and still work correctly.",
                file=sys.stderr,
            )
            torch.backends.cudnn.enabled = False
