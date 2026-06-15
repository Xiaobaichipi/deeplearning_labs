"""Per-session state management with automatic disk reload."""

import functools
import math
import os
import traceback
import uuid
import numpy as np
import pandas as pd
import torch
from flask import current_app, jsonify, session

from .data_utils import load_data


ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def clean_nan(obj):
    """Recursively replace NaN/Infinity with None for safe JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, np.floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return clean_nan(obj.tolist())
    return obj


def json_ok(data):
    return jsonify(clean_nan(data))


def get_data_id():
    if "data_id" not in session:
        session["data_id"] = uuid.uuid4().hex
    return session["data_id"]


def allowed_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS


class RouteError(Exception):
    """Carry an HTTP status code for @handle_errors."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def handle_errors(f):
    """Decorator: wrap route handler with try/except + JSON error response."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except RouteError as e:
            return jsonify({"error": e.message}), e.status_code
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    return wrapper


def get_sm():
    """Shortcut: current_app.config['session_manager']."""
    return current_app.config["session_manager"]


def ensure_data(sm, data_id):
    """Get data or raise RouteError."""
    df = sm.get_data(data_id)
    if df is None:
        raise RouteError("No data uploaded")
    return df


class SessionManager:
    """Manages per-session data, model, split, and training history.

    On cache miss for data, tries to reload the original uploaded file
    from disk (``upload_dir / data_id /``).
    """

    def __init__(self, upload_dir: str):
        self._upload_dir = upload_dir
        self._data: dict = {}
        self._models: dict = {}
        self._splits: dict = {}
        self._histories: dict = {}
        self._pending_params: dict = {}
        self._model_configs: dict = {}
        self._task_configs: dict = {}

    # -- data ----------------------------------------------------------------

    def get_data(self, data_id: str):
        """Return the DataFrame for *data_id*, loading from disk if needed."""
        df = self._data.get(data_id)
        if df is not None:
            return df
        return self._reload_from_disk(data_id)

    def set_data(self, data_id: str, df):
        self._data[data_id] = df

    # -- model ---------------------------------------------------------------

    def get_model(self, data_id: str):
        return self._models.get(data_id)

    def set_model(self, data_id: str, model: torch.nn.Module):
        self._models[data_id] = model

    # -- split ---------------------------------------------------------------

    def get_split(self, data_id: str):
        return self._splits.get(data_id)

    def set_split(self, data_id: str, split_result: dict):
        self._splits[data_id] = split_result

    # -- training history ----------------------------------------------------

    def get_history(self, data_id: str):
        return self._histories.get(data_id)

    def set_history(self, data_id: str, history: dict):
        self._histories[data_id] = history

    # -- model config ----------------------------------------------------------

    def set_model_config(self, data_id: str, config: dict):
        self._model_configs[data_id] = config

    def get_model_config(self, data_id: str):
        return self._model_configs.get(data_id)

    # -- pending training params ---------------------------------------------

    def set_pending_params(self, data_id: str, params: dict):
        self._pending_params[data_id] = params

    def get_pending_params(self, data_id: str):
        return self._pending_params.pop(data_id, None)

    # -- task config (time series settings) ----------------------------------

    def get_task_config(self, data_id: str):
        return self._task_configs.get(data_id)

    def set_task_config(self, data_id: str, config: dict):
        self._task_configs[data_id] = config

    # -- combined checks -----------------------------------------------------

    def has_model(self, data_id: str) -> bool:
        return data_id in self._models and data_id in self._splits

    # -- reset ---------------------------------------------------------------

    def reset(self, data_id: str):
        self._data.pop(data_id, None)
        self._models.pop(data_id, None)
        self._splits.pop(data_id, None)
        self._histories.pop(data_id, None)
        self._pending_params.pop(data_id, None)
        self._task_configs.pop(data_id, None)
        self._model_configs.pop(data_id, None)

    # -- internal ------------------------------------------------------------

    def _reload_from_disk(self, data_id: str):
        file_dir = os.path.join(self._upload_dir, data_id)
        if not os.path.isdir(file_dir):
            return None
        for fname in os.listdir(file_dir):
            fpath = os.path.join(file_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                try:
                    df = load_data(fpath)
                    self._data[data_id] = df
                    return df
                except Exception:
                    continue
        return None
