"""Per-session state management with automatic disk reload."""

import os
import pandas as pd
import torch

from .data_utils import load_data


ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


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

    # -- combined checks -----------------------------------------------------

    def has_model(self, data_id: str) -> bool:
        return data_id in self._models and data_id in self._splits

    # -- reset ---------------------------------------------------------------

    def reset(self, data_id: str):
        self._data.pop(data_id, None)
        self._models.pop(data_id, None)
        self._splits.pop(data_id, None)
        self._histories.pop(data_id, None)

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
