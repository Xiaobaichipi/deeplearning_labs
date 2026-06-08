"""Tests for utils/session.py — SessionManager state management."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from utils.session import SessionManager


# A tiny model for testing
class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def manager():
    """SessionManager backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SessionManager(tmpdir)


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


# =============================================================================
# Data (DataFrame) CRUD
# =============================================================================

class TestDataCRUD:
    def test_set_and_get_data(self, manager, sample_df):
        manager.set_data("s1", sample_df)
        result = manager.get_data("s1")
        assert result is not None
        assert result.shape == (3, 2)
        assert list(result.columns) == ["a", "b"]

    def test_get_data_returns_none_for_unknown_id(self, manager):
        assert manager.get_data("nonexistent") is None

    def test_get_data_returns_none_for_unknown_disk_dir(self, manager):
        # No files on disk → get_data should return None
        assert manager.get_data("no_files") is None

    def test_set_data_overwrites_previous(self, manager):
        df1 = pd.DataFrame({"x": [1]})
        df2 = pd.DataFrame({"y": [2]})
        manager.set_data("s1", df1)
        manager.set_data("s1", df2)
        result = manager.get_data("s1")
        assert list(result.columns) == ["y"]


# =============================================================================
# Model CRUD
# =============================================================================

class TestModelCRUD:
    def test_set_and_get_model(self, manager):
        model = _DummyModel()
        manager.set_model("s1", model)
        result = manager.get_model("s1")
        assert isinstance(result, nn.Module)
        assert isinstance(result, _DummyModel)

    def test_get_model_returns_none_for_untrained(self, manager):
        assert manager.get_model("no_model") is None

    def test_has_model_true_when_model_and_split_exist(self, manager):
        manager.set_model("s1", _DummyModel())
        manager.set_split("s1", {"dummy": True})
        assert manager.has_model("s1") is True

    def test_has_model_false_when_missing_model(self, manager):
        manager.set_split("s1", {"dummy": True})
        assert manager.has_model("s1") is False

    def test_has_model_false_when_missing_split(self, manager):
        manager.set_model("s1", _DummyModel())
        assert manager.has_model("s1") is False


# =============================================================================
# Split CRUD
# =============================================================================

class TestSplitCRUD:
    def test_set_and_get_split(self, manager):
        split = {"X_train": np.array([[1.0]]), "y_train": np.array([0.0]),
                 "X_test": np.array([[2.0]]), "y_test": np.array([1.0]),
                 "task_type": "regression", "input_dim": 1}
        manager.set_split("s1", split)
        result = manager.get_split("s1")
        assert result["task_type"] == "regression"
        assert result["input_dim"] == 1

    def test_get_split_returns_none(self, manager):
        assert manager.get_split("no_split") is None


# =============================================================================
# History CRUD
# =============================================================================

class TestHistoryCRUD:
    def test_set_and_get_history(self, manager):
        history = {"train_loss": [1.0, 0.5, 0.3]}
        manager.set_history("s1", history)
        result = manager.get_history("s1")
        assert result["train_loss"] == [1.0, 0.5, 0.3]

    def test_get_history_returns_none(self, manager):
        assert manager.get_history("no_history") is None


# =============================================================================
# Model Config CRUD
# =============================================================================

class TestModelConfigCRUD:
    def test_set_and_get_model_config(self, manager):
        config = {"model_type": "mlp", "epochs": 50}
        manager.set_model_config("s1", config)
        result = manager.get_model_config("s1")
        assert result["model_type"] == "mlp"

    def test_get_model_config_returns_none(self, manager):
        assert manager.get_model_config("no_config") is None


# =============================================================================
# Pending Params (single-use / pop semantics)
# =============================================================================

class TestPendingParams:
    def test_set_and_get_pending_params(self, manager):
        params = {"target_col": "y", "epochs": 50}
        manager.set_pending_params("s1", params)
        result = manager.get_pending_params("s1")
        assert result["target_col"] == "y"

    def test_get_pending_params_pops(self, manager):
        params = {"target_col": "y"}
        manager.set_pending_params("s1", params)
        manager.get_pending_params("s1")  # first get returns it
        result = manager.get_pending_params("s1")  # second get should return None
        assert result is None

    def test_get_pending_params_returns_none_for_no_params(self, manager):
        assert manager.get_pending_params("no_params") is None


# =============================================================================
# Reset
# =============================================================================

class TestReset:
    def test_reset_clears_all_state(self, manager, sample_df):
        manager.set_data("s1", sample_df)
        manager.set_model("s1", _DummyModel())
        manager.set_split("s1", {"dummy": True})
        manager.set_history("s1", {"train_loss": [1.0]})
        manager.set_pending_params("s1", {"target_col": "y"})

        manager.reset("s1")

        assert manager.get_data("s1") is None
        assert manager.get_model("s1") is None
        assert manager.get_split("s1") is None
        assert manager.get_history("s1") is None
        assert manager.get_pending_params("s1") is None

    def test_reset_does_not_affect_other_sessions(self, manager):
        manager.set_data("s1", pd.DataFrame({"a": [1]}))
        manager.set_data("s2", pd.DataFrame({"b": [2]}))

        manager.reset("s1")

        assert manager.get_data("s1") is None
        assert manager.get_data("s2") is not None


# =============================================================================
# Disk Reload
# =============================================================================

class TestDiskReload:
    def test_reload_from_disk_after_cache_cleared(self, manager, sample_df):
        """Data that was set and saved to disk can be reloaded after cache loss."""
        data_id = "disk_test"
        manager.set_data(data_id, sample_df)

        # SessionManager saves to disk on set_data — we manually write CSV
        # to simulate what the upload route does
        file_dir = os.path.join(manager._upload_dir, data_id)
        os.makedirs(file_dir, exist_ok=True)
        filepath = os.path.join(file_dir, "data.csv")
        sample_df.to_csv(filepath, index=False)

        # Clear in-memory cache
        manager._data.pop(data_id, None)

        # Should reload from disk
        result = manager.get_data(data_id)
        assert result is not None
        assert result.shape == (3, 2)

    def test_reload_returns_none_when_disk_empty(self, manager):
        """No files on disk → get_data returns None."""
        data_id = "empty_dir"
        file_dir = os.path.join(manager._upload_dir, data_id)
        os.makedirs(file_dir, exist_ok=True)  # empty dir

        result = manager.get_data(data_id)
        assert result is None

    def test_reload_skips_unsupported_extensions(self, manager):
        """Only .csv, .xls, .xlsx files are loaded."""
        data_id = "bad_ext"
        file_dir = os.path.join(manager._upload_dir, data_id)
        os.makedirs(file_dir, exist_ok=True)

        # Write a .txt file
        with open(os.path.join(file_dir, "data.txt"), "w") as f:
            f.write("not,valid\n1,2\n")

        result = manager.get_data(data_id)
        assert result is None

    def test_multiple_files_in_dir_reloads_first_valid(self, manager, sample_df):
        """When multiple CSV files exist, reload picks the first valid one."""
        data_id = "multi_files"
        file_dir = os.path.join(manager._upload_dir, data_id)
        os.makedirs(file_dir, exist_ok=True)

        sample_df.to_csv(os.path.join(file_dir, "a.csv"), index=False)
        sample_df.to_csv(os.path.join(file_dir, "b.csv"), index=False)

        manager._data.pop(data_id, None)
        result = manager.get_data(data_id)
        assert result is not None
        assert result.shape == (3, 2)
