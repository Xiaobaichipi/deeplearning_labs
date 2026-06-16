"""Tests for SplitResult dataclass dict-compatibility layer."""
import numpy as np
import pytest

from utils.data_utils import SplitResult


@pytest.fixture
def sr():
    return SplitResult(
        X_train=np.array([[1.0, 2.0], [3.0, 4.0]]),
        X_test=np.array([[5.0, 6.0]]),
        y_train=np.array([0, 1]),
        y_test=np.array([0]),
        feature_names=["a", "b"],
        target_name="target",
        task_type="classification",
        n_classes=2,
        input_dim=2,
        is_time_series=True,
        seq_len=10,
        pred_len=3,
        label_len=5,
    )


class TestDictCompatibility:
    def test_getitem(self, sr):
        assert sr["task_type"] == "classification"
        assert sr["input_dim"] == 2

    def test_setitem(self, sr):
        sr["y_scaler"] = {"method": "mean", "mean": 0.0}
        assert sr.y_scaler == {"method": "mean", "mean": 0.0}

    def test_get(self, sr):
        assert sr.get("task_type") == "classification"
        assert sr.get("nonexistent") is None
        assert sr.get("nonexistent", 42) == 42

    def test_contains_required(self, sr):
        assert "X_train" in sr
        assert "task_type" in sr

    def test_contains_optional_unset(self, sr):
        """Optional fields with default=None are NOT 'in' SplitResult."""
        assert "x_mark_train" not in sr
        assert "norm_params" not in sr

    def test_contains_optional_set(self, sr):
        """Optional fields with non-None value ARE 'in' SplitResult."""
        sr["norm_params"] = {"method": "minmax"}
        assert "norm_params" in sr

    def test_contains_non_default_false(self, sr):
        """Fields with non-None default (e.g. is_time_series=False) ARE in."""
        assert "is_time_series" in sr
        assert "seq_len" in sr

    def test_items(self, sr):
        items = dict(sr.items())
        assert items["task_type"] == "classification"
        assert items["input_dim"] == 2
        assert "X_train" in items

    def test_update(self, sr):
        sr.update(y_scaler={"method": "mean"}, norm_params={"method": "minmax"})
        assert sr.y_scaler == {"method": "mean"}
        assert sr.norm_params == {"method": "minmax"}

    def test_get_after_setitem(self, sr):
        sr["norm_params"] = {"method": "minmax"}
        assert sr.get("norm_params") == {"method": "minmax"}
