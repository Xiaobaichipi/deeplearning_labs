"""Tests for utils/project_manager.py — disk serialization round-trip."""
import os
import tempfile

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_regression

from utils.data_utils import SplitResult
from utils.model_utils import create_model, train_model
from utils.project_manager import ProjectManager


@pytest.fixture
def pm():
    """ProjectManager with a temporary projects directory."""
    tmpdir = tempfile.mkdtemp()
    yield ProjectManager(tmpdir)


@pytest.fixture
def sample_split():
    """A general (non-time-series) SplitResult."""
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
        is_time_series=False,
        target_encoder=LabelEncoder().fit([0, 1]),
    )


@pytest.fixture
def ts_split():
    """A time-series SplitResult with pred_len."""
    return SplitResult(
        X_train=np.random.randn(10, 3, 5).astype(np.float32),
        X_test=np.random.randn(5, 3, 5).astype(np.float32),
        y_train=np.random.randn(10, 3).astype(np.float32),
        y_test=np.random.randn(5, 3).astype(np.float32),
        feature_names=["f1", "f2", "f3", "f4", "f5", "target"],
        target_name="target",
        task_type="regression",
        n_classes=1,
        input_dim=5,
        is_time_series=True,
        seq_len=10,
        pred_len=3,
        label_len=5,
        time_col="date",
        y_scaler={"method": "mean", "mean": 100.0, "std": 20.0},
    )


class TestSplitRoundTrip:
    def test_general_split_round_trip(self, pm, sample_split):
        pid = pm.create_project("test-general")
        pm.save_split(pid, sample_split)
        loaded = pm.load_split(pid)
        assert loaded is not None
        assert isinstance(loaded, SplitResult)
        np.testing.assert_array_equal(loaded.X_train, sample_split.X_train)
        np.testing.assert_array_equal(loaded.X_test, sample_split.X_test)
        np.testing.assert_array_equal(loaded.y_train, sample_split.y_train)
        np.testing.assert_array_equal(loaded.y_test, sample_split.y_test)
        assert loaded.feature_names == sample_split.feature_names
        assert loaded.task_type == sample_split.task_type
        assert loaded.n_classes == sample_split.n_classes
        assert loaded.input_dim == sample_split.input_dim
        assert loaded.is_time_series is False

    def test_ts_split_round_trip(self, pm, ts_split):
        pid = pm.create_project("test-ts")
        pm.save_split(pid, ts_split)
        loaded = pm.load_split(pid)
        assert loaded is not None
        assert isinstance(loaded, SplitResult)
        np.testing.assert_array_equal(loaded.X_train, ts_split.X_train)
        np.testing.assert_array_equal(loaded.y_train, ts_split.y_train)
        assert loaded.is_time_series is True
        assert loaded.seq_len == 10
        assert loaded.pred_len == 3
        assert loaded.label_len == 5
        assert loaded.time_col == "date"

    def test_y_scaler_round_trip(self, pm, ts_split):
        pid = pm.create_project("test-scaler")
        pm.save_split(pid, ts_split)
        loaded = pm.load_split(pid)
        assert loaded.y_scaler == {"method": "mean", "mean": 100.0, "std": 20.0}

    def test_target_encoder_round_trip(self, pm, sample_split):
        pid = pm.create_project("test-encoder")
        pm.save_split(pid, sample_split)
        loaded = pm.load_split(pid)
        assert loaded.target_encoder is not None
        np.testing.assert_array_equal(loaded.target_encoder.classes_, [0, 1])

    def test_target_encoder_none(self, pm):
        """Regression models have target_encoder=None."""
        split = SplitResult(
            X_train=np.ones((2, 1)), X_test=np.ones((1, 1)),
            y_train=np.array([1.0, 2.0]), y_test=np.array([3.0]),
            feature_names=["x"], target_name="y",
            task_type="regression", n_classes=1, input_dim=1,
            is_time_series=False, target_encoder=None,
        )
        pid = pm.create_project("test-no-encoder")
        pm.save_split(pid, split)
        loaded = pm.load_split(pid)
        assert loaded.target_encoder is None

    def test_split_no_project(self, pm):
        """load_split returns None when no split file exists."""
        assert pm.load_split("nonexistent") is None

    def test_split_default_round_trip(self, pm):
        """SplitResult with only required fields round-trips safely."""
        split = SplitResult(
            X_train=np.array([[1.0]]), X_test=np.array([[2.0]]),
            y_train=np.array([0.0]), y_test=np.array([1.0]),
            feature_names=["x"], target_name="y",
            task_type="regression", n_classes=1, input_dim=1,
        )
        pid = pm.create_project("test-minimal")
        pm.save_split(pid, split)
        loaded = pm.load_split(pid)
        assert loaded.is_time_series is False
        assert loaded.y_scaler is None
        assert loaded.target_encoder is None


class TestSklearnSerialization:
    """Pickle round-trip for sklearn-backed models via ProjectManager."""

    def test_sklearn_save_load_round_trip(self, pm):
        """Train a random forest, save via pickle, load, and verify predictions match."""
        X, y = make_regression(n_samples=50, n_features=2, noise=0.1, random_state=42)
        X = X.astype(np.float32)

        model = create_model("random_forest_regressor", X.shape[1], 1, n_estimators=10)
        trained, _ = train_model(
            model, X, y, X, y, task_type="regression",
        )
        preds_before = trained.predict(X)

        pid = pm.create_project("test-sklearn-model")
        mid = pm.next_model_id(pid)
        meta = {
            "model_type": "random_forest_regressor",
            "_sklearn_backend": True,
            "input_dim": X.shape[1],
            "output_dim": 1,
        }
        pm.save_model(pid, mid, trained, meta, is_sklearn=True)

        loaded_model, loaded_meta = pm.load_model(pid, mid)
        assert loaded_model is not None
        assert loaded_meta["_sklearn_backend"] is True
        assert loaded_meta["model_type"] == "random_forest_regressor"

        preds_after = loaded_model.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-12)
