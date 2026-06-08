"""Tests for utils/data_utils.py — pure functions for data processing."""

import numpy as np
import pandas as pd
import pytest
from utils.data_utils import (clean_data, denormalize_target, fill_missing,
                               normalize_data, normalize_target, split_data)


class TestNormalizeTarget:
    def test_mean_normalization(self):
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_test = np.array([2.5, 4.5])

        y_train_norm, y_test_norm, params = normalize_target(y_train, y_test, method="mean")

        assert np.allclose(y_train_norm.mean(), 0.0, atol=1e-6)
        assert np.allclose(y_train_norm.std(), 1.0, atol=1e-6)
        assert params["method"] == "mean"

    def test_minmax_normalization(self):
        y_train = np.array([10.0, 20.0, 30.0, 40.0])
        y_test = np.array([15.0, 35.0])

        y_train_norm, y_test_norm, params = normalize_target(y_train, y_test, method="minmax")

        assert np.allclose(y_train_norm.min(), 0.0)
        assert np.allclose(y_train_norm.max(), 1.0)
        assert params["method"] == "minmax"

    def test_none_method_returns_unchanged(self):
        y_train = np.array([1.0, 2.0, 3.0])
        y_test = np.array([4.0, 5.0])

        y_train_norm, y_test_norm, params = normalize_target(y_train, y_test, method=None)

        assert np.allclose(y_train_norm, y_train)
        assert np.allclose(y_test_norm, y_test)
        assert params["method"] is None


class TestDenormalizeTarget:
    def test_denormalize_mean_restores_original_scale(self):
        y_train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_test = np.array([15.0, 45.0])

        _, y_test_norm, params = normalize_target(y_train, y_test, method="mean")
        y_test_restored = denormalize_target(y_test_norm, params)

        assert np.allclose(y_test_restored, y_test)

    def test_denormalize_minmax_restores_original_scale(self):
        y_train = np.array([10.0, 20.0, 30.0, 40.0])
        y_test = np.array([15.0, 35.0])

        _, y_test_norm, params = normalize_target(y_train, y_test, method="minmax")
        y_test_restored = denormalize_target(y_test_norm, params)

        assert np.allclose(y_test_restored, y_test)

    def test_denormalize_none_scaler_returns_unchanged(self):
        y = np.array([1.0, 2.0, 3.0])
        result = denormalize_target(y, None)
        assert np.allclose(result, y)

    def test_denormalize_roundtrip_preserves_precision(self):
        rng = np.random.RandomState(99)
        y_train = rng.uniform(100, 500, 50)
        y_test = rng.uniform(100, 500, 20)

        for method in ("mean", "minmax"):
            _, y_test_norm, params = normalize_target(y_train, y_test, method=method)
            y_test_restored = denormalize_target(y_test_norm, params)
            assert np.allclose(y_test_restored, y_test, atol=1e-5)


class TestNormalizeData:
    def test_minmax_normalizes_to_zero_one_range(self):
        X_train = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)
        X_test = np.array([[15.0], [35.0]], dtype=np.float32)

        X_train_norm, X_test_norm, params = normalize_data(X_train, X_test, method="minmax")

        assert np.allclose(X_train_norm.min(axis=0), [0.0])
        assert np.allclose(X_train_norm.max(axis=0), [1.0])
        assert params["method"] == "minmax"
        assert "min" in params
        assert "max" in params

    def test_mean_normalizes_to_zero_mean_unit_variance(self):
        X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        X_test = np.array([[2.5], [4.5]], dtype=np.float32)

        X_train_norm, X_test_norm, params = normalize_data(X_train, X_test, method="mean")

        assert np.allclose(X_train_norm.mean(axis=0), [0.0], atol=1e-6)
        assert np.allclose(X_train_norm.std(axis=0), [1.0], atol=1e-6)
        assert params["method"] == "mean"
        assert "mean" in params
        assert "std" in params

    def test_constant_column_survives_minmax(self):
        X_train = np.array([[5.0], [5.0], [5.0]], dtype=np.float32)
        X_test = np.array([[5.0]], dtype=np.float32)

        X_train_norm, X_test_norm, params = normalize_data(X_train, X_test, method="minmax")

        assert np.allclose(X_train_norm, [[0.0], [0.0], [0.0]])
        assert np.allclose(X_test_norm, [[0.0]])

    def test_constant_column_survives_mean(self):
        X_train = np.array([[5.0], [5.0], [5.0]], dtype=np.float32)
        X_test = np.array([[5.0]], dtype=np.float32)

        X_train_norm, X_test_norm, params = normalize_data(X_train, X_test, method="mean")

        assert np.allclose(X_train_norm, [[0.0], [0.0], [0.0]])
        assert np.allclose(X_test_norm, [[0.0]])

    def test_multiple_features_independently_normalized(self):
        rng = np.random.RandomState(42)
        X_train = rng.rand(10, 3).astype(np.float32)
        X_test = rng.rand(5, 3).astype(np.float32)

        X_train_norm, X_test_norm, params = normalize_data(X_train, X_test, method="minmax")

        assert X_train_norm.shape == (10, 3)
        assert X_test_norm.shape == (5, 3)
        assert np.allclose(X_train_norm.min(axis=0), [0.0, 0.0, 0.0])
        assert np.allclose(X_train_norm.max(axis=0), [1.0, 1.0, 1.0])

    def test_test_set_uses_training_statistics(self):
        X_train = np.array([[0.0], [10.0]], dtype=np.float32)
        X_test = np.array([[-5.0], [20.0]], dtype=np.float32)

        X_train_norm, X_test_norm, params = normalize_data(X_train, X_test, method="minmax")

        # Test values outside train range should be outside [0,1]
        assert X_test_norm[0, 0] < 0.0
        assert X_test_norm[1, 0] > 1.0


class TestSplitData:
    def test_regression_split_returns_correct_shapes(self):
        df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
                           "feat2": [10.0, 20.0, 30.0, 40.0, 50.0],
                           "target": [100.0, 200.0, 300.0, 400.0, 500.0]})

        result = split_data(df, "target", test_size=0.4)

        assert result["task_type"] == "regression"
        assert result["X_train"].shape[1] == 2
        assert result["X_test"].shape[1] == 2
        assert len(result["X_train"]) == 3
        assert len(result["X_test"]) == 2
        assert result["input_dim"] == 2
        assert result["target_encoder"] is None

    def test_classification_split_returns_encoder(self):
        df = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                           "target": ["cat", "dog", "cat", "dog", "cat", "dog"]})

        result = split_data(df, "target", test_size=0.5)

        assert result["task_type"] == "classification"
        assert result["n_classes"] == 2
        assert result["target_encoder"] is not None

    def test_split_preserves_feature_names(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0],
                           "C": [7.0, 8.0, 9.0], "y": [0.0, 1.0, 0.0]})

        result = split_data(df, "y", test_size=0.33)

        assert "A" in result["feature_names"]
        assert "B" in result["feature_names"]
        assert "C" in result["feature_names"]
        assert "y" == result["target_name"]

    def test_small_dataset_handles_minimal_split(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [10.0, 20.0]})
        result = split_data(df, "y", test_size=0.5)
        assert len(result["X_train"]) == 1
        assert len(result["X_test"]) == 1


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "num": [1.0, 2.0, 3.0, None, 5.0],
        "cat": ["a", "b", None, "b", "a"],
        "target": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


class TestCleanData:
    def test_drop_duplicates_removes_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
        cleaned, report = clean_data(df, drop_duplicates=True)
        assert len(cleaned) == 2
        assert any("duplicate" in r.lower() for r in report)

    def test_drop_columns_removes_specified_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        cleaned, report = clean_data(df, drop_duplicates=False, drop_columns=["a", "c"])
        assert list(cleaned.columns) == ["b"]

    def test_preserves_at_least_one_column_when_dropping_all(self):
        df = pd.DataFrame({"a": [1]})
        cleaned, report = clean_data(df, drop_duplicates=False, drop_columns=["a"])
        assert len(cleaned.columns) >= 1

    def test_outlier_removal_with_iqr(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 100.0]})
        cleaned, report = clean_data(df, drop_duplicates=False, handle_outliers=True,
                                     outlier_factor=1.5)
        assert len(cleaned) < len(df)
        assert any("outlier" in r.lower() for r in report)

    def test_no_outliers_when_data_is_uniform(self):
        df = pd.DataFrame({"val": [5.0, 5.0, 5.0, 5.0]})
        cleaned, report = clean_data(df, drop_duplicates=False, handle_outliers=True)
        assert len(cleaned) == 4


class TestFillMissing:
    def test_mean_fill_for_numeric_column(self, sample_df):
        filled, report = fill_missing(sample_df, strategy="mean")
        assert filled["num"].isnull().sum() == 0
        assert np.allclose(filled.loc[3, "num"], sample_df["num"].mean(), atol=1e-6)

    def test_mode_fill_for_categorical_column(self, sample_df):
        filled, report = fill_missing(sample_df, strategy="mode")
        assert filled["cat"].isnull().sum() == 0

    def test_constant_fill_with_custom_value(self, sample_df):
        filled, report = fill_missing(sample_df, strategy="constant", fill_value=99)
        assert filled.loc[3, "num"] == 99

    def test_ffill_forward_fill(self):
        df = pd.DataFrame({"a": [1.0, None, None, 4.0]})
        filled, report = fill_missing(df, strategy="ffill")
        assert filled["a"].tolist() == [1.0, 1.0, 1.0, 4.0]

    def test_bfill_backward_fill(self):
        df = pd.DataFrame({"a": [1.0, None, None, 4.0]})
        filled, report = fill_missing(df, strategy="bfill")
        assert filled["a"].tolist() == [1.0, 4.0, 4.0, 4.0]

    def test_fill_with_no_missing_values_produces_no_report(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        filled, report = fill_missing(df, strategy="mean")
        assert len(report) == 0
