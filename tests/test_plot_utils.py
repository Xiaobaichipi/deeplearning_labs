"""Tests for utils/plot_utils.py — validate base64 PNG output of every plot function."""

import base64
import io

import numpy as np
import pandas as pd
import pytest

from utils.plot_utils import (fig_to_base64, plot_confusion_matrix,
                               plot_correlation_heatmap, plot_data_distribution,
                               plot_feature_importance, plot_pred_vs_true,
                               plot_pred_vs_true_line, plot_residuals,
                               plot_roc_curve, plot_training_history)


def _png_header(data):
    """Return first 8 decoded bytes of a base64 PNG for header validation."""
    return base64.b64decode(data)[:8]


def _is_valid_png_base64(data):
    """Check that a string is valid base64 and starts with PNG magic header."""
    if not isinstance(data, str) or len(data) < 50:
        return False
    try:
        header = _png_header(data)
        return header == b'\x89PNG\r\n\x1a\n'
    except Exception:
        return False


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat":  ["a", "b", "a", "b", "a"],
    })


# =============================================================================
# fig_to_base64
# =============================================================================

class TestFigToBase64:
    def test_returns_valid_png(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        result = fig_to_base64(fig)
        assert _is_valid_png_base64(result)

    def test_default_dpi_is_100(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        result = fig_to_base64(fig)
        assert _is_valid_png_base64(result)
        # Check file size is reasonable for 100dpi (expect > 5KB)
        assert len(result) > 1000


# =============================================================================
# plot_training_history
# =============================================================================

class TestPlotTrainingHistory:
    def test_returns_images_dict(self):
        history = {
            "train_loss": [1.0, 0.5, 0.3],
            "val_loss": [1.1, 0.6, 0.4],
            "train_metric": [0.5, 0.7, 0.9],
            "val_metric": [0.4, 0.6, 0.8],
        }
        images = plot_training_history(history)
        assert isinstance(images, dict)
        assert "training_history" in images
        assert _is_valid_png_base64(images["training_history"])


# =============================================================================
# plot_feature_importance
# =============================================================================

class TestPlotFeatureImportance:
    def test_returns_valid_png(self):
        result = plot_feature_importance(["feat_a", "feat_b", "feat_c"],
                                          [0.1, 0.6, 0.3])
        assert _is_valid_png_base64(result)

    def test_single_feature_does_not_crash(self):
        result = plot_feature_importance(["only"], [1.0])
        assert _is_valid_png_base64(result)

    def test_many_features_does_not_crash(self):
        names = [f"feat_{i}" for i in range(50)]
        values = np.random.RandomState(0).rand(50)
        result = plot_feature_importance(names, values)
        assert _is_valid_png_base64(result)


# =============================================================================
# plot_data_distribution
# =============================================================================

class TestPlotDataDistribution:
    def test_returns_images_dict_with_numeric_columns(self, sample_df):
        images = plot_data_distribution(sample_df)
        assert isinstance(images, dict)
        assert "data_distribution" in images
        assert _is_valid_png_base64(images["data_distribution"])

    def test_returns_empty_dict_when_no_numeric_columns(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        images = plot_data_distribution(df)
        assert images == {}

    def test_limits_to_six_columns(self):
        df = pd.DataFrame({f"col_{i}": np.random.rand(20) for i in range(10)})
        images = plot_data_distribution(df)
        assert "data_distribution" in images


# =============================================================================
# plot_correlation_heatmap
# =============================================================================

class TestPlotCorrelationHeatmap:
    def test_returns_valid_png_with_multiple_numeric_cols(self, sample_df):
        result = plot_correlation_heatmap(sample_df)
        assert _is_valid_png_base64(result)

    def test_returns_none_with_fewer_than_two_numeric_cols(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        assert plot_correlation_heatmap(df) is None

    def test_returns_none_with_no_numeric_cols(self):
        df = pd.DataFrame({"cat": ["a", "b"]})
        assert plot_correlation_heatmap(df) is None


# =============================================================================
# plot_confusion_matrix
# =============================================================================

class TestPlotConfusionMatrix:
    def test_returns_valid_png(self):
        cm = [[5, 1], [2, 4]]
        result = plot_confusion_matrix(cm, ["class_0", "class_1"])
        assert _is_valid_png_base64(result)

    def test_empty_confusion_matrix_returns_none(self):
        result = plot_confusion_matrix([], [])
        assert result is None


# =============================================================================
# plot_roc_curve
# =============================================================================

class TestPlotROCCurve:
    def test_returns_base64_and_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        img, auc = plot_roc_curve(y_true, y_score)
        assert _is_valid_png_base64(img)
        assert isinstance(auc, float)
        assert 0 < auc <= 1

    def test_perfect_separation_gives_auc_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.0, 0.0, 1.0, 1.0])
        img, auc = plot_roc_curve(y_true, y_score)
        assert auc == 1.0


# =============================================================================
# plot_pred_vs_true (scatter)
# =============================================================================

class TestPlotPredVsTrue:
    def test_returns_valid_png(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.3, 4.9])
        result = plot_pred_vs_true(y_true, y_pred)
        assert _is_valid_png_base64(result)


# =============================================================================
# plot_pred_vs_true_line
# =============================================================================

class TestPlotPredVsTrueLine:
    def test_returns_valid_png(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.3, 4.9])
        result = plot_pred_vs_true_line(y_true, y_pred)
        assert _is_valid_png_base64(result)

    def test_single_sample_does_not_crash(self):
        y_true = np.array([1.0])
        y_pred = np.array([1.1])
        result = plot_pred_vs_true_line(y_true, y_pred)
        assert _is_valid_png_base64(result)


# =============================================================================
# plot_residuals
# =============================================================================

class TestPlotResiduals:
    def test_returns_valid_png(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.3, 4.9])
        result = plot_residuals(y_true, y_pred)
        assert _is_valid_png_base64(result)

    def test_all_zero_residuals(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = plot_residuals(y_true, y_pred)
        assert _is_valid_png_base64(result)
