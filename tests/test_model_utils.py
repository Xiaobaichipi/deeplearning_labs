"""Tests for utils/model_utils.py — training, prediction, and evaluation."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from utils.model_utils import (create_model, cross_validate_model, evaluate,
                                predict, train_model)
from utils.models import get_model_class


# ── Helpers ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def regression_data():
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
    y = y.astype(np.float32)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@pytest.fixture(scope="session")
def classification_data():
    X, y = make_classification(n_samples=200, n_features=4, n_classes=2,
                                n_informative=3, n_redundant=0, random_state=42)
    y = y.astype(np.int64)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def _train_mlp_regression(regression_data, epochs=5):
    """Quick-train an MLP for regression and return (model, history)."""
    input_dim = regression_data["X_train"].shape[1]
    model = create_model("mlp", input_dim, 1, hidden_layers=[8, 4], dropout=0.0)
    trained, history = train_model(
        model, regression_data["X_train"], regression_data["y_train"],
        regression_data["X_test"], regression_data["y_test"],
        task_type="regression", epochs=epochs, batch_size=32, lr=0.01,
        patience=100, device="cpu",
    )
    return trained, history


def _train_mlp_classifier(classification_data, epochs=5):
    """Quick-train an MLP for classification and return (model, history)."""
    input_dim = classification_data["X_train"].shape[1]
    n_classes = len(np.unique(classification_data["y_train"]))
    model = create_model("mlp", input_dim, n_classes, hidden_layers=[8, 4], dropout=0.0)
    trained, history = train_model(
        model, classification_data["X_train"], classification_data["y_train"],
        classification_data["X_test"], classification_data["y_test"],
        task_type="classification", epochs=epochs, batch_size=32, lr=0.01,
        patience=100, device="cpu",
    )
    return trained, history


# =============================================================================
# create_model
# =============================================================================

class TestCreateModel:
    def test_creates_mlp(self):
        model = create_model("mlp", 10, 1)
        assert model is not None
        # Should have layers (check it's a torch Module)
        assert list(model.parameters()) != []

    def test_creates_all_model_types(self):
        for model_type in ("mlp", "cnn", "rnn", "lstm", "gru", "transformer"):
            model = create_model(model_type, 10, 2, dropout=0.0)
            assert model is not None, f"{model_type} failed to create"

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("nonexistent", 10, 1)


# =============================================================================
# predict — regression
# =============================================================================

class TestPredictRegression:
    def test_returns_prediction_and_none_probs(self, regression_data):
        model, _ = _train_mlp_regression(regression_data, epochs=5)
        preds, probs = predict(model, regression_data["X_test"], task_type="regression")

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(regression_data["y_test"]),)
        assert probs is None

    def test_predictions_have_reasonable_values(self, regression_data):
        model, _ = _train_mlp_regression(regression_data, epochs=5)
        preds, probs = predict(model, regression_data["X_train"], task_type="regression")

        # Predictions should be finite floats
        assert np.all(np.isfinite(preds))
        assert preds.dtype == np.float64 or preds.dtype == np.float32


# =============================================================================
# predict — classification
# =============================================================================

class TestPredictClassification:
    def test_returns_predictions_and_probabilities(self, classification_data):
        model, _ = _train_mlp_classifier(classification_data, epochs=5)
        preds, probs = predict(model, classification_data["X_test"], task_type="classification")

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(classification_data["y_test"]),)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (len(classification_data["y_test"]), 2)

    def test_probabilities_sum_to_one(self, classification_data):
        model, _ = _train_mlp_classifier(classification_data, epochs=5)
        preds, probs = predict(model, classification_data["X_test"], task_type="classification")

        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predictions_are_valid_class_labels(self, classification_data):
        model, _ = _train_mlp_classifier(classification_data, epochs=5)
        preds, probs = predict(model, classification_data["X_test"], task_type="classification")

        n_classes = len(np.unique(classification_data["y_train"]))
        assert np.all(preds >= 0)
        assert np.all(preds < n_classes)


# =============================================================================
# cross_validate_model
# =============================================================================

class TestCrossValidate:
    def test_returns_correct_dict_structure(self, regression_data):
        result = cross_validate_model(
            model_type="mlp",
            input_dim=regression_data["X_train"].shape[1],
            output_dim=1,
            X=regression_data["X_train"],
            y=regression_data["y_train"],
            task_type="regression",
            model_params={"hidden_layers": [8, 4], "dropout": 0.0},
            n_splits=3,
            epochs=3,
            batch_size=32,
            lr=0.01,
            device="cpu",
        )

        assert "cv_scores" in result
        assert "mean_score" in result
        assert "std_score" in result
        assert "n_splits" in result
        assert result["n_splits"] == 3
        assert len(result["cv_scores"]) == 3

    def test_cv_scores_are_finite_floats(self, classification_data):
        n_classes = len(np.unique(classification_data["y_train"]))
        result = cross_validate_model(
            model_type="mlp",
            input_dim=classification_data["X_train"].shape[1],
            output_dim=n_classes,
            X=classification_data["X_train"],
            y=classification_data["y_train"],
            task_type="classification",
            model_params={"hidden_layers": [8], "dropout": 0.0},
            n_splits=3,
            epochs=3,
            batch_size=32,
            lr=0.01,
            device="cpu",
        )

        assert np.all(np.isfinite(result["cv_scores"]))
        assert 0.0 <= result["mean_score"] <= 1.0

    def test_cv_without_model_params_uses_empty_defaults(self, regression_data):
        # Must not crash when model_params=None
        result = cross_validate_model(
            model_type="mlp",
            input_dim=regression_data["X_train"].shape[1],
            output_dim=1,
            X=regression_data["X_train"],
            y=regression_data["y_train"],
            task_type="regression",
            model_params=None,
            n_splits=2,
            epochs=2,
            device="cpu",
        )
        assert len(result["cv_scores"]) == 2


# =============================================================================
# train_model regression MAE scale
# =============================================================================

class TestTrainMAEScale:
    def test_mae_is_mean_not_sum(self, regression_data):
        """Regression train_metric should be per-sample mean, not sum across samples."""
        model, history = _train_mlp_regression(regression_data, epochs=5)
        y_train = regression_data["y_train"]
        y_range = max(float(y_train.max() - y_train.min()), 1.0)
        # Mean MAE cannot be orders of magnitude larger than y's range.
        # Sum MAE (bug) ≈ n_samples * typical_error >> y_range.
        assert history["train_metric"][-1] < y_range * 2


# =============================================================================
# evaluate
# =============================================================================

class TestEvaluateRegression:
    def test_returns_correct_metric_keys(self, regression_data):
        model, _ = _train_mlp_regression(regression_data, epochs=5)
        result = evaluate(model, regression_data["X_test"], regression_data["y_test"],
                          task_type="regression", device="cpu")

        assert result["task_type"] == "regression"
        assert "mse" in result
        assert "rmse" in result
        assert "mae" in result
        assert "r2" in result
        assert "images" in result

    def test_metrics_are_non_negative(self, regression_data):
        model, _ = _train_mlp_regression(regression_data, epochs=5)
        result = evaluate(model, regression_data["X_test"], regression_data["y_test"],
                          task_type="regression", device="cpu")

        assert result["mse"] >= 0
        assert result["rmse"] >= 0
        assert result["mae"] >= 0

    def test_regression_images_contain_expected_keys(self, regression_data):
        model, _ = _train_mlp_regression(regression_data, epochs=5)
        result = evaluate(model, regression_data["X_test"], regression_data["y_test"],
                          task_type="regression", device="cpu")

        assert "pred_vs_true" in result["images"]
        assert "residuals" in result["images"]
        # Values should be valid base64 strings
        for key, img in result["images"].items():
            assert isinstance(img, str)
            assert len(img) > 100  # non-trivial base64

    def test_without_images_still_returns_metrics(self, regression_data):
        """evaluate always includes images (no image-less mode exists)."""
        model, _ = _train_mlp_regression(regression_data, epochs=5)
        result = evaluate(model, regression_data["X_test"], regression_data["y_test"],
                          task_type="regression", device="cpu")

        assert result["mse"] >= 0  # metrics exist regardless


class TestEvaluateClassification:
    def test_returns_correct_metric_keys(self, classification_data):
        model, _ = _train_mlp_classifier(classification_data, epochs=5)
        result = evaluate(model, classification_data["X_test"], classification_data["y_test"],
                          task_type="classification", device="cpu")

        assert result["task_type"] == "classification"
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "images" in result

    def test_accuracy_is_within_valid_range(self, classification_data):
        model, _ = _train_mlp_classifier(classification_data, epochs=5)
        result = evaluate(model, classification_data["X_test"], classification_data["y_test"],
                          task_type="classification", device="cpu")

        assert 0.0 <= result["accuracy"] <= 1.0

    def test_classification_images_contain_confusion_matrix(self, classification_data):
        model, _ = _train_mlp_classifier(classification_data, epochs=5)
        result = evaluate(model, classification_data["X_test"], classification_data["y_test"],
                          task_type="classification", device="cpu")

        assert "confusion_matrix" in result["images"]

    def test_binary_classification_includes_roc_curve(self, classification_data):
        """Binary classification should include ROC curve in images."""
        model, _ = _train_mlp_classifier(classification_data, epochs=5)
        result = evaluate(model, classification_data["X_test"], classification_data["y_test"],
                          task_type="classification", device="cpu")

        assert "roc_curve" in result["images"]

    def test_with_target_encoder_images_are_still_generated(self, classification_data):
        """evaluate accepts optional target_encoder without crashing."""
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["cat", "dog"])

        model, _ = _train_mlp_classifier(classification_data, epochs=5)
        result = evaluate(model, classification_data["X_test"], classification_data["y_test"],
                          task_type="classification", target_encoder=le, device="cpu")

        assert result["accuracy"] >= 0
        assert "confusion_matrix" in result["images"]
