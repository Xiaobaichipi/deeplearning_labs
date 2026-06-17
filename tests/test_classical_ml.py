"""Tests for classical ML models (sklearn backend wrappers)."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from utils.model_utils import create_model, train_model, predict, evaluate
from utils.models import uses_sklearn_backend, get_model_class


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def regression_data():
    X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@pytest.fixture(scope="session")
def classification_data():
    X, y = make_classification(n_samples=200, n_features=4, n_classes=2,
                                n_informative=3, n_redundant=0, random_state=42)
    X = X.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# ── Registry / metadata ──────────────────────────────────────────────────

CLASSICAL_REGRESSORS = [
    "random_forest_regressor",
    "xgboost_regressor",
    "lightgbm_regressor",
    "decision_tree_regressor",
]

CLASSICAL_CLASSIFIERS = [
    "random_forest_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
    "decision_tree_classifier",
]

ALL_CLASSICAL = CLASSICAL_REGRESSORS + CLASSICAL_CLASSIFIERS


class TestRegistry:
    def test_all_marked_sklearn_backend(self):
        for mt in ALL_CLASSICAL:
            assert uses_sklearn_backend(mt), f"{mt} should have uses_sklearn_backend=True"

    def test_classes_are_importable(self):
        for mt in ALL_CLASSICAL:
            cls = get_model_class(mt)
            assert cls is not None
            assert hasattr(cls, "fit")
            assert hasattr(cls, "predict")

    def test_sklearn_models_have_fit_not_forward(self):
        """Sklearn models should have fit() (not just forward() like PyTorch)."""
        for mt in ALL_CLASSICAL:
            cls = get_model_class(mt)
            model = cls(input_dim=4, output_dim=1)
            assert hasattr(model, "fit")
            # These are sklearn-backed — pipeline strategy uses .fit() not forward()
            assert getattr(model, "uses_sklearn_backend", False) is True


# ── Instantiation ────────────────────────────────────────────────────────

class TestInstantiation:
    def test_create_regressors(self):
        for mt in CLASSICAL_REGRESSORS:
            model = create_model(mt, input_dim=4, output_dim=1)
            assert model is not None

    def test_create_classifiers(self):
        for mt in CLASSICAL_CLASSIFIERS:
            model = create_model(mt, input_dim=4, output_dim=2)
            assert model is not None

    def test_create_with_params(self):
        model = create_model("random_forest_regressor", 4, 1,
                             n_estimators=50, max_depth=5, min_samples_split=4)
        assert model is not None
        assert model._model.n_estimators == 50
        assert model._model.max_depth == 5
        assert model._model.min_samples_split == 4

    def test_create_decision_tree_no_n_estimators(self):
        """Decision Tree should not require n_estimators."""
        model = create_model("decision_tree_regressor", 4, 1, max_depth=3)
        assert model is not None
        assert model._model.max_depth == 3


# ── Training & prediction ────────────────────────────────────────────────

class TestTraining:
    def test_regressor_fit_predict(self, regression_data):
        for mt in CLASSICAL_REGRESSORS:
            model = create_model(mt, regression_data["X_train"].shape[1], 1)
            trained, history = train_model(
                model,
                regression_data["X_train"], regression_data["y_train"],
                regression_data["X_test"], regression_data["y_test"],
                task_type="regression",
            )
            assert trained is not None
            assert "sklearn_backend" in history
            assert history["sklearn_backend"] is True
            # Should have a single-entry history (sklearn trains in one shot)
            assert len(history["train_loss"]) == 1

    def test_classifier_fit_predict(self, classification_data):
        for mt in CLASSICAL_CLASSIFIERS:
            model = create_model(mt, classification_data["X_train"].shape[1], 2)
            trained, history = train_model(
                model,
                classification_data["X_train"], classification_data["y_train"],
                classification_data["X_test"], classification_data["y_test"],
                task_type="classification",
            )
            assert trained is not None
            assert "sklearn_backend" in history

    def test_predict_returns_array(self, regression_data):
        model = create_model("random_forest_regressor", 4, 1, n_estimators=10)
        trained, _ = train_model(
            model,
            regression_data["X_train"], regression_data["y_train"],
            regression_data["X_test"], regression_data["y_test"],
            task_type="regression",
        )
        preds, _ = predict(trained, regression_data["X_test"], task_type="regression")
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(regression_data["X_test"])

    def test_classifier_predict_proba_available(self, classification_data):
        for mt in CLASSICAL_CLASSIFIERS:
            model = create_model(mt, classification_data["X_train"].shape[1], 2)
            assert hasattr(model, "predict_proba"), f"{mt} missing predict_proba"


# ── Evaluation ────────────────────────────────────────────────────────────

class TestEvaluation:
    def test_evaluate_regression(self, regression_data):
        model = create_model("random_forest_regressor", 4, 1, n_estimators=10)
        trained, _ = train_model(
            model,
            regression_data["X_train"], regression_data["y_train"],
            regression_data["X_test"], regression_data["y_test"],
            task_type="regression",
        )
        result = evaluate(trained, regression_data["X_test"],
                          regression_data["y_test"], task_type="regression")
        assert "mse" in result
        assert "rmse" in result
        assert "r2" in result
        assert result["task_type"] == "regression"

    def test_evaluate_classification(self, classification_data):
        model = create_model("random_forest_classifier", 4, 2, n_estimators=10)
        trained, _ = train_model(
            model,
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_test"], classification_data["y_test"],
            task_type="classification",
        )
        result = evaluate(trained, classification_data["X_test"],
                          classification_data["y_test"], task_type="classification")
        assert "accuracy" in result
        assert "precision" in result
        assert "f1_score" in result
        assert result["task_type"] == "classification"


# ── Task-type mismatch validation ─────────────────────────────────────────

class TestTaskTypeValidation:
    """train_model / evaluate should reject regressor↔classification mismatches."""

    def test_regressor_with_classification_raises(self, regression_data):
        """Regressor trained with task_type='classification' should raise."""
        model = create_model("random_forest_regressor", 4, 1, n_estimators=10)
        with pytest.raises(ValueError, match="regressor.*classification"):
            train_model(
                model,
                regression_data["X_train"], regression_data["y_train"],
                regression_data["X_test"], regression_data["y_test"],
                task_type="classification",
            )

    def test_classifier_with_regression_raises(self, classification_data):
        """Classifier trained with task_type='regression' should raise."""
        model = create_model("random_forest_classifier", 4, 2, n_estimators=10)
        with pytest.raises(ValueError, match="classifier.*regression"):
            train_model(
                model,
                classification_data["X_train"], classification_data["y_train"],
                classification_data["X_test"], classification_data["y_test"],
                task_type="regression",
            )

    def test_evaluate_regressor_classification_raises(self, regression_data):
        """evaluate on regressor with task_type='classification' should raise."""
        model = create_model("random_forest_regressor", 4, 1, n_estimators=10)
        trained, _ = train_model(
            model,
            regression_data["X_train"], regression_data["y_train"],
            regression_data["X_test"], regression_data["y_test"],
            task_type="regression",
        )
        with pytest.raises(ValueError, match="regressor.*classification"):
            evaluate(trained, regression_data["X_test"],
                     regression_data["y_test"], task_type="classification")

    def test_evaluate_classifier_regression_raises(self, classification_data):
        """evaluate on classifier with task_type='regression' should raise."""
        model = create_model("random_forest_classifier", 4, 2, n_estimators=10)
        trained, _ = train_model(
            model,
            classification_data["X_train"], classification_data["y_train"],
            classification_data["X_test"], classification_data["y_test"],
            task_type="classification",
        )
        with pytest.raises(ValueError, match="classifier.*regression"):
            evaluate(trained, classification_data["X_test"],
                     classification_data["y_test"], task_type="regression")

    def test_cross_val_regressor_classification_raises(self):
        """cross_validate_model with regressor and classification should raise."""
        from utils.model_utils import cross_validate_model
        with pytest.raises(ValueError, match="regressor.*classification"):
            cross_validate_model(
                "random_forest_regressor", 4, 1,
                np.random.randn(20, 4).astype(np.float32),
                np.random.randn(20),
                task_type="classification",
                n_splits=2,
            )

    def test_cross_val_classifier_regression_raises(self):
        """cross_validate_model with classifier and regression should raise."""
        from utils.model_utils import cross_validate_model
        with pytest.raises(ValueError, match="classifier.*regression"):
            cross_validate_model(
                "random_forest_classifier", 4, 2,
                np.random.randn(20, 4).astype(np.float32),
                np.random.randint(0, 2, 20),
                task_type="regression",
                n_splits=2,
            )


# ── String label encoding ─────────────────────────────────────────────────

class TestStringLabelEncoding:
    """Classification with string labels should work via target_encoder."""

    def test_train_with_string_labels(self):
        """train_model with string classification labels should succeed."""
        from sklearn.preprocessing import LabelEncoder
        X = np.random.randn(20, 2).astype(np.float32)
        y = np.array(["cat", "dog"] * 10)

        le = LabelEncoder()
        le.fit(y)

        model = create_model("random_forest_classifier", 2, 2, n_estimators=10)
        trained, history = train_model(
            model, X[:15], y[:15], X[15:], y[15:],
            task_type="classification",
            target_encoder=le,
        )
        assert trained is not None
        assert "train_loss" in history
        assert history["sklearn_backend"] is True

    def test_evaluate_with_string_labels(self):
        """evaluate with string classification labels should succeed."""
        from sklearn.preprocessing import LabelEncoder
        X = np.random.randn(20, 2).astype(np.float32)
        y = np.array(["cat", "dog"] * 10)

        le = LabelEncoder()
        le.fit(y)

        model = create_model("random_forest_classifier", 2, 2, n_estimators=10)
        trained, _ = train_model(
            model, X[:15], y[:15], X[15:], y[15:],
            task_type="classification",
            target_encoder=le,
        )
        result = evaluate(trained, X[15:], y[15:],
                          task_type="classification", target_encoder=le)
        assert "accuracy" in result
        assert result["task_type"] == "classification"
