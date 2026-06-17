"""Classical ML wrappers — Random Forest, XGBoost, LightGBM, Decision Tree.

All wrappers inherit from BaseModel, set ``pipeline = "small"`` and
``uses_sklearn_backend = True`` so the training pipeline dispatches to
``.fit()`` / ``.predict()`` instead of the PyTorch autograd loop.
"""

from .base import BaseModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np


# ── Base mixin ────────────────────────────────────────────────────────────────

class _ClassicalMLMixin:
    """Mixin that marks a model as sklearn-backed."""

    pipeline = "small"
    uses_sklearn_backend = True

    def predict_proba(self, X):
        """Return class probabilities (classification only)."""
        raise NotImplementedError


# ── Random Forest ────────────────────────────────────────────────────────────

class RandomForestRegressorWrapper(BaseModel, _ClassicalMLMixin):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self._model = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


class RandomForestClassifierWrapper(BaseModel, _ClassicalMLMixin):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self._model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


# ── XGBoost ──────────────────────────────────────────────────────────────────

class XGBRegressorWrapper(BaseModel, _ClassicalMLMixin):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        import xgboost as xgb
        self._model = xgb.XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


class XGBClassifierWrapper(BaseModel, _ClassicalMLMixin):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        import xgboost as xgb
        self._model = xgb.XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


# ── LightGBM ─────────────────────────────────────────────────────────────────

class LGBMRegressorWrapper(BaseModel, _ClassicalMLMixin):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        import lightgbm as lgb
        self._model = lgb.LGBMRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


class LGBMClassifierWrapper(BaseModel, _ClassicalMLMixin):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        import lightgbm as lgb
        self._model = lgb.LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


# ── Decision Tree ────────────────────────────────────────────────────────────

class DecisionTreeRegressorWrapper(BaseModel, _ClassicalMLMixin):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self._model = DecisionTreeRegressor(
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=42,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


class DecisionTreeClassifierWrapper(BaseModel, _ClassicalMLMixin):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        self._model = DecisionTreeClassifier(
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            random_state=42,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)
