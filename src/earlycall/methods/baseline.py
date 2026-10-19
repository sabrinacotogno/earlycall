"""Baselines. Any method that does not beat these is not worth its complexity.

BaseRate            ignores the run.
RidgeOnFeatures     bioprocess: estimates the final titer from summary features.
LogisticOnFeatures  solar: probability of a fault from the same summary features.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from earlycall.batch_data import BatchData
from earlycall.methods.base import BatchMonitor


class BaseRate(BatchMonitor):
    """Ignores the run.

    predict_quality: not enough information (NaN) for every run: never an alarm.
    predict_risk: the share of low runs in training.
    """

    name = "base_rate"

    def fit(self, train: BatchData) -> "BaseRate":
        self.n_params_ = train.n_params
        self._fit_threshold(train.y)
        return self

    def predict_quality(self, X: np.ndarray, t_index: int) -> np.ndarray:
        X = self._check_X(X, self.n_params_, t_index)
        return np.full(X.shape[0], np.nan)

    def predict_risk(self, X: np.ndarray, t_index: int) -> np.ndarray:
        X = self._check_X(X, self.n_params_, t_index)
        return np.full(X.shape[0], self.base_rate_)


# ------------------------------------------------- summary features (ridge, logistic)
def extract_features(X: np.ndarray) -> np.ndarray:
    """A few summary numbers per parameter, from the (partial) trajectories.

    X: (n_batches, n_params, n_time) -> (n_batches, 5 * n_params)
    Per parameter: last value, mean, min, max, and the recent change
    (last value minus the value 3 points earlier, divided by 3).
    """
    columns = []
    for p in range(X.shape[1]):
        curve = X[:, p, :]
        lookback = min(3, curve.shape[1] - 1)
        recent_change = (curve[:, -1] - curve[:, -1 - lookback]) / lookback
        columns += [curve[:, -1], curve.mean(axis=1), curve.min(axis=1), curve.max(axis=1), recent_change]
    return np.column_stack(columns)


class TrajectoryModel(BatchMonitor):
    """Shared by ridge, logistic regression and PLS: one model per cut time.

    The model used at t_index is trained on the training runs cut at that same t_index, so
    training runs and running batches are seen the same way. Nothing is invented.
    Each model is built the first time it is needed, then reused.
    Subclasses define _target (what the model learns) and _fit_model.
    """

    def _target(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _fit_model(self, X: np.ndarray, target: np.ndarray):
        raise NotImplementedError

    def fit(self, train: BatchData) -> "TrajectoryModel":
        self.n_params_ = train.n_params
        self._fit_threshold(train.y)
        known = ~np.isnan(train.y)
        self.train_X_ = train.X[known]
        self.train_target_ = self._target(train.y[known])
        self.models_per_cut_ = {}                                # t_index -> model
        return self

    def _model_at(self, t_index: int):
        """Model trained on the training runs cut at t_index (built once, then reused)."""
        if t_index not in self.models_per_cut_:
            cut = self.train_X_[:, :, :t_index]
            self.models_per_cut_[t_index] = self._fit_model(cut, self.train_target_)
        return self.models_per_cut_[t_index]


class RidgeOnFeatures(TrajectoryModel):
    """Bioprocess. Ridge regression on summary features: estimates the final quality (titer)."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0, low_quantile: float = 0.3, low_threshold: float | None = None):
        super().__init__(low_quantile, low_threshold)
        self.alpha = alpha

    def _target(self, y: np.ndarray) -> np.ndarray:
        return y                                                 # the final quality itself

    def _fit_model(self, X: np.ndarray, target: np.ndarray):
        model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        return model.fit(extract_features(X), target)

    def predict_quality(self, X: np.ndarray, t_index: int) -> np.ndarray:
        X = self._check_X(X, self.n_params_, t_index)            # only the past
        return self._model_at(t_index).predict(extract_features(X))


class LogisticOnFeatures(TrajectoryModel):
    """Solar. Logistic regression on summary features: probability of ending low (faulty)."""

    name = "logistic"

    def __init__(self, C: float = 1.0, low_quantile: float = 0.3, low_threshold: float | None = None):
        super().__init__(low_quantile, low_threshold)
        self.C = C                                               # inverse strength of the penalty

    def _target(self, y: np.ndarray) -> np.ndarray:
        return self.is_low(y).astype(int)                        # 1 = low, 0 = not low

    def _fit_model(self, X: np.ndarray, target: np.ndarray):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=1000))
        return model.fit(extract_features(X), target)

    def predict_risk(self, X: np.ndarray, t_index: int) -> np.ndarray:
        X = self._check_X(X, self.n_params_, t_index)            # only the past
        model = self._model_at(t_index)
        return model.predict_proba(extract_features(X))[:, 1]    # column 1 = class "low"
