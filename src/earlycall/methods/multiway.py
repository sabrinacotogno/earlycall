"""Multiway PLS: the classical approach to batch monitoring (Nomikos & MacGregor).

Batch-wise unfolding: each batch (n_params, n_time) becomes one row of length
n_params * n_time, then PLS works on these rows.

MultiwayPLSRegression (bioprocess): PLS regression estimates the final quality (titer).
MultiwayPLS (solar): PLS components fitted on the labels 1 = low, 0 = not low, then a
    logistic regression on the components gives the probability of ending low.

One model per cut time: see TrajectoryModel.
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression

from earlycall.methods.baseline import TrajectoryModel


def unfold(X: np.ndarray) -> np.ndarray:
    """(n_batches, n_params, n_time) -> (n_batches, n_params * n_time)."""
    return X.reshape(X.shape[0], -1)


def fit_pls(X: np.ndarray, target: np.ndarray, n_components: int) -> PLSRegression:
    """PLS on the unfolded trajectories (PLSRegression scales the inputs itself)."""
    features = unfold(X)
    n_components = min(n_components, features.shape[1])
    return PLSRegression(n_components=n_components).fit(features, target)


class MultiwayPLSRegression(TrajectoryModel):
    """Bioprocess. Estimates the final quality (titer) from the unfolded trajectory."""

    name = "multiway_pls"

    def __init__(self, n_components: int = 2, low_quantile: float = 0.3, low_threshold: float | None = None):
        super().__init__(low_quantile, low_threshold)
        self.n_components = n_components

    def _target(self, y: np.ndarray) -> np.ndarray:
        return y                                                 # the final quality itself

    def _fit_model(self, X: np.ndarray, target: np.ndarray):
        return fit_pls(X, target, self.n_components)

    def predict_quality(self, X: np.ndarray, t_index: int) -> np.ndarray:
        X = self._check_X(X, self.n_params_, t_index)            # only the past
        return np.ravel(self._model_at(t_index).predict(unfold(X)))


class MultiwayPLS(TrajectoryModel):
    """Solar. PLS components, then logistic regression: probability of ending low (faulty)."""

    name = "multiway_pls"

    def __init__(self, n_components: int = 2, low_quantile: float = 0.3, low_threshold: float | None = None):
        super().__init__(low_quantile, low_threshold)
        self.n_components = n_components

    def _target(self, y: np.ndarray) -> np.ndarray:
        return self.is_low(y).astype(int)                        # 1 = low, 0 = not low

    def _fit_model(self, X: np.ndarray, target: np.ndarray):
        pls = fit_pls(X, target, self.n_components)
        classifier = LogisticRegression(max_iter=1000).fit(pls.transform(unfold(X)), target)
        return pls, classifier

    def predict_risk(self, X: np.ndarray, t_index: int) -> np.ndarray:
        X = self._check_X(X, self.n_params_, t_index)            # only the past
        pls, classifier = self._model_at(t_index)
        components = pls.transform(unfold(X))
        return classifier.predict_proba(components)[:, 1]        # column 1 = class "low"
