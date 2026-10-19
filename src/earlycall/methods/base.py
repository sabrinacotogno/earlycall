"""Common interface: every method is compared through the same calls."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from earlycall.batch_data import BatchData


class BatchMonitor(ABC):
    """Learns from complete training batches, then judges runs that are still going.

    A run is low if its final quality is below `low_threshold_`, the `low_quantile`
    of the TRAINING qualities (0.3 -> the worst 30% of training runs are "low"),
    or a fixed `low_threshold` when one is given (solar: 0.5 for a yes/no label).

    Contract
    --------
    fit(train)                    train contains COMPLETE batches and their quality y.
                                  Every subclass calls self._fit_threshold(train.y) in fit.
    Then, depending on the dataset, one of:
    predict_quality(X, t_index)   bioprocess: estimated final quality (e.g. titer) per run;
                                  NaN = not enough information. Alarm = estimate < low_threshold_.
    predict_risk(X, t_index)      solar: probability that the run is low (faulty), in [0, 1].
                                  Alarm = risk > base_rate_ (share of low runs in training).
    X: (n_batches, n_params, n_time_or_more). Only X[:, :, :t_index] may be used: the rest
    is the future.
    """

    name: str = "base"

    def __init__(self, low_quantile: float = 0.3, low_threshold: float | None = None):
        self.low_quantile = low_quantile
        self.low_threshold = low_threshold      # fixed threshold; overrides low_quantile when given

    @abstractmethod
    def fit(self, train: BatchData) -> "BatchMonitor":
        ...

    def predict_quality(self, X: np.ndarray, t_index: int) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not estimate the final quality")

    def predict_risk(self, X: np.ndarray, t_index: int) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not give a probability")

    # ------------------------------------------------------------ shared helpers
    def _fit_threshold(self, y: np.ndarray) -> None:
        """Set low_threshold_, base_rate_ (share of low runs) and mean_quality_ from training y."""
        y = np.asarray(y, dtype=float)
        y = y[~np.isnan(y)]
        if y.size == 0:
            raise ValueError("No known quality in the training batches")
        if self.low_threshold is not None:      # e.g. 0.5 for a yes/no label coded 1 = good, 0 = bad
            self.low_threshold_ = float(self.low_threshold)
        else:
            self.low_threshold_ = float(np.quantile(y, self.low_quantile))
        self.base_rate_ = float(np.mean(y < self.low_threshold_))
        self.mean_quality_ = float(np.mean(y))

    def is_low(self, y: np.ndarray) -> np.ndarray:
        """True where a quality is below the training threshold (NaN gives False)."""
        return np.asarray(y, dtype=float) < self.low_threshold_

    @staticmethod
    def _check_X(X: np.ndarray, n_params: int, t_index: int) -> np.ndarray:
        """Shared input check; returns X cut at t_index."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 3 or X.shape[1] != n_params:
            raise ValueError(f"X must be (n_batches, {n_params}, n_time), got {X.shape}")
        if not 2 <= t_index <= X.shape[2]:
            raise ValueError(f"t_index must be in [2, {X.shape[2]}], got {t_index}")
        return X[:, :, :t_index]
