from earlycall.methods.base import BatchMonitor
from earlycall.methods.baseline import BaseRate, LogisticOnFeatures, RidgeOnFeatures
from earlycall.methods.correlation import BatchGrouper
from earlycall.methods.multiway import MultiwayPLS, MultiwayPLSRegression

__all__ = ["BatchMonitor", "BaseRate", "RidgeOnFeatures", "LogisticOnFeatures", "BatchGrouper",
           "MultiwayPLSRegression", "MultiwayPLS", "dataset_settings", "default_methods"]


def dataset_settings(name: str) -> dict:
    """Settings that depend on the dataset.

    Bioprocess: every method estimates the final titer (estimate_quality = True); alarm when
    the estimate is below the low threshold (30th percentile of training titers).
    Solar: quality is a yes/no label (1 = healthy, 0 = injected fault), so "low" = below 0.5;
    every method gives a probability of a fault (estimate_quality = False); alarm when it is
    above the share of faulty runs in the training set. The relative-power curves are noisy,
    so they are smoothed and grouped with a lower threshold.
    """
    if name.startswith("solar"):
        return {"low_threshold": 0.5, "smoothing_window": 4, "corr_threshold": 0.6, "estimate_quality": False}
    return {"low_threshold": None, "smoothing_window": 0, "corr_threshold": 0.8, "estimate_quality": True}


def default_methods(smoothing_window: int = 0, low_threshold: float | None = None, corr_threshold: float = 0.8,
                    estimate_quality: bool = False) -> list[BatchMonitor]:
    """The methods compared in the benchmark.

    smoothing_window: trailing moving average used by the correlation method only
        (number of time points, 0 = no smoothing).
    low_threshold: fixed "low" threshold for every method (None = bottom 30% of training quality).
    estimate_quality: True = bioprocess: base rate, ridge regression, BatchGrouper, PLS regression
        (each estimates the titer);
        False = solar: base rate, logistic regression, BatchGrouper, PLS + logistic.
    """
    grouper = BatchGrouper(corr_threshold=corr_threshold, min_group_size=3, smoothing_window=smoothing_window,
                           low_threshold=low_threshold)
    if estimate_quality:
        return [
            BaseRate(low_threshold=low_threshold),
            RidgeOnFeatures(alpha=1.0, low_threshold=low_threshold),
            grouper,
            MultiwayPLSRegression(n_components=2, low_threshold=low_threshold),
        ]
    return [
        BaseRate(low_threshold=low_threshold),
        LogisticOnFeatures(C=1.0, low_threshold=low_threshold),
        grouper,
        MultiwayPLS(n_components=2, low_threshold=low_threshold),
    ]
