"""Fair comparison of methods: grouped splits, scores as a function of time into the batch."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from earlycall.batch_data import BatchData
from earlycall.methods.base import BatchMonitor

METRICS = ["auc", "recall", "false_alarm", "brier"]


def cv_groups(data: BatchData) -> np.ndarray:
    """Label that must never be split between train and test.

    Solar: the date (all inverters on a day share the weather). Otherwise: the batch itself.
    """
    if "date" in data.meta.columns:
        return data.meta["date"].astype(str).to_numpy()
    return np.asarray(data.batch_ids)


def grouped_splits(groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) pairs where a group never appears on both sides."""
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(groups)), groups=groups))


def cut_points(n_time: int, n_cuts: int = 8) -> list[int]:
    """Evenly spaced t_index values from 10% to 100% of the run (at least 3 points)."""
    points = np.linspace(0.1 * n_time, n_time, n_cuts).astype(int)
    return sorted(set(np.clip(points, 3, n_time).tolist()))


# ------------------------------------------------------------------ scores
def auc_or_nan(is_low: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve: probability that a random low run gets a higher score than a
    random good run (1 = perfect ranking, 0.5 = coin flip). NaN if only one class is present."""
    both_classes = 0 < is_low.sum() < len(is_low)
    return roc_auc_score(is_low, score) if both_classes else np.nan


def alarm_rates(alarm: np.ndarray, is_low: np.ndarray) -> dict[str, float]:
    """recall = share of low runs with an alarm; false_alarm = share of good runs with an alarm."""
    return {
        "recall": alarm[is_low].mean() if is_low.any() else np.nan,
        "false_alarm": alarm[~is_low].mean() if (~is_low).any() else np.nan,
    }


def quality_scores(estimate: np.ndarray, is_low: np.ndarray, threshold: float,
                   mean_quality: float) -> dict[str, float]:
    """Bioprocess: scores of the estimated final quality (titer).

    alarm: estimate < threshold. NaN estimate (not enough information): no alarm.
    auc: lower estimate = riskier; a NaN estimate is ranked as a typical run (training mean).
    brier: NaN, there is no probability to score.
    """
    estimate, is_low = np.asarray(estimate, dtype=float), np.asarray(is_low, dtype=bool)
    alarm = estimate < threshold                                  # NaN < threshold is False
    ranking = -np.where(np.isnan(estimate), mean_quality, estimate)
    return {"auc": auc_or_nan(is_low, ranking), **alarm_rates(alarm, is_low), "brier": np.nan}


def risk_scores(risk: np.ndarray, is_low: np.ndarray, alarm_level: float) -> dict[str, float]:
    """Solar: scores of the probability of a fault.

    alarm: risk > alarm_level = share of faulty runs in TRAINING (riskier than a typical run).
        A risk equal to it (no information) gives no alarm.
    auc: higher risk = riskier.
    brier: mean squared error of the probabilities (lower is better).
    """
    risk, is_low = np.asarray(risk, dtype=float), np.asarray(is_low, dtype=bool)
    alarm = risk > alarm_level
    return {"auc": auc_or_nan(is_low, risk), **alarm_rates(alarm, is_low),
            "brier": brier_score_loss(is_low, risk)}


def auc_by_cause(risk: np.ndarray, is_low: np.ndarray, causes: np.ndarray) -> dict[str, float]:
    """Solar: AUC for each cause alone, low runs with that cause against all good runs.

    causes: one label per run (e.g. fault_type). Returns {"auc_<cause>": value}.
    """
    risk, is_low, causes = np.asarray(risk, dtype=float), np.asarray(is_low, dtype=bool), np.asarray(causes)
    result = {}
    for cause in sorted(set(causes[is_low])):
        keep = (is_low & (causes == cause)) | ~is_low
        result["auc_" + str(cause).replace(" ", "_")] = auc_or_nan(is_low[keep], risk[keep])
    return result


# -------------------------------------------------------------- evaluation
def scores_at_cut(fitted: BatchMonitor, X: np.ndarray, is_low: np.ndarray, t_index: int,
                  estimate_quality: bool, causes: np.ndarray | None) -> dict[str, float]:
    """All scores of one fitted method on the test runs, with data up to t_index.

    estimate_quality: True = bioprocess (predict_quality), False = solar (predict_risk).
    causes: solar only, one fault type per test run (adds one AUC per fault type).
    """
    if estimate_quality:
        estimate = fitted.predict_quality(X, t_index)
        return quality_scores(estimate, is_low, fitted.low_threshold_, fitted.mean_quality_)
    risk = fitted.predict_risk(X, t_index)
    scores = risk_scores(risk, is_low, alarm_level=fitted.base_rate_)
    if causes is not None:
        scores.update(auc_by_cause(risk, is_low, causes))
    return scores


def evaluate_one_fold(method: BatchMonitor, train: BatchData, test: BatchData, t_indices: list[int],
                      cause_column: str | None = None, estimate_quality: bool = False) -> list[dict]:
    """Fit a fresh copy of `method` on train, score it on the test runs at every cut point."""
    fitted = copy.deepcopy(method).fit(train)
    known = ~np.isnan(test.y)
    X, is_low = test.X[known], fitted.is_low(test.y[known])
    causes = test.meta[cause_column].to_numpy()[known] if cause_column else None
    rows = []
    for t_index in t_indices:
        rows.append({"t_index": t_index,
                     "time": float(test.time[t_index - 1]),
                     "percent": round(float(test.percent[t_index - 1])),   # % of the batch duration
                     **scores_at_cut(fitted, X, is_low, t_index, estimate_quality, causes)})
    return rows


def evaluate(methods: list[BatchMonitor], data: BatchData, t_indices: list[int] | None = None,
             n_splits: int = 5, cause_column: str | None = None,
             estimate_quality: bool = False) -> pd.DataFrame:
    """Every method, every fold, every cut point.

    Returns one row per (method, fold, t_index): method, fold, t_index, time, percent, *METRICS,
    plus auc_<cause> columns when cause_column is given.
    """
    t_indices = t_indices or cut_points(data.n_time)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(grouped_splits(cv_groups(data), n_splits)):
        train, test = data.subset(train_idx), data.subset(test_idx)
        for method in methods:
            for row in evaluate_one_fold(method, train, test, t_indices, cause_column, estimate_quality):
                rows.append({"method": method.name, "fold": fold, **row})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean over folds, per method and cut point (every score column, including auc_<cause>)."""
    scores = [c for c in results.columns if c in METRICS or c.startswith("auc_")]
    return results.groupby(["method", "t_index", "time", "percent"], as_index=False)[scores].mean()


def grouping_score(labels: np.ndarray, true_groups: np.ndarray) -> float:
    """Adjusted Rand index between found groups and the true recipes (simulated data only).

    1 = same partition, 0 = no better than chance. Ungrouped batches (-1) form their own label.
    """
    return float(adjusted_rand_score(np.asarray(true_groups).astype(str), np.asarray(labels)))
