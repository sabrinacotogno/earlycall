"""BatchGrouper: groups of runs that are pairwise correlated on EVERY parameter.

Optional smoothing (smoothing_window > 0): every curve is first replaced by its trailing
moving average (mean of the current and previous points). It uses past points only, so a
running batch and a finished one are smoothed the same way. smoothing_window = 0: no smoothing.

Fit (complete training batches)
    1. For each parameter, Pearson correlation between all pairs of batches.
    2. Two batches are "similar" if their correlation is >= corr_threshold on every
       parameter, i.e. if the MINIMUM correlation over parameters passes the threshold.
    3. A group = a set of batches that are ALL similar to each other (a clique),
       with at least min_group_size members. Exact search: list every maximal clique.
       A batch found in several cliques goes to the clique with the highest average
       correlation (computed once, on the full cliques), so the result is unique.
       Batches left ungrouped are grouped again among themselves, until nothing changes.
       If the exact search takes longer than max_exact_seconds, a greedy search is
       used instead (fast, but the result depends on the order batches are tried).
    4. Per group: reference profile (mean), band (std), quality range (min, max of y),
       mean quality and risk = share of members that ended low.
       y is used only here, never for grouping.

Monitor (run still going, data up to t_index)
    5. Correlate the partial run with each group's reference profile cut at t_index.
       Assigned group = all parameters >= corr_threshold, best mean correlation; -1 if none.
    6. Bioprocess (predict_quality): estimated quality = mean quality of the assigned group;
       NaN (not enough information) if no group fits.
       Solar (predict_risk): risk = share of low runs in the assigned group; the training
       base rate if no group fits.
    7. Diagnosis (diagnose): the assigned group, its past runs, how many ended low and,
       when a cause is known for the training runs (e.g. fault_type), the most common
       cause among the low ones.
"""
from __future__ import annotations

import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd

from earlycall.batch_data import BatchData
from earlycall.methods.base import BatchMonitor


# --------------------------------------------------------------- smoothing
def trailing_moving_average(X: np.ndarray, window: int) -> np.ndarray:
    """Replace each point by the mean of itself and the previous window-1 points.

    X: (n_batches, n_params, n_time). Only past points are used, so cutting a batch at time t
    and then smoothing gives the same values as smoothing first and cutting after.
    The first points use the points available (fewer than `window`).
    window 0 or 1: X is returned unchanged.
    """
    if window <= 1:
        return X
    smoothed = np.empty(X.shape, dtype=float)
    for p in range(X.shape[1]):
        table = pd.DataFrame(X[:, p, :].T)          # rows = time points, columns = batches
        smoothed[:, p, :] = table.rolling(window, min_periods=1).mean().to_numpy().T
    return smoothed


# ------------------------------------------------------------ correlations
def correlation_with_reference(curves: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Pearson correlation of each curve (row) with one reference curve.

    curves: (n_batches, n_time), reference: (n_time,). Returns (n_batches,).
    A flat curve or reference gives 0 (no evidence of similarity).
    """
    curves_centered = curves - curves.mean(axis=1, keepdims=True)
    reference_centered = reference - reference.mean()
    numerator = curves_centered @ reference_centered
    denominator = np.linalg.norm(curves_centered, axis=1) * np.linalg.norm(reference_centered)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = numerator / denominator
    return np.nan_to_num(correlation, nan=0.0)


def pairwise_correlation(X: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of batches, per parameter.

    X: (n_batches, n_params, n_time), no NaN. Returns (n_params, n_batches, n_batches).
    Flat curves give NaN in np.corrcoef: they are set to 0 (no evidence of similarity).
    """
    matrices = []
    for p in range(X.shape[1]):
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.corrcoef(X[:, p, :])
        matrices.append(np.nan_to_num(matrix, nan=0.0))
    return np.stack(matrices)


# ------------------------------------------------------------------ groups
def link_matrix(similarity: np.ndarray, threshold: float) -> np.ndarray:
    """True where two batches are linked: similarity >= threshold (a batch is not linked to itself)."""
    linked = similarity >= threshold
    np.fill_diagonal(linked, False)
    return linked


def all_cliques(linked: np.ndarray, max_seconds: float) -> list[list[int]] | None:
    """Every maximal clique: a set of batches all linked to each other, that no other batch can join.

    Exact search (networkx.find_cliques). Returns None if it takes longer than max_seconds.
    """
    graph = nx.from_numpy_array(linked.astype(int))
    start = time.perf_counter()
    cliques = []
    for clique in nx.find_cliques(graph):
        cliques.append(sorted(clique))
        if time.perf_counter() - start > max_seconds:
            return None
    return cliques


def average_correlation(members: list[int], similarity: np.ndarray) -> float:
    """Mean similarity over all pairs of members."""
    pairs = [similarity[i, j] for i in members for j in members if i < j]
    return float(np.mean(pairs))


def resolve_overlaps(cliques: list[list[int]], similarity: np.ndarray) -> list[list[int]]:
    """A batch found in several cliques stays only in the one with the highest average correlation.

    Averages are computed once, on the full cliques, so the order of the cliques does not
    matter. On an exact tie, the first clique in the list wins.
    """
    averages = [average_correlation(members, similarity) for members in cliques]
    best_clique = {}
    for c, members in enumerate(cliques):
        for batch in members:
            if batch not in best_clique or averages[c] > averages[best_clique[batch]]:
                best_clique[batch] = c
    return [[batch for batch in members if best_clique[batch] == c]
            for c, members in enumerate(cliques)]


def one_pass(batches: list[int], similarity: np.ndarray, threshold: float, min_group_size: int,
             max_seconds: float) -> list[list[int]] | None:
    """One round of exact grouping on a subset of batches. None if the search is too slow."""
    sub_similarity = similarity[np.ix_(batches, batches)]
    cliques = all_cliques(link_matrix(sub_similarity, threshold), max_seconds)
    if cliques is None:
        return None
    large = sorted(members for members in cliques if len(members) >= min_group_size)
    groups = resolve_overlaps(large, sub_similarity)
    # back from positions in the subset to the original batch numbers
    return [[batches[i] for i in members] for members in groups if len(members) >= min_group_size]


def exact_groups(similarity: np.ndarray, threshold: float, min_group_size: int,
                 max_seconds: float) -> list[np.ndarray] | None:
    """Unique, non-overlapping groups from all maximal cliques. None if the search is too slow.

    Recycling: batches left ungrouped (their clique dissolved) are grouped again among
    themselves, pass after pass, until a pass finds no new group.
    """
    deadline = time.perf_counter() + max_seconds
    ungrouped = list(range(len(similarity)))
    groups = []
    while len(ungrouped) >= min_group_size:
        seconds_left = deadline - time.perf_counter()
        new_groups = one_pass(ungrouped, similarity, threshold, min_group_size, seconds_left)
        if new_groups is None:
            return None
        if not new_groups:
            break
        groups += new_groups
        grouped_now = {batch for members in new_groups for batch in members}
        ungrouped = [batch for batch in ungrouped if batch not in grouped_now]
    return [np.array(sorted(members)) for members in groups]


def grow_group(seed: int, linked: np.ndarray, similarity: np.ndarray, available: set[int]) -> list[int]:
    """Start from `seed` and add batches, most similar first, if linked to ALL members."""
    candidates = [j for j in available if j != seed and linked[seed, j]]
    candidates.sort(key=lambda j: similarity[seed, j], reverse=True)
    group = [seed]
    for j in candidates:
        if all(linked[j, member] for member in group):
            group.append(j)
    return group


def greedy_groups(similarity: np.ndarray, threshold: float, min_group_size: int) -> list[np.ndarray]:
    """Fast fallback for large datasets. Groups are valid cliques, but not unique.

    Try seeds in order of how many links they have; grow a group around the first seed
    that gives a large enough group; remove its members; repeat.
    """
    linked = link_matrix(similarity, threshold)
    available = set(range(len(similarity)))
    groups = []

    while len(available) >= min_group_size:
        seeds = sorted(available, key=lambda i: linked[i, list(available)].sum(), reverse=True)
        new_group = None
        for seed in seeds:
            group = grow_group(seed, linked, similarity, available)
            if len(group) >= min_group_size:
                new_group = group
                break
        if new_group is None:
            break
        groups.append(np.array(sorted(new_group)))
        available -= set(new_group)

    return groups


def find_groups(similarity: np.ndarray, threshold: float, min_group_size: int,
                max_seconds: float = 30.0) -> tuple[list[np.ndarray], str]:
    """Exact groups if the search finishes within max_seconds, greedy groups otherwise.

    Returns (groups, method) with method = "exact" or "greedy".
    """
    groups = exact_groups(similarity, threshold, min_group_size, max_seconds)
    if groups is not None:
        return groups, "exact"
    warnings.warn(f"Exact grouping took more than {max_seconds} s: using the greedy search instead")
    return greedy_groups(similarity, threshold, min_group_size), "greedy"


# ----------------------------------------------------------------- method
class BatchGrouper(BatchMonitor):
    """Correlation-based grouping and monitoring (see module docstring).

    Fitted attributes
    -----------------
    groups_ : list of index arrays into the training batches
    grouping_method_ : "exact" or "greedy" (see find_groups)
    labels_ : (n_train,) group of each training batch, -1 = ungrouped
    similarity_ : (n_train, n_train) minimum correlation over parameters
    reference_, band_std_ : (n_groups, n_params, n_time) mean and std profile per group
    group_range_ : (n_groups, 2) min and max final quality of the members
    group_mean_ : (n_groups,) mean final quality of the members
    group_risk_ : (n_groups,) share of members below the low threshold
    train_ids_, train_y_, train_meta_ : the training runs, used by diagnose
    (references and bands are computed on the smoothed curves when smoothing_window > 0)
    """

    name = "correlation"

    def __init__(self, corr_threshold: float = 0.8, min_group_size: int = 3, smoothing_window: int = 0,
                 low_quantile: float = 0.3, low_threshold: float | None = None,
                 max_exact_seconds: float = 30.0):
        super().__init__(low_quantile, low_threshold)
        self.corr_threshold = corr_threshold
        self.min_group_size = min_group_size
        self.smoothing_window = smoothing_window
        self.max_exact_seconds = max_exact_seconds

    # ------------------------------------------------------------------ fit
    def fit(self, train: BatchData) -> "BatchGrouper":
        if np.isnan(train.X).any():
            raise ValueError("BatchGrouper needs data without NaN: clean it first")
        self._fit_threshold(train.y)
        self.n_params_ = train.n_params
        curves = trailing_moving_average(train.X, self.smoothing_window)

        self.similarity_ = pairwise_correlation(curves).min(axis=0)   # worst parameter per pair
        self.groups_, self.grouping_method_ = find_groups(
            self.similarity_, self.corr_threshold, self.min_group_size, self.max_exact_seconds)
        self.labels_ = self._labels(train.n_batches)
        self._describe_groups(curves, train.y)
        self.train_ids_ = np.asarray(train.batch_ids)
        self.train_y_ = np.asarray(train.y, dtype=float)
        self.train_meta_ = train.meta
        return self

    def _labels(self, n_batches: int) -> np.ndarray:
        labels = np.full(n_batches, -1)
        for g, members in enumerate(self.groups_):
            labels[members] = g
        return labels

    def _describe_groups(self, curves: np.ndarray, y: np.ndarray) -> None:
        """Reference (mean), band (std), quality range and risk of each group."""
        references, stds, ranges, means, risks = [], [], [], [], []
        for members in self.groups_:
            member_curves = curves[members]
            quality = y[members]
            quality = quality[~np.isnan(quality)]
            references.append(member_curves.mean(axis=0))
            stds.append(member_curves.std(axis=0))
            ranges.append([quality.min(), quality.max()] if quality.size else [np.nan, np.nan])
            means.append(quality.mean() if quality.size else np.nan)
            risks.append(self.is_low(quality).mean() if quality.size else self.base_rate_)
        shape = (0, curves.shape[1], curves.shape[2])
        self.reference_ = np.array(references) if references else np.empty(shape)
        self.band_std_ = np.array(stds) if stds else np.empty(shape)
        self.group_range_ = np.array(ranges) if ranges else np.empty((0, 2))
        self.group_mean_ = np.array(means, dtype=float)
        self.group_risk_ = np.array(risks, dtype=float)

    @property
    def n_groups_(self) -> int:
        return len(self.groups_)

    # -------------------------------------------------------------- monitor
    def group_correlations(self, X: np.ndarray, t_index: int) -> np.ndarray:
        """Correlation of each partial run with each group reference, per parameter.

        Returns (n_batches, n_groups, n_params).
        """
        X = self._check_X(X, self.n_params_, t_index)
        X = trailing_moving_average(X, self.smoothing_window)   # same smoothing as in fit
        references = self.reference_[:, :, :t_index]
        result = np.zeros((X.shape[0], self.n_groups_, self.n_params_))
        for g in range(self.n_groups_):
            for p in range(self.n_params_):
                result[:, g, p] = correlation_with_reference(X[:, p, :], references[g, p])
        return result

    def assign(self, X: np.ndarray, t_index: int) -> np.ndarray:
        """Group label of each run, -1 if no group fits on every parameter."""
        n_batches = np.asarray(X).shape[0]
        if self.n_groups_ == 0 or t_index < 3:
            return np.full(n_batches, -1)

        correlations = self.group_correlations(X, t_index)
        labels = np.full(n_batches, -1)
        for b in range(n_batches):
            fits_all = (correlations[b] >= self.corr_threshold).all(axis=1)
            if fits_all.any():
                mean_corr = np.where(fits_all, correlations[b].mean(axis=1), -np.inf)
                labels[b] = int(np.argmax(mean_corr))
        return labels

    @staticmethod
    def _value_of_group(labels: np.ndarray, per_group: np.ndarray, if_no_group: float) -> np.ndarray:
        """For each run, the value of its group; `if_no_group` for runs with label -1."""
        return np.array([per_group[g] if g >= 0 else if_no_group for g in labels], dtype=float)

    def predict_quality(self, X: np.ndarray, t_index: int) -> np.ndarray:
        """Bioprocess: mean final quality of the assigned group; NaN (not enough information) if none fits."""
        return self._value_of_group(self.assign(X, t_index), self.group_mean_, np.nan)

    def predict_risk(self, X: np.ndarray, t_index: int) -> np.ndarray:
        """Solar: share of faulty runs in the assigned group; the training share if none fits."""
        return self._value_of_group(self.assign(X, t_index), self.group_risk_, self.base_rate_)

    def diagnose(self, X: np.ndarray, t_index: int, cause_column: str | None = None) -> pd.DataFrame:
        """Where each risk comes from: the assigned group and the past runs in it.

        One row per run:
            group      assigned group (-1 = no group fits)
            risk       same value as predict_risk
            n_members  number of past runs in the group
            members    their batch ids
            n_low      how many of them ended low
        cause_column: a column of the training meta (e.g. "fault_type"). Adds likely_cause =
            the most common cause among the low members (alphabetical first on a tie;
            "" when no member ended low or no group fits).
        """
        rows = [self._describe_match(g, cause_column) for g in self.assign(X, t_index)]
        return pd.DataFrame(rows)

    def _describe_match(self, group: int, cause_column: str | None) -> dict:
        """One row of diagnose for a run assigned to `group` (-1 = none)."""
        if group < 0:
            members, low = np.array([], dtype=int), np.array([], dtype=bool)
            risk = self.base_rate_
        else:
            members = self.groups_[group]
            low = self.is_low(self.train_y_[members])
            risk = self.group_risk_[group]
        row = {"group": int(group), "risk": risk, "n_members": len(members),
               "members": self.train_ids_[members].tolist(), "n_low": int(low.sum())}
        if cause_column is not None:
            causes = self.train_meta_[cause_column].to_numpy()[members][low]
            row["likely_cause"] = str(pd.Series(causes).mode().iloc[0]) if len(causes) else ""
        return row
