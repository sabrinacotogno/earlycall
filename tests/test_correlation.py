import numpy as np
import pytest

from earlycall.batch_data import BatchData
from earlycall.data.simulate import simulate_batches
from earlycall.evaluation import grouping_score
from earlycall.methods.correlation import (BatchGrouper, exact_groups, find_groups, greedy_groups,
                                          pairwise_correlation, trailing_moving_average)


def test_pairwise_correlation_shape_and_diagonal():
    X = np.random.default_rng(0).normal(size=(5, 2, 30))
    C = pairwise_correlation(X)
    assert C.shape == (2, 5, 5)
    assert np.allclose(np.diagonal(C, axis1=1, axis2=2), 1.0)


def test_find_groups_needs_every_pair_linked():
    # 0-1 and 1-2 are linked, but not 0-2: no group of 3
    S = np.eye(3)
    S[0, 1] = S[1, 0] = 0.9
    S[1, 2] = S[2, 1] = 0.9
    groups, method = find_groups(S, threshold=0.8, min_group_size=3)
    assert groups == [] and method == "exact"


def two_overlapping_cliques(c_prefers_second: bool) -> np.ndarray:
    """Batches 0,1,2 and 2,3,4 are two cliques sharing batch 2."""
    S = np.eye(5)
    first, second = (0.85, 0.95) if c_prefers_second else (0.95, 0.85)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        S[i, j] = S[j, i] = first
    for i, j in [(2, 3), (2, 4), (3, 4)]:
        S[i, j] = S[j, i] = second
    return S


def test_shared_batch_goes_to_clique_with_highest_average():
    groups = exact_groups(two_overlapping_cliques(True), 0.8, 2, max_seconds=10)
    assert [list(g) for g in groups] == [[0, 1], [2, 3, 4]]
    groups = exact_groups(two_overlapping_cliques(False), 0.8, 2, max_seconds=10)
    assert [list(g) for g in groups] == [[0, 1, 2], [3, 4]]


def test_shared_batch_goes_to_more_similar_clique_even_if_smaller():
    # 0,1,2 (very similar) and 2,3,4,5 (less similar, but larger) share batch 2
    S = np.eye(6)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        S[i, j] = S[j, i] = 0.95
    for i in [2, 3, 4, 5]:
        for j in [2, 3, 4, 5]:
            if i != j:
                S[i, j] = 0.85
    groups = exact_groups(S, 0.8, 2, max_seconds=10)
    assert [list(g) for g in groups] == [[0, 1, 2], [3, 4, 5]]


def test_ungrouped_batches_are_recycled():
    # cliques A,B,C (0.95), C,D,E (0.85), D,E,F (0.82): one pass keeps only A,B,C;
    # the second pass groups D,E,F
    S = np.eye(6)
    A, B, C, D, E, F = range(6)
    for members, value in [((A, B, C), 0.95), ((C, D, E), 0.85), ((D, E, F), 0.82)]:
        for i in members:
            for j in members:
                if i != j:
                    S[i, j] = value
    groups = exact_groups(S, 0.8, 3, max_seconds=10)
    assert [list(g) for g in groups] == [[A, B, C], [D, E, F]]


def test_exact_groups_do_not_depend_on_batch_order():
    S = two_overlapping_cliques(True)
    order = np.array([4, 2, 0, 3, 1])
    shuffled = exact_groups(S[np.ix_(order, order)], 0.8, 2, max_seconds=10)
    back = sorted(sorted(order[g].tolist()) for g in shuffled)
    assert back == [[0, 1], [2, 3, 4]]


def test_too_slow_exact_search_falls_back_to_greedy():
    S = two_overlapping_cliques(True)
    groups, method = find_groups(S, 0.8, 2, max_seconds=-1)
    assert method == "greedy"
    assert [list(g) for g in groups] == [list(g) for g in greedy_groups(S, 0.8, 2)]


def test_rising_and_falling_runs_never_share_a_group():
    t = np.linspace(0, 1, 50)
    up, down = t, 1 - t
    X = np.stack([[up], [up + 0.01 * t**2], [up * 2], [down], [down * 3], [down + 0.5]])
    data = BatchData(X=X, time=t, params=["p"], batch_ids=list("abcdef"), y=np.arange(6.0))
    model = BatchGrouper(corr_threshold=0.9, min_group_size=3).fit(data)
    rising_groups = set(model.labels_[:3]) - {-1}
    falling_groups = set(model.labels_[3:]) - {-1}
    assert not (rising_groups & falling_groups)


def test_recovers_recipes_on_clean_synthetic_data():
    data = simulate_batches(scenario="clean", missing_rate=0.0, n_faulty=0, seed=1)
    model = BatchGrouper(corr_threshold=0.9, min_group_size=3).fit(data)
    assert grouping_score(model.labels_, data.meta["recipe"]) > 0.8


def test_monitoring_returns_one_risk_per_run():
    data = simulate_batches(scenario="clean", missing_rate=0.0, n_faulty=0, seed=1)
    model = BatchGrouper().fit(data)
    risk = model.predict_risk(data.X[:5], t_index=40)
    assert risk.shape == (5,)
    assert ((risk >= 0) & (risk <= 1)).all()


def test_smoothing_window_zero_changes_nothing():
    data = simulate_batches(scenario="clean", missing_rate=0.0, n_faulty=0, seed=1)
    plain = BatchGrouper().fit(data)
    zero = BatchGrouper(smoothing_window=0).fit(data)
    assert np.array_equal(plain.labels_, zero.labels_)
    assert np.allclose(plain.similarity_, zero.similarity_)


def test_smoothing_uses_only_past_points():
    X = np.random.default_rng(0).normal(size=(3, 2, 20))
    smoothed_then_cut = trailing_moving_average(X, 5)[:, :, :12]
    cut_then_smoothed = trailing_moving_average(X[:, :, :12], 5)
    assert np.allclose(smoothed_then_cut, cut_then_smoothed)


def test_smoothing_reduces_jitter():
    X = np.random.default_rng(0).normal(size=(3, 1, 200))
    smoothed = trailing_moving_average(X, 5)
    assert np.diff(smoothed, axis=2).std() < np.diff(X, axis=2).std()


def test_diagnose_points_to_past_runs_and_likely_cause():
    import pandas as pd
    t = np.linspace(0, 1, 50)
    up, down = t, 1 - t
    X = np.stack([[up], [up + 0.01 * t**2], [up * 2], [down], [down * 3], [down + 0.5]])
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])            # 1 = healthy, 0 = faulty
    meta = pd.DataFrame({"fault_type": ["tracker", "tracker", "none", "none", "none", "none"]})
    data = BatchData(X=X, time=t, params=["p"], batch_ids=list("abcdef"), y=y, meta=meta)
    model = BatchGrouper(corr_threshold=0.9, min_group_size=3, low_threshold=0.5).fit(data)
    table = model.diagnose(np.stack([[up * 3]]), t_index=50, cause_column="fault_type")
    row = table.iloc[0]
    assert sorted(row["members"]) == ["a", "b", "c"]
    assert row["n_low"] == 2 and row["likely_cause"] == "tracker"
    assert row["risk"] == pytest.approx(2 / 3)
    assert row["risk"] == model.predict_risk(np.stack([[up * 3]]), t_index=50)[0]


def test_predict_quality_is_group_mean_or_nan():
    t = np.linspace(0, 1, 50)
    up = t
    X = np.stack([[up], [up + 0.01 * t**2], [up * 2]])
    data = BatchData(X=X, time=t, params=["p"], batch_ids=list("abc"), y=np.array([1.0, 2.0, 6.0]))
    model = BatchGrouper(corr_threshold=0.9, min_group_size=3).fit(data)
    new_runs = np.stack([[up * 3], [1 - up]])                          # rising: fits; falling: no group
    assert np.allclose(model.predict_quality(new_runs, t_index=50), [3.0, np.nan], equal_nan=True)
