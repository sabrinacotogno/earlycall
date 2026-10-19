import numpy as np
import pytest

from earlycall.batch_data import BatchData
from earlycall.data.simulate import simulate_batches
from earlycall.evaluation import evaluate
from earlycall.methods import BaseRate, BatchMonitor, default_methods


def make_data(y):
    n = len(y)
    return BatchData(X=np.zeros((n, 2, 5)), time=np.arange(5), params=["a", "b"],
                     batch_ids=[f"r{i}" for i in range(n)], y=np.asarray(y, dtype=float))


def test_template_refuses_incomplete_method():
    class Incomplete(BatchMonitor):                         # no fit
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_base_rate_threshold_and_risk():
    model = BaseRate(low_quantile=0.3).fit(make_data(range(1, 11)))
    assert model.low_threshold_ == pytest.approx(3.7)   # 30th percentile of 1..10
    assert model.base_rate_ == pytest.approx(0.3)       # 1, 2, 3 are low
    assert np.allclose(model.predict_risk(np.zeros((4, 2, 5)), t_index=3), 0.3)


def test_risk_never_reads_the_future():
    model = BaseRate().fit(make_data([1, 2, 3, 4]))
    with pytest.raises(ValueError):
        model.predict_risk(np.zeros((1, 2, 5)), t_index=6)


def test_full_benchmark_runs_on_synthetic_data():
    data = simulate_batches(scenario="clean", n_batches=30, missing_rate=0.0, n_faulty=0, seed=2)
    results = evaluate(default_methods(), data, t_indices=[20, 60], n_splits=3)
    assert set(results["method"]) == {m.name for m in default_methods()}
    assert len(results) == 4 * 3 * 2   # methods x folds x cut points


def test_fixed_low_threshold_for_yes_no_label():
    from earlycall.methods import BatchGrouper
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 1, 20)).cumsum(axis=2)
    y = np.array([1.0] * 8 + [0.0] * 4)                     # 1 = good, 0 = faulty (33% faulty)
    data = BatchData(X, np.arange(20), ["p"], [f"b{i}" for i in range(12)], y)
    model = BatchGrouper(low_threshold=0.5).fit(data)
    assert model.low_threshold_ == 0.5
    assert model.base_rate_ == 4 / 12


def test_trajectory_models_train_one_model_per_cut_on_the_past_only():
    from earlycall.methods import LogisticOnFeatures, MultiwayPLS, MultiwayPLSRegression, RidgeOnFeatures
    data = simulate_batches(scenario="clean", n_batches=30, missing_rate=0.0, n_faulty=0, seed=3)
    X = data.X[:5]
    garbage_future = X.copy()
    garbage_future[:, :, 40:] = 1e6                              # absurd values after t_index = 40
    for model, prediction in [(LogisticOnFeatures(), "predict_risk"), (MultiwayPLS(), "predict_risk"),
                              (RidgeOnFeatures(), "predict_quality"), (MultiwayPLSRegression(), "predict_quality")]:
        model.fit(data)
        predict = getattr(model, prediction)
        early = predict(X, t_index=40)
        assert np.allclose(early, predict(garbage_future, t_index=40))             # future never read
        predict(X, t_index=80)
        assert sorted(model.models_per_cut_) == [40, 80]                            # one model per cut


def test_every_risk_is_a_probability_of_ending_low():
    from earlycall.methods import LogisticOnFeatures, MultiwayPLS
    data = simulate_batches(scenario="clean", n_batches=30, missing_rate=0.0, n_faulty=0, seed=3)
    for method in (LogisticOnFeatures(), MultiwayPLS()):
        model = method.fit(data)
        assert set(model.train_target_) == {0, 1}                        # trained on low / not low
        risk = model.predict_risk(data.X, t_index=60)
        assert ((risk >= 0) & (risk <= 1)).all()
        # low runs get a higher risk than good runs on average (training data)
        assert risk[model.train_target_ == 1].mean() > risk[model.train_target_ == 0].mean()


def test_auc_by_cause():
    from earlycall.evaluation import auc_by_cause
    risk = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.3])
    is_low = np.array([True, True, True, False, False, False])
    causes = np.array(["tracker", "tracker", "string", "none", "none", "none"])
    result = auc_by_cause(risk, is_low, causes)
    assert result["auc_tracker"] == 1.0                    # both tracker runs above all healthy ones
    assert result["auc_string"] == pytest.approx(1 / 3)   # 0.2 is above only 0.1


def test_bioprocess_benchmark_scores_estimated_quality():
    data = simulate_batches(scenario="clean", n_batches=30, missing_rate=0.0, n_faulty=0, seed=2)
    methods = default_methods(estimate_quality=True)
    assert [m.name for m in methods] == ["base_rate", "ridge", "correlation", "multiway_pls"]
    results = evaluate(methods, data, t_indices=[20, 60], n_splits=3, estimate_quality=True)
    assert len(results) == 4 * 3 * 2
    base = results[results["method"] == "base_rate"]
    # no information: never an alarm, ranking = coin flip (NaN where a fold has one class only)
    assert (base["recall"].fillna(0) == 0).all() and (base["false_alarm"].fillna(0) == 0).all()
    assert (base["auc"].dropna() == 0.5).all()


def test_ridge_and_pls_regression_estimate_quality():
    from earlycall.methods import MultiwayPLSRegression, RidgeOnFeatures
    data = simulate_batches(scenario="clean", n_batches=30, missing_rate=0.0, n_faulty=0, seed=3)
    for method in (RidgeOnFeatures(), MultiwayPLSRegression()):
        model = method.fit(data)
        assert np.array_equal(model.train_target_, data.y)              # trained on the quality itself
        full = model.predict_quality(data.X, t_index=data.n_time)
        assert np.corrcoef(full, data.y)[0, 1] > 0.5                    # estimates follow the truth


def test_quality_scores_alarm_below_threshold_and_no_alarm_without_information():
    from earlycall.evaluation import quality_scores
    estimate = np.array([1.0, np.nan, 3.0, 2.5])
    is_low = np.array([True, True, False, False])
    scores = quality_scores(estimate, is_low, threshold=2.0, mean_quality=2.8)
    assert scores["recall"] == 0.5          # 1.0 < 2 raises an alarm, NaN does not
    assert scores["false_alarm"] == 0.0
    assert scores["auc"] == 0.75            # NaN ranked as 2.8: above 2.5, below 3.0


def test_risk_alarm_above_training_share_of_low_runs():
    from earlycall.evaluation import risk_scores
    risk = np.array([0.6, 0.27, 0.35, 0.1])
    is_low = np.array([True, True, False, False])
    scores = risk_scores(risk, is_low, alarm_level=0.27)   # 0.27 = share of low runs in training
    assert scores["recall"] == 0.5          # 0.6 > 0.27 alarm; 0.27 (no information) no alarm
    assert scores["false_alarm"] == 0.5     # 0.35 > 0.27 alarm


def test_solar_alarm_level_comes_from_training():
    rng = np.random.default_rng(0)
    y = np.array([0.0] * 3 + [1.0] * 7)     # 1 = healthy, 0 = faulty: 30% faulty
    model = BaseRate(low_threshold=0.5).fit(make_data(y))
    assert model.base_rate_ == pytest.approx(0.3)
    y[:1] = 1.0                             # 20% faulty
    assert BaseRate(low_threshold=0.5).fit(make_data(y)).base_rate_ == pytest.approx(0.2)
