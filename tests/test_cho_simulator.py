import numpy as np
import pandas as pd

from earlycall.data.cho_simulator import SCENARIOS, generate, make_design


def test_design_cycles_conditions_and_counts_runs():
    design = make_design(9, SCENARIOS["base"], anomaly_rate=0.0, rng=np.random.default_rng(0))
    assert len(design) == 9
    assert [c for _, c, _ in design[:3]] == SCENARIOS["base"]
    assert all(faults == [] for _, _, faults in design)


def test_generate_writes_three_consistent_files(tmp_path):
    generate(tmp_path, n_runs=3, scenario="base", anomaly_rate=0.0, seed=0)
    online = pd.read_csv(tmp_path / "online_timeseries.csv")
    offline = pd.read_csv(tmp_path / "offline_samples.csv")
    meta = pd.read_csv(tmp_path / "runs_metadata.csv")

    assert len(meta) == 3
    assert set(online.run_id) == set(offline.run_id) == set(meta.run_id)
    assert offline.groupby("run_id").day.max().eq(14).all()
    assert "titer_gL" in offline.columns
