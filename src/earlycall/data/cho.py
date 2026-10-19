"""Simulated CHO fed-batch lab exports -> run-level batches.

Input: the three CSV files written by cho_simulator.py (one folder per scenario).
One batch = one bioreactor run (14 days). Quality = final titer (day 14).

Two sampling rates, as in a real lab:
    online   sensors every 15 min -> averaged per time step
    offline  daily samples        -> interpolated between sample days, never beyond

Rules:
- titer is the OUTCOME: never an input
- control setpoints reveal the recipe directly (temperature, cumulated feed, agitation):
  not used as inputs; controlled variables (pH, dissolved oxygen) are flat by design: not used
- runs are aligned on clock time (hours since inoculation). Aligning on process phases
  (e.g. when temperature reaches its new setpoint) would be more realistic: next step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from earlycall.batch_data import BatchData
from earlycall.preprocessing import clean_tables

# pH and dissolved oxygen are held at their setpoints by the controllers: their curves are
# noise around a constant and carry no shape information, so they are not used either.
ONLINE_PARAMS = ["capacitance_pF_cm", "O2_flow_mLmin"]
OFFLINE_PARAMS = ["VCD_1e6_per_mL", "glucose_gL", "lactate_gL", "ammonium_mM"]
PARAMS = OFFLINE_PARAMS + ONLINE_PARAMS
QUALITY = "titer_gL"
DURATION_H = 336.0


def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read online_timeseries.csv, offline_samples.csv, runs_metadata.csv."""
    raw_dir = Path(raw_dir)
    online = pd.read_csv(raw_dir / "online_timeseries.csv")
    offline = pd.read_csv(raw_dir / "offline_samples.csv")
    meta = pd.read_csv(raw_dir / "runs_metadata.csv")
    return online, offline, meta


def remove_spikes(online: pd.DataFrame, params: list[str], window: int = 9, k: float = 5.0):
    """Points far from the local median -> NaN.

    "Far" = more than k times the noise level, the noise level being the standard
    deviation of (signal - local median) over the run.
    """
    online = online.sort_values(["run_id", "time_h"]).copy()
    n_spikes = 0
    for param in params:
        local_median = online.groupby("run_id")[param].transform(
            lambda s: s.rolling(window, center=True, min_periods=1).median()
        )
        residual = online[param] - local_median
        noise_level = residual.groupby(online["run_id"]).transform("std")
        spike = residual.abs() > k * noise_level.clip(lower=1e-9)
        online.loc[spike, param] = np.nan
        n_spikes += int(spike.sum())
    return online, n_spikes


def online_on_grid(online: pd.DataFrame, grid: np.ndarray, step_h: float) -> pd.DataFrame:
    """Average the online sensors over each time step: one row per run and grid time."""
    online = online.copy()
    online["time"] = (online["time_h"] // step_h) * step_h
    averaged = online.groupby(["run_id", "time"], as_index=False)[ONLINE_PARAMS].mean()
    return averaged[averaged["time"].isin(grid)]


def offline_on_grid(offline: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Interpolate the daily samples onto the grid, run by run (NaN outside the sampled range)."""
    pieces = []
    for run_id, samples in offline.groupby("run_id"):
        piece = pd.DataFrame({"run_id": run_id, "time": grid})
        for param in OFFLINE_PARAMS:
            known = samples[["time_h", param]].dropna()
            piece[param] = np.interp(grid, known["time_h"], known[param], left=np.nan, right=np.nan)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def final_titer(offline: pd.DataFrame, day: int = 14) -> pd.Series:
    """Final titer per run. NaN when the day-14 sample is missing: never invented."""
    last_day = offline[offline["day"] == day].set_index("run_id")
    return last_day[QUALITY]


def prepare_scenario(raw_dir: str | Path, step_h: float = 3.0) -> tuple[BatchData, dict]:
    """Full pipeline for one scenario folder: load -> clean -> common grid -> BatchData."""
    online, offline, meta = load_raw(raw_dir)
    online, n_spikes = remove_spikes(online, ONLINE_PARAMS)

    grid = np.arange(0, DURATION_H + step_h, step_h)
    values = offline_on_grid(offline, grid).merge(online_on_grid(online, grid, step_h),
                                                  on=["run_id", "time"], how="left")
    values = values.rename(columns={"run_id": "batch_id"})[["batch_id", "time", *PARAMS]]

    batches = pd.DataFrame({
        "batch_id": meta["run_id"],
        "quality": meta["run_id"].map(final_titer(offline)),
        "condition": meta["condition"],
        "anomalies": meta["anomalies"],
        "is_faulty": meta["anomalies"] != "none",
    })
    values, batches, report = clean_tables(values, batches, PARAMS, max_gap=2)
    report["spikes_removed"] = n_spikes
    return BatchData.from_tables(values, batches), report
