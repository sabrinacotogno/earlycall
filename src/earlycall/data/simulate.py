"""Simple synthetic fed-batch curves with known ground truth (used by the tests).

Not a realistic bioreactor model (for that, see cho_simulator.py): simple curves with the
right qualitative shapes, so that we KNOW which runs belong together.

Parameters simulated (all functions of time t in hours):
    vcd       viable cell density: logistic growth K / (1 + exp(-r (t - t_mid))),
              then exponential decline at rate kd after t_death
    glucose   consumed in proportion to the cells present, clipped at 0
    lactate   produced with growth, consumed after t_mid with time constant lac_tau
    ph        decreases when lactate is high
Quality y   = final yield, depends on the recipe + noise.
Noise       = Gaussian, with std `noise` x the amplitude of each curve.

Scenarios (each one stresses a different weakness):
    "clean"       3 recipes differing in growth timing, decline and lactate consumption
    "level_shift" adds recipe "A_low": the shape of A at 60% of its level, with lower yield
                  -> Pearson cannot separate it from A, a level-aware method can
    "time_shift"  random start delays of 0-24 h for every run
                  -> correlation and Euclidean distances suffer, time warping would not
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from earlycall.batch_data import BatchData

PARAMS = ["vcd", "glucose", "lactate", "ph"]

RECIPES: dict[str, dict[str, float]] = {
    # good: steady growth, long plateau, lactate consumed
    "A": {"K": 20.0, "r": 0.06, "t_mid": 120.0, "t_death": 280.0, "kd": 0.004, "lac_tau": 50.0, "yield_mean": 3.0},
    # medium: fast growth, early decline
    "B": {"K": 16.0, "r": 0.09, "t_mid": 80.0, "t_death": 170.0, "kd": 0.010, "lac_tau": 120.0, "yield_mean": 2.4},
    # poor: slow growth, lactate never consumed
    "C": {"K": 22.0, "r": 0.035, "t_mid": 170.0, "t_death": 400.0, "kd": 0.0, "lac_tau": 1e6, "yield_mean": 1.8},
}

SCENARIOS = ("clean", "level_shift", "time_shift")

GLUCOSE_0 = 6.0           # g/L at inoculation
GLUCOSE_UPTAKE = 0.0012   # g/L consumed per (1e6 cells/mL x h)
LACTATE_YIELD = 0.12      # g/L lactate per 1e6 cells/mL during growth
PH_0, PH_SLOPE = 7.10, 0.15


def simulate_batch(
    recipe: dict[str, float],
    time: np.ndarray,
    rng: np.random.Generator,
    level_scale: float = 1.0,
    time_shift: float = 0.0,
    noise: float = 0.03,
) -> np.ndarray:
    """Simulate one run. Returns an array of shape (len(PARAMS), len(time)).

    level_scale multiplies the amplitudes (same shape, different level).
    time_shift delays the whole run (hours).
    noise is the std of the Gaussian noise, as a fraction of each curve's amplitude.
    """
    # run-to-run variability around the recipe
    K = recipe["K"] * level_scale * (1 + 0.05 * rng.standard_normal())
    r = recipe["r"] * (1 + 0.05 * rng.standard_normal())
    t_mid = recipe["t_mid"] + time_shift + 2.0 * rng.standard_normal()

    t_death = recipe["t_death"] + time_shift
    vcd = K / (1 + np.exp(-r * (time - t_mid)))
    vcd = vcd * np.exp(-recipe["kd"] * np.clip(time - t_death, 0.0, None))

    # glucose: consumed in proportion to the cells present (cumulated over time)
    dt = np.diff(time, prepend=time[0])
    glucose = np.clip(GLUCOSE_0 - GLUCOSE_UPTAKE * np.cumsum(vcd * dt), 0.0, None)

    # lactate: follows growth, then is consumed after the growth midpoint
    after_mid = np.clip(time - t_mid, 0.0, None)
    lactate = LACTATE_YIELD * vcd * np.exp(-after_mid / recipe["lac_tau"])

    ph = PH_0 - PH_SLOPE * lactate

    curves = np.vstack([vcd, glucose, lactate, ph])
    amplitude = curves.max(axis=1, keepdims=True) - curves.min(axis=1, keepdims=True)
    curves = curves + noise * amplitude * rng.standard_normal(curves.shape)
    curves[:3] = np.clip(curves[:3], 0.0, None)   # cells and concentrations cannot be negative
    return curves


def add_missing(X: np.ndarray, rng: np.random.Generator, rate: float = 0.01, max_gap: int = 4) -> np.ndarray:
    """Insert random NaN gaps (1..max_gap points) so that cleaning has something to do.

    `rate` is the probability that a gap starts at a given point. Gaps never touch
    the first or last point, so that interpolation can always fill them.
    """
    X = X.copy()
    n_batches, n_params, n_time = X.shape
    starts = rng.random(X.shape) < rate
    starts[:, :, 0] = False
    starts[:, :, n_time - max_gap - 1:] = False
    for b, p, t in zip(*np.nonzero(starts)):
        X[b, p, t:t + rng.integers(1, max_gap + 1)] = np.nan
    return X


def add_faulty(X: np.ndarray, rng: np.random.Generator, n_faulty: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt a few runs (frozen sensor or drift on one parameter).

    These are SENSOR faults: the process itself is fine, so the run's quality y is
    not changed. Returns (X, is_faulty) with is_faulty a boolean array (n_batches,).
    """
    X = X.copy()
    n_batches, n_params, n_time = X.shape
    is_faulty = np.zeros(n_batches, dtype=bool)
    for b in rng.choice(n_batches, size=min(n_faulty, n_batches), replace=False):
        p = rng.integers(n_params)
        t0 = rng.integers(n_time // 4, n_time // 2)
        if rng.random() < 0.5:   # frozen: the sensor repeats its last value
            X[b, p, t0:] = X[b, p, t0]
        else:                    # drift: slowly growing offset, up to 30% of the signal range
            span = np.nanmax(X[b, p]) - np.nanmin(X[b, p])
            X[b, p, t0:] += np.linspace(0, 0.3 * span, n_time - t0)
        is_faulty[b] = True
    return X, is_faulty


def simulate_batches(
    n_batches: int = 50,
    duration_h: float = 336.0,
    step_h: float = 3.0,
    scenario: str = "clean",
    noise: float = 0.03,
    missing_rate: float = 0.01,
    n_faulty: int = 2,
    seed: int = 0,
) -> BatchData:
    """Simulate a full dataset for one scenario.

    meta columns: recipe (true group), is_faulty, time_shift_h, level_scale.
    batch_ids: "run_000", "run_001", ...
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")

    rng = np.random.default_rng(seed)
    time = np.arange(0, duration_h + step_h, step_h)

    # (label, recipe parameters, level_scale) cycled over the runs
    groups = [(name, RECIPES[name], 1.0) for name in RECIPES]
    if scenario == "level_shift":
        groups.append(("A_low", RECIPES["A"], 0.6))   # same shape as A, lower level and yield

    X = np.empty((n_batches, len(PARAMS), len(time)))
    y = np.empty(n_batches)
    rows = []
    for b in range(n_batches):
        label, recipe, level = groups[b % len(groups)]
        shift = rng.uniform(0, 24) if scenario == "time_shift" else 0.0
        X[b] = simulate_batch(recipe, time, rng, level_scale=level, time_shift=shift, noise=noise)
        y[b] = recipe["yield_mean"] * level * (1 + noise * rng.standard_normal())
        rows.append({"recipe": label, "time_shift_h": shift, "level_scale": level})

    X, is_faulty = add_faulty(X, rng, n_faulty)
    X = add_missing(X, rng, missing_rate)

    meta = pd.DataFrame(rows)
    meta.insert(1, "is_faulty", is_faulty)
    return BatchData(
        X=X,
        time=time,
        params=PARAMS,
        batch_ids=[f"run_{b:03d}" for b in range(n_batches)],
        y=y,
        meta=meta,
    )
