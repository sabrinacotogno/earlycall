"""Kaggle solar plant files -> inverter runs with injected faults.

One batch = one run: one inverter (SOURCE_KEY) over the run period 06:00-18:30 of one day.

Solar has no quality measured independently of the curves (energy / irradiation is computed
from the same signals the methods see). So the truth is created: known faults are injected
into 30% of the runs, and quality = 1 (healthy) or 0 (faulty).
  failed string : power x 0.93 from a random hour between 08:00 and 14:00 (level drop)
  stuck tracker : from 13:00, power falls gradually to 60% at 18:30 (shape change)

The weather is removed by comparing each inverter with its peers:
relative_power = power / median power of all inverters at the same date and time.
(irradiation and module temperature come from one sensor per plant: identical for all inverters.)

Cross-validation groups = date: all inverters of a plant see the same weather on a
given day, so splitting runs at random would leak information.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from earlycall.batch_data import BatchData
from earlycall.preprocessing import clean_tables

PARAMS = ["ac_power", "irradiation", "module_temperature"]       # cleaned and gap-filled
INPUTS = ["relative_power"]                                       # what the methods see
DATE_FORMATS = ["%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S"]   # the files do not all use the same one
STEP = "15min"


# ------------------------------------------------------------------ loading
def parse_timestamps(text: pd.Series) -> pd.Series:
    """Parse DATE_TIME with an explicit format (never let pandas guess day vs month)."""
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if parsed.notna().all():
            return parsed
    raise ValueError(f"Unknown date format, e.g. {text.iloc[0]!r}")


def load_generation(path: str | Path) -> pd.DataFrame:
    """One row per inverter and timestamp: timestamp, inverter, ac_power (kW)."""
    df = pd.read_csv(path)
    return pd.DataFrame({
        "timestamp": parse_timestamps(df["DATE_TIME"]),
        "inverter": df["SOURCE_KEY"].astype(str),
        "ac_power": df["AC_POWER"].astype(float),
    })


def load_weather(path: str | Path) -> pd.DataFrame:
    """One row per timestamp: irradiation (kW/m2), module_temperature (degC)."""
    df = pd.read_csv(path)
    return pd.DataFrame({
        "timestamp": parse_timestamps(df["DATE_TIME"]),
        "irradiation": df["IRRADIATION"].astype(float),
        "module_temperature": df["MODULE_TEMPERATURE"].astype(float),
    })


def load_plant(raw_dir: str | Path, plant: int) -> pd.DataFrame:
    """Generation and weather of one plant, merged on timestamp."""
    raw_dir = Path(raw_dir)
    generation = load_generation(raw_dir / f"Plant_{plant}_Generation_Data.csv")
    weather = load_weather(raw_dir / f"Plant_{plant}_Weather_Sensor_Data.csv")
    weather = weather.drop_duplicates("timestamp")
    return generation.merge(weather, on="timestamp", how="left")


# ----------------------------------------------------------------- cleaning
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates(["inverter", "timestamp"])
    return df, before - len(df)


def remove_impossible_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Negative power or irradiation is a sensor error -> NaN."""
    df = df.copy()
    bad = (df["ac_power"] < 0) | (df["irradiation"] < 0)
    df.loc[bad, ["ac_power", "irradiation"]] = float("nan")
    return df, int(bad.sum())


def remove_frozen_power(df: pd.DataFrame, n_repeats: int = 4) -> tuple[pd.DataFrame, int]:
    """A non-zero power value repeated n_repeats times in a row = frozen sensor -> NaN."""
    df = df.sort_values(["inverter", "timestamp"]).copy()
    same_as_previous = df.groupby("inverter")["ac_power"].diff().eq(0)
    run_length = same_as_previous.groupby(
        (~same_as_previous).cumsum()
    ).cumsum()
    frozen = (run_length >= n_repeats - 1) & (df["ac_power"] > 0)
    df.loc[frozen, "ac_power"] = float("nan")
    return df, int(frozen.sum())


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Basic data-quality pass. Returns (clean table, report of what was found)."""
    df, n_duplicates = remove_duplicates(df)
    df, n_impossible = remove_impossible_values(df)
    df, n_frozen = remove_frozen_power(df)
    report = {"duplicates": n_duplicates, "impossible_values": n_impossible, "frozen_points": n_frozen}
    return df, report


# ----------------------------------------------------------- inverter runs
def keep_daylight(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep rows between start and end (clock time, e.g. "06:00" and "18:30")."""
    clock = df["timestamp"].dt.strftime("%H:%M")
    return df[(clock >= start) & (clock <= end)]


def complete_grid(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """One row per inverter, date and 15-min slot; missing timestamps become NaN rows."""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date.astype(str)
    slots = pd.date_range(f"2000-01-01 {start}", f"2000-01-01 {end}", freq=STEP).strftime("%H:%M")
    df["slot"] = df["timestamp"].dt.strftime("%H:%M")

    full_index = pd.MultiIndex.from_product(
        [sorted(df["inverter"].unique()), sorted(df["date"].unique()), slots],
        names=["inverter", "date", "slot"],
    )
    df = df.set_index(["inverter", "date", "slot"])[PARAMS].reindex(full_index).reset_index()
    return df


def slot_to_hours(slot: pd.Series) -> pd.Series:
    """"06:15" -> 6.25"""
    hours = slot.str.slice(0, 2).astype(int)
    minutes = slot.str.slice(3, 5).astype(int)
    return hours + minutes / 60


def daily_quality(values: pd.DataFrame) -> pd.Series:
    """Per batch: sum of AC power / sum of irradiation (both over the same 15-min slots)."""
    totals = values.groupby("batch_id")[["ac_power", "irradiation"]].sum()
    return totals["ac_power"] / totals["irradiation"]


def to_inverter_runs(df: pd.DataFrame, plant: int, start: str = "06:00", end: str = "18:30",
                     min_daily_irradiation: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the values and batches tables (see batch_data.py).

    Runs with almost no sun (sum of irradiation below min_daily_irradiation) are dropped:
    their normalised quality is meaningless.
    """
    grid = complete_grid(keep_daylight(df, start, end), start, end)
    grid["batch_id"] = grid["inverter"] + "_" + grid["date"]
    grid["time"] = slot_to_hours(grid["slot"])

    values = grid[["batch_id", "time", *PARAMS]].copy()
    batches = grid[["batch_id", "inverter", "date"]].drop_duplicates("batch_id").reset_index(drop=True)
    batches.insert(1, "plant", plant)

    sunny = values.groupby("batch_id")["irradiation"].sum() >= min_daily_irradiation
    batches = batches[batches["batch_id"].map(sunny)].reset_index(drop=True)
    values = values[values["batch_id"].isin(batches["batch_id"])]
    return values, batches

 
# ------------------------------------------------------------ fault injection
def inject_faults(values: pd.DataFrame, batches: pd.DataFrame, share: float = 0.3,
                  string_loss: float = 0.07, tracker_loss: float = 0.4, tracker_start: float = 13.0,
                  seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Damage ac_power of `share` of the batches; quality = 1 healthy, 0 faulty."""
    rng = np.random.default_rng(seed)
    n = len(batches)
    batches = batches.copy()
    faulty = rng.random(n) < share
    batches["fault_type"] = np.where(faulty, rng.choice(["failed string", "stuck tracker"], n), "none")
    batches["onset"] = np.select(
        [batches["fault_type"] == "failed string", batches["fault_type"] == "stuck tracker"],
        [rng.uniform(8, 14, n), tracker_start],
        default=np.nan,
    )
    batches.insert(1, "quality", np.where(faulty, 0.0, 1.0))

    values = values.copy()
    by_batch = batches.set_index("batch_id")
    fault_type = values["batch_id"].map(by_batch["fault_type"])
    onset = values["batch_id"].map(by_batch["onset"])

    string_hit = (fault_type == "failed string") & (values["time"] >= onset)
    values.loc[string_hit, "ac_power"] *= 1 - string_loss

    tracker = fault_type == "stuck tracker"
    end = values["time"].max()
    factor = 1 - tracker_loss * ((values["time"] - tracker_start) / (end - tracker_start)).clip(0, 1)
    values.loc[tracker, "ac_power"] *= factor[tracker]
    return values, batches


def add_relative_power(values: pd.DataFrame, batches: pd.DataFrame, min_peer_power: float = 50.0) -> pd.DataFrame:
    """relative_power = power / median power of all inverters at the same date and time.

    At dawn and dusk (median below min_peer_power kW) the ratio is unstable: set to 1.0.
    """
    values = values.merge(batches[["batch_id", "date"]], on=["batch_id","date"])
    peer_median = values.groupby(["date", "time"])["ac_power"].transform("median")
    values["relative_power"] = (values["ac_power"] / peer_median).where(peer_median >= min_peer_power, 1.0)
    return values.drop(columns="date")


# ------------------------------------------------------------------ pipeline
def prepare_plant(raw_dir: str | Path, plant: int) -> tuple[BatchData, dict]:
    """Load -> clean -> inverter runs -> fill gaps -> inject faults -> relative power -> BatchData."""
    df = load_plant(raw_dir, plant)
    df, report = clean(df)

    values, batches = to_inverter_runs(df, plant)
    values, batches, gap_report = clean_tables(values, batches, PARAMS, max_gap=2)
    report.update(gap_report)

    values, batches = inject_faults(values, batches)
    values = add_relative_power(values, batches)
    report["faults"] = batches["fault_type"].value_counts().to_dict()
    return BatchData.from_tables(values[["batch_id", "time", *INPUTS]], batches), report
