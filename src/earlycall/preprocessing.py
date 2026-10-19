"""Cleaning steps shared by all datasets, on the `values` table
(columns batch_id, time, <params>, one row per batch and time point)."""
from __future__ import annotations

import pandas as pd


def fill_short_gaps_in_series(series: pd.Series, max_gap: int) -> pd.Series:
    """Linear interpolation of the holes that are at most `max_gap` points long.

    Longer holes, and holes at the start or end, stay NaN.
    (pandas' interpolate(limit=...) would partly fill long holes too.)
    """
    is_missing = series.isna()
    hole_id = (is_missing != is_missing.shift()).cumsum()
    hole_length = is_missing.groupby(hole_id).transform("sum")
    interpolated = series.interpolate(limit_area="inside")
    short_hole = is_missing & (hole_length <= max_gap)
    return series.where(~short_hole, interpolated)


def fill_short_gaps(values: pd.DataFrame, params: list[str], max_gap: int = 3) -> pd.DataFrame:
    """Fill short holes inside each batch, parameter by parameter."""
    values = values.sort_values(["batch_id", "time"]).copy()
    for param in params:
        values[param] = values.groupby("batch_id")[param].transform(
            lambda s: fill_short_gaps_in_series(s, max_gap)
        )
    return values


def find_incomplete_batches(values: pd.DataFrame, params: list[str]) -> list[str]:
    """Batches that still have at least one missing value."""
    has_nan = values[params].isna().any(axis=1)
    return sorted(values.loc[has_nan, "batch_id"].astype(str).unique())


def drop_batches(values: pd.DataFrame, batches: pd.DataFrame, to_drop: list[str]):
    """Remove the given batches from both tables."""
    keep_values = ~values["batch_id"].astype(str).isin(to_drop)
    keep_batches = ~batches["batch_id"].astype(str).isin(to_drop)
    return values[keep_values].reset_index(drop=True), batches[keep_batches].reset_index(drop=True)


def clean_tables(values: pd.DataFrame, batches: pd.DataFrame, params: list[str],
                 max_gap: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fill short gaps, then drop the batches that still have holes.

    Returns (values, batches, report). The report says what was done, so nothing
    is dropped silently.
    """
    n_missing_before = int(values[params].isna().sum().sum())
    values = fill_short_gaps(values, params, max_gap)
    n_missing_after = int(values[params].isna().sum().sum())

    incomplete = find_incomplete_batches(values, params)
    values, batches = drop_batches(values, batches, incomplete)

    report = {
        "missing_values_before": n_missing_before,
        "missing_values_filled": n_missing_before - n_missing_after,
        "batches_dropped": len(incomplete),
        "batches_kept": len(batches),
    }
    return values, batches, report
