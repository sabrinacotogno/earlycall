"""BatchData: the single data container shared by every module.

In memory, the data is a 3-D array:
    X has shape (n_batches, n_params, n_time)   -- batch first, like scikit-learn samples.

On disk, it is two CSV files:
    <name>_values.csv    one row per batch and time point: batch_id, time, <param 1>, <param 2>, ...
    <name>_batches.csv   one row per batch: batch_id, quality, <extra info such as recipe or date>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BatchData:
    """Aligned multivariate batch trajectories.

    X : (n_batches, n_params, n_time) float array. NaN allowed before cleaning.
    time : (n_time,) common time grid (hours since start, or hour of day for solar).
    params : names of the parameters, in the order of axis 1 of X.
    batch_ids : identifier of each batch, in the order of axis 0 of X.
    y : (n_batches,) final quality (yield, normalised energy). NaN if unknown.
    meta : one row per batch (true recipe, inverter, date...).
           Methods never read it: it is for evaluation and plots only.
    """

    X: np.ndarray
    time: np.ndarray
    params: list[str]
    batch_ids: list[str]
    y: np.ndarray
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.time = np.asarray(self.time, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.params = list(self.params)
        self.batch_ids = [str(b) for b in self.batch_ids]
        self.meta = self.meta.reset_index(drop=True)

        if self.X.ndim != 3:
            raise ValueError(f"X must be (n_batches, n_params, n_time), got {self.X.shape}")
        n_batches, n_params, n_time = self.X.shape
        if len(self.time) != n_time:
            raise ValueError(f"time has {len(self.time)} points, X has {n_time}")
        if len(self.params) != n_params:
            raise ValueError(f"{len(self.params)} param names for {n_params} params")
        if len(self.batch_ids) != n_batches or len(self.y) != n_batches:
            raise ValueError("batch_ids and y must have one entry per batch")
        if len(set(self.batch_ids)) != n_batches:
            raise ValueError("batch_ids must be unique")
        if not self.meta.empty and len(self.meta) != n_batches:
            raise ValueError("meta must have one row per batch")

    # ------------------------------------------------------------------ shape
    @property
    def n_batches(self) -> int:
        return self.X.shape[0]

    @property
    def n_params(self) -> int:
        return self.X.shape[1]

    @property
    def n_time(self) -> int:
        return self.X.shape[2]

    @property
    def percent(self) -> np.ndarray:
        """Each time point as % of the batch duration: 0 at the first point, 100 at the last.

        Example (CHO, 0 to 336 h): 108 h -> 32%. Example (solar, 6:00 to 18:30): 12:15 -> 50%.
        """
        return 100 * (self.time - self.time[0]) / (self.time[-1] - self.time[0])

    # --------------------------------------------------------------- selection
    def subset(self, idx) -> "BatchData":
        """Keep only the batches at positions `idx` (e.g. a train or test split)."""
        idx = np.asarray(idx, dtype=int)
        meta = self.meta.iloc[idx] if not self.meta.empty else self.meta
        return BatchData(
            X=self.X[idx],
            time=self.time,
            params=self.params,
            batch_ids=[self.batch_ids[i] for i in idx],
            y=self.y[idx],
            meta=meta,
        )

    def truncate(self, t_index: int) -> "BatchData":
        """Keep the first `t_index` time points: what is known at that moment of the run."""
        if not 1 <= t_index <= self.n_time:
            raise ValueError(f"t_index must be in [1, {self.n_time}], got {t_index}")
        return BatchData(
            X=self.X[:, :, :t_index],
            time=self.time[:t_index],
            params=self.params,
            batch_ids=self.batch_ids,
            y=self.y,
            meta=self.meta,
        )

    # -------------------------------------------------------- tables <-> array
    @classmethod
    def from_tables(cls, values: pd.DataFrame, batches: pd.DataFrame) -> "BatchData":
        """Build BatchData from the two tables.

        values  : columns batch_id, time, then one column per parameter.
                  Every batch must be on the same time grid (missing points become NaN).
        batches : columns batch_id, quality, then any extra info (kept as meta).
                  The order of this table gives the order of the batches.
        """
        values = values.copy()
        batches = batches.copy()
        values["batch_id"] = values["batch_id"].astype(str)
        batches["batch_id"] = batches["batch_id"].astype(str)

        batch_ids = batches["batch_id"].tolist()
        params = [c for c in values.columns if c not in ("batch_id", "time")]
        times = np.sort(values["time"].unique())

        missing = set(batch_ids) - set(values["batch_id"])
        if missing:
            raise ValueError(f"Batches without values: {sorted(missing)[:5]}")

        # one 2-D table (batches x times) per parameter, stacked into 3-D
        arrays = []
        for param in params:
            wide = values.pivot(index="batch_id", columns="time", values=param)
            wide = wide.reindex(index=batch_ids, columns=times)
            arrays.append(wide.to_numpy(dtype=float))
        X = np.stack(arrays, axis=1)

        meta = batches.drop(columns=["batch_id", "quality"])
        return cls(X=X, time=times, params=params, batch_ids=batch_ids,
                   y=batches["quality"].to_numpy(dtype=float), meta=meta)

    def to_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Inverse of from_tables: returns (values, batches)."""
        pieces = []
        for b, batch_id in enumerate(self.batch_ids):
            piece = pd.DataFrame(self.X[b].T, columns=self.params)
            piece.insert(0, "time", self.time)
            piece.insert(0, "batch_id", batch_id)
            pieces.append(piece)
        values = pd.concat(pieces, ignore_index=True)

        batches = self.meta.copy()
        batches.insert(0, "quality", self.y)
        batches.insert(0, "batch_id", self.batch_ids)
        return values, batches

    # ------------------------------------------------------------------- io
    def save(self, folder: str | Path, name: str) -> None:
        """Write <folder>/<name>_values.csv and <folder>/<name>_batches.csv."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        values, batches = self.to_tables()
        values.to_csv(folder / f"{name}_values.csv", index=False)
        batches.to_csv(folder / f"{name}_batches.csv", index=False)

    @classmethod
    def load(cls, folder: str | Path, name: str) -> "BatchData":
        """Read the two CSV files written by save()."""
        folder = Path(folder)
        values = pd.read_csv(folder / f"{name}_values.csv")
        batches = pd.read_csv(folder / f"{name}_batches.csv")
        return cls.from_tables(values, batches)
