import numpy as np
import pandas as pd

from earlycall.preprocessing import clean_tables, fill_short_gaps


def test_short_gaps_are_filled_long_gaps_are_not():
    values = pd.DataFrame({"batch_id": "a", "time": range(8),
                           "x": [0, np.nan, 2, 3, np.nan, np.nan, np.nan, 7]})
    filled = fill_short_gaps(values, ["x"], max_gap=2)
    assert filled["x"].iloc[1] == 1
    assert filled["x"].iloc[4:7].isna().all()


def test_incomplete_batches_are_dropped_and_reported():
    values = pd.DataFrame({"batch_id": ["a"] * 3 + ["b"] * 3, "time": [0, 1, 2] * 2,
                           "x": [1, 2, 3, np.nan, 5, 6]})
    batches = pd.DataFrame({"batch_id": ["a", "b"], "quality": [1.0, 2.0]})
    values, batches, report = clean_tables(values, batches, ["x"])
    assert list(batches["batch_id"]) == ["a"]
    assert report["batches_dropped"] == 1
