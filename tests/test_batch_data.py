import numpy as np
import pandas as pd
import pytest

from earlycall.batch_data import BatchData


def make_data():
    rng = np.random.default_rng(0)
    return BatchData(
        X=rng.normal(size=(4, 2, 10)),
        time=np.arange(10),
        params=["p0", "p1"],
        batch_ids=["b0", "b1", "b2", "b3"],
        y=rng.normal(size=4),
        meta=pd.DataFrame({"group": list("aabb")}),
    )


def test_shapes_are_checked():
    with pytest.raises(ValueError):
        BatchData(X=np.zeros((2, 3)), time=[0, 1, 2], params=["a"], batch_ids=["x", "y"], y=[1, 2])


def test_subset_and_truncate():
    data = make_data()
    sub = data.subset([0, 2])
    assert sub.X.shape == (2, 2, 10)
    assert sub.batch_ids == ["b0", "b2"]
    assert list(sub.meta["group"]) == ["a", "b"]
    assert data.truncate(4).X.shape == (4, 2, 4)


def test_tables_roundtrip():
    data = make_data()
    values, batches = data.to_tables()
    assert list(values.columns) == ["batch_id", "time", "p0", "p1"]
    assert list(batches.columns) == ["batch_id", "quality", "group"]
    back = BatchData.from_tables(values, batches)
    assert np.allclose(back.X, data.X)
    assert back.batch_ids == data.batch_ids


def test_save_load_roundtrip(tmp_path):
    data = make_data()
    data.save(tmp_path, "demo")
    back = BatchData.load(tmp_path, "demo")
    assert np.allclose(back.X, data.X)
    assert np.allclose(back.y, data.y)
    assert list(back.meta["group"]) == list(data.meta["group"])
