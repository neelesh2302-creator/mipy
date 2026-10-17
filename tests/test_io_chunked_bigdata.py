import numpy as np
import pandas as pd
import pytest

from mipy import io as mio
from mipy.bigdata import simulate_cohort, run_engine_benchmark
from mipy.chunked import impute_chunked

ENG = [e for e in mio.ENGINES if mio.engine_available(e)]


@pytest.fixture(scope="module")
def small():
    return simulate_cohort(500, n_features=4, seed=3)


def _wide(c):
    d = pd.DataFrame(c["X"], columns=c["features"])
    d.insert(0, "visit", c["visit"])
    d.insert(0, "patient_id", c["patient_id"])
    return d


@pytest.mark.parametrize("engine", ENG)
def test_round_trip(engine, small):
    w = _wide(small)
    lg = mio.wide_to_long(w, engine=engine)
    back = mio.long_to_wide(lg, engine=engine)
    if not isinstance(back, pd.DataFrame):
        back = back.to_pandas()
    back = back.sort_values(["patient_id", "visit"]).reset_index(drop=True)
    assert len(back) == len(w)
    a = w[small["features"]].to_numpy(dtype=float)
    b = back[small["features"]].to_numpy(dtype=float)
    assert np.allclose(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9))


@pytest.mark.parametrize("engine", ENG)
def test_match_reports_orphans(engine, small):
    vis = _wide(small)[["patient_id", "visit"]]
    clin = pd.DataFrame({"patient_id": small["clin_id"], "age": small["clin_age"]})
    _, rep = mio.match_ids(vis, clin, engine=engine)
    assert rep["left_only"] >= 1 and rep["right_only"] >= 1
    assert rep["matched_ids"] == rep["left_ids"] - rep["left_only"]


@pytest.mark.parametrize("method", ["Mean", "LOCF", "softImpute", "sklearn-II"])
def test_chunked_imputation(method, small):
    out = impute_chunked(small["X"], small["codes"], method=method, chunk_rows=300,
                         sample_rows=1000)
    assert not np.isnan(out).any()
    obs = np.isfinite(small["X"])
    assert np.allclose(out[obs], small["X"][obs])


def test_engine_benchmark_runs():
    df = run_engine_benchmark(sizes=(2000,), n_features=3, log=lambda *a: None)
    assert (df.status == "ok").all()
