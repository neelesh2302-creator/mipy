"""Data handling at the scale of millions of patients.

Times and measures the memory of the steps that break conventional
software on very large cohorts: matching patient identifiers between
tables, reshaping between long tables and analysis matrices, and imputing
in chunks, with pandas, polars and DuckDB (SQL).
"""
from __future__ import annotations

import gc
import os
import tempfile
import threading
import time

import numpy as np
import pandas as pd

from . import io as mio
from .chunked import impute_chunked

BYTES_PER_LONG_ROW = {"pandas": 64, "polars": 32, "duckdb": 24}


def simulate_cohort(n_patients, n_visits=4, n_features=10, missing=0.10, seed=0,
                    orphan_frac=0.001, keep_truth_rows=200_000):
    """A synthetic longitudinal cohort with realistic identifiers.

    Returns a dict: ``patient_id`` (int64, one per row), ``visit`` (int8),
    ``X`` (float32 rows x features, NaN = missing), ``truth`` (true values of
    the first rows, for scoring), and a clinical table (``clin_id``,
    ``clin_age``, ``clin_sex``) in which a share ``orphan_frac`` of IDs has
    no visits and the same share of visiting patients has no clinical row.
    """
    rs = np.random.default_rng(seed)
    ids = (10 ** 9 + rs.permutation(n_patients).astype(np.int64) * 7)
    ids.sort()
    keep = np.ones((n_patients, n_visits), dtype=bool)
    keep[:, 1:] = rs.random((n_patients, n_visits - 1)) > 0.10     # 10% visits lost
    pid = np.repeat(ids, n_visits)[keep.ravel()]
    visit = np.tile(np.arange(1, n_visits + 1, dtype=np.int8), n_patients)[keep.ravel()]
    n_rows = len(pid)
    W = rs.normal(size=(3, n_features)).astype(np.float32)
    X = np.empty((n_rows, n_features), dtype=np.float32)
    codes = np.repeat(np.arange(n_patients), keep.sum(1))
    step = 2_000_000
    pat_eff = rs.normal(size=(n_patients, 3)).astype(np.float32)
    for a in range(0, n_rows, step):
        b = min(a + step, n_rows)
        lat = pat_eff[codes[a:b]] + 0.5 * rs.normal(size=(b - a, 3)).astype(np.float32)
        X[a:b] = 8 + lat @ W + 0.4 * rs.normal(size=(b - a, n_features)).astype(np.float32)
    t_rows = min(keep_truth_rows, n_rows)
    truth = X[:t_rows].copy()
    for a in range(0, n_rows, step):
        b = min(a + step, n_rows)
        blk = X[a:b]
        blk[rs.random(blk.shape) < missing] = np.nan
    n_orph = max(1, int(orphan_frac * n_patients))
    clin_id = np.concatenate([ids[n_orph:], 10 ** 9 - 1 - np.arange(n_orph, dtype=np.int64)])
    return {"patient_id": pid, "visit": visit, "codes": codes, "X": X, "truth": truth,
            "features": [f"f{j}" for j in range(n_features)],
            "clin_id": clin_id,
            "clin_age": rs.integers(30, 85, len(clin_id)).astype(np.int16),
            "clin_sex": rs.integers(0, 2, len(clin_id)).astype(np.int8)}


class PeakMemory:
    """Context manager: peak increase in resident memory (MB) while active."""

    def __init__(self, interval=0.02):
        import psutil
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0

    def __enter__(self):
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self._stop = False
        self._t = threading.Thread(target=self._watch, daemon=True)
        self._t.start()
        return self

    def _watch(self):
        while not self._stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop = True
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)
        self.mb = (self.peak - self.base) / 2 ** 20
        return False


def _available_bytes():
    import psutil
    return psutil.virtual_memory().available


def _frames(c, engine):
    cols = {"patient_id": c["patient_id"], "visit": c["visit"]}
    cols.update({f: c["X"][:, j] for j, f in enumerate(c["features"])})
    clin = {"patient_id": c["clin_id"], "age": c["clin_age"], "sex": c["clin_sex"]}
    if engine == "polars":
        import polars as pl
        return pl.DataFrame(cols), pl.DataFrame(clin)
    return pd.DataFrame(cols, copy=False), pd.DataFrame(clin)


def run_engine_benchmark(sizes=(10_000, 100_000, 1_000_000, 10_000_000),
                         engines=("pandas", "polars", "duckdb"), n_features=10,
                         impute_methods=("Mean", "LOCF", "softImpute", "sklearn-II"),
                         step_time_limit=900, memory_fraction=0.6, seed=2026,
                         log=print, done=None, on_row=None):
    """Time and memory of ID matching, wide->long, long->wide and chunked
    imputation at each cohort size. Steps predicted to exceed the time
    limit, or to need more than ``memory_fraction`` of the free memory,
    are skipped and reported as such. ``done`` maps (size, engine, step) to
    a finished row (for resuming). Returns a DataFrame."""
    rows = list(done.values()) if done else []
    done = dict(done or {})
    last = {(k[1], k[2]): (k[0], r["seconds"]) for k, r in sorted(done.items())
            if r.get("status") == "ok"}
    for n in sizes:
        coh = None
        for engine in [e for e in engines if mio.engine_available(e)] + ["numpy"]:
            steps = (["match", "wide_to_long", "long_to_wide"] if engine != "numpy"
                     else [f"impute:{m}" for m in impute_methods])
            for step in steps:
                key = (n, engine, step)
                if key in done:
                    continue
                prev = last.get((engine, step))
                pred = prev[1] * n / prev[0] if prev else 0.0
                rows_est = n * 3.7
                mem_need = (rows_est * n_features * BYTES_PER_LONG_ROW.get(engine, 16)
                            if engine != "numpy" else rows_est * n_features * 12)
                if pred > step_time_limit:
                    row = {"patients": n, "engine": engine, "step": step, "seconds": np.nan,
                           "peak_MB": np.nan, "status": f"skipped (predicted {pred/60:.0f} min)"}
                elif mem_need > memory_fraction * _available_bytes():
                    row = {"patients": n, "engine": engine, "step": step, "seconds": np.nan,
                           "peak_MB": np.nan,
                           "status": f"skipped (needs about {mem_need/2**30:.1f} GB)"}
                else:
                    if coh is None:
                        coh = simulate_cohort(n, n_features=n_features, seed=seed)
                    row = _run_step(coh, n, engine, step, seed)
                    if row["status"] == "ok":
                        last[(engine, step)] = (n, row["seconds"])
                log(f"  {n:>12,} patients  {engine:<7} {step:<22} {row['status']:<8} "
                    + (f"{row['seconds']:8.1f}s {row['peak_MB']:8.0f} MB" if row["status"] == "ok" else ""))
                rows.append(row)
                done[key] = row
                if on_row is not None:
                    on_row(key, row)
                gc.collect()
        del coh
        gc.collect()
    return pd.DataFrame(rows)


def _run_step(c, n, engine, step, seed):
    row = {"patients": n, "engine": engine, "step": step, "rows": len(c["patient_id"])}
    try:
        if engine == "numpy":
            method = step.split(":", 1)[1]
            with PeakMemory() as pm:
                t0 = time.time()
                out = impute_chunked(c["X"], c["codes"], method=method, seed=seed)
                sec = time.time() - t0
            k = len(c["truth"])
            miss = np.isnan(c["X"][:k])
            err = (out[:k] - c["truth"])[miss]
            sd = np.nanstd(c["truth"], axis=0).mean()
            row.update(seconds=sec, peak_MB=pm.mb, status="ok",
                       nrmse=float(np.sqrt(np.mean(err ** 2)) / sd))
            del out
            return row
        con = None
        if engine == "duckdb":
            import duckdb
            con = duckdb.connect()
            tmp = tempfile.mkdtemp(prefix="mipy_duck_")
            lim = int(0.5 * _available_bytes() / 2 ** 20)
            con.execute(f"SET memory_limit='{lim}MB'")
            con.execute(f"SET temp_directory='{tmp}'")
        wide, clin = _frames(c, engine)
        visits = wide[["patient_id", "visit"]] if engine != "polars" else wide.select(["patient_id", "visit"])
        with PeakMemory() as pm:
            t0 = time.time()
            if step == "match":
                joined, rep = mio.match_ids(visits, clin, on="patient_id", engine=engine, con=con)
                row["detail"] = (f"matched {rep['matched_ids']:,}; visits without clinical row "
                                 f"{rep['left_only']:,}; clinical rows without visits {rep['right_only']:,}")
                ok = rep["left_only"] > 0 and rep["right_only"] > 0
                del joined
            elif step == "wide_to_long":
                lg = mio.wide_to_long(wide, engine=engine, con=con)
                n_long = lg.height if engine == "polars" else len(lg)
                row["detail"] = f"{n_long:,} long rows"
                ok = n_long == int(np.isfinite(c["X"]).sum())
                del lg
            else:
                lg = mio.wide_to_long(wide, engine=engine, con=con)
                t0 = time.time()                       # time the pivot only
                back = mio.long_to_wide(lg, engine=engine, con=con)
                nb = back.height if engine == "polars" else len(back)
                row["detail"] = f"{nb:,} rows rebuilt"
                bsum = (float(np.nansum(back.select(c["features"]).to_numpy())) if engine == "polars"
                        else float(np.nansum(back[c["features"]].to_numpy(dtype=float))))
                n_any = int(np.isfinite(c["X"]).any(axis=1).sum())   # rows with any value
                ok = nb == n_any and np.isclose(bsum, float(np.nansum(c["X"], dtype=np.float64)), rtol=1e-4)
                del lg, back
            sec = time.time() - t0
        row.update(seconds=sec, peak_MB=pm.mb, status="ok" if ok else "failed check")
        if con is not None:
            con.close()
    except MemoryError:
        row.update(seconds=np.nan, peak_MB=np.nan, status="out of memory")
    except Exception as e:
        row.update(seconds=np.nan, peak_MB=np.nan, status=f"error: {type(e).__name__}")
        row["detail"] = str(e)[:200]
    return row
