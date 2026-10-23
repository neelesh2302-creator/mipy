"""MIPY: Missing-data Imputation in Python for high-dimensional longitudinal data.

Quick start
-----------
>>> import mipy
>>> res = mipy.analyze("my_data.csv")       # benchmark, impute, figures, report
>>> res.show()                              # tables and figures (inline in Jupyter)
>>> res.completed                           # your table with the missing values filled in

Try it on the bundled example (a synthetic cohort laid out like GSE65622):

>>> df = mipy.example_cohort()
>>> res = mipy.analyze(df, out="example_results", methods="quick", repeats=1)

Lower-level functions
---------------------
>>> mipy.list_methods()                         # methods and their evidence
>>> done = mipy.impute(df, method="softImpute", id_col="Patient_ID",
...                    time_col="sampling_point")
>>> draws = mipy.impute(df, method="MICE-Long", id_col="Patient_ID",
...                     time_col="sampling_point", m=5)   # list of 5 completions
>>> res = mipy.benchmark(df, methods=["Mean", "softImpute", "MICE-Long"],
...                      id_col="Patient_ID", time_col="sampling_point")
"""
from __future__ import annotations

__version__ = "2.1.1"

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .context import ImputeContext
from .evaluate import evaluate, evaluate_per_protein, mcse_mean, structure_ratio
from .imputers import METHODS, available, run
from .literature import literature
from .mi import fmi_summary, predictive_coverage, rubin
from .schema import build_variable_schema
from .data import DataError, prepare_data, read_table
from .datasets import example_cohort
from .analysis import Result, analyze
from . import (bigdata, chunked, datasets, io, masking, rbridge, runner,  # noqa: E402,F401
               survival)

__all__ = ["analyze", "Result", "prepare_data", "read_table", "DataError", "example_cohort",
           "impute", "benchmark", "evaluate", "list_methods", "literature", "rubin",
           "ImputeContext", "METHODS", "fmi_summary", "predictive_coverage",
           "build_variable_schema", "structure_ratio", "evaluate_per_protein", "__version__"]


def list_methods(main_only=False) -> pd.DataFrame:
    """The available methods, the package and function each calls, and
    the evidence behind each as implemented: "yes" (a peer-reviewed
    publication describes it), "adapted" (a published method changed in
    engine, stopping rule or unit of application) or "no"."""
    rows = []
    for m in METHODS.values():
        if main_only and not m.main:
            continue
        rows.append({"method": m.name, "family": m.family, "package": m.package,
                     "function": m.function, "peer-reviewed": m.evidence,
                     "multiple imputation": "yes" if m.multiple else "no",
                     "uses patients": "yes" if m.longitudinal else "no",
                     "new in MIPY": "yes" if m.new else "no",
                     "available": "yes" if available(m.name) else f"needs {m.needs}"})
    return pd.DataFrame(rows)


def _prepare(data, id_col, time_col, features):
    from pathlib import Path
    if isinstance(data, (str, Path)):
        data = read_table(data)
    if isinstance(data, np.ndarray):
        X = np.asarray(data, dtype=float)
        return None, X, None, None, None, [f"x{k}" for k in range(X.shape[1])]
    df = data.copy()
    order = None
    if id_col is not None:
        keys = [id_col] + ([time_col] if time_col else [])
        order = np.lexsort([df[k].to_numpy() for k in reversed(keys)])
        df = df.iloc[order].reset_index(drop=True)
    if features is None:
        skip = {id_col, time_col}
        features = [c for c in df.columns if c not in skip
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = df[features].to_numpy(dtype=float)
    patient = df[id_col].to_numpy() if id_col else None
    time = df[time_col].to_numpy() if time_col else None
    return df, X, patient, time, order, list(features)


def impute(data, method="softImpute", id_col=None, time_col=None, features=None,
           m: int = 1, seed: int = 2026, log2: bool = False, **settings):
    """Impute a wide table (one row per patient-visit) or a numeric array.

    Parameters
    ----------
    data : DataFrame or ndarray
    method : a name from ``list_methods()``
    id_col, time_col : patient and visit columns (needed by LOCF, MICE-Long, ...)
    features : columns to impute (default: every numeric column except id/time)
    m : number of completions; m > 1 needs a multiple-imputation method
    log2 : impute on the log2(x + 1) scale and transform back
    settings : any ImputeContext field (mice_iter, knn_k, gamma, ...)

    Returns a DataFrame like ``data`` (rows in the original order), or a
    list of m DataFrames when m > 1.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; see mipy.list_methods()")
    info = METHODS[method]
    if m > 1 and not info.multiple:
        raise ValueError(f"{method} is single imputation; use a multiple-imputation method "
                         f"({', '.join(k for k, v in METHODS.items() if v.multiple)})")
    if not available(method):
        raise ImportError(f"{method} needs the package {info.needs!r}")
    df, X, patient, time, order, feats = _prepare(data, id_col, time_col, features)
    if info.longitudinal and patient is None:
        raise ValueError(f"{method} needs id_col (and preferably time_col)")
    Xa = np.where(X > -1, np.log2(X + 1), np.nan) if log2 else X
    ctx = ImputeContext(seed=seed, patient=patient, time=time, m=max(m, 1), **settings)
    ctx = ctx.with_(schema=build_variable_schema(Xa, ctx.max_predictors, ctx.min_cor))
    out, draws = run(method, Xa, ctx)
    comps = draws if (m > 1 and draws) else [out]
    if log2:
        comps = [2 ** c - 1 for c in comps]
    if df is None:
        return comps if m > 1 else comps[0]
    res = []
    inv = np.argsort(order) if order is not None else None
    for C in comps:
        d = df.copy()
        d[feats] = C
        if inv is not None:
            d = d.iloc[inv].reset_index(drop=True)
        res.append(d)
    return res if m > 1 else res[0]


def benchmark(data, methods: Optional[Sequence[str]] = None, id_col=None, time_col=None,
              features=None, mechanisms: Iterable[str] = ("MCAR", "MAR", "MNAR"),
              rate=0.2, strength=2.0, repeats=1, seed=2026, log2=False, m=5,
              clinical=None, n_workers=1) -> pd.DataFrame:
    """Mask observed values under each mechanism, impute with each method,
    and score the hidden values. Returns one row per repeat x mechanism x
    method with median per-column nRMSE, r, bias, the within-patient SD
    ratio (if patients are known) and, for multiple imputation, the
    coverage of the imputation intervals."""
    from .masking import Amputer
    from .runner import CheckpointStore, run_tasks, score_record
    import tempfile
    df, X, patient, time, order, feats = _prepare(data, id_col, time_col, features)
    if log2:
        X = np.where(X > -1, np.log2(X + 1), np.nan)
    methods = [mm for mm in (methods or [k for k, v in METHODS.items() if v.main])
               if available(mm) and (patient is not None or not METHODS[mm].longitudinal)]
    amp_ = Amputer(X, time=time, clinical=clinical, rate=rate, strength=strength)
    pat = patient if patient is not None else np.arange(len(X))
    store = CheckpointStore(tempfile.mkdtemp(prefix="mipy_bench_"))
    tasks = []
    from functools import partial
    for rep in range(1, repeats + 1):
        for j, mech in enumerate(mechanisms):
            amp = amp_.ampute(mech, seed + 1000 * (rep - 1) + 101 * (j + 1))
            sch = build_variable_schema(amp["X"])
            ctx = ImputeContext(seed=seed, patient=patient, time=time, schema=sch, m=m)
            for name in methods:
                tasks.append({"key": (rep, mech, name), "name": name, "X": amp["X"], "ctx": ctx,
                              "ckpt": f"b_{rep}_{mech}_{name}",
                              "post": partial(score_record, X_true=X, mask=amp["mask"], patient=pat)})
    res = run_tasks(tasks, store, n_workers=n_workers, log=lambda *a: None)
    rows = []
    for (rep, mech, name), r in res.items():
        rows.append({"repeat": rep, "mechanism": mech, "method": name, **r})
    return pd.DataFrame(rows).sort_values(["repeat", "mechanism", "NRMSE"]).reset_index(drop=True)
