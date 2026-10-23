"""Running many imputations with checkpoints and, optionally, in parallel.

Each finished run is saved at once under a fingerprint of the data, the
method (with its revision) and every setting that changes the result, so
an interrupted run resumes where it stopped. The fingerprint is the same
as in MIPY v1, so v1's saved results are found and reused.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .context import ImputeContext


def fingerprint(X, name, settings):
    h = hashlib.sha256()
    a = np.ascontiguousarray(np.asarray(X, dtype=float))
    h.update(str(a.shape).encode())
    h.update(np.nan_to_num(a, nan=-1.2345e300).tobytes())
    h.update(str(name).encode())
    h.update(repr(tuple(settings)).encode())
    return h.hexdigest()[:16]


class CheckpointStore:
    """Atomic pickle checkpoints in ``folder``; also looks in ``legacy``
    folders (read-only) for results saved by earlier versions."""

    def __init__(self, folder, legacy=(), fresh=False):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.legacy = [Path(p) for p in legacy if p]
        self.fresh = fresh

    def load(self, name):
        if self.fresh:
            return None
        for d in [self.folder, *self.legacy]:
            p = d / f"{name}.pkl"
            if p.is_file():
                try:
                    with open(p, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    continue
        return None

    def save(self, name, obj):
        tmp = self.folder / f"{name}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.folder / f"{name}.pkl")


def run_one(name, X, ctx: ImputeContext, keep_draws=True):
    """Run a method and return a result record (never raises)."""
    from . import imputers
    t0 = time.time()
    try:
        out, draws = imputers.run(name, X, ctx)
        rec = {"imputed": out, "draws": draws if keep_draws else None, "ok": True,
               "error": "", "m": len(draws) if draws else 1,
               "seconds": time.time() - t0}
        if imputers.LAST_DIAGNOSTICS:               # e.g. effective sample sizes
            rec["diag"] = dict(imputers.LAST_DIAGNOSTICS)
        return rec
    except Exception as e:                          # recorded, not raised
        return {"imputed": None, "draws": None, "ok": False, "m": 0,
                "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}


def _task(args):
    import warnings
    key, name, X, ctx, keep_draws, post = args
    with warnings.catch_warnings():             # library warnings (convergence, empty slices)
        warnings.simplefilter("ignore")         # would flood the output of worker processes
        rec = run_one(name, X, ctx, keep_draws)
        if post is not None and rec["ok"]:
            rec = post(rec)
    return key, rec


def _init_worker():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"                         # also limits torch (GAIN, HyperImpute)
    try:                                            # one BLAS thread per worker
        from threadpoolctl import threadpool_limits
        globals()["_TPL"] = threadpool_limits(limits=1)
    except Exception:
        pass


class _HideMain:
    """While worker processes start, hide the calling script from them, so a
    script without an ``if __name__ == "__main__":`` guard is not re-run in
    every worker (tasks only need the mipy package). No effect in Jupyter."""

    def __enter__(self):
        import sys
        self.mod = sys.modules.get("__main__")
        self.saved = {}
        if self.mod is not None:
            if hasattr(self.mod, "__file__"):
                self.saved["__file__"] = self.mod.__file__
                del self.mod.__file__
            self.saved["__spec__"] = getattr(self.mod, "__spec__", None)
            self.mod.__spec__ = None
        return self

    def __exit__(self, *exc):
        for a, v in self.saved.items():
            setattr(self.mod, a, v)
        return False


def run_tasks(tasks, store: CheckpointStore, n_workers=1, on_done=None, log=print):
    """Run a list of tasks, each a dict with keys: key, name, X, ctx,
    ckpt (checkpoint name) and optional keep_draws, post (a picklable
    function applied to the record in the worker, e.g. scoring).

    Finished checkpoints are loaded instead of recomputed. Remaining tasks
    run in ``n_workers`` processes (1 = in this process). If the worker
    pool fails (e.g. memory), the rest run one by one here.
    """
    results, todo = {}, []
    for t in tasks:
        rec = store.load(t["ckpt"])
        if rec is not None:
            results[t["key"]] = rec
            if on_done:
                on_done(t, rec, loaded=True)
        else:
            todo.append(t)
    if not todo:
        return results

    def _finish(t, rec):
        store.save(t["ckpt"], rec)
        results[t["key"]] = rec
        if on_done:
            on_done(t, rec, loaded=False)

    by_key = {t["key"]: t for t in todo}
    args = [(t["key"], t["name"], t["X"], t["ctx"], t.get("keep_draws", True), t.get("post"))
            for t in todo]
    import multiprocessing as mp
    if mp.parent_process() is not None:              # already inside a worker
        n_workers = 1
    if n_workers > 1 and len(todo) > 1:
        try:
            # "spawn" everywhere: the only method on Windows, and safe inside Jupyter
            with _HideMain(), ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                                  mp_context=mp.get_context("spawn")) as ex:
                futs = [ex.submit(_task, a) for a in args]
                for f in as_completed(futs):
                    key, rec = f.result()
                    _finish(by_key.pop(key), rec)
        except Exception as e:                       # BrokenProcessPool etc.
            log(f"  parallel run stopped ({type(e).__name__}); finishing one by one")
    for key, t in list(by_key.items()):
        _, rec = _task((key, t["name"], t["X"], t["ctx"], t.get("keep_draws", True), t.get("post")))
        _finish(t, rec)
    return results


def score_record(rec, X_true, mask, patient):
    """Reduce a run record to its scores (used for repeated maskings, where
    keeping every completed matrix would take gigabytes). Same fields as
    MIPY v1's repeat checkpoints."""
    from .evaluate import evaluate_per_protein, structure_ratio
    if not rec.get("ok"):
        return {"ok": False, "error": rec.get("error", ""), "seconds": rec.get("seconds", 0)}
    out = rec["imputed"]
    truth = np.where(mask, X_true, np.nan)
    pp = evaluate_per_protein(truth, out, mask)
    res = {"ok": True, "NRMSE": float(pp["nrmse"].median()),
           "r": float(pp["r"].median()), "bias": float(pp["bias"].median()),
           "structure_ratio": structure_ratio(X_true, out, mask, patient),
           "seconds": rec["seconds"]}
    if rec.get("draws") and len(rec["draws"]) > 1:
        from .mi import predictive_coverage
        cov = predictive_coverage(truth, rec["draws"], mask)
        res["coverage"] = cov["coverage"]
    if rec.get("diag"):
        res["diag"] = rec["diag"]
    return res


def default_workers(max_workers=4):
    n = os.cpu_count() or 1
    return max(1, min(max_workers, n - 1))
