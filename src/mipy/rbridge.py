"""Agreement between MIPY's Python implementations and the original R
packages (softImpute, missForest, mice norm.boot, imputeLCMD).

MIPY writes the masked matrices to a folder, runs the bundled R script
with Rscript, reads the R completions back and compares them with the
Python ones on the same cells.
"""
from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from .context import ImputeContext
from .evaluate import evaluate_per_protein
from .imputers import (_soft_fit, minprob_params_r, qrilc_params_r, run as run_method,
                       softimpute_setup)
from .mi import fmi_summary


def find_rscript(hint=None):
    """Path of Rscript, or None. Looks at ``hint``, the PATH, R_HOME and the
    usual Windows install folders (newest version first)."""
    cands = [hint] if hint else []
    w = shutil.which("Rscript")
    if w:
        cands.append(w)
    if os.environ.get("R_HOME"):
        cands.append(str(Path(os.environ["R_HOME"]) / "bin" / "Rscript"))
    if platform.system() == "Windows":
        for root in [os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                     str(Path.home() / "AppData" / "Local" / "Programs")]:
            hits = sorted(glob.glob(str(Path(root) / "R" / "R-*" / "bin" / "Rscript.exe")),
                          reverse=True)
            cands.extend(hits)
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
        if c and Path(c + ".exe").is_file():
            return str(c) + ".exe"
    return None


def r_script_path():
    return str(resources.files("mipy").joinpath("r/mipy_r_agreement.R"))


def write_inputs(folder, amputed: dict, schemas: dict, seed=2026, m=5, mice_maxit=5,
                 mf_ntree=100, mf_maxiter=10, run=("softImpute", "missForest", "mice", "lcmd")):
    """Write the masked matrices and settings the R script needs.
    ``amputed`` maps mechanism -> {"X": masked matrix, ...}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lams = {}
    for mech, amp in amputed.items():
        Xa = amp["X"]
        cols = [f"p{k}" for k in range(Xa.shape[1])]
        pd.DataFrame(Xa, columns=cols).to_csv(folder / f"X_{mech}.csv", index=False, na_rep="NA")
        mu, sd, Zs, M, lam = softimpute_setup(Xa, seed)
        pd.DataFrame(np.where(M, np.nan, Zs), columns=cols).to_csv(
            folder / f"Z_{mech}.csv", index=False, na_rep="NA")
        (folder / f"lambda_{mech}.txt").write_text(repr(float(lam)))
        lams[mech] = lam
        p = Xa.shape[1]
        pm = np.zeros((p, p), dtype=int)
        for k, preds in schemas[mech].items():
            if np.isnan(Xa[:, k]).any():
                pm[k, preds] = 1
        pd.DataFrame(pm, columns=cols).to_csv(folder / f"pred_{mech}.csv", index=False)
    cfg = pd.DataFrame({"key": ["mechs", "seed", "m", "mice_maxit", "mf_ntree", "mf_maxiter", "run"],
                        "value": [" ".join(amputed), seed, m, mice_maxit, mf_ntree, mf_maxiter,
                                  " ".join(run)]})
    cfg.to_csv(folder / "config.csv", index=False)
    for f in folder.glob("status.csv"):
        f.unlink()
    return lams


def run_r(folder, rscript=None, timeout=6 * 3600, log=print):
    """Run the R script on the prepared folder. Returns the status table
    (empty if R could not be run)."""
    rs = find_rscript(rscript)
    if rs is None:
        log("  Rscript not found: install R (https://cran.r-project.org) or set R_HOME")
        return pd.DataFrame(columns=["step", "mech", "status", "seconds"])
    cmd = [rs, "--vanilla", r_script_path(), str(folder)]
    log(f"  running {Path(rs).name} on {folder}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            log("  R stopped with an error:\n" + (p.stderr or "")[-1500:])
    except subprocess.TimeoutExpired:
        log("  R timed out; partial results are used")
    st = Path(folder) / "status.csv"
    return pd.read_csv(st) if st.is_file() else pd.DataFrame(columns=["step", "mech", "status", "seconds"])


def _rho(a, b):
    """Spearman correlation of per-protein nRMSE (robust to a few proteins
    with very large errors)."""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    g = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[g], b[g])[0]) if g.sum() > 2 else float("nan")


def _read(folder, name):
    f = Path(folder) / name
    if not f.is_file():
        return None
    return pd.read_csv(f).to_numpy(dtype=float)


def compare(folder, amputed: dict, X_true, python_results: dict, seed=2026, m=5, lams=None):
    """Compare R outputs with Python for each mechanism.

    python_results maps (mechanism, method) -> completed matrix (or record
    with "imputed"/"draws"). Returns (summary DataFrame, details dict)."""
    rows = []
    details = {}
    st_file = Path(folder) / "status.csv"
    status = pd.read_csv(st_file) if st_file.is_file() else pd.DataFrame(columns=["step", "mech", "status"])

    def _mice_fn(mech):
        ok = status[(status["mech"].astype(str) == mech) & (status["status"] == "ok")]["step"]
        if "mice" in set(ok):
            return "norm.boot"
        if "mice-ridge" in set(ok):
            return "ls.meth = ridge"
        if "mice-norm" in set(ok):
            return "mice norm"
        return ""

    for mech, amp in amputed.items():
        Xa, mask = amp["X"], amp["mask"]
        truth = np.where(mask, X_true, np.nan)
        miss = np.isnan(Xa)

        def acc(Xc):
            pp = evaluate_per_protein(truth, Xc, mask)
            return pp, float(pp["nrmse"].median())

        # softImpute: same lambda, same standardisation -> cell-level agreement
        si = _read(folder, f"si_{mech}.csv")
        if si is not None:
            mu, sd, Zs, M, lam = softimpute_setup(Xa, seed)
            Zf = np.where(M, 0.0, Zs)
            py = _soft_fit(Zf, M, lam, Zf)
            d = (si - py)[miss]
            r_cells = float(np.corrcoef(si[miss], py[miss])[0, 1])
            ppR, nR = acc(si * sd + mu)
            ppP, nP = acc(py * sd + mu)
            rows.append({"mechanism": mech, "method": "softImpute", "comparison": "same cells",
                         "cell r": r_cells, "max |diff| (SD units)": float(np.max(np.abs(d))),
                         "nRMSE R": nR, "nRMSE Python": nP,
                         "protein nRMSE rho": _rho(ppR["nrmse"], ppP["nrmse"])})
        # missForest (randomForest in R vs extremely randomised trees in Python)
        mf = _read(folder, f"mf_{mech}.csv")
        py = _pyres(python_results, mech, "missForest")
        if mf is not None and py is not None:
            ppR, nR = acc(mf)
            ppP, nP = acc(py[0])
            rows.append({"mechanism": mech, "method": "missForest", "comparison": "accuracy",
                         "cell r": float(np.corrcoef(mf[mask], py[0][mask])[0, 1]),
                         "nRMSE R": nR, "nRMSE Python": nP,
                         "protein nRMSE rho": _rho(ppR["nrmse"], ppP["nrmse"])})
        # mice norm.boot vs MICE-Bayes
        draws = [_read(folder, f"mice_{mech}_{l}.csv") for l in range(1, m + 1)]
        py = _pyres(python_results, mech, "MICE-Bayes")
        if all(x is not None for x in draws) and py is not None:
            draws = [np.where(np.isnan(x), np.nanmean(x, 0), x) for x in draws]
            meanR = np.mean(draws, axis=0)
            ppR, nR = acc(meanR)
            ppP, nP = acc(py[0])
            fR = fmi_summary(draws, Xa)["FMI"]
            fP = fmi_summary(py[1], Xa)["FMI"] if py[1] else np.nan
            rows.append({"mechanism": mech, "method": "MICE-Bayes vs mice norm.boot",
                         "comparison": "accuracy and FMI" + (f" ({_mice_fn(mech)})"
                                                            if _mice_fn(mech) not in ("", "norm.boot") else ""),
                         "cell r": float(np.corrcoef(meanR[mask], py[0][mask])[0, 1]),
                         "nRMSE R": nR, "nRMSE Python": nP,
                         "protein nRMSE rho": _rho(ppR["nrmse"], ppP["nrmse"]),
                         "FMI R": fR, "FMI Python": fP})
        # imputeLCMD: deterministic parameters must match exactly
        pm = _read(folder, f"par_minprob_{mech}.csv")
        if pm is not None:
            c_py, s_py = minprob_params_r(Xa.T)
            rows.append({"mechanism": mech, "method": "MinProb (imputeLCMD)", "comparison": "parameters",
                         "max |diff| (parameters)": float(max(np.max(np.abs(pm[:, 0] - c_py)),
                                                             abs(pm[0, 1] - s_py)))})
            mpR = _read(folder, f"minprob_{mech}.csv")
            if mpR is not None:
                outP, _ = run_method("MinProb-R", Xa, ImputeContext(seed=seed))
                rows[-1].update({"nRMSE R": acc(mpR)[1], "nRMSE Python": acc(outP)[1]})
        pq = _read(folder, f"par_qrilc_{mech}.csv")
        if pq is not None:
            par = qrilc_params_r(Xa.T)[["mean", "sd", "upper"]].to_numpy()
            rows.append({"mechanism": mech, "method": "QRILC (imputeLCMD)", "comparison": "parameters",
                         "max |diff| (parameters)": float(np.nanmax(np.abs(pq - par)))})
            qrR = _read(folder, f"qrilc_{mech}.csv")
            if qrR is not None:
                outP, _ = run_method("QRILC-R", Xa, ImputeContext(seed=seed))
                rows[-1].update({"nRMSE R": acc(qrR)[1], "nRMSE Python": acc(outP)[1]})
    return pd.DataFrame(rows), details


def _pyres(python_results, mech, name):
    r = python_results.get((mech, name))
    if r is None:
        return None
    if isinstance(r, dict):
        return (r["imputed"], r.get("draws")) if r.get("ok", True) and r.get("imputed") is not None else None
    return (np.asarray(r), None)
