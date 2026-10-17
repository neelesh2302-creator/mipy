"""Scoring imputations against hidden true values.

Every accuracy measure is computed within a column (protein) on its masked
cells and summarised by the median across columns. nRMSE = RMSE / SD of
the true values, so 1.00 is what the column's own mean would achieve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_per_protein(truth, imputed, mask, names=None):
    rows = []
    names = names if names is not None else [f"p{k}" for k in range(truth.shape[1])]
    for k in range(truth.shape[1]):
        sel = mask[:, k]
        if sel.sum() < 3:
            continue
        t, q = truth[sel, k], imputed[sel, k]
        g = np.isfinite(t) & np.isfinite(q)
        if g.sum() < 3:
            continue
        t, q = t[g], q[g]
        sd = np.std(t)
        rmse = float(np.sqrt(np.mean((t - q) ** 2)))
        r = (float(np.corrcoef(t, q)[0, 1])
             if np.std(t) > 1e-12 and np.std(q) > 1e-12 else np.nan)
        rows.append({"protein": names[k], "n_masked": int(g.sum()),
                     "rmse": rmse, "nrmse": rmse / sd if sd > 1e-12 else np.nan,
                     "mae": float(np.mean(np.abs(t - q))), "r": r,
                     "bias": float(np.mean(q - t)), "sd_truth": float(sd),
                     "sd_imputed": float(np.std(q))})
    return pd.DataFrame(rows)


def structure_ratio(X_true, X_imp, mask, patient):
    """Pooled within-patient SD of the completed matrix divided by that of
    the true values, on cells whose true value is known, over patient-column
    groups with at least 2 such cells and at least one masked cell."""
    known = np.isfinite(X_true)
    num_ = den = 0.0
    patient = np.asarray(patient)
    for s in np.unique(patient):
        rws = patient == s
        T = np.where(known[rws], X_true[rws], np.nan)
        Q = np.where(known[rws], X_imp[rws], np.nan)
        use = (known[rws].sum(0) >= 2) & mask[rws].any(0)
        if not use.any():
            continue
        T, Q = T[:, use], Q[:, use]
        num_ += np.nansum((Q - np.nanmean(Q, 0)) ** 2)
        den += np.nansum((T - np.nanmean(T, 0)) ** 2)
    return float(np.sqrt(num_ / den)) if den > 0 else np.nan


def pooled_r(truth, imputed, mask):
    """Correlation over all masked cells pooled across columns. Shown only
    to demonstrate why it misleads."""
    t, q = truth[mask], imputed[mask]
    g = np.isfinite(t) & np.isfinite(q)
    return float(np.corrcoef(t[g], q[g])[0, 1])


def summarise(pp: pd.DataFrame) -> dict:
    return {"NRMSE_median": float(pp["nrmse"].median()),
            "NRMSE_IQR": float(pp["nrmse"].quantile(.75) - pp["nrmse"].quantile(.25)),
            "r_median": float(pp["r"].median()),
            "MAE_median": float(pp["mae"].median()),
            "bias_median": float(pp["bias"].median()),
            "pct_nrmse_below_1": float(100 * (pp["nrmse"] < 1.0).mean())}


def evaluate(X_true, imputed, mask, patient=None, names=None) -> dict:
    """One-call summary of an imputation.

    X_true  : the matrix before masking (NaN = never observed)
    imputed : the completed matrix
    mask    : the cells that were masked
    patient : patient of each row; adds the within-patient SD ratio
    """
    X_true = np.asarray(X_true, dtype=float)
    imputed = np.asarray(imputed, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    truth = np.where(mask, X_true, np.nan)
    pp = evaluate_per_protein(truth, imputed, mask, names)
    out = summarise(pp)
    if patient is not None:
        out["structure_ratio"] = structure_ratio(X_true, imputed, mask, patient)
    return out


def mcse_mean(values):
    """Monte Carlo standard error of a mean over repeats (Morris et al. 2019)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
