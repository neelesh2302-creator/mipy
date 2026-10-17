"""Multiple-imputation summaries: Rubin's rules and calibration of the
imputation uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _st


def rubin(q, u):
    """Rubin's rules for m estimates q with variances u.
    Returns (Qbar, Ubar, B, T, FMI, df)."""
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    m = len(q)
    qbar, ubar = float(np.mean(q)), float(np.mean(u))
    b = float(np.var(q, ddof=1)) if m > 1 else 0.0
    t = ubar + (1 + 1 / m) * b
    fmi = (1 + 1 / m) * b / t if t > 0 else np.nan
    df = (m - 1) / fmi ** 2 if fmi and fmi > 0 else np.inf
    return qbar, ubar, b, t, fmi, df


def fmi_summary(draws, X_masked):
    """FMI for (a) the mean of the imputed cells and (b) each column mean,
    summarised by the median and IQR across columns (as in MIPY v1)."""
    miss = np.isnan(X_masked)
    D = [np.asarray(d, dtype=float) for d in draws]
    vals = [d[miss] for d in D]
    qa, ua, ba, ta, fa, dfa = rubin([v.mean() for v in vals],
                                    [v.var(ddof=1) / v.size for v in vals])
    fmis = []
    for k in range(D[0].shape[1]):
        if not miss[:, k].any():
            continue
        cols = [d[:, k] for d in D]
        fmis.append(rubin([c.mean() for c in cols], [c.var(ddof=1) / c.size for c in cols])[4])
    fmis = np.asarray(fmis, dtype=float)
    return {"m": len(D), "Qbar": qa, "Ubar": ua, "B": ba, "T": ta, "FMI": fa, "df": dfa,
            "median FMI, protein means": float(np.nanmedian(fmis)),
            "IQR FMI, protein means": float(np.nanpercentile(fmis, 75) - np.nanpercentile(fmis, 25))}


def predictive_coverage(truth, draws, mask, level=0.95):
    """How well the spread of the m completions describes the error.

    For each masked cell, a prediction interval is formed from the m draws,
    mean +/- t_{m-1} * s * sqrt(1 + 1/m), and checked against the hidden
    true value. A well-calibrated multiple imputation covers about `level`
    of the cells; a method whose draws are too similar covers fewer.
    Single imputation (one draw) has no interval and is reported as 0.
    """
    D = np.stack([np.asarray(d, dtype=float)[mask] for d in draws])
    t = np.asarray(truth, dtype=float)[mask]
    m = D.shape[0]
    if m < 2:
        return {"coverage": 0.0, "width": 0.0, "mean_abs_z": np.nan, "m": m}
    mean = D.mean(0)
    s = D.std(0, ddof=1)
    half = _st.t.ppf(0.5 + level / 2, m - 1) * s * np.sqrt(1 + 1 / m)
    g = np.isfinite(t)
    cover = np.abs(t[g] - mean[g]) <= half[g]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(t[g] - mean[g]) / (s[g] * np.sqrt(1 + 1 / m))
    return {"coverage": float(cover.mean()), "width": float(np.median(2 * half[g])),
            "mean_abs_z": float(np.nanmedian(z)), "m": m}


def pool_estimates(estimates: pd.DataFrame, by: str, est="beta", se="se"):
    """Pool per-draw estimates (columns: draw, <by>, beta, se) with Rubin's
    rules. Returns one row per <by> with pooled beta, se, df and p-value."""
    rows = []
    for key, g in estimates.groupby(by):
        q, u = g[est].to_numpy(), (g[se].to_numpy() ** 2)
        qbar, ubar, b, t, fmi, df = rubin(q, u)
        se_p = np.sqrt(t)
        dfx = df if np.isfinite(df) else 1e6
        p = 2 * _st.t.sf(abs(qbar / se_p), dfx) if se_p > 0 else np.nan
        rows.append({by: key, "beta": qbar, "se": se_p, "df": df, "fmi": fmi, "p": p})
    return pd.DataFrame(rows)
