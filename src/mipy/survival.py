"""Survival endpoints: integrity checks, a reconciled progression endpoint,
and univariate Cox models pooled over multiple imputations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _st

from .mi import rubin


def audit_endpoints(ep: pd.DataFrame) -> dict:
    """ep: one row per patient with t_RFS, e_RFS, t_MFS, e_MFS.
    Counts the inconsistencies between the two endpoints."""
    both = (ep.e_RFS == 1) & (ep.e_MFS == 1)
    return {"patients": len(ep),
            "metastasis_without_relapse": int(((ep.e_MFS == 1) & (ep.e_RFS == 0)).sum()),
            "relapse_after_metastasis": int((both & (ep.t_RFS > ep.t_MFS)).sum()),
            "non_positive_times": int(((ep.t_RFS <= 0) | (ep.t_MFS <= 0)).sum())}


def reconcile(ep: pd.DataFrame) -> pd.DataFrame:
    """A progression endpoint consistent by construction: an event is any
    relapse or metastasis, its time the earliest recorded event time;
    patients without an event are censored at the shorter follow-up.
    Patients whose earliest time is zero or negative (event before the
    first sample) are excluded and flagged. Exploratory: it does not
    replace checking the source records."""
    t_ev = np.where(ep.e_RFS == 1, ep.t_RFS, np.inf)
    t_ev = np.minimum(t_ev, np.where(ep.e_MFS == 1, ep.t_MFS, np.inf))
    event = np.isfinite(t_ev).astype(int)
    t_cens = np.minimum(ep.t_RFS, ep.t_MFS)
    T = np.where(event == 1, t_ev, t_cens).astype(float)
    out = pd.DataFrame({"T": T, "E": event}, index=ep.index)
    out["excluded"] = out["T"] <= 0
    return out


def univariate_cox(time, event, X, max_iter=30, tol=1e-9):
    """Cox model with one covariate at a time, for every column of X
    (Breslow ties, Newton-Raphson, vectorised over columns). Columns are
    standardised, so the hazard ratio is per SD. Returns a DataFrame with
    beta, se, z, p for each column."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    X = np.asarray(X, dtype=float)
    sd = X.std(0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - X.mean(0)) / sd
    order = np.argsort(-time, kind="mergesort")
    t, e, Z = time[order], event[order], Xs[order]
    uniq_t, first = np.unique(-t, return_index=True)   # descending times: risk set = prefix
    last = np.r_[first[1:], len(t)] - 1
    beta = np.zeros(Z.shape[1])
    for _ in range(max_iter):
        w = np.exp(np.clip(Z * beta, -50, 50))
        S0 = np.cumsum(w, 0)[last]
        S1 = np.cumsum(w * Z, 0)[last]
        S2 = np.cumsum(w * Z * Z, 0)[last]
        d = np.add.reduceat(e, first)
        sx = np.add.reduceat(Z * e[:, None], first, axis=0)
        grad = (sx - d[:, None] * S1 / S0).sum(0)
        info = (d[:, None] * (S2 / S0 - (S1 / S0) ** 2)).sum(0)
        stepv = np.where(info > 0, grad / np.maximum(info, 1e-12), 0.0)
        stepv = np.clip(stepv, -2, 2)
        beta = beta + stepv
        if np.max(np.abs(stepv)) < tol:
            break
    w = np.exp(np.clip(Z * beta, -50, 50))
    S0 = np.cumsum(w, 0)[last]
    S1 = np.cumsum(w * Z, 0)[last]
    S2 = np.cumsum(w * Z * Z, 0)[last]
    d = np.add.reduceat(e, first)
    info = (d[:, None] * (S2 / S0 - (S1 / S0) ** 2)).sum(0)
    se = np.where(info > 0, 1 / np.sqrt(np.maximum(info, 1e-12)), np.nan)
    z = beta / se
    return pd.DataFrame({"beta": beta, "se": se, "z": z, "p": 2 * _st.norm.sf(np.abs(z))})


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = np.isfinite(p).sum()
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    q = np.full_like(p, np.nan)
    ranked = p[order][:n] * n / np.arange(1, n + 1)
    q[order[:n]] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q


def pooled_univariate_cox(time, event, completions, names):
    """Univariate Cox per protein on each completion, pooled with Rubin's
    rules when there is more than one completion."""
    per = [univariate_cox(time, event, C) for C in completions]
    rows = []
    for k, name in enumerate(names):
        q = [r.loc[k, "beta"] for r in per]
        u = [r.loc[k, "se"] ** 2 for r in per]
        if len(per) == 1:
            b, s = q[0], np.sqrt(u[0])
            p = 2 * _st.norm.sf(abs(b / s)) if s > 0 else np.nan
            fmi = 0.0
        else:
            qbar, ubar, B, T, fmi, df = rubin(q, u)
            b, s = qbar, np.sqrt(T)
            p = 2 * _st.t.sf(abs(b / s), df if np.isfinite(df) else 1e6) if s > 0 else np.nan
        rows.append({"protein": name, "log_hr": b, "se": s, "p": p, "fmi": fmi})
    out = pd.DataFrame(rows)
    out["q"] = bh_fdr(out["p"].to_numpy())
    return out
