"""Predictor selection for chained equations (quickpred, van Buuren 2018, 9.1.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def masked_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """corr(A_j, B_k) for every column pair, using the rows where A_j is
    observed. B must be complete. Returns a (p_A x p_B) array."""
    W = np.isfinite(A).astype(float)
    A0 = np.where(W > 0, A, 0.0)
    n = W.sum(0)[:, None]
    sa, saa = A0.sum(0)[:, None], (A0 ** 2).sum(0)[:, None]
    sb, sbb, sab = W.T @ B, W.T @ (B ** 2), A0.T @ B
    num_ = n * sab - sa * sb
    den = np.sqrt(np.clip(n * saa - sa ** 2, 0, None) * np.clip(n * sbb - sb ** 2, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num_ / den
    return np.where(np.isfinite(r), r, 0.0)


def build_variable_schema(Xa: np.ndarray, max_predictors: int = 25,
                          min_cor: float = 0.10) -> dict:
    """For each column k: up to ``max_predictors`` other columns whose
    correlation with k's value, or with whether k is missing, is at least
    ``min_cor`` (the strongest ones). Returns {k: [j, ...]}."""
    Xa = np.asarray(Xa, dtype=float)
    pc = pd.DataFrame(Xa).corr(min_periods=10).to_numpy(copy=True)
    np.fill_diagonal(pc, 0.0)
    pc = np.where(np.isfinite(pc), np.abs(pc), 0.0)
    R = np.isnan(Xa).astype(float)
    pr = np.abs(masked_corr(Xa, R))
    C = np.maximum(pc, pr.T)
    np.fill_diagonal(C, -1.0)
    schema = {}
    for k in range(Xa.shape[1]):
        order = np.argsort(-C[k])
        keep = [j for j in order if C[k, j] >= min_cor][:max_predictors]
        schema[k] = keep if keep else list(order[:max_predictors])
    return schema
