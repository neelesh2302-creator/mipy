"""Imputation of very large cohorts in chunks of patients.

The model is fitted once (on all rows for simple statistics, on a random
subsample of rows for model-based methods) and then applied chunk by chunk,
so memory stays bounded by the chunk size rather than the cohort size.
"""
from __future__ import annotations

import numpy as np


def _patient_chunks(patient_codes, chunk_rows):
    """Row ranges that never split a patient (rows sorted by patient)."""
    n = len(patient_codes)
    starts = [0]
    while starts[-1] < n:
        end = min(starts[-1] + chunk_rows, n)
        while end < n and patient_codes[end] == patient_codes[end - 1]:
            end += 1
        starts.append(end)
    return list(zip(starts[:-1], starts[1:]))


class ChunkedMean:
    name = "Mean"

    def fit(self, X, sample_rows=None, seed=0):
        s = np.zeros(X.shape[1])
        c = np.zeros(X.shape[1])
        for a in range(0, len(X), 1_000_000):          # streaming column means
            blk = X[a:a + 1_000_000]
            ok = np.isfinite(blk)
            s += np.where(ok, blk, 0).sum(0)
            c += ok.sum(0)
        self.mu = np.where(c > 0, s / np.maximum(c, 1), 0.0)
        return self

    def transform(self, Xc, codes_c=None):
        return np.where(np.isfinite(Xc), Xc, self.mu)


class ChunkedLOCF:
    name = "LOCF"

    def fit(self, X, sample_rows=200_000, seed=0):
        rs = np.random.default_rng(seed)
        idx = rs.choice(len(X), size=min(sample_rows, len(X)), replace=False)
        self.med = np.nan_to_num(np.nanmedian(X[idx], axis=0))
        return self

    def transform(self, Xc, codes_c):
        out = Xc.copy()
        same_prev = np.r_[False, codes_c[1:] == codes_c[:-1]]
        for _ in range(50):                              # forward fill within patient
            hole = np.isnan(out[1:]) & same_prev[1:, None]
            if not hole.any():
                break
            out[1:][hole] = out[:-1][hole]
        same_next = np.r_[codes_c[:-1] == codes_c[1:], False]
        for _ in range(50):                              # backward fill
            hole = np.isnan(out[:-1]) & same_next[:-1, None]
            if not hole.any():
                break
            out[:-1][hole] = out[1:][hole]
        return np.where(np.isnan(out), self.med, out)


class ChunkedSoftImpute:
    """softImpute fitted on a subsample of rows; each chunk is completed by
    projecting onto the fitted low-rank subspace (a few EM steps with the
    subspace held fixed)."""
    name = "softImpute"

    def fit(self, X, sample_rows=50_000, seed=0, rank=None):
        from .imputers import softimpute_setup, _soft_fit
        rs = np.random.default_rng(seed)
        idx = np.sort(rs.choice(len(X), size=min(sample_rows, len(X)), replace=False))
        Xs = np.asarray(X[idx], dtype=float)
        mu, sd, Zs, M, lam = softimpute_setup(Xs, seed)
        Zf = np.where(M, 0.0, Zs)
        Zhat = _soft_fit(Zf, M, lam, Zf)
        U, d, Vt = np.linalg.svd(Zhat, full_matrices=False)
        dd = np.maximum(d - lam, 0.0)
        r = int((dd > 1e-8).sum()) or 1
        self.mu, self.sd, self.V, self.r = mu, sd, Vt[:r].T, r
        self.shrink = dd[:r] / d[:r]                    # softImpute's singular-value shrinkage
        return self

    def transform(self, Xc, codes_c=None, n_iter=10):
        Z = (Xc - self.mu) / self.sd
        M = np.isnan(Z)
        cur = np.where(M, 0.0, Z)
        for _ in range(n_iter):
            low = ((cur @ self.V) * self.shrink) @ self.V.T
            cur = np.where(M, low, Z)
        return cur * self.sd + self.mu


class ChunkedIterative:
    """scikit-learn IterativeImputer fitted on a subsample, applied per chunk."""
    name = "sklearn-II"

    def fit(self, X, sample_rows=50_000, seed=0, max_iter=5):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        rs = np.random.default_rng(seed)
        idx = rs.choice(len(X), size=min(sample_rows, len(X)), replace=False)
        self.it = IterativeImputer(max_iter=max_iter, random_state=seed,
                                   initial_strategy="median", keep_empty_features=True)
        self.it.fit(np.asarray(X[idx], dtype=float))
        return self

    def transform(self, Xc, codes_c=None):
        return self.it.transform(np.asarray(Xc, dtype=float))


CHUNKED = {"Mean": ChunkedMean, "LOCF": ChunkedLOCF,
           "softImpute": ChunkedSoftImpute, "sklearn-II": ChunkedIterative}


def impute_chunked(X, patient_codes, method="softImpute", chunk_rows=500_000,
                   sample_rows=50_000, seed=0, out=None):
    """Complete a large matrix (rows sorted by patient) chunk by chunk.
    Writes into ``out`` (allocated if None) and returns it."""
    imp = CHUNKED[method]()
    if method in ("Mean",):
        imp.fit(X)
    else:
        imp.fit(X, sample_rows=sample_rows, seed=seed)
    out = np.empty_like(X) if out is None else out
    for a, b in _patient_chunks(patient_codes, chunk_rows):
        out[a:b] = imp.transform(X[a:b], patient_codes[a:b])
    return out
