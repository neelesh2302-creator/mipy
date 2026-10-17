"""Masking (amputation) of observed cells, so imputations can be scored
against the hidden true values.

The MCAR, MAR and MNAR designs reproduce MIPY v1 exactly (same random
stream), so masks, and therefore saved results, are identical. LOD is new
in v2: a hard detection limit, where the lowest values are removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _expit(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -50, 50)))


class Amputer:
    """Masks a share of the observed cells of X under a chosen mechanism.

    Parameters
    ----------
    X : (n, p) array on the analysis scale, NaN = missing
    time : sampling point of each row (MAR driver)
    clinical : DataFrame of clinical covariates, one row per row of X (MAR driver)
    rate : share of the observed cells to mask
    strength : slope beta of the logistic masking score
    min_obs : each column keeps at least this many observed cells
    """

    MECHANISMS = ("MCAR", "MAR", "MNAR", "LOD")

    def __init__(self, X, time=None, clinical=None, rate=0.2, strength=2.0, min_obs=12):
        self.X = np.asarray(X, dtype=float)
        self.OBS = np.isfinite(self.X)
        self.n, self.p = self.X.shape
        self.rate, self.strength, self.min_obs = rate, strength, min_obs
        mu = np.nanmean(self.X, axis=0)
        sd = np.nanstd(self.X, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
        self.Z = (self.X - mu) / sd
        self.time = np.zeros(self.n) if time is None else np.asarray(time)
        self.clinical = (pd.DataFrame(index=range(self.n)) if clinical is None
                         else pd.DataFrame(clinical).reset_index(drop=True))

    def _calibrate(self, score, target):
        s = score[self.OBS]
        lo, hi = -30.0, 30.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if _expit(s + mid).mean() < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def aux_score(self):
        """MAR driver: one standardised score per row from the sampling
        point, the clinical covariates and the row's mean observed level."""
        parts = [pd.Series(self.time, dtype=float)]
        for c in self.clinical.columns:
            parts.append(pd.to_numeric(self.clinical[c], errors="coerce").astype(float))
        parts.append(pd.Series(np.nanmean(self.Z, axis=1)))
        A = pd.concat(parts, axis=1)
        A = (A - A.mean()) / A.std(ddof=0).replace(0, 1)
        a = A.fillna(0).mean(axis=1).to_numpy()
        return (a - a.mean()) / (a.std() if a.std() > 0 else 1.0)

    def _floor(self, mask, rng):
        for k in range(self.p):                      # keep min_obs per column
            left = int((self.OBS[:, k] & ~mask[:, k]).sum())
            need = min(self.min_obs, int(self.OBS[:, k].sum())) - left
            if need > 0:
                idx = np.flatnonzero(mask[:, k])
                mask[rng.choice(idx, size=min(need, len(idx)), replace=False), k] = False
        return mask

    def ampute(self, mech, seed, strength=None):
        """Returns {"X": masked matrix, "mask": masked cells, "truth": true
        values at masked cells (NaN elsewhere)}."""
        beta = self.strength if strength is None else strength
        rng = np.random.default_rng(seed)
        n, p = self.n, self.p
        if mech == "LOD":
            score = np.nan_to_num(self.Z) + rng.normal(0.0, 0.3, (n, p))
            cut = np.quantile(score[self.OBS], self.rate)
            mask = self.OBS & (score < cut)
        else:
            if mech == "MCAR":
                score = np.zeros((n, p))
            elif mech == "MAR":
                score = np.repeat(beta * self.aux_score()[:, None], p, axis=1)
            elif mech == "MNAR":                     # left-censoring: low values lost
                score = -beta * np.nan_to_num(self.Z)
            else:
                raise ValueError(f"unknown mechanism {mech!r}")
            alpha = self._calibrate(score, self.rate)
            mask = self.OBS & (rng.random((n, p)) < _expit(score + alpha))
        mask = self._floor(mask, rng)
        Xa = np.where(mask, np.nan, self.X)
        truth = np.where(mask, self.X, np.nan)
        return {"X": Xa, "mask": mask, "truth": truth}

    def summary(self, amp):
        m = amp["mask"]
        return {"cells masked": int(m.sum()),
                "% of observed": round(100 * m.sum() / self.OBS.sum(), 2),
                "matrix missing after (%)": round(100 * np.isnan(amp["X"]).mean(), 2),
                "mean z of masked cells": round(float(self.Z[m].mean()), 3)}
