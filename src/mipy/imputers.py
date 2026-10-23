"""Imputation methods.

Every imputer has the signature ``fn(X, ctx) -> (completed, draws)``:

* ``X``         2-D float array, NaN = missing (rows = patient-visits)
* ``ctx``       :class:`mipy.context.ImputeContext`
* ``completed`` the completed matrix (for multiple imputation: the mean
                of the completions)
* ``draws``     list of m completed matrices for multiple-imputation
                methods, otherwise ``None``

The thirteen methods of MIPY v1 are reproduced here exactly (same
algorithm, same random-number stream), so results saved by the v1
notebook remain valid. Five methods are new in v2: MICE-Long,
MICE-LongMNAR and the multiple-imputation versions of softImpute,
missForest and miceforest.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats as _st
from sklearn.impute import KNNImputer, SimpleImputer

from .context import ImputeContext


def _named(Xm):
    return pd.DataFrame(Xm, columns=[f"p{k}" for k in range(Xm.shape[1])])


def _schema(ctx, Xm):
    if ctx.schema is None:
        from .schema import build_variable_schema
        return build_variable_schema(Xm, ctx.max_predictors, ctx.min_cor)
    return ctx.schema


# ---------------------------------------------------------------- simple
def imp_mean(Xm, ctx=None):
    return SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(Xm), None


def imp_median(Xm, ctx=None):
    return SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(Xm), None


def imp_locf(Xm, ctx):
    """Last observation carried forward within the patient, then backward,
    then the protein median. Rows must be ordered by patient and visit."""
    patient = ctx.rows_by_patient()
    d = pd.DataFrame(Xm)
    d = d.groupby(patient).ffill()
    d = d.groupby(patient).bfill()
    return d.fillna(pd.DataFrame(Xm).median()).to_numpy(dtype=float, copy=True), None


def imp_knn(Xm, ctx):
    return KNNImputer(n_neighbors=ctx.knn_k, weights="distance",
                      keep_empty_features=True).fit_transform(Xm), None


# ---------------------------------------------------------------- softImpute
def _soft_fit(Zs, M, lam, cur, max_iter=200, tol=1e-5):
    """Iterate the soft-thresholded SVD of Mazumder et al. (2010)."""
    for _ in range(max_iter):
        U, d, Vt = np.linalg.svd(cur, full_matrices=False)
        low = (U * np.maximum(d - lam, 0.0)) @ Vt
        new = np.where(M, low, Zs)
        delta = np.sum((new - cur) ** 2) / max(np.sum(cur ** 2), 1e-12)
        cur = new
        if delta < tol:
            break
    return cur


def softimpute_setup(Xm, seed):
    """Standardise, then choose lambda by holding out 10% of the observed
    cells and scanning a 15-point grid (warm starts, largest first).
    Returns (mu, sd, Zs, M, lambda)."""
    M = np.isnan(Xm)
    mu, sd = np.nanmean(Xm, 0), np.nanstd(Xm, 0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    Zs = (Xm - mu) / sd
    Zf = np.where(M, 0.0, Zs)
    d1 = np.linalg.svd(Zf, compute_uv=False)[0]
    rs = np.random.default_rng(seed)
    obs = np.argwhere(~M)
    hold = obs[rs.random(len(obs)) < 0.10]
    Mv = M.copy()
    Mv[hold[:, 0], hold[:, 1]] = True
    cur, best = np.where(Mv, 0.0, Zs), None
    for lam in d1 * np.geomspace(0.5, 0.01, 15):
        cur = _soft_fit(np.where(Mv, 0.0, Zs), Mv, lam, cur, max_iter=100)
        err = np.mean((cur[hold[:, 0], hold[:, 1]] - Zs[hold[:, 0], hold[:, 1]]) ** 2)
        if best is None or err < best[0]:
            best = (err, lam)
    return mu, sd, Zs, M, best[1]


def imp_softimpute(Xm, ctx):
    """softImpute (Mazumder, Hastie & Tibshirani 2010) on column-standardised
    data, lambda chosen by cross-validation on held-out observed cells."""
    mu, sd, Zs, M, lam = softimpute_setup(Xm, ctx.seed)
    Zf = np.where(M, 0.0, Zs)
    return _soft_fit(Zf, M, lam, Zf) * sd + mu, None


def imp_softimpute_mi(Xm, ctx):
    """Multiple imputation with softImpute. For each of m completions a
    different 10% of the observed cells is held out, the model is refitted
    without them (parameter uncertainty), and every imputed value receives
    a residual draw whose SD is the protein's error on its held-out cells."""
    mu, sd, Zs, M, lam = softimpute_setup(Xm, ctx.seed)
    obs = np.argwhere(~M)
    draws = []
    for l in range(ctx.m):
        rs = np.random.default_rng(ctx.seed + 500 + l)
        hold = obs[rs.random(len(obs)) < ctx.mi_holdout]
        Mh = M.copy()
        Mh[hold[:, 0], hold[:, 1]] = True
        Zh = np.where(Mh, 0.0, Zs)
        fit = _soft_fit(Zh, Mh, lam, Zh)
        res = np.full(Zs.shape, np.nan)
        res[hold[:, 0], hold[:, 1]] = fit[hold[:, 0], hold[:, 1]] - Zs[hold[:, 0], hold[:, 1]]
        sig = _column_sigma(res)
        out = np.where(M, fit + rs.normal(0.0, 1.0, Zs.shape) * sig, Zs)
        draws.append(out * sd + mu)
    return np.mean(draws, axis=0), draws


def _column_sigma(res, min_n=5):
    """Per-column residual SD from a matrix of residuals (NaN = none);
    columns with fewer than min_n residuals use the pooled SD."""
    pooled = float(np.nanstd(res)) if np.isfinite(res).any() else 1.0
    cnt = np.isfinite(res).sum(0)
    with np.errstate(invalid="ignore"):
        s = np.sqrt(np.nanmean(res ** 2, axis=0))
    return np.where((cnt >= min_n) & np.isfinite(s), s, pooled)


# ---------------------------------------------------------------- chained equations
def _iterative(Xm, make_estimator, schema, n_imp, ctx, noise=False):
    """Chained equations (FCS) with the quickpred predictor set per column,
    proteins visited from fewest to most missing. With n_imp > 1 every
    completion is fitted on a bootstrap sample of the rows. noise=True adds
    a residual draw with SD sqrt(1/alpha) of a Bayesian ridge model (mice's
    norm.boot); noise="oob" uses the out-of-bootstrap residual SD instead,
    which suits tree models whose in-sample residuals are too small."""
    M = np.isnan(Xm)
    order = np.argsort(M.sum(axis=0))
    draws = []
    for rep in range(n_imp):
        rs = np.random.default_rng(ctx.seed + rep)
        cur = SimpleImputer(strategy="median",
                            keep_empty_features=True).fit_transform(Xm)
        boot = rs.integers(0, Xm.shape[0], Xm.shape[0]) if n_imp > 1 else None
        for _ in range(ctx.mice_iter):
            for k in order:
                miss = M[:, k]
                if not miss.any() or len(schema[k]) == 0:
                    continue
                tr = np.flatnonzero(~miss)
                obs_rows = tr
                if boot is not None:
                    tb = boot[~miss[boot]]          # bootstrap rows, with repeats
                    tr = tb if len(np.unique(tb)) >= 8 else tr
                est = make_estimator(int(rs.integers(1 << 30)))
                pr = schema[k]
                est.fit(cur[np.ix_(tr, pr)], cur[tr, k])
                pred = est.predict(cur[np.ix_(np.flatnonzero(miss), pr)])
                if noise is True:
                    sigma = float(np.sqrt(1.0 / getattr(est, "alpha_", np.inf)))
                    if not np.isfinite(sigma) or sigma <= 0:
                        resid = cur[tr, k] - est.predict(cur[np.ix_(tr, pr)])
                        sigma = float(np.std(resid))
                    pred = pred + rs.normal(0.0, sigma, size=pred.shape)
                elif noise == "oob":
                    oob = np.setdiff1d(obs_rows, tr)
                    if len(oob) >= 5:
                        resid = cur[oob, k] - est.predict(cur[np.ix_(oob, pr)])
                    else:
                        resid = cur[tr, k] - est.predict(cur[np.ix_(tr, pr)])
                    sigma = float(np.std(resid))
                    pred = pred + rs.normal(0.0, sigma, size=pred.shape)
                cur[miss, k] = pred
        draws.append(cur.copy())
    return np.mean(draws, axis=0), draws


def imp_mice_bayes(Xm, ctx):
    """Proper multiple imputation (bootstrap + Bayesian-ridge residual draw,
    as mice's norm.boot), m = ctx.m."""
    from sklearn.linear_model import BayesianRidge
    return _iterative(Xm, lambda s: BayesianRidge(), _schema(ctx, Xm), ctx.m, ctx, noise=True)


def imp_missforest(Xm, ctx):
    """Random-forest conditional model (Stekhoven & Buhlmann 2012), with
    extremely randomised trees, single imputation."""
    from sklearn.ensemble import ExtraTreesRegressor
    mean, _ = _iterative(
        Xm, lambda s: ExtraTreesRegressor(n_estimators=50, max_depth=12,
                                          random_state=s, n_jobs=1),
        _schema(ctx, Xm), 1, ctx)
    return mean, None


def imp_missforest_mi(Xm, ctx):
    """missForest as proper multiple imputation: bootstrap rows per
    completion and an out-of-bootstrap residual draw."""
    from sklearn.ensemble import ExtraTreesRegressor
    return _iterative(
        Xm, lambda s: ExtraTreesRegressor(n_estimators=50, max_depth=12,
                                          random_state=s, n_jobs=1),
        _schema(ctx, Xm), ctx.m, ctx, noise="oob")


def imp_sklearn_ii(Xm, ctx):
    """scikit-learn IterativeImputer, default BayesianRidge, single imputation."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    it = IterativeImputer(max_iter=ctx.mice_iter, random_state=ctx.seed,
                          n_nearest_features=ctx.max_predictors,
                          initial_strategy="median", keep_empty_features=True)
    return it.fit_transform(Xm), None


def imp_miceforest(Xm, ctx):
    """miceforest with LightGBM, the same predictor sets, direct prediction
    (mean_match_candidates=0) and no saved models (tenfold less memory,
    identical values)."""
    import miceforest as mf
    schema = _schema(ctx, Xm)
    d = _named(Xm)
    names = list(d.columns)
    vs = {names[k]: [names[j] for j in schema[k]]
          for k in range(Xm.shape[1]) if np.isnan(Xm[:, k]).any()}
    kern = mf.ImputationKernel(d, num_datasets=1, random_state=ctx.seed,
                               variable_schema=vs, mean_match_candidates=0,
                               save_all_iterations_data=False)
    kern.mice(ctx.mice_iter, verbose=False, learning_rate=0.05,
              n_estimators=200, num_leaves=31, num_threads=1, verbosity=-1)
    out = kern.complete_data(0).to_numpy(dtype=float, copy=True)
    del kern
    gc.collect()
    return out, None


def holdout_mi(single_fn: Callable, Xm, ctx):
    """Turn any single imputer into multiple imputation. For each of m
    completions a different share (ctx.mi_holdout) of the observed cells is
    hidden and the imputer is rerun with a new seed; the hidden cells give
    each column's residual SD, and every imputed value receives a draw with
    that SD. Approximate (subsampling rather than bootstrap), but usable
    with black-box imputers."""
    M = np.isnan(Xm)
    obs = np.argwhere(~M)
    draws = []
    for l in range(ctx.m):
        rs = np.random.default_rng(ctx.seed + 600 + l)
        hold = obs[rs.random(len(obs)) < ctx.mi_holdout]
        Xh = Xm.copy()
        Xh[hold[:, 0], hold[:, 1]] = np.nan
        fit, _ = single_fn(Xh, ctx.with_(seed=ctx.seed + l))
        fit = np.asarray(fit, dtype=float)
        res = np.full(Xm.shape, np.nan)
        res[hold[:, 0], hold[:, 1]] = fit[hold[:, 0], hold[:, 1]] - Xm[hold[:, 0], hold[:, 1]]
        sig = _column_sigma(res)
        draws.append(np.where(M, fit + rs.normal(0.0, 1.0, Xm.shape) * sig, Xm))
    return np.mean(draws, axis=0), draws


def imp_miceforest_mi(Xm, ctx):
    return holdout_mi(imp_miceforest, Xm, ctx)


# ---------------------------------------------------------------- left-censored
def imp_minprob(Xm, ctx, q=0.01, tune_sigma=1.0):
    """MinProb (Lazar et al. 2016) applied per protein: missing values drawn
    from a normal centred on the protein's 1st percentile, SD = tune_sigma x
    the median protein SD."""
    rs = np.random.default_rng(ctx.seed)
    out = Xm.copy()
    sd_med = float(np.nanmedian(np.nanstd(Xm, axis=0)))
    for k in range(Xm.shape[1]):
        miss = np.isnan(Xm[:, k])
        if miss.any():
            mu = float(np.nanquantile(Xm[:, k], q))
            out[miss, k] = rs.normal(mu, tune_sigma * sd_med, miss.sum())
    return out, None


def imp_qrilc(Xm, ctx):
    """QRILC (Lazar et al. 2016) applied per protein: a normal fitted to the
    observed quantiles, draws truncated at the missing fraction."""
    rs = np.random.default_rng(ctx.seed)
    out = Xm.copy()
    for k in range(Xm.shape[1]):
        miss = np.isnan(Xm[:, k])
        nm = int(miss.sum())
        if nm == 0:
            continue
        obs = np.sort(Xm[~miss, k])
        pfrac = nm / len(miss)
        qs = pfrac + (1 - pfrac) * (np.arange(1, len(obs) + 1) - 0.5) / len(obs)
        z = _st.norm.ppf(np.clip(qs, 1e-6, 1 - 1e-6))
        sigma, mu = np.polyfit(z, obs, 1) if len(obs) >= 3 else (np.std(obs), np.mean(obs))
        sigma = float(abs(sigma)) if sigma and np.isfinite(sigma) else float(np.std(obs) or 1.0)
        upper = (_st.norm.ppf(pfrac) if pfrac < 1 else np.inf)
        out[miss, k] = mu + sigma * _st.truncnorm.rvs(-np.inf, upper, size=nm, random_state=rs)
    return out, None


# R-faithful parameter estimates (imputeLCMD 2.1), used by the R-agreement stage.
def minprob_params_r(D, q=0.01, tune_sigma=1.0):
    """imputeLCMD::impute.MinProb parameters for a features x samples matrix
    D: the q-quantile of each sample (column) and the SD (median SD of the
    features observed in more than half of the samples)."""
    D = np.asarray(D, dtype=float)
    centre = np.array([np.nanquantile(D[:, i], q) for i in range(D.shape[1])])
    frac_obs = np.isfinite(D).mean(axis=1)
    rows = D[frac_obs > 0.5]
    sds = np.array([np.nanstd(r, ddof=1) for r in rows if np.isfinite(r).sum() > 1])
    return centre, float(np.nanmedian(sds)) * tune_sigma


def qrilc_params_r(D):
    """imputeLCMD::impute.QRILC parameters (mean, sd and truncation point)
    for each sample (column) of a features x samples matrix D, reproducing
    its quantile-regression grid exactly."""
    D = np.asarray(D, dtype=float)
    out = []
    upper_q = 0.99
    for i in range(D.shape[1]):
        x = D[:, i]
        p_na = float(np.mean(~np.isfinite(x)))
        frm, to = p_na + 0.001, upper_q + 0.001
        by = (upper_q - p_na) / (upper_q * 100)
        n_steps = int(np.floor((to - frm) / by + 1e-10)) + 1
        qn = _st.norm.ppf(frm + by * np.arange(n_steps))
        probs = 0.001 + 0.01 * np.arange(int(np.floor((0.991 - 0.001) / 0.01 + 1e-10)) + 1)
        qx = np.nanquantile(x, probs)
        slope, intercept = np.polyfit(qn, qx, 1)
        upper = _st.norm.ppf(p_na + 0.001, loc=intercept, scale=slope)
        out.append((float(intercept), float(slope), float(upper), p_na))
    return pd.DataFrame(out, columns=["mean", "sd", "upper", "p_missing"])


def imp_minprob_r(Xm, ctx, q=0.01, tune_sigma=1.0):
    """MinProb exactly as imputeLCMD 2.1 applies it to a features x samples
    matrix: each sample (row of Xm) gets draws centred on its own q-quantile,
    SD = median SD of the proteins observed in more than half the samples."""
    D = Xm.T
    centre, sd = minprob_params_r(D, q, tune_sigma)
    rs = np.random.default_rng(ctx.seed)
    out = Xm.copy()
    for i in range(Xm.shape[0]):
        miss = np.isnan(Xm[i])
        if miss.any():
            out[i, miss] = rs.normal(centre[i], sd, int(miss.sum()))
    return out, None


def imp_qrilc_r(Xm, ctx, tune_sigma=1.0):
    """QRILC exactly as imputeLCMD 2.1 applies it to a features x samples
    matrix: per sample, draws from a normal with the quantile-regression
    mean and SD (times tune_sigma), truncated above at the missing fraction."""
    par = qrilc_params_r(Xm.T)
    rs = np.random.default_rng(ctx.seed)
    out = Xm.copy()
    for i in range(Xm.shape[0]):
        miss = np.isnan(Xm[i])
        if not miss.any():
            continue
        mu, sdv, up = par.loc[i, "mean"], par.loc[i, "sd"], par.loc[i, "upper"]
        scale = abs(sdv) * tune_sigma
        out[i, miss] = mu + scale * _st.truncnorm.rvs(-np.inf, (up - mu) / scale,
                                                      size=int(miss.sum()), random_state=rs)
    return out, None


# ---------------------------------------------------------------- AutoML / GAN
def imp_hyperimpute(Xm, ctx):
    """HyperImpute with a two-model candidate pool (linear, random forest)."""
    from hyperimpute.plugins.imputers import Imputers
    plug = Imputers().get("hyperimpute",
                          regression_seed=["linear_regression",
                                           "random_forest_regressor"],
                          classifier_seed=["logistic_regression"],
                          n_inner_iter=ctx.hyper_inner_iter, select_patience=2,
                          random_state=ctx.seed)
    return np.asarray(plug.fit_transform(_named(Xm)), dtype=float), None


def imp_gain(Xm, ctx):
    from hyperimpute.plugins.imputers import Imputers
    plug = Imputers().get("gain", n_epochs=ctx.gain_epochs, random_state=ctx.seed)
    return np.asarray(plug.fit_transform(_named(Xm)), dtype=float), None


# ---------------------------------------------------------------- new in MIPY v2
def _expit(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -50, 50)))


def _calibrate_selection(pi, gamma, grid=200):
    """Intercepts a_k such that E_z[expit(a_k - gamma z)] = pi_k, z ~ N(0,1)."""
    z = _st.norm.ppf((np.arange(grid) + 0.5) / grid)
    lo = np.full(len(pi), -30.0)
    hi = np.full(len(pi), 30.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        val = _expit(mid[:, None] - gamma * z[None, :]).mean(axis=1)
        lo = np.where(val < pi, mid, lo)
        hi = np.where(val < pi, hi, mid)
    return (lo + hi) / 2


def _longitudinal_index(patient):
    """Previous and next row of the same patient (-1 if none), patient codes
    and visit counts. Rows must be ordered by patient and visit."""
    codes = pd.factorize(np.asarray(patient))[0]
    n = len(codes)
    prev = np.full(n, -1)
    nxt = np.full(n, -1)
    same_prev = np.r_[False, codes[1:] == codes[:-1]]
    prev[same_prev] = np.flatnonzero(same_prev) - 1
    same_next = np.r_[codes[:-1] == codes[1:], False]
    nxt[same_next] = np.flatnonzero(same_next) + 1
    counts = np.bincount(codes)
    return codes, prev, nxt, counts


# Diagnostics of the last run in this process (read by runner.run_one).
# Filled only when asked for with ctx.extra["ess"] = True; computing them
# draws no random numbers, so the imputations are the same either way.
LAST_DIAGNOSTICS: dict = {}


def _ess_summary(values):
    """Summary of effective sample sizes of the importance resampling."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {}
    return {"ess_median": float(np.median(v)), "ess_mean": float(v.mean()),
            "ess_p10": float(np.percentile(v, 10)), "ess_p01": float(np.percentile(v, 1)),
            "ess_min": float(v.min()), "ess_below10_pct": float(100 * np.mean(v < 10)),
            "ess_cells": int(v.size)}


def _iterative_long(Xm, ctx, gamma=0.0, n_imp=None, n_candidates=50):
    """Longitudinal chained equations (MIPY v2).

    For protein k the predictors are its quickpred set plus three
    within-patient terms rebuilt from the current completion at every step:
    the same protein at the patient's previous visit, at the next visit,
    and its leave-one-out patient mean; the visit index is added too. Each
    completion is fitted on a patient-level (cluster) bootstrap and every
    imputed value is a Bayesian-ridge predictive draw.

    gamma > 0 turns the draw into a selection-model (MNAR) draw: candidates
    from the MAR predictive distribution are resampled with weights
    expit(a_k - gamma z), z being the standardised candidate value, so low
    values are favoured as under a detection limit; a_k matches each
    protein's missing fraction. gamma = 0 is MAR. gamma is a sensitivity
    parameter, not something the data can estimate.

    With ctx.extra["ess"] = True and gamma > 0, the effective sample size
    (sum w)^2 / sum w^2 of the candidate weights is recorded for every
    imputed cell of the last sweep of every completion and summarised in
    LAST_DIAGNOSTICS (C = n_candidates is the largest possible value).
    """
    from sklearn.linear_model import BayesianRidge
    want_ess = bool((ctx.extra or {}).get("ess")) and gamma > 0
    ess_vals = []
    n_imp = ctx.m if n_imp is None else n_imp
    schema = _schema(ctx, Xm)
    patient = ctx.rows_by_patient()
    tvals = (np.zeros(len(Xm)) if ctx.time is None
             else pd.to_numeric(pd.Series(np.asarray(ctx.time)), errors="coerce")
             .fillna(0).to_numpy(dtype=float))
    codes, prev, nxt, counts = _longitudinal_index(patient)
    n_pat = len(counts)
    rows_of = [np.flatnonzero(codes == c) for c in range(n_pat)]
    M = np.isnan(Xm)
    order = np.argsort(M.sum(axis=0))
    mu_o = np.nanmean(Xm, 0)
    sd_o = np.nanstd(Xm, 0)
    sd_o[~np.isfinite(sd_o) | (sd_o < 1e-12)] = 1.0
    a_k = _calibrate_selection(M.mean(0), gamma) if gamma > 0 else None
    draws = []
    for rep in range(n_imp):
        rs = np.random.default_rng(ctx.seed + 7919 * (rep + 1))
        cur = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(Xm)
        if n_imp > 1:
            bp = rs.integers(0, n_pat, n_pat)
            boot_rows = np.concatenate([rows_of[c] for c in bp])
        else:
            boot_rows = np.arange(len(Xm))
        for sweep in range(ctx.mice_iter):
            for k in order:
                miss = M[:, k]
                if not miss.any():
                    continue
                col = cur[:, k]
                psum = np.bincount(codes, weights=col, minlength=n_pat)
                cnt = counts[codes]
                loo = np.where(cnt > 1, (psum[codes] - col) / np.maximum(cnt - 1, 1), mu_o[k])
                lag = np.where(prev >= 0, col[np.maximum(prev, 0)], loo)
                lead = np.where(nxt >= 0, col[np.maximum(nxt, 0)], loo)
                F = np.column_stack([cur[:, schema[k]], lag, lead, loo, tvals])
                tr = boot_rows[~miss[boot_rows]]
                if len(np.unique(tr)) < 8:
                    tr = np.flatnonzero(~miss)
                est = BayesianRidge().fit(F[tr], cur[tr, k])
                mi_rows = np.flatnonzero(miss)
                mu = est.predict(F[mi_rows])
                sigma = float(np.sqrt(1.0 / est.alpha_)) if est.alpha_ > 0 else float(np.std(cur[tr, k]))
                if gamma > 0:
                    cand = mu[:, None] + sigma * rs.normal(size=(len(mi_rows), n_candidates))
                    w = _expit(a_k[k] - gamma * (cand - mu_o[k]) / sd_o[k])
                    w = w / w.sum(axis=1, keepdims=True)
                    if want_ess and sweep == ctx.mice_iter - 1:
                        ess_vals.append(1.0 / (w ** 2).sum(axis=1))   # weights sum to 1
                    if n_imp > 1:
                        u = rs.random(len(mi_rows))[:, None]
                        pick = (np.cumsum(w, axis=1) < u).sum(axis=1)
                        val = cand[np.arange(len(mi_rows)), np.minimum(pick, n_candidates - 1)]
                    else:
                        val = (w * cand).sum(axis=1)
                else:
                    val = mu + rs.normal(0.0, sigma, size=mu.shape) if n_imp > 1 else mu
                cur[mi_rows, k] = val
        draws.append(cur.copy())
    if want_ess:
        LAST_DIAGNOSTICS.update(gamma=float(gamma), candidates=int(n_candidates),
                                **_ess_summary(np.concatenate(ess_vals) if ess_vals else []))
    return np.mean(draws, axis=0), (draws if n_imp > 1 else None)


def imp_mice_long(Xm, ctx):
    return _iterative_long(Xm, ctx, gamma=0.0)


def imp_mice_long_mnar(Xm, ctx):
    return _iterative_long(Xm, ctx, gamma=ctx.gamma)


# ---------------------------------------------------------------- registry
@dataclass(frozen=True)
class MethodInfo:
    name: str
    fn: Callable
    family: str
    peer_reviewed: bool
    package: str
    function: str
    cite: str
    needs: Optional[str] = None       # optional module required
    multiple: bool = False            # returns m completions
    longitudinal: bool = False        # uses the patient structure
    slow: bool = False
    new: bool = False                 # introduced by MIPY
    main: bool = True                 # part of the main benchmark tables
    rev: str = ""                     # revision; part of the checkpoint fingerprint
    adapted: bool = False             # published method changed here (see evidence)

    @property
    def evidence(self) -> str:
        """Evidence behind the method as implemented, as in the MIPY paper:
        "yes" (a peer-reviewed publication describes it), "adapted" (a
        published method changed in engine, stopping rule or unit of
        application) or "no" (no peer-reviewed description; this includes
        the methods new in MIPY)."""
        if self.adapted:
            return "adapted"
        if self.new:
            return "no"
        return "yes" if self.peer_reviewed else "no"


_M = MethodInfo
METHODS = {m.name: m for m in [
    _M("Mean", imp_mean, "Naive", False, "scikit-learn", "SimpleImputer(strategy='mean')", "pedregosa2011sklearn"),
    _M("Median", imp_median, "Naive", False, "scikit-learn", "SimpleImputer(strategy='median')", "pedregosa2011sklearn"),
    _M("LOCF", imp_locf, "Longitudinal", False, "pandas", "groupby().ffill().bfill()", "", longitudinal=True),
    _M("KNN", imp_knn, "Neighbour", True, "scikit-learn", "KNNImputer", "troyanskaya2001"),
    _M("softImpute", imp_softimpute, "Low-rank", True, "mipy (numpy)", "soft-thresholded SVD, CV lambda", "mazumder2010softimpute", rev="cv-lambda-v2"),
    _M("MICE-Bayes", imp_mice_bayes, "Chained", True, "mipy (scikit-learn)", "BayesianRidge in FCS loop", "vanbuuren2011mice", multiple=True, rev="proper-draws-v2", adapted=True),
    _M("missForest", imp_missforest, "Chained", True, "mipy (scikit-learn)", "ExtraTreesRegressor in FCS loop", "stekhoven2012missforest", slow=True, adapted=True),
    _M("sklearn-II", imp_sklearn_ii, "Chained", False, "scikit-learn", "IterativeImputer", "pedregosa2011sklearn"),
    _M("miceforest", imp_miceforest, "Chained", False, "miceforest", "ImputationKernel().mice()", "wilson2024miceforest", needs="miceforest", slow=True, rev="one-thread-v2"),
    _M("HyperImpute", imp_hyperimpute, "AutoML", True, "hyperimpute", "Imputers().get('hyperimpute')", "jarrett2022hyperimpute", needs="hyperimpute", slow=True),
    _M("GAIN", imp_gain, "Generative", True, "hyperimpute", "Imputers().get('gain')", "yoon2018gain", needs="hyperimpute", slow=True),
    _M("MinProb", imp_minprob, "Left-censored", True, "mipy (numpy)", "MinProb, per protein", "lazar2016", adapted=True),
    _M("QRILC", imp_qrilc, "Left-censored", True, "mipy (scipy)", "QRILC, per protein", "lazar2016", adapted=True),
    _M("MICE-Long", imp_mice_long, "Longitudinal chained", False, "mipy", "FCS + within-patient terms, cluster bootstrap", "resche2018", multiple=True, longitudinal=True, new=True, rev="long-v1"),
    _M("MICE-LongMNAR", imp_mice_long_mnar, "Selection-model MNAR", False, "mipy", "MICE-Long + selection-model draws (gamma)", "tompsett2018", multiple=True, longitudinal=True, new=True, rev="longmnar-v1"),
    _M("MinProb-R", imp_minprob_r, "Left-censored", True, "mipy (numpy)", "MinProb, per sample (imputeLCMD)", "lazar2016", main=False),
    _M("QRILC-R", imp_qrilc_r, "Left-censored", True, "mipy (scipy)", "QRILC, per sample (imputeLCMD)", "lazar2016", main=False),
    _M("softImpute-MI", imp_softimpute_mi, "Low-rank", True, "mipy (numpy)", "softImpute + holdout residual draws", "mazumder2010softimpute", multiple=True, new=True, main=False, rev="mi-v1"),
    _M("missForest-MI", imp_missforest_mi, "Chained", True, "mipy (scikit-learn)", "missForest + bootstrap + OOB residual draws", "stekhoven2012missforest", multiple=True, slow=True, new=True, main=False, rev="mi-v1"),
    _M("miceforest-MI", imp_miceforest_mi, "Chained", False, "miceforest", "miceforest + holdout residual draws", "wilson2024miceforest", needs="miceforest", multiple=True, slow=True, new=True, main=False, rev="mi-v1"),
]}


def available(name: str) -> bool:
    """True if the optional package a method needs can be imported."""
    need = METHODS[name].needs
    if need is None:
        return True
    import importlib.util
    return importlib.util.find_spec(need) is not None


def run(name: str, X, ctx: Optional[ImputeContext] = None):
    """Run one method; observed cells are restored afterwards."""
    ctx = ctx or ImputeContext()
    X = np.asarray(X, dtype=float)
    LAST_DIAGNOSTICS.clear()
    out, draws = METHODS[name].fn(X.copy(), ctx)
    out = np.asarray(out, dtype=float)
    if out.shape != X.shape:
        raise ValueError(f"{name} returned shape {out.shape}, expected {X.shape}")
    if np.isnan(out).any():
        raise ValueError(f"{name} left {int(np.isnan(out).sum())} cells missing")
    obs = ~np.isnan(X)
    out[obs] = X[obs]
    if draws:
        draws = [np.where(obs, X, np.asarray(d, dtype=float)) for d in draws]
    return out, draws
