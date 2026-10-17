"""Figures for :func:`mipy.analyze`.

Every function returns a matplotlib ``Figure`` built without pyplot, so it
works the same in scripts, terminals without a display, Jupyter and
worker processes. ``save_figure`` writes PNG and PDF copies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MECH_LABEL = {"MCAR": "MCAR (completely at random)", "MAR": "MAR (depends on visit and clinical data)",
              "MNAR": "MNAR (low values lost)", "LOD": "LOD (below a detection limit)"}
MECH_COLOR = {"MCAR": "#4C72B0", "MAR": "#DD8452", "MNAR": "#C44E52", "LOD": "#8172B3"}


def _figure(w, h):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=100)
    fig.set_layout_engine("constrained")
    return fig


def method_colors(names=None):
    """A fixed colour for every method, so a method looks the same in every
    figure (tab20 has 20 colours, one per method in mipy.METHODS)."""
    from matplotlib import colormaps
    from .imputers import METHODS
    cmap = colormaps["tab20"]
    order = list(METHODS)
    cols = {n: cmap(i % 20) for i, n in enumerate(order)}
    for i, n in enumerate(names or []):
        if n not in cols:
            cols[n] = cmap((len(order) + i) % 20)
    return cols


def save_figure(fig, folder, name, dpi=200):
    """Write ``name``.png and ``name``.pdf into ``folder``; returns the PNG path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / f"{name}.png"
    fig.savefig(png, dpi=dpi)
    fig.savefig(folder / f"{name}.pdf")
    return png


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)


def _hbox(ax, data):
    """Horizontal boxplot on every matplotlib version ('vert' was replaced
    by 'orientation' in matplotlib 3.10)."""
    kw = dict(showfliers=False, patch_artist=True, widths=0.6)
    try:
        return ax.boxplot(data, orientation="horizontal", **kw)
    except TypeError:
        return ax.boxplot(data, vert=False, **kw)


def _panel_label(ax, letter):
    ax.set_title(letter, loc="left", fontsize=10, fontweight="bold")


# ---------------------------------------------------------------- data
def missingness(X, features, time=None, max_cols=400, max_rows=600):
    """Where the values are missing: per feature, per visit, the pattern,
    and whether low-level features miss more often."""
    from scipy.stats import spearmanr
    X = np.asarray(X, dtype=float)
    M = np.isnan(X)
    fig = _figure(10, 7.2)
    axs = fig.subplots(2, 2)
    a = axs[0, 0]
    pf = 100 * M.mean(0)
    a.hist(pf, bins=min(40, max(10, len(pf) // 5)), color="#4C72B0", edgecolor="white")
    a.set_xlabel("% missing in the feature", fontsize=9)
    a.set_ylabel("features", fontsize=9)
    a.set_title(f"A  Missing values per feature (overall {100 * M.mean():.1f}%)", loc="left", fontsize=10)
    _style(a)

    a = axs[0, 1]
    if time is not None and len(np.unique(time)) <= 40:
        t = np.asarray(time)
        vals = np.unique(t)
        pv = [100 * M[t == v].mean() for v in vals]
        nv = [int((t == v).sum()) for v in vals]
        lab = [str(int(v)) if float(v).is_integer() else f"{v:g}" for v in vals]
        bars = a.bar(range(len(vals)), pv, color="#4C72B0")
        a.set_xticks(range(len(vals)), lab)
        for b, n in zip(bars, nv):
            a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n}", ha="center",
                   va="bottom", fontsize=7)
        a.set_xlabel("visit", fontsize=9)
        a.set_ylabel("% missing", fontsize=9)
        a.set_title("B  Missing values by visit", loc="left", fontsize=10)
    else:
        a.hist(100 * M.mean(1), bins=30, color="#4C72B0", edgecolor="white")
        a.set_xlabel("% missing in the row", fontsize=9)
        a.set_ylabel("rows", fontsize=9)
        a.set_title("B  Missing values per row", loc="left", fontsize=10)
    _style(a)

    a = axs[1, 0]
    order = np.argsort(-pf, kind="stable")
    cols = order[:max_cols] if len(order) > max_cols else order
    rows = np.arange(X.shape[0])
    if len(rows) > max_rows:
        rows = np.linspace(0, X.shape[0] - 1, max_rows).astype(int)
    a.imshow(M[np.ix_(rows, cols)], aspect="auto", cmap="Greys", interpolation="nearest")
    a.set_xlabel(f"features, most missing first{' (first %d)' % max_cols if len(order) > max_cols else ''}",
                 fontsize=9)
    a.set_ylabel("rows (patient visits)", fontsize=9)
    a.set_title("C  Pattern (black = missing)", loc="left", fontsize=10)
    a.tick_params(labelsize=7)

    a = axs[1, 1]
    med = np.nanmedian(np.where(M, np.nan, X), axis=0) if (~M).any() else np.zeros(X.shape[1])
    g = np.isfinite(med)
    a.scatter(med[g], pf[g], s=8, alpha=0.6, color="#C44E52", edgecolor="none")
    rho = np.nan
    if g.sum() >= 5 and np.ptp(pf[g]) > 0:
        rho = float(spearmanr(med[g], pf[g])[0])
    a.set_xlabel("median observed value of the feature (analysis scale)", fontsize=9)
    a.set_ylabel("% missing", fontsize=9)
    a.set_title("D  Do low-level features miss more?", loc="left", fontsize=10)
    a.text(0.98, 0.95, f"Spearman rho = {rho:.2f}" if np.isfinite(rho) else "rho not defined",
           transform=a.transAxes, ha="right", va="top", fontsize=8)
    _style(a)
    return fig


def masking(zvals: dict, rate):
    """Standardised true value of the hidden cells under each mechanism:
    MNAR and LOD hide low values, MCAR does not."""
    mechs = list(zvals)
    fig = _figure(max(5, 1.6 * len(mechs) + 2), 3.6)
    a = fig.subplots()
    data = [np.asarray(zvals[m])[np.isfinite(zvals[m])] for m in mechs]
    parts = a.violinplot(data, showmedians=True, showextrema=False)
    for b, m in zip(parts["bodies"], mechs):
        b.set_facecolor(MECH_COLOR.get(m, "#999999"))
        b.set_alpha(0.6)
    a.axhline(0, color="black", lw=0.8, ls="--")
    a.set_xticks(range(1, len(mechs) + 1), mechs)
    a.set_ylabel("z-score of the hidden true value", fontsize=9)
    a.set_title(f"Values hidden by each masking design ({int(round(100 * rate))}% of observed values)",
                loc="left", fontsize=10)
    _style(a)
    return fig


# ---------------------------------------------------------------- accuracy
def accuracy(summary: pd.DataFrame, colors=None):
    """Mean nRMSE (with Monte Carlo SE) of every method under each mechanism."""
    mechs = list(dict.fromkeys(summary["mechanism"]))
    colors = colors or method_colors(list(summary["method"].unique()))
    nm = summary["method"].nunique()
    fig = _figure(3.3 * len(mechs) + 1.2, 0.28 * nm + 1.6)
    axs = np.atleast_1d(fig.subplots(1, len(mechs), sharex=False))
    for a, mech in zip(axs, mechs):
        s = summary[(summary["mechanism"] == mech) & summary["nRMSE"].notna()].sort_values("nRMSE", ascending=False)
        y = np.arange(len(s))
        err = s["MCSE"].fillna(0).to_numpy()
        for yi, (_, r), e in zip(y, s.iterrows(), err):
            a.errorbar(r["nRMSE"], yi, xerr=1.96 * e if e > 0 else None, fmt="o",
                       color=colors.get(r["method"], "grey"), ms=6, capsize=2)
        a.set_yticks(y, s["method"])
        a.axvline(1.0, color="grey", ls="--", lw=0.8)
        if len(s):
            best = s.iloc[-1]
            a.annotate("best", (best["nRMSE"], y[-1]), xytext=(6, 0), textcoords="offset points",
                       va="center", fontsize=7)
        a.set_xlabel("nRMSE (lower is better)", fontsize=9)
        a.text(1.0, 1.0, " feature mean", transform=a.get_xaxis_transform(), fontsize=7,
               color="grey", va="top", ha="left")
        a.set_title(MECH_LABEL.get(mech, mech), loc="left", fontsize=10)
        _style(a)
    return fig


def protein_nrmse(per_protein: pd.DataFrame, colors=None, clip_q=0.98):
    """Spread of nRMSE across features (each feature's mean over repeats)."""
    mechs = list(dict.fromkeys(per_protein["mechanism"]))
    colors = colors or method_colors(list(per_protein["method"].unique()))
    nm = per_protein["method"].nunique()
    fig = _figure(3.3 * len(mechs) + 1.2, 0.3 * nm + 1.6)
    axs = np.atleast_1d(fig.subplots(1, len(mechs)))
    hi = np.nanquantile(per_protein["nRMSE"], clip_q) if len(per_protein) else 2
    for a, mech in zip(axs, mechs):
        d = per_protein[per_protein["mechanism"] == mech].dropna(subset=["nRMSE"])
        meds = d.groupby("method")["nRMSE"].median().sort_values(ascending=False)
        data = [d.loc[d["method"] == mth, "nRMSE"].to_numpy() for mth in meds.index]
        if not data:
            continue
        bp = _hbox(a, data)
        for patch, mth in zip(bp["boxes"], meds.index):
            patch.set_facecolor(colors.get(mth, "grey"))
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("black")
        a.set_yticks(range(1, len(meds) + 1), meds.index)
        a.axvline(1.0, color="grey", ls="--", lw=0.8)
        a.set_xlim(0, max(1.2, min(hi, 4)))
        a.set_xlabel("nRMSE per feature", fontsize=9)
        a.set_title(MECH_LABEL.get(mech, mech), loc="left", fontsize=10)
        _style(a)
    return fig


def sd_ratio(summary: pd.DataFrame):
    """Within-patient SD of the completed data / that of the true values."""
    s = summary.dropna(subset=["SD ratio"])
    mechs = list(dict.fromkeys(s["mechanism"]))
    methods = (s.groupby("method")["SD ratio"].apply(lambda v: np.nanmean(np.abs(np.log(v))))
               .sort_values(ascending=False).index.tolist())
    fig = _figure(6.5, 0.28 * len(methods) + 1.6)
    a = fig.subplots()
    off = np.linspace(-0.2, 0.2, max(1, len(mechs)))
    for k, mech in enumerate(mechs):
        d = s[s["mechanism"] == mech].set_index("method")
        y = [methods.index(mm) + off[k] for mm in d.index]
        a.scatter(d["SD ratio"], y, color=MECH_COLOR.get(mech, "grey"), label=mech, s=22, zorder=3)
    a.axvline(1.0, color="grey", ls="--", lw=0.8)
    a.set_yticks(range(len(methods)), methods)
    a.set_xlabel("within-patient SD ratio (1 = visit-to-visit variation kept)", fontsize=8)
    a.legend(fontsize=7, frameon=False, loc="best")
    a.set_title("Does the method keep each patient's visit-to-visit variation?", loc="left", fontsize=9)
    _style(a)
    return fig


def coverage(summary: pd.DataFrame, level=95):
    """Share of hidden values inside the multiple-imputation intervals."""
    s = summary.dropna(subset=["coverage (%)"])
    mechs = list(dict.fromkeys(s["mechanism"]))
    methods = sorted(s["method"].unique())
    fig = _figure(max(6.5, 1.1 * len(methods) + 3), 3.6)
    a = fig.subplots()
    w = min(0.8 / max(1, len(mechs)), 0.22)
    for k, mech in enumerate(mechs):
        d = s[s["mechanism"] == mech].set_index("method").reindex(methods)
        x = np.arange(len(methods)) + (k - (len(mechs) - 1) / 2) * w
        a.bar(x, d["coverage (%)"], width=w, color=MECH_COLOR.get(mech, "grey"), label=mech)
    a.axhline(level, color="black", ls="--", lw=0.8)
    a.text(len(methods) - 0.45, level, f"{level}% target", va="bottom", ha="right", fontsize=7)
    a.set_xticks(range(len(methods)), methods, fontsize=8)
    a.set_ylabel("% of hidden values inside the interval", fontsize=8)
    a.set_ylim(0, 100)
    a.set_xlim(-0.6, len(methods) - 0.4)
    a.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    a.set_title("Multiple imputation: are the 95% intervals honest?", loc="left", fontsize=9)
    _style(a)
    return fig


def runtime(scores: pd.DataFrame, colors=None):
    """Mean seconds per imputation run."""
    s = scores[scores["ok"]].groupby("method")["seconds"].mean().sort_values()
    colors = colors or method_colors(list(s.index))
    fig = _figure(6, 0.26 * len(s) + 1.4)
    a = fig.subplots()
    a.barh(range(len(s)), s.to_numpy().clip(min=1e-3), color=[colors.get(m, "grey") for m in s.index])
    a.set_yticks(range(len(s)), s.index)
    a.set_xscale("log")
    a.set_xlabel("seconds per run (log scale)", fontsize=8)
    for i, v in enumerate(s.to_numpy()):
        a.text(max(v, 1e-3), i, f" {v:.1f} s", va="center", fontsize=7)
    a.set_title("Run time on this table", loc="left", fontsize=9)
    _style(a)
    return fig


# ---------------------------------------------------------------- results
def imputed_vs_observed(X, completed, method):
    """Imputed values of the truly missing cells against the observed values."""
    X = np.asarray(X, dtype=float)
    C = np.asarray(completed, dtype=float)
    M = np.isnan(X)
    fig = _figure(9.5, 3.6)
    a, b = fig.subplots(1, 2)
    obs, imp = X[~M], C[M]
    lo, hi = np.nanquantile(np.r_[obs, imp], [0.002, 0.998]) if imp.size else np.nanquantile(obs, [0.002, 0.998])
    bins = np.linspace(lo, hi, 60)
    a.hist(obs, bins=bins, density=True, alpha=0.55, color="#4C72B0", label=f"observed (n={obs.size:,})")
    if imp.size:
        a.hist(imp, bins=bins, density=True, alpha=0.55, color="#C44E52", label=f"imputed (n={imp.size:,})")
    a.set_xlabel("value (analysis scale)", fontsize=8)
    a.set_ylabel("density", fontsize=8)
    a.legend(fontsize=7, frameon=False)
    a.set_title(f"A  {method}: imputed vs observed values", loc="left", fontsize=9)
    _style(a)
    mu = np.nanmean(np.where(M, np.nan, X), 0)
    sd = np.nanstd(np.where(M, np.nan, X), 0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    zs = []
    for k in range(X.shape[1]):
        if M[:, k].any():
            zs.append(np.mean((C[M[:, k], k] - mu[k]) / sd[k]))
    zs = np.asarray(zs)
    if zs.size:
        b.hist(zs, bins=40, color="#C44E52", alpha=0.75, edgecolor="white")
        b.axvline(0, color="black", ls="--", lw=0.8)
        b.text(0.98, 0.95, f"median {np.median(zs):+.2f} SD", transform=b.transAxes, ha="right",
               va="top", fontsize=8)
    b.set_xlabel("mean imputed value of a feature, in SD from its observed mean", fontsize=8)
    b.set_ylabel("features", fontsize=8)
    b.set_title("B  Below 0: imputed lower than observed", loc="left", fontsize=9)
    _style(b)
    return fig


def survival_volcano(tab: pd.DataFrame, q_cut=0.10, n_label=6):
    """Univariate Cox screen: log hazard ratio per SD against -log10 p."""
    t = tab.dropna(subset=["log_hr", "p"])
    fig = _figure(6.2, 4.4)
    a = fig.subplots()
    sig = t["q"] < q_cut
    y = -np.log10(t["p"].clip(lower=1e-300))
    a.scatter(t.loc[~sig, "log_hr"], y[~sig], s=10, color="#999999", alpha=0.6, edgecolor="none",
              label="not significant")
    a.scatter(t.loc[sig, "log_hr"], y[sig], s=16, color="#C44E52", edgecolor="none",
              label=f"FDR q < {q_cut:.2f}")
    for i, (_, r) in enumerate(t.nsmallest(n_label, "p").iterrows()):
        a.annotate(str(r["feature"])[:18], (r["log_hr"], -np.log10(max(r["p"], 1e-300))),
                   xytext=(4, 4 if i % 2 == 0 else -10), textcoords="offset points", fontsize=7,
                   ha="left" if r["log_hr"] < 0 else "right")
    a.axvline(0, color="black", lw=0.6)
    a.set_xlabel("log hazard ratio per SD (baseline sample)", fontsize=8)
    a.set_ylabel("-log10 p", fontsize=8)
    a.legend(fontsize=7, frameon=False)
    a.set_title("Exploratory survival screen", loc="left", fontsize=9)
    _style(a)
    return fig
