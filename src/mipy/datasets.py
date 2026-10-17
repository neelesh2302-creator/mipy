"""Datasets: the GSE65622 analysis files and a calibrated synthetic
longitudinal cohort with complete ground truth."""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

GSE_FILES = {"RFS": "RFS_data.csv", "MFS": "MFS_data.csv"}


def load_gse65622(endpoint="RFS", path=None) -> pd.DataFrame:
    """Load a GSE65622 analysis file (80 patients, 288 visits, 507 proteins,
    as used by MIML and MIPY). ``path`` may be the CSV itself or its folder;
    by default the copy bundled in ``mipy/data`` is used if present.

    The files are derived from the public GEO series GSE65622
    (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65622)."""
    fname = GSE_FILES[endpoint]
    cands = []
    if path is not None:
        p = Path(path)
        cands += [p] if p.suffix.lower() == ".csv" else [p / fname]
    cands.append(Path(str(resources.files("mipy").joinpath("data/" + fname))))
    for c in cands:
        if c.is_file():
            return pd.read_csv(c)
    raise FileNotFoundError(
        f"{fname} not found. Pass path=... or copy the file into the package's "
        f"data folder ({resources.files('mipy').joinpath('data')}).")


def simulate_longitudinal(n_patients=200, n_visits=4, n_proteins=200, rank=5,
                          icc=0.6, visit_effect=0.15, seed=2026, calibrate_from=None):
    """A longitudinal proteomic cohort with complete true values.

    Each protein k for patient i at visit j is
        mu_k + sd_k * (sqrt(icc_k) * b_ik + sqrt(1 - icc_k) * e_ijk + trend_kj)
    where b_i and e_ij have a rank-``rank`` correlation structure across
    proteins plus noise. If ``calibrate_from`` = (X, patient) is given
    (X on the analysis scale, NaN allowed), protein means, SDs and
    within-patient correlations (ICC) are taken from it.

    Returns dict(X, patient, time, proteins); X has no missing values.
    """
    rs = np.random.default_rng(seed)
    if calibrate_from is not None:
        Xr, pat = calibrate_from
        Xr = np.asarray(Xr, dtype=float)
        pick = np.sort(rs.choice(Xr.shape[1], size=min(n_proteins, Xr.shape[1]), replace=False))
        Xr = Xr[:, pick]
        n_proteins = Xr.shape[1]
        mu = np.nanmean(Xr, 0)
        sd = np.nanstd(Xr, 0)
        sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        df = pd.DataFrame(Xr)
        g = df.groupby(np.asarray(pat))
        between = g.mean().var(ddof=1).to_numpy()
        within = g.var(ddof=1).mean().to_numpy()
        iccs = np.clip(between / np.maximum(between + within, 1e-12), 0.05, 0.95)
        iccs = np.where(np.isfinite(iccs), iccs, icc)
    else:
        mu = rs.uniform(6, 12, n_proteins)
        sd = rs.uniform(0.4, 1.5, n_proteins)
        iccs = np.clip(rs.normal(icc, 0.15, n_proteins), 0.05, 0.95)

    def lowrank(n):
        L = rs.normal(size=(n, rank)) @ rs.normal(size=(rank, n_proteins)) / np.sqrt(rank)
        L = L + 0.7 * rs.normal(size=(n, n_proteins))
        return (L - L.mean(0)) / L.std(0)

    b = lowrank(n_patients)
    n_rows = n_patients * n_visits
    e = lowrank(n_rows)
    trend = visit_effect * rs.normal(size=(n_visits, n_proteins))
    pat = np.repeat(np.arange(n_patients), n_visits)
    vis = np.tile(np.arange(1, n_visits + 1), n_patients)
    Z = np.sqrt(iccs) * b[pat] + np.sqrt(1 - iccs) * e + trend[vis - 1]
    X = mu + sd * Z
    return {"X": X, "patient": pat, "time": vis,
            "proteins": [f"sim{k}" for k in range(n_proteins)]}


def example_cohort(n_patients=80, n_proteins=120, seed=2026, save=None) -> pd.DataFrame:
    """A synthetic cohort laid out like the GSE65622 analysis files, for
    trying MIPY and as a template for your own table.

    Columns: ``Patient_ID``, ``sampling_point`` (visits 1, 2, 4, 5; some
    patients miss visits), ``Prot_001`` ... (positive, right-skewed
    intensities, about 10% missing, mostly low values lost below a
    per-protein detection limit), ``RFS`` (days from the sample to relapse
    or censoring), ``event`` (1 = relapse), and the clinical columns
    ``trg_score``, ``ypn_stage``, ``ypt_stage`` and ``batch``. Five
    proteins (Prot_001 to Prot_005) are linked to relapse risk.

    The values are simulated; they are not patient data. ``save`` writes
    the table to a CSV file as well.
    """
    rs = np.random.default_rng(seed)
    visits = np.array([1, 2, 4, 5])
    sim = simulate_longitudinal(n_patients=n_patients, n_visits=len(visits),
                                n_proteins=n_proteins, rank=5, icc=0.6, seed=seed)
    Z = sim["X"]                                       # log2 scale, complete
    pat = sim["patient"]
    vis = visits[sim["time"] - 1]
    present = (vis == 1) | (rs.random(len(vis)) < 0.85)
    # relapse risk from the baseline level of five proteins and nodal stage
    base = Z[vis == 1][:, :5]
    zb = (base - base.mean(0)) / base.std(0)
    ypn = np.clip(np.round(rs.normal(0.6 + 0.5 * zb[:, 0], 0.8)), 0, 2).astype(int)
    lin = zb @ np.array([0.8, 0.6, 0.5, -0.6, 0.4]) + 0.4 * ypn
    t_event = rs.exponential(3000 * np.exp(-(lin - lin.mean())))
    t_cens = rs.uniform(900, 3200, n_patients)
    T = np.minimum(t_event, t_cens)
    E = (t_event <= t_cens).astype(int)
    day = {1: 0, 2: 14, 4: 90, 5: 180}               # days from the first sample
    offs = np.array([day[v] for v in vis])
    rfs = np.round(T[pat] - offs)
    present &= (rfs > 0) | (vis == 1)
    rfs = np.maximum(rfs, 1)
    # detection limit per protein, then a little random loss
    X = Z.copy()
    mu = Z.mean(0)                                     # low-abundance proteins lose more values
    rank = (np.argsort(np.argsort(-mu)) + 0.5) / n_proteins
    lod_q = np.clip(0.11 * rank * rs.uniform(0.3, 1.7, n_proteins), 0, 0.3)
    lod = np.array([np.quantile(Z[:, k], lod_q[k]) for k in range(n_proteins)])
    below = (Z < lod) & (rs.random(Z.shape) < 0.85)
    X[below] = np.nan
    X[rs.random(Z.shape) < 0.035] = np.nan
    raw = np.round(2.0 ** X - 1.0, 4)
    names = [f"Prot_{k + 1:03d}" for k in range(n_proteins)]
    df = pd.DataFrame(raw, columns=names)
    df.insert(0, "sampling_point", vis)
    df.insert(0, "Patient_ID", pat + 1)
    df["RFS"] = rfs.astype(int)
    df["event"] = E[pat]
    trg = np.clip(np.round(rs.normal(3 - 0.6 * zb[:, 1], 1.0)), 1, 5).astype(int)
    ypt = np.clip(np.round(rs.normal(2 + 0.4 * ypn, 0.9)), 0, 4).astype(int)
    df["trg_score"] = trg[pat]
    df["ypn_stage"] = ypn[pat]
    df["ypt_stage"] = ypt[pat]
    df["batch"] = rs.integers(1, 4, n_patients)[pat]
    df = df[present].reset_index(drop=True)
    if save is not None:
        df.to_csv(save, index=False)
    return df
