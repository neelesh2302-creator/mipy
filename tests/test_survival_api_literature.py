import numpy as np
import pandas as pd
import pytest

import mipy
from mipy.survival import univariate_cox, reconcile, audit_endpoints, bh_fdr


def test_cox_matches_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rs = np.random.default_rng(2)
    n = 120
    x = rs.normal(size=n)
    t = rs.exponential(np.exp(-0.7 * x))
    # continuous times: lifelines uses Efron's tie handling, MIPY Breslow's
    e = (rs.random(n) < 0.8).astype(int)
    ours = univariate_cox(t, e, x[:, None]).iloc[0]
    cph = lifelines.CoxPHFitter().fit(pd.DataFrame({"T": t, "E": e, "x": (x - x.mean()) / x.std()}),
                                      "T", "E")
    assert np.isclose(ours.beta, cph.params_["x"], atol=2e-3)
    assert np.isclose(ours.se, cph.standard_errors_["x"], atol=2e-3)


def test_reconcile_is_consistent():
    ep = pd.DataFrame({"t_RFS": [100, 200, 300, 50], "e_RFS": [0, 1, 0, 1],
                       "t_MFS": [80, 250, 300, -5], "e_MFS": [1, 1, 0, 1]})
    a = audit_endpoints(ep)
    assert a["metastasis_without_relapse"] == 1 and a["non_positive_times"] == 1
    r = reconcile(ep)
    assert list(r.E) == [1, 1, 0, 1] and list(r["T"]) == [80, 200, 300, -5]
    assert list(r.excluded) == [False, False, False, True]


def test_bh():
    q = bh_fdr(np.array([0.01, 0.04, 0.03, 0.2]))
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)


def test_public_api(cohort):
    df = pd.DataFrame(cohort["X"], columns=[f"P{k}" for k in range(cohort["X"].shape[1])])
    df.insert(0, "visit", cohort["time"])
    df.insert(0, "pid", cohort["patient"])
    df = df.sample(frac=1, random_state=1)            # rows shuffled on purpose
    out = mipy.impute(df, method="MICE-Long", id_col="pid", time_col="visit", mice_iter=2)
    assert list(out.index) == list(range(len(df))) or len(out) == len(df)
    assert out.filter(like="P").notna().all().all()
    assert (out["pid"].to_numpy() == df["pid"].to_numpy()).all()
    many = mipy.impute(df, method="MICE-Long", id_col="pid", time_col="visit", m=3, mice_iter=2)
    assert len(many) == 3
    with pytest.raises(ValueError):
        mipy.impute(df, method="softImpute", m=3)
    tab = mipy.list_methods()
    assert {"method", "peer-reviewed", "multiple imputation"} <= set(tab.columns)
    assert len(mipy.literature()) >= 10


def test_evidence_classes_match_paper():
    """Evidence behind each method as in Table 2 of the MIPY paper."""
    ev = mipy.list_methods().set_index("method")["peer-reviewed"]
    assert set(ev) <= {"yes", "adapted", "no"}
    for name in ("MICE-Bayes", "missForest", "MinProb", "QRILC"):
        assert ev[name] == "adapted"
    for name in ("KNN", "softImpute", "HyperImpute", "GAIN", "MinProb-R", "QRILC-R"):
        assert ev[name] == "yes"
    for name in ("Mean", "Median", "LOCF", "sklearn-II", "miceforest", "MICE-Long",
                 "MICE-LongMNAR", "softImpute-MI", "missForest-MI", "miceforest-MI"):
        assert ev[name] == "no"


def test_benchmark_api(cohort):
    df = pd.DataFrame(cohort["full"], columns=[f"P{k}" for k in range(cohort["X"].shape[1])])
    df.insert(0, "visit", cohort["time"])
    df.insert(0, "pid", cohort["patient"])
    res = mipy.benchmark(df, methods=["Mean", "softImpute"], id_col="pid", time_col="visit",
                         mechanisms=("MCAR",), m=2)
    assert set(res.method) == {"Mean", "softImpute"}
    assert res.set_index("method").loc["softImpute", "NRMSE"] < res.set_index("method").loc["Mean", "NRMSE"]
