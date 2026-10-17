import numpy as np
import pytest

import mipy
from mipy.imputers import METHODS, available

FAST = [k for k, v in METHODS.items() if not v.slow and available(k)]


@pytest.fixture(scope="module")
def ctx(cohort):
    sch = mipy.build_variable_schema(cohort["X"])
    return mipy.ImputeContext(seed=7, patient=cohort["patient"], time=cohort["time"],
                              schema=sch, mice_iter=2, m=3)


@pytest.mark.parametrize("name", FAST)
def test_completes_and_keeps_observed(name, cohort, ctx):
    X = cohort["X"]
    out, draws = mipy.run(name, X, ctx)
    assert out.shape == X.shape
    assert not np.isnan(out).any()
    obs = ~np.isnan(X)
    assert np.array_equal(out[obs], X[obs])
    if METHODS[name].multiple:
        assert draws is not None and len(draws) == ctx.m
        assert all(np.array_equal(d[obs], X[obs]) for d in draws)


@pytest.mark.parametrize("name", FAST)
def test_reproducible(name, cohort, ctx):
    a, _ = mipy.run(name, cohort["X"], ctx)
    b, _ = mipy.run(name, cohort["X"], ctx)
    assert np.array_equal(a, b)


def test_mi_draws_differ(cohort, ctx):
    for name in ("MICE-Bayes", "MICE-Long", "softImpute-MI"):
        _, draws = mipy.run(name, cohort["X"], ctx)
        miss = np.isnan(cohort["X"])
        assert np.std([d[miss].mean() for d in draws]) > 0


def test_longitudinal_beats_cross_sectional(cohort, ctx):
    """With strong within-patient correlation, using the patient's other
    visits should help."""
    X, full = cohort["X"], cohort["full"]
    miss = np.isnan(X)
    err = {}
    for name in ("MICE-Bayes", "MICE-Long"):
        out, _ = mipy.run(name, X, ctx)
        err[name] = np.sqrt(np.mean((out[miss] - full[miss]) ** 2))
    assert err["MICE-Long"] < err["MICE-Bayes"]


def test_mnar_gamma_shifts_down(cohort, ctx):
    X = cohort["X"]
    miss = np.isnan(X)
    a, _ = mipy.run("MICE-LongMNAR", X, ctx.with_(gamma=0.0))
    b, _ = mipy.run("MICE-LongMNAR", X, ctx.with_(gamma=2.0))
    assert b[miss].mean() < a[miss].mean()


def test_locf_needs_patients(cohort):
    with pytest.raises(ValueError):
        mipy.run("LOCF", cohort["X"], mipy.ImputeContext())


def test_minprob_qrilc_r_parameters():
    """The R-faithful parameter estimates on a hand-checkable case."""
    from mipy.imputers import minprob_params_r, qrilc_params_r
    D = np.arange(1, 21, dtype=float).reshape(10, 2)
    centre, sd = minprob_params_r(D)
    assert np.allclose(centre, np.quantile(D, 0.01, axis=0))
    assert np.isclose(sd, np.median(np.std(D, axis=1, ddof=1)))
    par = qrilc_params_r(D)
    assert (par["sd"] > 0).all() and (par["p_missing"] == 0).all()
