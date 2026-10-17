import numpy as np

import mipy
from mipy.masking import Amputer


def test_rates_and_direction(cohort):
    X = cohort["X"]
    am = Amputer(X, time=cohort["time"], rate=0.2)
    for mech in ("MCAR", "MAR", "MNAR", "LOD"):
        amp = am.ampute(mech, 11)
        rate = amp["mask"].sum() / np.isfinite(X).sum()
        assert 0.17 < rate < 0.21, (mech, rate)
        assert not (amp["mask"] & np.isnan(X)).any()
    z_mcar = am.Z[am.ampute("MCAR", 1)["mask"]].mean()
    z_mnar = am.Z[am.ampute("MNAR", 1)["mask"]].mean()
    z_lod = am.Z[am.ampute("LOD", 1)["mask"]].mean()
    assert abs(z_mcar) < 0.15 and z_mnar < -0.3 and z_lod < z_mnar


def test_min_obs_floor(cohort):
    am = Amputer(cohort["X"], rate=0.9, min_obs=12)
    amp = am.ampute("MCAR", 3)
    left = np.isfinite(amp["X"]).sum(0)
    assert (left >= np.minimum(12, np.isfinite(cohort["X"]).sum(0))).all()


def test_mean_imputation_scores_one(cohort):
    X = cohort["X"]
    amp = Amputer(X, rate=0.2).ampute("MCAR", 5)
    out, _ = mipy.run("Mean", amp["X"])
    ev = mipy.evaluate(X, out, amp["mask"], patient=cohort["patient"])
    assert 0.9 < ev["NRMSE_median"] < 1.2
    assert np.isfinite(ev["structure_ratio"])


def test_rubin_and_coverage():
    q = [1.0, 1.2, 0.8, 1.1]
    u = [0.01] * 4
    qbar, ubar, b, t, fmi, df = mipy.rubin(q, u)
    assert np.isclose(qbar, 1.025) and np.isclose(t, ubar + 1.25 * b) and 0 < fmi < 1
    rs = np.random.default_rng(0)
    mu = rs.normal(size=(400, 5))
    truth = mu + rs.normal(size=mu.shape)            # truth and draws from the same
    mask = np.ones_like(truth, dtype=bool)           # predictive distribution
    draws = [mu + rs.normal(size=mu.shape) for _ in range(5)]
    cov = mipy.predictive_coverage(truth, draws, mask)["coverage"]
    assert 0.9 < cov < 0.99
