import numpy as np
import pytest

import mipy
from mipy import rbridge
from mipy.masking import Amputer

pytestmark = pytest.mark.skipif(rbridge.find_rscript() is None, reason="R not installed")


def test_softimpute_and_lcmd_agree_with_r(tmp_path, cohort):
    X = cohort["X"]
    amp = {"MCAR": Amputer(X, rate=0.2).ampute("MCAR", 9)}
    sch = {"MCAR": mipy.build_variable_schema(amp["MCAR"]["X"])}
    rbridge.write_inputs(tmp_path, amp, sch, run=("softImpute", "lcmd"))
    st = rbridge.run_r(tmp_path, log=lambda *a: None)
    if not (st.status == "ok").all():
        pytest.skip("R packages not available: " + "; ".join(st[st.status != "ok"].step))
    tab, _ = rbridge.compare(tmp_path, amp, X, {})
    si = tab[tab.method == "softImpute"].iloc[0]
    assert si["cell r"] > 0.995      # both stop at a convergence tolerance
    lc = tab[tab.method.str.contains("imputeLCMD")]
    assert (lc["max |diff| (parameters)"] < 1e-8).all()


def test_mice_runs_even_when_norm_boot_stops(tmp_path, cohort):
    """Few observed values and many predictors make mice's norm.boot stop on
    rank-deficient bootstrap samples; the R script must still deliver the
    m completions (rerun with ls.meth = "ridge", or mice's norm) and compare() must use them."""
    X = cohort["X"].copy()
    X[:90, :4] = np.nan                     # four proteins with 30 observed values
    amp = {"MCAR": Amputer(X, rate=0.1).ampute("MCAR", 3)}
    sch = {"MCAR": mipy.build_variable_schema(amp["MCAR"]["X"], max_predictors=25, min_cor=0.0)}
    rbridge.write_inputs(tmp_path, amp, sch, m=2, mice_maxit=2, run=("mice",))
    st = rbridge.run_r(tmp_path, log=lambda *a: None)
    if not (st[st.step.str.startswith("package:mice")].status == "ok").all():
        pytest.skip("R package mice not available")
    ok = set(st[st.status == "ok"].step)
    assert ok & {"mice", "mice-ridge", "mice-norm"}
    ctx = mipy.ImputeContext(seed=1, schema=sch["MCAR"], mice_iter=2, m=2)
    rec = mipy.runner.run_one("MICE-Bayes", amp["MCAR"]["X"], ctx)
    tab, _ = rbridge.compare(tmp_path, amp, X, {("MCAR", "MICE-Bayes"): rec}, m=2)
    row = tab[tab.method == "MICE-Bayes vs mice norm.boot"]
    assert len(row) == 1 and np.isfinite(row["nRMSE R"].iloc[0])
