import numpy as np
import pytest


@pytest.fixture(scope="session")
def cohort():
    """A small longitudinal cohort: 30 patients x 4 visits x 40 proteins,
    10% missing, with the complete truth kept."""
    rs = np.random.default_rng(1)
    n_pat, V, p = 30, 4, 40
    pat = np.repeat(np.arange(n_pat), V)
    tim = np.tile([1, 2, 4, 5], n_pat)
    L = rs.normal(size=(n_pat * V, 3)) @ rs.normal(size=(3, p))
    L += np.repeat(rs.normal(size=(n_pat, p)), V, 0) * 0.8
    full = 8 + L + 0.3 * rs.normal(size=L.shape)
    X = full.copy()
    X[rs.random(X.shape) < 0.1] = np.nan
    return {"X": X, "full": full, "patient": pat, "time": tim}
