"""The evidence behind Python imputation packages.

Each entry was checked against the publication or the package page (URL
given) in October 2026. "Peer-reviewed" means a journal or conference
paper describing the package or the exact method it implements; preprints
and software DOIs are listed but do not count.
"""
from __future__ import annotations

import pandas as pd

PACKAGES = [
    dict(package="scikit-learn", methods="SimpleImputer, KNNImputer, IterativeImputer",
         package_paper="Pedregosa et al. 2011, JMLR 12:2825-2830 (general library paper)",
         method_paper="IterativeImputer: none of its own (docs cite mice, van Buuren 2011); "
                      "KNNImputer: Troyanskaya et al. 2001, Bioinformatics 17:520-525",
         peer_reviewed="library yes; IterativeImputer no", longitudinal="not stated",
         high_dimensional="not stated",
         url="https://scikit-learn.org/stable/modules/impute.html"),
    dict(package="miceforest", methods="MICE with LightGBM, mean matching",
         package_paper="none found", method_paper="none found", peer_reviewed="no",
         longitudinal="not stated", high_dimensional="not stated",
         url="https://pypi.org/project/miceforest/"),
    dict(package="hyperimpute", methods="HyperImpute; plugins incl. GAIN, MissForest, MICE, SoftImpute",
         package_paper="Jarrett et al. 2022, ICML, PMLR 162:9916-9937",
         method_paper="GAIN: Yoon et al. 2018, ICML, PMLR 80:5689-5698", peer_reviewed="yes",
         longitudinal="no (cross-sectional UCI data)", high_dimensional="no (at most 91 features)",
         url="https://proceedings.mlr.press/v162/jarrett22a.html"),
    dict(package="fancyimpute", methods="SoftImpute, KNN, IterativeSVD, MatrixFactorization, NuclearNorm",
         package_paper="none (software DOI 10.5281/zenodo.51773)",
         method_paper="Mazumder et al. 2010 (SoftImpute); Troyanskaya et al. 2001 (KNN)",
         peer_reviewed="methods only", longitudinal="not stated", high_dimensional="not stated",
         url="https://pypi.org/project/fancyimpute/"),
    dict(package="missingpy", methods="KNNImputer, MissForest",
         package_paper="none", method_paper="Stekhoven & Buhlmann 2012, Bioinformatics 28:112-118",
         peer_reviewed="methods only", longitudinal="not stated",
         high_dimensional="method paper yes (p > n proteomics)",
         url="https://pypi.org/project/missingpy/"),
    dict(package="missforest", methods="MissForest (LightGBM default)",
         package_paper="none (software DOI 10.5281/zenodo.13368883)",
         method_paper="Stekhoven & Buhlmann 2012", peer_reviewed="methods only",
         longitudinal="not stated", high_dimensional="method paper yes",
         url="https://pypi.org/project/missforest/"),
    dict(package="statsmodels", methods="MICEData / MICE (OLS, PMM, pooling)",
         package_paper="Seabold & Perktold 2010, Proc. SciPy, 92-96 (general library paper)",
         method_paper="MICE module: none of its own", peer_reviewed="library only",
         longitudinal="not stated", high_dimensional="not stated",
         url="https://www.statsmodels.org/stable/imputation.html"),
    dict(package="autoimpute", methods="univariate, Bayesian regression, PMM, MICE, LOCF/NOCB",
         package_paper="none found", method_paper="none found", peer_reviewed="no",
         longitudinal="time-series interpolation, not evaluated", high_dimensional="not stated",
         url="https://pypi.org/project/autoimpute/"),
    dict(package="impyute", methods="mean, KNN, MICE, EM, LOCF, moving window",
         package_paper="none found", method_paper="none found", peer_reviewed="no",
         longitudinal="time-series methods, not evaluated", high_dimensional="not stated",
         url="https://pypi.org/project/impyute/"),
    dict(package="datawig", methods="deep-learning imputation for mixed tables",
         package_paper="Biessmann et al. 2019, JMLR 20(175):1-6; Biessmann et al. 2018, CIKM 2017-2025",
         method_paper="as package", peer_reviewed="yes", longitudinal="not stated",
         high_dimensional="not stated", url="https://www.jmlr.org/papers/v20/18-753.html"),
    dict(package="MIDASpy", methods="multiple imputation with denoising autoencoders",
         package_paper="Lall & Robinson 2023, J Stat Softw 107(9):1-38",
         method_paper="Lall & Robinson 2022, Political Analysis 30:179-196", peer_reviewed="yes",
         longitudinal="no", high_dimensional="yes (349 columns)",
         url="https://www.jstatsoft.org/v107/i09"),
    dict(package="Pympute", methods="per-variable model selection among 10 learners",
         package_paper="Vafaei Sadr et al. 2025, Sci Rep 15:17176",
         method_paper="as package", peer_reviewed="yes", longitudinal="not stated",
         high_dimensional="no (43-45 variables, about 10,800 patients)",
         url="https://www.nature.com/articles/s41598-025-02276-5"),
    dict(package="PyPOTS", methods="time-series imputation (SAITS, BRITS, CSDI, ...)",
         package_paper="preprint only (arXiv:2305.18811)", method_paper="per model",
         peer_reviewed="no (preprint)", longitudinal="time series, not evaluated in the paper",
         high_dimensional="no", url="https://arxiv.org/abs/2305.18811"),
    dict(package="ImputeGAP", methods="40+ time-series imputation algorithms",
         package_paper="preprint only (arXiv:2503.15250)", method_paper="per algorithm",
         peer_reviewed="no (preprint)", longitudinal="time series, not evaluated",
         high_dimensional="not stated", url="https://arxiv.org/abs/2503.15250"),
    dict(package="mipy (this work)", methods="13 benchmarked methods + MICE-Long, MICE-LongMNAR, MI wrappers",
         package_paper="this manuscript", method_paper="see Table 1", peer_reviewed="under review",
         longitudinal="yes (repeated visits)", high_dimensional="yes (507 proteins, 80 patients)",
         url=""),
]


def literature() -> pd.DataFrame:
    return pd.DataFrame(PACKAGES)
