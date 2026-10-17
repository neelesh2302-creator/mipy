# Methods

Every imputer has the same signature, `fn(X, ctx) -> (completed, draws)`, and
is listed in `mipy.METHODS` with its family, the package and function it calls,
whether a peer-reviewed publication describes it, and whether it returns
multiple completions. `mipy.list_methods()` shows the table.

## New in MIPY

**MICE-Long.** Chained equations (Bayesian ridge) in which protein *k* is
predicted from its quickpred predictor set plus three within-patient terms,
rebuilt from the current completion at every step: the same protein at the
patient's previous visit, at the next visit, and the patient's leave-one-out
mean; the visit index is added too. Each of the *m* completions is fitted on a
patient-level bootstrap and every value is a predictive draw, so the
completions are proper multiple imputations. Related to the multilevel
chained-equations approaches reviewed by Resche-Rigon & White (2018).

**MICE-LongMNAR.** MICE-Long with selection-model draws: candidate values from
the MAR predictive distribution are resampled with weights
`expit(a_k - gamma * z)`, where `z` is the standardised candidate value and
`a_k` matches protein *k*'s missing fraction. Low values are favoured, as under
a detection limit. `gamma = 0` is MAR; `gamma` cannot be estimated from the
data and should be varied (sensitivity analysis), in the spirit of NARFCS
(Tompsett et al. 2018).

**Multiple-imputation versions.** `softImpute-MI` refits on a different 90% of
the observed cells for each completion and adds residual draws with each
protein's held-out error; `missForest-MI` bootstraps rows and adds
out-of-bootstrap residual draws; `miceforest-MI` wraps miceforest in the same
held-out-residual scheme (`mipy.imputers.holdout_mi` does this for any
imputer).

## Ported from R

| MIPY | R original | Fidelity |
|---|---|---|
| softImpute | `softImpute::softImpute(type = "svd")` | same algorithm and penalty; cell-level r > 0.99 |
| MinProb-R, QRILC-R | `imputeLCMD::impute.MinProb`, `impute.QRILC` | identical parameters (to 1e-13); same draw distribution |
| MinProb, QRILC | the same, applied per protein rather than per sample | protein-specific detection limits |
| MICE-Bayes | `mice(method = "norm.boot")` | same scheme; accuracy and FMI compared |
| missForest | `missForest::missForest` | extremely randomised trees instead of random forests; accuracy compared |
