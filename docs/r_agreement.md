# Agreement with R

```python
from mipy import rbridge
rbridge.write_inputs("r_check", amputed, schemas)
status = rbridge.run_r("r_check")              # finds Rscript; installs R packages
table, _ = rbridge.compare("r_check", amputed, X_true, python_results)
```

The bundled script `mipy/r/mipy_r_agreement.R` runs softImpute, missForest,
mice (norm.boot) and imputeLCMD on exactly the masked cells MIPY wrote.

| R package | What is compared | Why |
|---|---|---|
| softImpute | every imputed cell | same standardisation and penalty, so the fits should coincide |
| imputeLCMD (MinProb, QRILC) | fitted parameters | deterministic, so they must match to rounding error |
| missForest | accuracy, per-protein agreement | R uses randomForest, MIPY uses scikit-learn trees |
| mice (norm.boot) | accuracy, FMI | MIPY's MICE-Bayes draws from a Bayesian ridge, mice from least squares |

## When mice's norm.boot stops

`norm.boot` fits ordinary least squares (QR) on a bootstrap sample of the
rows where the variable is observed. If a protein has few observed values
and many predictors, a bootstrap sample can contain fewer distinct rows than
predictors. `lm.fit` then returns `NA` for the aliased coefficients, the
imputations become `NA`, the next variables lose their complete rows, and
mice stops with `missing value where TRUE/FALSE needed`.

On GSE65622 (as few as 42 observed values per protein, 25 predictors) this
happens under every masking mechanism. The R script then reruns the same
`norm.boot` with mice's own ridge option, `ls.meth = "ridge"` (penalty
1e-5 times the diagonal of X'X; see `?mice::estimice`). The status table
records the step as `mice-ridge`, and `compare()` marks those rows. MIPY's
MICE-Bayes does not have the problem because the Bayesian ridge fit is
defined at any rank.

Agreement of per-protein accuracy is reported as a Spearman correlation,
so that a single protein with a very large error in one implementation does
not decide the figure.
