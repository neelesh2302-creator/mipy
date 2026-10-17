# Benchmarking

`mipy.benchmark()` hides a share of the observed values, imputes them with each
method and scores the hidden values:

* **nRMSE** within each column (RMSE / SD of the true values; 1 = the column's
  own mean), summarised by the median across columns. Pooling cells across
  columns inflates correlations (a constant imputer reaches r > 0.9), so all
  scores are within-column.
* **Within-patient SD ratio**: pooled within-patient SD of the completed values
  over that of the true values (1 = preserved, < 1 flattened, > 1 inflated).
* **Coverage** (multiple imputation): share of hidden values inside the 95%
  interval formed from the completions.

Masking designs (`mipy.masking.Amputer`): MCAR; MAR (driven by visit, clinical
covariates and the sample's mean level); MNAR (logistic in the cell's own
value, slope `strength`); LOD (hard detection limit). Every column keeps at
least `min_obs` observed values.
