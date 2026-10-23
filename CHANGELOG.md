# Changelog

## 2.1.1 (2026-10-24)

For the revised MIPY paper.

* `Amputer.ampute("PROBIT", ...)`: a second not-at-random masking design with a
  probit selection, Pr(masked) = Phi(alpha - beta z). `Amputer.matched_probit_slope()`
  gives the slope whose masked cells have the same expected mean z as the
  logistic MNAR design, and `Amputer.expected_masked_z()` the expected mean z of
  either design. MCAR, MAR, MNAR and LOD masks are unchanged.
* MICE-LongMNAR can record the effective sample size of its importance
  resampling (`ctx.extra["ess"] = True`); the summary is returned in the run
  record as `diag`. Recording it draws no random numbers, so the imputations
  are identical with or without it.
* `mipy.mi.fmi_summary` documents the within-imputation variance it uses.
* `list_methods()`, the methods table of `analyze()` and the README give the
  evidence behind each method as "yes", "adapted" or "no", as in the paper:
  MICE-Bayes, missForest, MinProb and QRILC are now "adapted" (a published
  method changed in engine, stopping rule or unit of application), and the
  multiple-imputation versions new in MIPY are "no". `MethodInfo.evidence`
  gives the class. Results and checkpoints are unchanged.
* `notebooks/MIPY_v2_pipeline.ipynb` is the pipeline of the revised paper, and
  `paper/output` holds the files of its run of 21 October 2026.
* Known issue: hyperimpute's GAIN plugin (0.1.1 to 0.1.17) does not pass
  `n_epochs`, batch size or hint rate to the model it trains, so GAIN always
  trains with the model's defaults (1,000 iterations, batch 256, hint rate 0.9)
  and `gain_epochs` has no effect. The analysis pipeline records the settings
  actually used.

## 2.1.0 (2026-10-16)

For users with their own data.

* `mipy.analyze()`: one call from a table to a benchmark of the methods, the
  completed data, figures (PNG and PDF), tables (CSV and Excel) and a
  self-contained HTML report. Every stage after the data check is protected,
  so a failing method or figure is reported and the rest still runs.
* `mipy.prepare_data()` and `mipy.read_table()`: read CSV (comma, semicolon or
  tab), Excel, TSV and Parquet; find the ID, visit, feature, clinical and
  survival columns by name and content; read `NA`, `<LOD`, `n/a`, decimal
  commas; turn visit labels such as `V2` or `week 4` into numbers; accept long
  tables; explain every decision; clear `DataError` messages.
* Patients are ordered by their ID, so the row order of the file does not
  change the results.
* `mipy.example_cohort()`: a synthetic cohort laid out like the GSE65622
  analysis files, for trying MIPY and as a template.
* Command line: `mipy analyze | check | impute | example | methods`.
* Figures module `mipy.plots` (no pyplot; works in scripts, terminals and
  Jupyter) and report module `mipy.report`.
* `Result.show()` displays tables and figures inline in Jupyter.
* Exploratory survival screen when the table has a survival time and event.
* matplotlib and openpyxl are now core dependencies.
* GAIN now trains for 1,000 iterations by default (hyperimpute's default; was
  100). The paper pipeline sets its own value, so its results do not change.
* `mipy.impute()` also accepts a file path.

## 2.0.0 (2026-10-15)

* MICE-Long and MICE-LongMNAR; multiple-imputation versions of softImpute,
  missForest and miceforest; Rubin's rules and interval coverage.
* LOD masking; repeated maskings with Monte Carlo standard errors.
* Agreement with the R packages (softImpute, missForest, mice, imputeLCMD),
  with mice's ridge fallback when `norm.boot` stops on rank-deficient samples.
* Data handling with pandas, polars and DuckDB; chunked imputation.
* Survival endpoint checks and pooled univariate Cox models.
