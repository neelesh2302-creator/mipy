# Changelog

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
