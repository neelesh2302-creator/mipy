# MIPY: Missing-data Imputation in Python

[![PyPI](https://img.shields.io/pypi/v/mipy-impute.svg)](https://pypi.org/project/mipy-impute/)
[![Python](https://img.shields.io/pypi/pyversions/mipy-impute.svg)](https://pypi.org/project/mipy-impute/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/neelesh2302-creator/mipy/blob/main/LICENSE)

MIPY finds out which imputation method works best on **your** data, fills in
the missing values with it, and gives you the tables, figures and a report to
show for it. It is built for **high-dimensional longitudinal data**: many
variables (for example hundreds of proteins) measured repeatedly on the same
patients, as in the serum proteomics series GSE65622.

```python
import mipy
res = mipy.analyze("my_data.csv")      # one line: benchmark, impute, figures, report
```

That call:

1. reads the table and works out which column is the patient ID, which is the
   visit, which are features, clinical covariates and survival times (and
   tells you, so you can correct it);
2. hides 20% of the values you **do** have, under three assumptions about why
   values go missing (completely at random, depending on visit and clinical
   data, or because they are low), repeats this three times, and asks every
   method to fill them in;
3. scores each method against the hidden true values (nRMSE within each
   feature; 1.00 = filling in the feature's mean), with Monte Carlo standard
   errors, the share of features where the method beats the mean, whether
   each patient's visit-to-visit variation is kept, and, for multiple
   imputation, whether the 95% intervals are honest;
4. fills in your real missing values with the best method (and gives *m*
   completed tables from the best multiple-imputation method);
5. runs an exploratory survival screen when the table has a survival time and
   an event column;
6. writes everything into one folder, with an HTML report that explains each
   result in plain language.

![Accuracy of the methods on the example cohort](https://raw.githubusercontent.com/neelesh2302-creator/mipy/main/docs/images/accuracy.png)

## Install

```bash
pip install mipy-impute
```

Python 3.9 or newer, on Windows, macOS or Linux. Optional extras add methods:

```bash
pip install "mipy-impute[ml]"        # miceforest (LightGBM)
pip install "mipy-impute[automl]"    # HyperImpute and GAIN (needs PyTorch)
pip install "mipy-impute[bigdata]"   # polars and DuckDB for very large tables
```

Methods whose package is missing are skipped with a message that says what to
install; nothing fails.

## Try it in one minute

```python
import mipy

df = mipy.example_cohort()             # synthetic cohort laid out like GSE65622
res = mipy.analyze(df, out="example_results", methods="quick", repeats=1)
res.show()                             # tables and figures; inline in Jupyter
```

Or from a terminal (Command Prompt, PowerShell, or a shell):

```bash
mipy example example.csv
mipy analyze example.csv --methods quick --repeats 1 --open
```

## Your own data

One row per patient visit, like this (any column names work):

| Patient_ID | sampling_point | EN-RAGE | GDF-15 | ... | RFS | event | trg_score |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 59.08 | 1203.4 | ... | 2727 | 0 | 4 |
| 1 | 2 | 52.19 |  | ... | 2697 | 0 | 4 |
| 2 | 1 |  | 980.1 | ... | 1579 | 1 | 2 |

* an **ID** column and a **visit** column (optional, but needed by the
  longitudinal methods),
* many **numeric feature** columns with empty cells where values are missing
  (`NA`, `n/a`, `<LOD`, `ND` and similar entries are read as missing),
* optional **clinical** columns (few distinct values: stage, grade, batch),
* optional **survival** columns (a time and a 0/1 event).

CSV, Excel, TSV and Parquet files work, and so do pandas DataFrames. Check how
MIPY reads a file before running anything:

```bash
mipy check my_data.csv
```

If a column is detected wrongly, say which is which:

```python
res = mipy.analyze("my_data.xlsx", id_col="Patient_ID", time_col="sampling_point",
                   features="EN-RAGE:Prot_496",            # a range of columns, or a list
                   clinical=["trg_score", "ypn_stage"], survival=("RFS", "event"),
                   out="my_results")
```

More in [docs/your_data.md](https://github.com/neelesh2302-creator/mipy/blob/main/docs/your_data.md).

## What you get

```
my_results/
  report.html            all results, figures embedded, with explanations
  figures/               fig01_missingness ... fig09_survival  (PNG and PDF)
  tables/                accuracy_summary.csv, decision.csv, ... and mipy_results.xlsx
  completed_data/        completed_<best method>.csv and completed_<MI method>_draw1..5.csv
  run_log.txt            every step, warnings and error details
```

| Figure | Shows |
|---|---|
| missingness | where values are missing, and whether low-level features miss more often |
| masking | which values each masking design hid |
| accuracy | nRMSE of every method under every mechanism, with Monte Carlo intervals |
| nrmse_per_feature | the spread of the error across features |
| sd_ratio | whether each patient's visit-to-visit variation is kept |
| coverage | calibration of multiple imputation (95% intervals) |
| runtime | seconds per run |
| imputed_vs_observed | your real missing values as filled in, against the observed values |
| survival | exploratory Cox screen (when survival columns exist) |

In Python, `res` holds the same results: `res.summary`, `res.decision`,
`res.completed`, `res.completed_draws`, `res.survival`, `res.tables`,
`res.figures`, `res.warnings`.

## Choosing methods

| `methods=` | Methods | Time on 288 visits x 507 proteins, 9 maskings* |
|---|---|---|
| `"quick"` | Mean, LOCF, KNN, softImpute, MinProb, QRILC, MICE-Long | about 5 min |
| `"standard"` (default) | quick + Median, MICE-Bayes, sklearn-II, MICE-LongMNAR | 10 to 20 min |
| `"complete"` | standard + missForest, miceforest, HyperImpute, GAIN | 2 h or more |
| `"all"` | complete + the multiple-imputation versions and the exact imputeLCMD variants | longer |

\*one process on a 2-core machine; the example cohort takes about 3 minutes
with "standard". `n_workers="auto"` runs several methods at once, and runs
already finished are reused if you stop and start again.

| Method | Family | Multiple imputation | Uses patients | Peer-reviewed description |
|---|---|---|---|---|
| Mean, Median | naive | no | no | no |
| LOCF | longitudinal rule | no | yes | no |
| KNN | neighbour | no | no | yes (Troyanskaya et al. 2001) |
| softImpute | low-rank | no | no | yes (Mazumder et al. 2010) |
| MICE-Bayes | chained equations | yes | no | adapted (van Buuren & Groothuis-Oudshoorn 2011) |
| missForest | chained equations | no | no | adapted (Stekhoven & Buhlmann 2012) |
| sklearn-II | chained equations | no | no | no (scikit-learn IterativeImputer) |
| miceforest | chained equations | no | no | no |
| HyperImpute | AutoML | no | no | yes (Jarrett et al. 2022) |
| GAIN | generative | no | no | yes (Yoon et al. 2018) |
| MinProb, QRILC | left-censored | no | no | adapted (Lazar et al. 2016) |
| **MICE-Long** | longitudinal chained | yes | yes | new in MIPY |
| **MICE-LongMNAR** | selection-model MNAR | yes | yes | new in MIPY |
| softImpute-MI, missForest-MI, miceforest-MI | MI versions | yes | no | new in MIPY |

*Adapted*: a published method changed in engine, stopping rule or unit of
application. MICE-Bayes uses Bayesian ridge in place of mice's least squares,
missForest uses extremely randomised trees and five fixed sweeps in place of
the stopping rule, and MinProb and QRILC work per protein rather than per
sample. The exact imputeLCMD versions are `MinProb-R` and `QRILC-R`.

* **MICE-Long**: chained equations that use each patient's previous visit,
  next visit and own mean, with a patient-level bootstrap (proper multiple
  imputation).
* **MICE-LongMNAR**: the same with selection-model draws for values lost
  below a detection limit; `settings={"gamma": ...}` sets the sensitivity
  parameter.
* softImpute, missForest, mice's `norm.boot` and imputeLCMD's MinProb and
  QRILC are implemented in Python and checked against the R packages
  (identical parameters; cell-level agreement r > 0.99 for softImpute).

## Lower-level functions

```python
done  = mipy.impute(df, method="softImpute", id_col="Patient_ID", time_col="sampling_point", log2=True)
draws = mipy.impute(df, method="MICE-Long", id_col="Patient_ID", time_col="sampling_point", m=5, log2=True)
bench = mipy.benchmark(df, methods=["Mean", "softImpute", "MICE-Long"], id_col="Patient_ID",
                       time_col="sampling_point", log2=True, mechanisms=("MCAR", "MNAR"), repeats=3)

from mipy.mi import pool_estimates      # Rubin's rules over the m completions
pooled = pool_estimates(estimates, by="term")   # columns: draw, term, beta, se
```

`mipy.list_methods()` lists every method with the function it calls and its
evidence; `mipy.literature()` lists 14 Python imputation packages and the
publication behind each.

## Reproducing the MIPY paper

`notebooks/MIPY_v2_pipeline.ipynb` runs the full GSE65622 analysis of the
paper (all methods, multiple imputation, MNAR sensitivity with logistic and
probit designs, agreement with R, data handling up to a million patients, a
synthetic second cohort) and writes every number, table and figure. The files
of the paper's run (21 October 2026) are in [`paper/output`](https://github.com/neelesh2302-creator/mipy/tree/main/paper).
The GSE65622 analysis files themselves are not in the repository.

## Tests

```bash
pip install "mipy-impute[all,test]"
pytest
```

## Citation

See [CITATION.cff](https://github.com/neelesh2302-creator/mipy/blob/main/CITATION.cff).
If you use MIPY, please cite the MIPY paper and the publication of each method
you report (`mipy.list_methods()` gives the reference key).

## Licence

MIT
