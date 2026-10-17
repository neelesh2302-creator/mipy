# Analyse your own data

## The table

One row per patient visit (wide layout), as in the GSE65622 files:

| column type | example | needed? |
|---|---|---|
| patient ID | `Patient_ID`, `subject`, `participant_id` | for the longitudinal methods (LOCF, MICE-Long, MICE-LongMNAR) and the within-patient SD ratio |
| visit | `sampling_point`, `visit`, `timepoint`, `week` | recommended; numbers or labels such as `V1`, `week 4`, `baseline` |
| features | proteins, metabolites, genes: numeric | yes, at least 2 |
| clinical | `trg_score`, `ypn_stage`, `batch` | optional; used by the MAR masking |
| survival | `RFS` + `event`, `OS_time` + `OS_status` | optional; turns on the exploratory survival screen |

Other columns (free text, dates) are kept in the output and otherwise ignored.

Values can be raw intensities or already transformed. With `log2="auto"`
(the default) MIPY works on log2(x + 1) when every value is above -1 and the
largest is above 50, as for raw intensities, and writes the completed values
back on the original scale.

Missing values can be empty cells or entries such as `NA`, `n/a`, `NaN`,
`-`, `ND`, `<LOD`, `BDL`, `not detected`. Numbers written with a decimal comma
(`12,5`) are read too. MIPY says how many entries it turned into missing
values.

## Files

```python
res = mipy.analyze("data.csv")                    # comma, semicolon or tab separated
res = mipy.analyze("data.xlsx", sheet="proteins") # Excel: first sheet by default
res = mipy.analyze(df)                            # a pandas DataFrame
```

Long tables (one row per patient x visit x feature):

```python
res = mipy.analyze("long.csv", layout="long", id_col="patient", time_col="visit",
                   feature_col="protein", value_col="intensity")
```

## What MIPY detects, and how to change it

Run a check first; nothing is imputed:

```python
p = mipy.prepare_data("data.csv")
print(p)            # the summary table
print(p.notes)      # every decision, e.g. "ID column: 'Patient_ID' (found by name ...)"
print(p.warnings)   # anything you should look at
```

or `mipy check data.csv` in a terminal. Then override what is wrong:

| argument | meaning |
|---|---|
| `id_col="PID"` | the patient ID column (`False`: there is none) |
| `time_col="Visit"` | the visit column (`False`: none; rows of a patient are taken in file order) |
| `features=["A", "B"]` or `features="A:Z"` | the columns to impute (a list, or a range from one column to another) |
| `clinical=["stage", "batch"]` | clinical columns (`False`: none) |
| `survival=("OS_months", "death")` | survival time and 0/1 event (`False`: skip the screen) |
| `max_missing=0.9` | features missing in more than this share are left as they are |
| `log2=True / False` | force the log2(x + 1) scale on or off |

## Choosing what to run

| argument | default | meaning |
|---|---|---|
| `methods` | `"standard"` | `"quick"`, `"standard"`, `"complete"`, `"all"` or a list of names |
| `mechanisms` | MCAR, MAR, MNAR | add `"LOD"` for a hard detection limit |
| `repeats` | 3 | maskings per mechanism; more repeats, smaller Monte Carlo error |
| `rate` | 0.2 | share of the observed values hidden |
| `m` | 5 | completions for multiple imputation |
| `export` | `"auto"` | method for the completed table; `"auto"` = best under the most plausible mechanism |
| `n_workers` | 1 | `"auto"` runs methods in parallel |
| `settings` | none | method settings, e.g. `{"gamma": 1.5, "mice_iter": 10, "knn_k": 10}` |

### How the export method is chosen

MIPY compares each feature's median value with its share of missing values
(Spearman rho, figure "missingness", panel D). If low-level features clearly
miss more often (rho <= -0.3), values were probably lost below a detection
limit, and the best method under MNAR is used. Otherwise MAR is the working
assumption and the best method under MAR is used. The report states the rule
and the rho. Pass `export="MICE-LongMNAR"` (or any method) to choose yourself.

## Large tables

* Start with `methods="quick", repeats=1` to see the run time.
* `n_workers="auto"` uses several CPU cores.
* Stopping is safe: finished runs are saved in `out/.mipy_cache` and reused
  when you run the same call again (`resume=False` starts over).

## When something goes wrong

| message | what to do |
|---|---|
| `file not found` | check the path; on Windows write `r"C:\Users\me\data.csv"` or use forward slashes |
| `column 'X' not found` | the message lists the columns; check spelling and spaces |
| `no numeric feature columns` | the features are stored as text; check the separator, or pass `features=` |
| `only N rows have feature values` | at least 10 rows are needed |
| `<method> skipped: needs the package ...` | install the package named, or ignore it |
| `<method> failed in k of n runs` | the other methods are still compared; the report lists the error |

Every stage after the data check is protected: if one figure, method or the
survival screen fails, the rest still runs and the problem is listed under
"Please read" in the report and in `res.warnings`.
