# Files of the MIPY paper

`output/` holds what the pipeline (`notebooks/MIPY_v2_pipeline.ipynb`) wrote
in the run used for the revised paper (21 October 2026, profile "complete",
mipy 2.1.1):

| Folder | Contents |
|---|---|
| `output/manuscript` | `numbers.tex` (every number quoted in the text), `tables.tex`, `decision_guide.tex` and the generated supplementary tables |
| `output/results` | every table as CSV, including per-protein errors and the effective sample sizes of MICE-LongMNAR (`s4_ess_gamma.csv`) |
| `output/figures` | every figure as PDF and PNG |
| `output/provenance` | `checks.json` (the integrity checks), `environment.json` (versions, hardware, GAIN settings actually used), `numbers.json` |

The GSE65622 analysis files are not here; the paper's data statement says
how to obtain them. Files with one row per patient are not copied.

To refresh `output/` after a new run, from the `mipy_pkg` folder:

```
python paper\copy_results.py
```
