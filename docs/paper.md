# Reproducing the paper

Open `notebooks/MIPY_v2_pipeline.ipynb` from the repository (CELL 1 finds the
package in `../src`; otherwise install mipy-impute 2.1.1 or later). Put the
two GSE65622 analysis files in `Missing_Python/data`, set `PROFILE` in the
second cell, and run. Results go to `mipy_run/`: figures, CSV results,
checkpoints, and `manuscript/numbers.tex`, `tables.tex`, `supplementary.tex`
and `decision_guide.tex` for the LaTeX manuscript.

The files of the run used for the revised paper (21 October 2026) are in
`paper/output`. After a new run, `python paper/copy_results.py` (from the
repository folder) replaces them; files with one row per patient are not
copied.
