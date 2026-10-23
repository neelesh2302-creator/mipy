"""Copy the files of a pipeline run into paper/output for the repository.

Run from the mipy_pkg folder after the notebook has finished:

    python paper\\copy_results.py

By default the run is read from ..\\mipy_run (Missing_Python\\mipy_run when
mipy_pkg sits in Missing_Python); give another folder as the first argument.
Only summary files are copied. Checkpoints, completed data and files with one
row per patient (the survival endpoints) are left out, and so are the
figures and tables the paper no longer uses.
"""
import fnmatch
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "mipy_run"
OUT = HERE / "output"

COPY = {"manuscript": ("*.tex",),
        "results": ("*.csv",),
        "figures": ("*.pdf", "*.png"),
        "provenance": ("*.json",)}
LEAVE_OUT = ("s1_endpoint_pairs.csv", "s1_nonpositive_times.csv", "s1_endpoint_order.csv",
             "s1_event_crosstab.csv", "table13_survival.csv", "s13_cox_*.csv",
             "s13_top20_jaccard.csv",                         # survival screen, not in the paper
             "fig03_*", "fig07_*", "fig08_*", "fig09_*", "fig14_*", "fig15_*")

if not (RUN / "manuscript" / "numbers.tex").is_file():
    sys.exit(f"No finished run in {RUN} (manuscript\\numbers.tex is missing). "
             "Give the run folder: python paper\\copy_results.py C:\\path\\to\\mipy_run")

if OUT.exists():
    shutil.rmtree(OUT)                                   # no stale files from older runs
n, size, skipped = 0, 0, []
for sub, patterns in COPY.items():
    src = RUN / sub
    if not src.is_dir():
        continue
    (OUT / sub).mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if not f.is_file() or not any(fnmatch.fnmatch(f.name, p) for p in patterns):
            continue
        if any(fnmatch.fnmatch(f.name, p) for p in LEAVE_OUT):
            skipped.append(f.name)
            continue
        shutil.copy2(f, OUT / sub / f.name)
        n += 1
        size += f.stat().st_size
print(f"copied {n} files ({size / 1e6:.1f} MB) from {RUN} to {OUT}")
if skipped:
    print(f"left out {len(skipped)}: {', '.join(skipped)}")
