"""Command line: ``mipy analyze my_data.csv`` (also ``python -m mipy ...``).

    mipy analyze DATA [--id COL] [--time COL] [--out DIR] [--methods standard] ...
    mipy check   DATA [--id COL] [--time COL] ...    what MIPY detects, nothing imputed
    mipy impute  DATA --method softImpute [--m 5] [--output FILE]
    mipy example [FILE]                               write the example cohort as CSV
    mipy methods                                      list the methods
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cols(v):
    if v is None:
        return None
    if v.lower() in ("none", "false", "no"):
        return False
    return v


def _common(p):
    p.add_argument("data", help="CSV, Excel, TSV or Parquet file; one row per patient visit")
    p.add_argument("--id", dest="id_col", help="patient ID column (default: found by name; 'none' = no ID)")
    p.add_argument("--time", dest="time_col", help="visit column (default: found by name; 'none' = no visit)")
    p.add_argument("--features", help="columns to impute: 'A,B,C' or a range 'first:last' (default: numeric columns)")
    p.add_argument("--clinical", help="clinical columns 'A,B' or 'none' (default: found automatically)")
    p.add_argument("--survival", nargs=2, metavar=("TIME", "EVENT"), help="survival time and event columns")
    p.add_argument("--no-survival", action="store_true", help="skip the survival screen")
    p.add_argument("--log2", default="auto", choices=["auto", "yes", "no"], help="analyse on log2(x + 1) (default auto)")
    p.add_argument("--max-missing", type=float, default=0.9, help="skip features missing in more than this share (0.9)")
    p.add_argument("--sheet", default=0, help="Excel sheet name or number (default: the first)")
    p.add_argument("--layout", default="wide", choices=["wide", "long"], help="wide (default) or long table")
    p.add_argument("--feature-col", help="long layout: column with the feature names")
    p.add_argument("--value-col", help="long layout: column with the values")


def _prep_kwargs(a):
    sheet = int(a.sheet) if str(a.sheet).isdigit() else a.sheet
    surv = False if a.no_survival else (tuple(a.survival) if a.survival else None)
    return dict(id_col=_cols(a.id_col), time_col=_cols(a.time_col), features=a.features,
                clinical=_cols(a.clinical), survival=surv,
                log2={"auto": "auto", "yes": True, "no": False}[a.log2], max_missing=a.max_missing,
                sheet=sheet, layout=a.layout, feature_col=a.feature_col, value_col=a.value_col)


def main(argv=None):
    from . import __version__
    for stream in (sys.stdout, sys.stderr):          # names such as TNF-\u03b1 on any console
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="mipy", description="MIPY: benchmark and impute missing values in "
                                 "longitudinal high-dimensional data (e.g. proteomics).")
    ap.add_argument("--version", action="version", version=f"mipy {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    pa = sub.add_parser("analyze", aliases=["analyse", "run"], help="benchmark the methods, impute, write figures and a report")
    _common(pa)
    pa.add_argument("--out", default="mipy_results", help="output folder (default mipy_results)")
    pa.add_argument("--methods", default="standard",
                    help="quick | standard | complete | all | comma list of methods (default standard)")
    pa.add_argument("--mechanisms", default="MCAR,MAR,MNAR", help="masking designs (default MCAR,MAR,MNAR; also LOD)")
    pa.add_argument("--repeats", type=int, default=3, help="maskings per mechanism (default 3)")
    pa.add_argument("--rate", type=float, default=0.2, help="share of observed values hidden (default 0.2)")
    pa.add_argument("--strength", type=float, default=2.0, help="MAR/MNAR masking slope (default 2)")
    pa.add_argument("--m", type=int, default=5, help="completions for multiple imputation (default 5)")
    pa.add_argument("--export", default="auto", help="'auto', a method name, or 'none'")
    pa.add_argument("--workers", default="1", help="parallel processes: a number or 'auto' (default 1)")
    pa.add_argument("--seed", type=int, default=2026)
    pa.add_argument("--fresh", action="store_true", help="ignore runs saved by an earlier call")
    pa.add_argument("--open", action="store_true", help="open the report in the browser at the end")
    pa.add_argument("--quiet", action="store_true", help="less output")

    pc = sub.add_parser("check", help="show how MIPY reads the table (nothing is imputed)")
    _common(pc)

    pi = sub.add_parser("impute", help="impute with one method, no benchmark")
    _common(pi)
    pi.add_argument("--method", default="softImpute", help="method name (mipy methods)")
    pi.add_argument("--m", type=int, default=1, help="completions (> 1 needs a multiple-imputation method)")
    pi.add_argument("--output", help="output CSV (default: <data>_<method>.csv; with --m > 1, _1.._m)")
    pi.add_argument("--seed", type=int, default=2026)

    pe = sub.add_parser("example", help="write the example cohort (GSE65622 layout) as a CSV file")
    pe.add_argument("file", nargs="?", default="mipy_example.csv")
    pe.add_argument("--patients", type=int, default=80)
    pe.add_argument("--proteins", type=int, default=120)
    pe.add_argument("--seed", type=int, default=2026)

    sub.add_parser("methods", help="list the imputation methods")

    a = ap.parse_args(argv)
    if a.cmd is None:
        ap.print_help()
        return 0
    try:
        return _dispatch(a)
    except KeyboardInterrupt:
        print("\nstopped. Finished runs are saved; run the same command again to continue.")
        return 130
    except Exception as e:
        from .data import DataError
        kind = "data problem" if isinstance(e, DataError) else type(e).__name__
        print(f"\nmipy: {kind}: {e}", file=sys.stderr)
        return 2


def _dispatch(a):
    import pandas as pd
    if a.cmd == "methods":
        from . import list_methods
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(list_methods().to_string(index=False))
        return 0
    if a.cmd == "example":
        from .datasets import example_cohort
        df = example_cohort(n_patients=a.patients, n_proteins=a.proteins, seed=a.seed, save=a.file)
        print(f"wrote {a.file}: {df.shape[0]} rows x {df.shape[1]} columns "
              f"(Patient_ID, sampling_point, Prot_001..., RFS, event, clinical columns)")
        print(f"next: mipy analyze {a.file} --methods quick --repeats 1")
        return 0
    if a.cmd == "check":
        from .data import prepare_data
        prep = prepare_data(a.data, **_prep_kwargs(a))
        print(prep.summary().to_string(index=False))
        for n in prep.notes:
            print("  -", n)
        for w in prep.warnings:
            print("  !", w)
        print("\nlooks right? run: mipy analyze", a.data, "(add --id/--time/--features to change)")
        return 0
    if a.cmd == "impute":
        from .analysis import _resolve_methods
        from .data import prepare_data
        from .imputers import METHODS, run
        from .context import ImputeContext
        from .schema import build_variable_schema
        prep = prepare_data(a.data, **_prep_kwargs(a))
        names, skipped = _resolve_methods([a.method], prep)
        if not names:
            raise ValueError(f"{a.method} cannot run here: {list(skipped.values())[0]}")
        name = names[0]
        if a.m > 1 and not METHODS[name].multiple:
            raise ValueError(f"{name} is single imputation; for --m > 1 use one of: "
                             + ", ".join(k for k, v in METHODS.items() if v.multiple))
        ctx = ImputeContext(seed=a.seed, patient=prep.patient, time=prep.time, m=max(a.m, 1))
        ctx = ctx.with_(schema=build_variable_schema(prep.X, ctx.max_predictors, ctx.min_cor))
        out, draws = run(name, prep.X, ctx)
        stem = Path(a.output) if a.output else Path(a.data).with_name(f"{Path(a.data).stem}_{name}.csv")
        comps = draws if (a.m > 1 and draws) else [out]
        for l, C in enumerate(comps, 1):
            fn = stem if len(comps) == 1 else stem.with_name(f"{stem.stem}_{l}{stem.suffix or '.csv'}")
            prep.completed_table(C).to_csv(fn, index=False)
            print("wrote", fn)
        return 0
    # analyze
    from .analysis import analyze
    workers = a.workers if a.workers == "auto" else int(a.workers)
    res = analyze(a.data, out=a.out, methods=a.methods, mechanisms=a.mechanisms, repeats=a.repeats,
                  rate=a.rate, strength=a.strength, m=a.m,
                  export=None if a.export.lower() == "none" else a.export, n_workers=workers,
                  seed=a.seed, resume=not a.fresh, show=False, verbose=not a.quiet, **_prep_kwargs(a))
    print()
    print(res.text_summary())
    if a.open:
        res.open_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
