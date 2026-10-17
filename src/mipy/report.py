"""The HTML report of :func:`mipy.analyze`: one self-contained file (figures
embedded), readable in any browser and easy to send to a colleague."""
from __future__ import annotations

import base64
import datetime as _dt
import html
from pathlib import Path

import numpy as np
import pandas as pd

CSS = """
:root{--ink:#1d2433;--muted:#5b6475;--line:#dfe3ea;--bg:#ffffff;--soft:#f5f7fa;--accent:#2f5d9e;--warn:#9a5b00;--warnbg:#fff7e6}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:36px 0 10px;padding-top:10px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 6px}
.sub{color:var(--muted);margin:0 0 18px}
.box{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:14px 0}
.box.warn{background:var(--warnbg);border-color:#f1d7a6}
.box ul{margin:6px 0 0;padding-left:20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}
.kpi{border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kpi b{display:block;font-size:20px}
.kpi span{color:var(--muted);font-size:13px}
.tw{overflow-x:auto;margin:8px 0 4px}
table{border-collapse:collapse;font-size:13px;min-width:50%}
th,td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:right;white-space:nowrap}
th{background:var(--soft);font-weight:600}
td:first-child,th:first-child{text-align:left}
td.l,th.l{text-align:left}
tr.best td{background:#eaf3ea}
figure{margin:16px 0 26px}
figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
figcaption{color:var(--muted);font-size:13.5px;margin-top:6px}
code{background:var(--soft);padding:1px 5px;border-radius:4px;font-size:13px}
nav a{margin-right:12px;color:var(--accent);text-decoration:none;font-size:14px}
.small{font-size:13px;color:var(--muted)}
@media (max-width:600px){h1{font-size:22px}th,td{padding:4px 6px}}
"""

INTRO = """MIPY hid a share of the values that <i>are</i> observed in your table, under different
assumptions about why values go missing, asked every method to fill them in, and compared the
filled-in values with the hidden true ones. The error is the <b>nRMSE</b>: the root mean squared
error divided by the standard deviation of the hidden values, computed within each feature and
summarised by the median across features. <b>1.00 is what filling in each feature's mean
achieves; lower is better.</b> Each masking was repeated; the Monte Carlo standard error (MCSE)
shows how much the mean would move with other random maskings."""

MECH_TEXT = {"MCAR": "values hidden completely at random",
             "MAR": "values hidden more often at some visits and for some clinical profiles",
             "MNAR": "low values hidden more often (as below a detection limit)",
             "LOD": "every value below a detection limit hidden"}


def _img(path):
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def png_width(path, dpi_scale=2):
    """Display width in CSS pixels of a PNG saved at 200 dpi (its size at 100 dpi)."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        return int.from_bytes(head[16:20], "big") // dpi_scale
    except Exception:
        return 900


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "&ndash;"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{nd}f}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, nd=None, best_col=None, max_rows=200, left=()):
    """DataFrame -> HTML table; float columns rounded (nd: {column: digits})."""
    nd = nd or {}
    if df is None or len(df) == 0:
        return "<p class='small'>(empty)</p>"
    d = df.head(max_rows)
    cols = list(d.columns)
    head = "".join(f"<th class='{'l' if c in left else ''}'>{html.escape(str(c))}</th>" for c in cols)
    body = []
    best_idx = None
    if best_col is not None and best_col in d and d[best_col].notna().any():
        best_idx = d[best_col].astype(float).idxmin()
    for i, r in d.iterrows():
        cls = " class='best'" if i == best_idx else ""
        tds = "".join(f"<td class='{'l' if c in left else ''}'>{_fmt(r[c], nd.get(c, 3))}</td>" for c in cols)
        body.append(f"<tr{cls}>{tds}</tr>")
    more = f"<p class='small'>first {max_rows} of {len(df)} rows; all rows in the CSV file</p>" if len(df) > max_rows else ""
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{more}"


def _figure(res, name, num):
    if name not in res.figures:
        return ""
    cap = html.escape(res.figure_notes.get(name, ""))
    w = png_width(res.figures[name])
    return (f"<figure><img alt='{html.escape(name)}' style='width:{w}px' src='{_img(res.figures[name])}'>"
            f"<figcaption><b>Figure {num}.</b> {cap} <span class='small'>(file: figures/"
            f"{html.escape(Path(res.figures[name]).name)} and .pdf)</span></figcaption></figure>")


SUMMARY_COLS = ["method", "family", "nRMSE", "MCSE", "best in", "% features nRMSE < 1", "r", "bias",
                "SD ratio", "coverage (%)", "seconds", "runs ok"]
ND = {"nRMSE": 3, "MCSE": 3, "% features nRMSE < 1": 1, "r": 3, "bias": 3, "SD ratio": 2,
      "coverage (%)": 1, "seconds": 1, "interval width": 3, "runner-up nRMSE": 3, "its coverage (%)": 1,
      "log_hr": 3, "HR per SD": 2, "se": 3, "p": 4, "q": 3, "fmi": 2}


def _sci(df, cols=("p", "q")):
    d = df.copy()
    for c in cols:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:.2e}" if isinstance(v, float) and np.isfinite(v) and v < 1e-3
                            else (f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else "–"))
    return d


def write_report(res, path) -> Path:
    path = Path(path)
    prep = res.prepared
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    rec = res.recommendation or {}
    best = rec.get("best", {})
    fig_order = [n for n in ["accuracy", "nrmse_per_feature", "sd_ratio", "coverage", "runtime",
                             "missingness", "masking", "imputed_vs_observed", "survival"] if n in res.figures]
    fnum = {n: i for i, n in enumerate(fig_order, 1)}
    P = []
    P.append(f"<h1>MIPY imputation report</h1><p class='sub'>{html.escape(str(res.settings.get('data', '')))}"
             f" &middot; {now} &middot; MIPY {html.escape(str(res.versions.get('mipy', '')))} &middot; "
             f"run time {res.seconds / 60:.1f} min</p>")
    P.append("<nav><a href='#key'>Key results</a><a href='#accuracy'>Accuracy</a><a href='#data'>Your data</a>"
             "<a href='#completed'>Completed data</a>"
             + ("<a href='#survival'>Survival</a>" if res.survival is not None else "")
             + "<a href='#methods'>Methods</a><a href='#settings'>Settings</a></nav>")

    # ------------------------------------------------ key results
    P.append("<h2 id='key'>Key results</h2>")
    if prep is not None:
        P.append("<div class='kpis'>"
                 f"<div class='kpi'><b>{len(prep.X):,}</b><span>rows (patient visits)</span></div>"
                 f"<div class='kpi'><b>{prep.n_patients:,}</b><span>patients</span></div>"
                 f"<div class='kpi'><b>{len(prep.features):,}</b><span>features imputed</span></div>"
                 f"<div class='kpi'><b>{100 * np.isnan(prep.X).mean():.2f}%</b><span>values missing</span></div>"
                 f"<div class='kpi'><b>{len(res.methods)}</b><span>methods compared</span></div></div>")
    items = []
    for mech, v in best.items():
        row = res.decision[res.decision["mechanism"] == mech]
        extra = ""
        if len(row) and row.iloc[0]["runner-up"]:
            r0 = row.iloc[0]
            extra = (f"; best in {html.escape(str(r0['best in']))} maskings; next: {html.escape(str(r0['runner-up']))} "
                     f"({_fmt(r0['runner-up nRMSE'])})")
        items.append(f"<li><b>{mech}</b>, {MECH_TEXT.get(mech, '')}: <b>{html.escape(v['method'])}</b>, "
                     f"nRMSE {v['nRMSE']:.3f}{extra}</li>")
    if items:
        P.append("<div class='box'><b>Most accurate method under each assumption</b><ul>" + "".join(items) + "</ul></div>")
    if res.completed_method:
        mi = (f" For analyses that should carry the imputation uncertainty, use the {len(res.completed_draws)} "
              f"completions from <b>{html.escape(res.mi_method)}</b> and pool the results with Rubin's rules."
              if res.mi_method else "")
        P.append(f"<div class='box'><b>Completed data.</b> The missing values were filled in with "
                 f"<b>{html.escape(res.completed_method)}</b>: {html.escape(str(rec.get('export reason', '')))}. "
                 f"Files are in <code>completed_data/</code>.{mi}</div>")
    if res.warnings:
        P.append("<div class='box warn'><b>Please read</b><ul>"
                 + "".join(f"<li>{html.escape(w)}</li>" for w in res.warnings) + "</ul></div>")

    # ------------------------------------------------ accuracy
    P.append("<h2 id='accuracy'>Accuracy of the methods</h2>")
    P.append(f"<p>{INTRO}</p>")
    if "accuracy" in res.figures:
        P.append(_figure(res, "accuracy", fnum["accuracy"]))
    if len(res.overview):
        ov = res.overview.reset_index()
        P.append("<h3>Mean nRMSE, all methods and mechanisms</h3>")
        P.append(_table(ov, nd={c: 3 for c in ov.columns}))
    if len(res.decision):
        P.append("<h3>Decision guide</h3>")
        P.append(_table(res.decision, nd=ND, left=("mechanism", "best method", "runner-up",
                                                    "best multiple-imputation method")))
    for mech in [m for m in ["MCAR", "MAR", "MNAR", "LOD"] if len(res.summary) and m in set(res.summary["mechanism"])]:
        s = res.summary[res.summary["mechanism"] == mech]
        s = s[[c for c in SUMMARY_COLS if c in s.columns]].reset_index(drop=True)
        P.append(f"<h3>{mech}: {MECH_TEXT.get(mech, '')}</h3>")
        P.append(_table(s, nd=ND, best_col="nRMSE", left=("method", "family")))
    P.append("<p class='small'>best in: how many of the maskings the method won. % features nRMSE &lt; 1: "
             "share of features where the method beats the feature mean. r: median correlation of "
             "imputed and true values within a feature. bias: median of imputed minus true (analysis "
             "scale). SD ratio: within-patient variation kept (1 = all of it). coverage: share of hidden "
             "values inside the 95% multiple-imputation interval.</p>")
    for n in ["nrmse_per_feature", "sd_ratio", "coverage", "runtime"]:
        P.append(_figure(res, n, fnum.get(n, 0)))

    # ------------------------------------------------ data
    P.append("<h2 id='data'>Your data</h2>")
    P.append(_table(res.data_summary, left=("item", "value")))
    if res.notes:
        P.append("<div class='box'><b>What MIPY found and did</b><ul>"
                 + "".join(f"<li>{html.escape(n)}</li>" for n in res.notes) + "</ul>"
                 "<p class='small'>Pass <code>id_col</code>, <code>time_col</code>, <code>features</code>, "
                 "<code>clinical</code> or <code>survival</code> to <code>mipy.analyze</code> to change "
                 "any of these choices.</p></div>")
    P.append(_figure(res, "missingness", fnum.get("missingness", 0)))
    if len(res.masking):
        P.append("<h3>The maskings</h3>")
        P.append(_table(res.masking, nd={"% of observed": 2, "matrix missing after (%)": 2,
                                         "mean z of masked cells": 3}, left=("mechanism",)))
    P.append(_figure(res, "masking", fnum.get("masking", 0)))

    # ------------------------------------------------ completed
    P.append("<h2 id='completed'>Completed data</h2>")
    if res.completed_method:
        P.append(f"<p>Method: <b>{html.escape(res.completed_method)}</b>. The file "
                 f"<code>completed_data/completed_{html.escape(res.completed_method)}.csv</code> is your "
                 f"table with the same rows, columns and order, the missing feature values filled in "
                 f"(on the original scale).</p>")
        if res.mi_method:
            P.append(f"<p>Multiple imputation: <b>{html.escape(res.mi_method)}</b>, files "
                     f"<code>completed_{html.escape(res.mi_method)}_draw1.csv</code> to "
                     f"<code>_draw{len(res.completed_draws)}.csv</code>. Run your analysis on each file and "
                     f"pool the estimates with Rubin's rules (<code>mipy.mi.pool_estimates</code>).</p>")
        if len(res.fmi):
            P.append("<p>Fraction of missing information (FMI): how much of the uncertainty of a mean "
                     "comes from the missing values.</p>" + _table(res.fmi, nd={c: 3 for c in res.fmi.columns}))
        P.append(_figure(res, "imputed_vs_observed", fnum.get("imputed_vs_observed", 0)))
    else:
        P.append("<p>No completed table was written (export was switched off or no method could run).</p>")

    # ------------------------------------------------ survival
    if res.survival is not None:
        P.append("<h2 id='survival'>Exploratory survival screen</h2>")
        sv = res.survival
        P.append(f"<p>Univariate Cox models, one feature at a time, on each patient's first sample; hazard "
                 f"ratios per SD; false discovery rate by Benjamini-Hochberg. {int((sv['q'] < 0.10).sum())} of "
                 f"{len(sv)} features have q &lt; 0.10. This is a screen for hypotheses, not a prognostic "
                 f"model: check the survival endpoint and validate any finding in other data.</p>")
        P.append(_figure(res, "survival", fnum.get("survival", 0)))
        P.append("<h3>Top 25 features</h3>" + _table(_sci(sv.head(25)), nd=ND, left=("feature",)))

    # ------------------------------------------------ methods
    P.append("<h2 id='methods'>Methods</h2>")
    if "methods" in res.tables:
        P.append(_table(res.tables["methods"], left=tuple(res.tables["methods"].columns)))
    P.append("<p class='small'>MICE-Long and MICE-LongMNAR are introduced by MIPY. Please cite the MIPY paper "
             "and the publication behind each method you report (<code>mipy.list_methods()</code>).</p>")

    # ------------------------------------------------ settings
    P.append("<h2 id='settings'>Settings and software</h2>")
    st = pd.DataFrame([(k, str(v)) for k, v in res.settings.items()], columns=["setting", "value"])
    P.append(_table(st, left=("setting", "value")))
    vs = pd.DataFrame([(k, str(v)) for k, v in res.versions.items()], columns=["package", "version"])
    P.append(_table(vs, left=("package", "version")))
    P.append("<p class='small'>Every table is in <code>tables/</code> (CSV, and all in mipy_results.xlsx), every "
             "figure in <code>figures/</code> (PNG and PDF). Re-running with the same data and settings "
             "gives the same numbers.</p>")

    doc = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>MIPY report</title><style>{CSS}</style></head><body><main>"
           + "\n".join(x for x in P if x) + "</main></body></html>")
    path.write_text(doc, encoding="utf-8")
    return path


def result_html_summary(res) -> str:
    """Short HTML shown when a Result is displayed in Jupyter."""
    rec = res.recommendation or {}
    parts = [f"<div style='font-family:sans-serif;max-width:900px'><h3 style='margin:4px 0'>MIPY analysis</h3>"]
    if res.prepared is not None:
        p = res.prepared
        parts.append(f"<p style='margin:2px 0'>{len(p.X):,} rows &middot; {p.n_patients:,} patients &middot; "
                     f"{len(p.features):,} features &middot; {100 * np.isnan(p.X).mean():.2f}% missing &middot; "
                     f"{len(res.methods)} methods</p>")
    if len(res.decision):
        parts.append(_table(res.decision, nd=ND, left=("mechanism", "best method", "runner-up",
                                                        "best multiple-imputation method")))
    if len(res.overview):
        parts.append("<p style='margin:8px 0 2px'><b>Mean nRMSE</b> (lower is better; 1 = feature mean)</p>")
        ov = res.overview.reset_index()
        parts.append(_table(ov, nd={c: 3 for c in ov.columns}))
    if res.completed_method:
        short = {"MCAR": "best under MCAR", "MAR": "best under MAR", "MNAR": "best under MNAR",
                 "LOD": "best under LOD"}.get(rec.get("plausible mechanism"), "see the report")
        if rec.get("export chosen by user"):
            short = "your choice"
        parts.append(f"<p>Completed data: <b>{html.escape(res.completed_method)}</b> "
                     f"({html.escape(short)})"
                     + (f"; {len(res.completed_draws)} completions from <b>{html.escape(res.mi_method)}</b>"
                        if res.mi_method else "") + ".</p>")
    if res.warnings:
        parts.append(f"<p style='color:#9a5b00'>{len(res.warnings)} warning(s): see <code>res.warnings</code> "
                     f"or the report.</p>")
    if res.report_path:
        parts.append(f"<p>Report: <code>{html.escape(str(res.report_path))}</code> &middot; "
                     f"<code>res.show()</code> shows every figure.</p>")
    parts.append("</div>")
    return "".join(parts)
