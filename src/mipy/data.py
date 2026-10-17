"""Reading and checking a user's table before imputation.

``prepare_data`` takes a file path or a DataFrame laid out like the GSE65622
files (one row per patient visit: an ID column, a visit column, many
numeric feature columns, and optionally clinical and survival columns),
works out which column is which, cleans what can be cleaned, and explains
every decision in ``PreparedData.notes``. Problems that make an analysis
impossible raise ``DataError`` with a message that says how to fix them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd


class DataError(ValueError):
    """The table cannot be analysed as given; the message says why."""


MISSING_MARKERS = {"", "na", "n/a", "nan", "none", "null", "-", "--", ".", "?", "nd",
                   "n.d.", "not detected", "<lod", "< lod", "lod", "bdl", "<loq", "missing", "#n/a",
                   "#value!", "#div/0!", "inf", "-inf"}

_ID_EXACT = {"patientid", "patient", "subjectid", "subject", "id", "participantid", "participant",
             "caseid", "individualid", "individual", "pid", "personid", "donorid", "donor",
             "animalid", "mouseid", "studyid", "usubjid", "subjid", "patientnumber", "patientno"}
_ID_PART = ("patient", "subject", "participant", "donor", "individual")
_TIME_EXACT = {"samplingpoint", "visit", "visitnumber", "visitno", "visitid", "timepoint", "time",
               "week", "weeks", "day", "days", "month", "months", "cycle", "sampling", "tp",
               "occasion", "wave", "visitday", "studyday", "visitweek", "avisitn", "avisit",
               "timepointid", "sampletime", "samplingtime", "followupvisit", "sessionnumber", "session"}
_TIME_PART = ("visit", "timepoint", "samplingpoint", "sampling_point")
_EVENT_EXACT = {"event", "status", "censor", "censored", "dead", "death", "died", "vitalstatus",
                "eventstatus", "fustat", "relapse", "recurrence", "progression", "osevent",
                "pfsevent", "rfsevent", "mfsevent", "dfsevent", "efsevent", "osstatus", "pfsstatus",
                "rfsstatus", "mfsstatus", "dfsstatus", "efsstatus", "censoring", "eventindicator"}
_SURV_EXACT = {"rfs", "mfs", "os", "pfs", "dfs", "efs", "dss", "css", "ttp", "ttr", "survival",
               "survivaltime", "survtime", "futime", "followup", "followuptime", "timetoevent",
               "tte", "ostime", "pfstime", "rfstime", "mfstime", "dfstime", "efstime", "osdays",
               "osmonths", "pfsdays", "pfsmonths", "rfsdays", "rfsmonths", "survivaldays",
               "survivalmonths", "eventtime", "timetodeath", "timetorelapse", "timetoprogression"}


def _key(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# ---------------------------------------------------------------- reading
def read_table(path, sheet=0) -> pd.DataFrame:
    """Read CSV, TSV/TXT, Excel (.xlsx/.xls), Parquet or Feather into a
    DataFrame. CSV files separated by ';' or tabs, and files saved in
    Latin-1 (common for Excel exports on Windows), are read as well."""
    p = Path(path).expanduser()
    if not p.exists():
        raise DataError(f"file not found: {p}. Check the path; on Windows use r'C:\\...' "
                        f"or forward slashes.")
    if p.is_dir():
        raise DataError(f"{p} is a folder; give the data file itself.")
    suf = p.suffix.lower()
    try:
        if suf in (".xlsx", ".xlsm", ".xls"):
            try:
                return pd.read_excel(p, sheet_name=sheet)
            except ImportError as e:
                raise DataError(f"reading {suf} files needs an extra package: {e}. "
                                f"Install it (pip install openpyxl) or save the sheet as CSV.") from e
        if suf == ".parquet":
            return pd.read_parquet(p)
        if suf == ".feather":
            return pd.read_feather(p)
        if suf in (".pkl", ".pickle"):
            raise DataError("pickle files are not read, for safety; save the table as CSV.")
    except DataError:
        raise
    except ImportError as e:
        raise DataError(f"reading {suf} files needs an extra package ({e}); save the table as CSV.") from e
    # text formats
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            sep = "\t" if suf in (".tsv", ".tab") else ","
            d = pd.read_csv(p, sep=sep, encoding=enc, low_memory=False)
            if d.shape[1] <= 2:                         # probably another separator
                d2 = pd.read_csv(p, sep=None, engine="python", encoding=enc)
                if d2.shape[1] > d.shape[1]:
                    d = d2
            return d
        except UnicodeDecodeError as e:
            last = e
            continue
        except pd.errors.EmptyDataError as e:
            raise DataError(f"{p.name} is empty.") from e
    raise DataError(f"could not read {p.name} as text ({last}).")


# ---------------------------------------------------------------- helpers
def _to_numeric(s: pd.Series):
    """Convert a column to numbers when most entries are numbers. Returns
    (series, converted?, n_values_turned_missing)."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float), True, 0
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(float).replace([np.inf, -np.inf], np.nan)
        return out, True, int(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).sum())
    st = s.astype("string").str.strip()
    low = st.str.lower()
    given = (st.notna() & (st != "")).fillna(False).astype(bool)   # something was written in the cell
    st = st.mask(low.isin(MISSING_MARKERS))
    present = st.notna()
    if present.sum() == 0:
        return pd.Series(np.nan, index=s.index), True, 0
    num = pd.to_numeric(st, errors="coerce")
    ok = num.notna().sum() / present.sum()
    if ok < 0.8:                                     # decimal commas?
        num2 = pd.to_numeric(st.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                             errors="coerce")
        if num2.notna().sum() / present.sum() > ok:
            num, ok = num2, num2.notna().sum() / present.sum()
    if ok < 0.8:
        return s, False, 0
    lost = int((given & num.isna()).sum())
    return num.astype(float).replace([np.inf, -np.inf], np.nan), True, lost


def _numeric_time(s: pd.Series):
    """Visit column -> numbers: as is if numeric, else the number inside the
    label ('V2', 'week 4', 'Day 10'), else the order of first appearance."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        return num.to_numpy(dtype=float), "numeric"
    txt = s.astype("string")
    ext = pd.to_numeric(txt.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")
    base = txt.str.lower().str.strip()
    if ext.notna().all():
        return ext.to_numpy(dtype=float), "number taken from each label"
    named = {"baseline": 0, "screening": -1, "pre": 0, "pretreatment": 0, "bl": 0}
    mapped = base.map(lambda v: named.get(re.sub(r"[^a-z]", "", str(v)), np.nan))
    comb = ext.fillna(mapped)
    if comb.notna().all():
        return comb.to_numpy(dtype=float), "number taken from each label (baseline = 0)"
    codes, _ = pd.factorize(s, sort=False)
    return codes.astype(float) + 1, "order of first appearance (labels have no numbers)"


def _resolve_cols(df, cols, what) -> List[str]:
    """Accept a list of names, a single name, or 'first:last' (a range)."""
    if cols is None:
        return []
    if isinstance(cols, str):
        if ":" in cols and cols not in df.columns:
            a, b = cols.split(":", 1)
            names = list(df.columns)
            for c in (a, b):
                if c not in names:
                    raise DataError(f"{what}: column {c!r} not found. Columns start with "
                                    f"{names[:6]}")
            i, j = names.index(a), names.index(b)
            if j < i:
                i, j = j, i
            return names[i:j + 1]
        cols = [c.strip() for c in cols.split(",")] if "," in cols and cols not in df.columns else [cols]
    out = []
    for c in cols:
        if c not in df.columns:
            close = [x for x in df.columns if _key(x) == _key(c)]
            if close:
                c = close[0]
            else:
                raise DataError(f"{what}: column {c!r} not found. Available columns include "
                                f"{list(df.columns[:8])}{' ...' if df.shape[1] > 8 else ''}")
        out.append(c)
    return out


def _id_by_content(df: pd.DataFrame):
    """The first column, if it looks like a patient ID: whole numbers or
    text, each value on 1.5 to 50 rows on average, most values repeated."""
    if df.shape[1] < 3 or len(df) < 6:
        return None
    c = df.columns[0]
    s = df[c].dropna()
    if pd.api.types.is_float_dtype(s) and not np.allclose(s, np.round(s)):
        return None
    if pd.api.types.is_bool_dtype(s):
        return None
    vc = s.value_counts()
    if len(vc) < 2 or not (1.5 <= len(s) / len(vc) <= 50) or (vc >= 2).mean() < 0.5:
        return None
    return c


def _visit_by_content(df: pd.DataFrame, id_col):
    """The column right after the ID, if it looks like a visit: few distinct
    values, and (ID, value) pairs almost all unique."""
    cols = list(df.columns)
    i = cols.index(id_col)
    if i + 1 >= len(cols):
        return None
    c = cols[i + 1]
    s = df[c]
    if s.isna().any() or not (2 <= s.nunique() <= 30):
        return None
    if df[[id_col, c]].duplicated().mean() > 0.05:
        return None
    return c


def _is_binary(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna().unique()
    return len(v) > 0 and set(np.round(v, 9)).issubset({0.0, 1.0}) and s.notna().mean() > 0.5


# ---------------------------------------------------------------- result
@dataclass
class PreparedData:
    """A checked table, ready for imputation (see :func:`prepare_data`)."""
    table: pd.DataFrame                  # the input table, rows in the original order (cleaned)
    order: np.ndarray                    # analysis row i is table row order[i]
    X: np.ndarray                        # features on the analysis scale, rows in analysis order
    features: List[str]
    log2: bool
    id_col: Optional[str] = None
    time_col: Optional[str] = None
    patient: Optional[np.ndarray] = None  # integer patient code per analysis row
    time: Optional[np.ndarray] = None     # numeric visit per analysis row
    clinical_cols: List[str] = field(default_factory=list)
    clinical: Optional[pd.DataFrame] = None
    survival: Optional[tuple] = None      # (time column, event column)
    longitudinal: bool = False
    excluded_features: dict = field(default_factory=dict)   # reason -> [columns]
    ignored_columns: List[str] = field(default_factory=list)
    empty_rows: int = 0
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------ scale
    def to_original_scale(self, Xa):
        Xa = np.asarray(Xa, dtype=float)
        return 2.0 ** Xa - 1.0 if self.log2 else Xa

    def completed_table(self, Xa) -> pd.DataFrame:
        """The input table with the feature columns replaced by a completed
        matrix (analysis scale, analysis row order), in the original row
        order and on the original scale."""
        out = self.table.copy()
        vals = np.full((len(out), len(self.features)), np.nan)
        vals[self.order] = self.to_original_scale(Xa)
        out[self.features] = vals
        return out

    # ------------------------------------------------------------ report
    @property
    def n_patients(self):
        return int(len(np.unique(self.patient))) if self.patient is not None else len(self.X)

    def summary(self) -> pd.DataFrame:
        X = self.X
        rows = [("rows (patient visits)", len(X)),
                ("patients", self.n_patients if self.id_col else "not given (each row on its own)"),
                ("features imputed", len(self.features)),
                ("missing cells", f"{int(np.isnan(X).sum()):,} of {X.size:,} "
                                  f"({100 * np.isnan(X).mean():.2f}%)"),
                ("features with any missing value", int(np.isnan(X).any(0).sum())),
                ("ID column", self.id_col or "none"),
                ("visit column", self.time_col or "none"),
                ("longitudinal methods", "used" if self.longitudinal else "skipped (needs repeated visits)"),
                ("clinical columns", ", ".join(map(str, self.clinical_cols)) or "none"),
                ("survival columns", " / ".join(map(str, self.survival)) if self.survival else "none"),
                ("analysis scale", "log2(x + 1)" if self.log2 else "as given")]
        if self.time is not None and self.time_col:
            vis = pd.Series(self.time).value_counts().sort_index()
            rows.append(("visits (value: rows)", ", ".join(f"{_fmt(k)}: {v}" for k, v in vis.items())[:300]))
        for why, cols in self.excluded_features.items():
            rows.append((f"not imputed: {why}", f"{len(cols)} ({', '.join(map(str, cols[:5]))}"
                                                f"{' ...' if len(cols) > 5 else ''})"))
        if self.ignored_columns:
            rows.append(("other columns (kept, not used)", ", ".join(map(str, self.ignored_columns[:10]))
                         + (" ..." if len(self.ignored_columns) > 10 else "")))
        return pd.DataFrame(rows, columns=["item", "value"]).astype({"value": str})

    def __repr__(self):
        return "PreparedData\n" + self.summary().to_string(index=False)


def _fmt(v):
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


# ---------------------------------------------------------------- main
def prepare_data(data, id_col=None, time_col=None, features=None, clinical=None, survival=None,
                 log2: Union[str, bool] = "auto", max_missing: float = 0.9, sheet=0,
                 layout: str = "wide", feature_col=None, value_col=None) -> PreparedData:
    """Check a table and get it ready for imputation.

    Parameters
    ----------
    data : path (CSV, Excel, TSV, Parquet) or DataFrame. One row per patient
        visit; ``layout="long"`` accepts one row per patient x visit x feature
        (give ``feature_col`` and ``value_col``).
    id_col, time_col : patient and visit columns. Found automatically from
        common names (Patient_ID, subject, visit, sampling_point, timepoint, ...)
        when not given; ``False`` means "there is none".
    features : columns to impute: a list, or ``"first:last"`` for a range.
        Default: every numeric column that is not an ID, visit, clinical or
        survival column.
    clinical : clinical covariates (used by the MAR masking). Default:
        numeric columns with at most 10 distinct whole-number values, and
        text columns with few categories. ``False`` = none.
    survival : (time column, event column) for the exploratory survival
        screen. Default: found from common names (RFS, OS, PFS, ... with
        event / status). ``False`` = none.
    log2 : "auto" (log2(x + 1) when every value is above -1 and the largest
        is above 50, as for raw intensities), True or False.
    max_missing : features missing in more than this share of rows are not
        imputed (they are kept, unchanged, in the output).
    """
    notes, warns = [], []
    if isinstance(data, (str, Path)):
        df = read_table(data, sheet=sheet)
        notes.append(f"read {Path(data).name}: {df.shape[0]} rows x {df.shape[1]} columns")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(np.asarray(data, dtype=float),
                          columns=[f"x{k + 1}" for k in range(np.asarray(data).shape[1])])
        id_col = False if id_col is None else id_col
        time_col = False if time_col is None else time_col
    else:
        try:                                          # polars and similar
            df = data.to_pandas()
        except Exception as e:
            raise DataError(f"data must be a file path or a pandas DataFrame, not {type(data).__name__}") from e
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DataError("the table is empty.")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        dup = sorted(set(df.columns[df.columns.duplicated()]))
        seen = {}
        cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                cols.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                cols.append(c)
        df.columns = cols
        warns.append(f"repeated column names were renamed: {dup[:5]}")
    df = df.dropna(axis=0, how="all").reset_index(drop=True)

    # ---------------------------------------------------- long layout
    if layout == "long":
        if id_col in (None, False) or time_col in (None, False) or feature_col is None or value_col is None:
            raise DataError("layout='long' needs id_col, time_col, feature_col and value_col.")
        id_col, time_col, feature_col, value_col = [
            _resolve_cols(df, [c], "long layout")[0] for c in (id_col, time_col, feature_col, value_col)]
        from .io import long_to_wide
        others = [c for c in df.columns if c not in (id_col, time_col, feature_col, value_col)]
        vals = df[[id_col, time_col, feature_col, value_col]].copy()
        vals[value_col] = _to_numeric(vals[value_col])[0]
        wide = long_to_wide(vals, id_col, time_col, feature_col, value_col)
        feat_names = [c for c in wide.columns if c not in (id_col, time_col)]
        if others:
            extra = df.groupby([id_col, time_col], sort=False)[others].first().reset_index()
            wide = wide.merge(extra, on=[id_col, time_col], how="left")
        notes.append(f"long table reshaped to {wide.shape[0]} rows x {len(feat_names)} features")
        df = wide
        if features is None:
            features = feat_names
    elif layout != "wide":
        raise DataError("layout must be 'wide' or 'long'")

    cols = list(df.columns)
    used = set()

    # ---------------------------------------------------- ID
    if id_col is None:
        cands = [c for c in cols if _key(c) in _ID_EXACT] or \
                [c for c in cols if any(p in _key(c) for p in _ID_PART)
                 and df[c].nunique(dropna=True) < len(df)]
        by_content = None if cands else _id_by_content(df)
        if cands:
            id_col = cands[0]
            notes.append(f"ID column: '{id_col}' (found by name; pass id_col=... to change)")
        elif by_content:
            id_col = by_content
            notes.append(f"ID column: '{id_col}' (the first column, whose values repeat over rows; "
                         f"pass id_col=... to change, or id_col=False if it is not an ID)")
        else:
            id_col = False
            notes.append("no ID column found: every row is treated as a separate patient "
                         "(pass id_col=... if there is one)")
    elif id_col is not False:
        id_col = _resolve_cols(df, [id_col], "id_col")[0]
    if id_col:
        used.add(id_col)
        if df[id_col].isna().any():
            k = int(df[id_col].isna().sum())
            df = df[df[id_col].notna()].reset_index(drop=True)
            warns.append(f"{k} rows without a patient ID were removed")

    # ---------------------------------------------------- visit
    if time_col is None:
        cands = [c for c in cols if c not in used and (_key(c) in _TIME_EXACT or
                 any(p in _key(c) for p in _TIME_PART))]
        cands = [c for c in cands if df[c].nunique(dropna=True) <= max(30, int(0.05 * len(df)))]
        nxt = _visit_by_content(df, id_col) if (id_col and not cands) else None
        if cands and id_col:
            time_col = cands[0]
            notes.append(f"visit column: '{time_col}' (found by name; pass time_col=... to change)")
        elif nxt:
            time_col = nxt
            notes.append(f"visit column: '{time_col}' (the column after the ID, with few distinct "
                         f"values that differ within a patient; pass time_col=... to change)")
        else:
            time_col = False
            if id_col:
                notes.append("no visit column found: rows of a patient are taken in file order "
                             "(pass time_col=... if there is one)")
    elif time_col is not False:
        time_col = _resolve_cols(df, [time_col], "time_col")[0]
    if time_col:
        used.add(time_col)
        if df[time_col].isna().any():
            k = int(df[time_col].isna().sum())
            df = df[df[time_col].notna()].reset_index(drop=True)
            warns.append(f"{k} rows without a visit value were removed")

    # ---------------------------------------------------- survival
    surv = None
    if survival not in (None, False):
        if isinstance(survival, str) or len(survival) != 2:
            raise DataError("survival must be a pair: (time column, event column), e.g. ('RFS', 'event')")
        st_, se_ = _resolve_cols(df, list(survival), "survival")
        if not _is_binary(df[se_]):
            raise DataError(f"survival event column {se_!r} must hold 0 (censored) and 1 (event)")
        surv = (st_, se_)
    elif survival is None:
        ev = [c for c in cols if c not in used and (_key(c) in _EVENT_EXACT or _key(c).startswith("e")
              and _key(c)[1:] in _SURV_EXACT) and _is_binary(df[c])]
        tm = [c for c in cols if c not in used and c not in ev and (_key(c) in _SURV_EXACT or
              (_key(c).startswith("t") and _key(c)[1:] in _SURV_EXACT))]
        tm = [c for c in tm if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
        if ev and tm:
            pair = None
            for t in tm:                               # match name stems: OS / OS_event
                for e in ev:
                    if _key(t).replace("time", "")[:3] and _key(t).replace("time", "")[:3] in _key(e):
                        pair = (t, e)
                        break
                if pair:
                    break
            surv = pair or (tm[0], ev[0])
            notes.append(f"survival columns: time '{surv[0]}', event '{surv[1]}' (found by name; "
                         f"pass survival=False to skip the survival screen)")
    if surv:
        used.update(surv)

    # ---------------------------------------------------- clinical and features
    conv = {}
    lost_total = 0
    for c in cols:
        if c in used:
            continue
        s, ok, lost = _to_numeric(df[c])
        conv[c] = (s, ok)
        lost_total += lost
    if clinical is False:
        clin = []
    elif clinical is not None:
        clin = _resolve_cols(df, clinical, "clinical")
    else:
        clin = []
        given = set(_resolve_cols(df, features, "features")) if features is not None else set()
        for c, (s, ok) in conv.items():
            if c in given:
                continue
            nun = s.nunique(dropna=True) if ok else df[c].nunique(dropna=True)
            if ok and 1 < nun <= 10 and np.allclose(s.dropna(), np.round(s.dropna())):
                clin.append(c)
            elif not ok and 1 < nun <= 20:
                clin.append(c)
        if clin:
            notes.append(f"clinical columns (few distinct values): {', '.join(clin[:8])}"
                         f"{' ...' if len(clin) > 8 else ''} (pass clinical=... to change)")
    used.update(clin)
    if features is not None:
        feats = _resolve_cols(df, features, "features")
        bad = [c for c in feats if c in (id_col, time_col)]
        if bad:
            raise DataError(f"features include the ID/visit column(s) {bad}")
        nonnum = [c for c in feats if not conv.get(c, (None, False))[1]]
        if nonnum:
            raise DataError(f"these feature columns are mostly text, not numbers: {nonnum[:5]}. "
                            f"Remove them from features, or fix the values.")
    else:
        feats = [c for c, (s, ok) in conv.items() if ok and c not in used]
    ignored = [c for c in cols if c not in used and c not in feats]
    if lost_total:
        warns.append(f"{lost_total} entries that were not numbers (such as '<LOD' or 'n/a') "
                     f"were treated as missing")

    if not feats:
        raise DataError("no numeric feature columns to impute were found. Pass features=[...] "
                        "or check that the values are numbers.")
    F = pd.DataFrame({c: conv[c][0] if c in conv else _to_numeric(df[c])[0] for c in feats})
    for c in feats:                                  # keep cleaned numbers in the output table
        df[c] = F[c].to_numpy()

    excluded = {}
    miss = F.isna().mean()
    allmiss = list(miss.index[miss >= 1.0])
    many = list(miss.index[(miss > max_missing) & (miss < 1.0)])
    sd = F.std(ddof=0)
    const = [c for c in feats if c not in allmiss and c not in many and not (sd[c] > 1e-12)]
    if allmiss:
        excluded["no observed value"] = allmiss
    if many:
        excluded[f"more than {int(100 * max_missing)}% missing"] = many
    if const:
        excluded["the same value in every row"] = const
    drop = set(allmiss) | set(many) | set(const)
    feats = [c for c in feats if c not in drop]
    if drop:
        warns.append(f"{len(drop)} feature column(s) are not imputed (kept unchanged): "
                     + "; ".join(f"{k}: {len(v)}" for k, v in excluded.items()))
    if len(feats) < 2:
        raise DataError(f"only {len(feats)} usable feature column(s); imputation needs at least 2.")

    # ---------------------------------------------------- rows, order
    Xraw = F[feats].to_numpy(dtype=float)
    empty = np.isnan(Xraw).all(1)
    n_empty = int(empty.sum())
    if n_empty:
        warns.append(f"{n_empty} row(s) have no observed feature value; they are kept in the "
                     f"output but left missing")
    keep = np.flatnonzero(~empty)
    if len(keep) < 10:
        raise DataError(f"only {len(keep)} rows have feature values; at least 10 are needed.")

    if id_col:                                       # patients in ID order, so row order in the file
        try:                                         # does not change the results
            codes, uniq = pd.factorize(df[id_col], sort=True)
        except TypeError:                            # mixed numbers and text
            codes, uniq = pd.factorize(df[id_col].astype(str), sort=True)
    else:
        codes = np.arange(len(df))
    tnum = None
    if time_col:
        tnum, how = _numeric_time(df[time_col])
        if how != "numeric":
            notes.append(f"visit values converted to numbers: {how}")
    key_t = tnum if tnum is not None else np.arange(len(df), dtype=float)
    order = keep[np.lexsort((np.arange(len(df))[keep], key_t[keep], codes[keep]))]
    patient = codes[order] if id_col else None
    if tnum is not None:
        time = tnum[order]
    elif id_col:                                     # visit number within the patient, in file order
        time = pd.Series(codes[order]).groupby(codes[order]).cumcount().to_numpy(dtype=float) + 1
    else:
        time = None
    longitudinal = False
    if id_col:
        per = pd.Series(patient).value_counts()
        longitudinal = bool((per >= 2).sum() >= 3)
        if not longitudinal:
            notes.append("fewer than 3 patients have 2 or more visits: longitudinal methods "
                         "(LOCF, MICE-Long, MICE-LongMNAR) are skipped")
        if time_col:
            dup = pd.DataFrame({"p": patient, "t": time}).duplicated().sum()
            if dup:
                warns.append(f"{int(dup)} rows repeat a patient-visit pair already in the table; "
                             f"they are kept and ordered as in the file")

    # ---------------------------------------------------- scale
    obs = Xraw[order]
    finite = obs[np.isfinite(obs)]
    if log2 == "auto":
        use_log = bool(finite.size and finite.min() > -1 and finite.max() > 50)
        notes.append("values analysed as log2(x + 1) (all values above -1, largest above 50, as "
                     "for raw intensities); results are written back on the original scale"
                     if use_log else "values analysed as given (they look already transformed; "
                     "pass log2=True to change)")
    else:
        use_log = bool(log2)
        if use_log and finite.size and (finite <= -1).any():
            k = int((finite <= -1).sum())
            if k > 0.01 * finite.size:
                raise DataError(f"log2=True needs values above -1, but {k} values are -1 or less. "
                                f"Use log2=False.")
            warns.append(f"{k} values of -1 or less have no log2(x + 1) and were treated as missing")
    with np.errstate(invalid="ignore", divide="ignore"):
        X = np.where(obs > -1, np.log2(np.maximum(obs, -1 + 1e-12) + 1), np.nan) if use_log else obs.copy()

    pmiss = np.isnan(X).mean()
    if pmiss == 0:
        notes.append("the features have no missing values: the benchmark still runs (it hides "
                     "observed values), and the completed table equals the input")
    elif pmiss > 0.5:
        warns.append(f"{100 * pmiss:.0f}% of the feature values are missing; every method will "
                     f"be uncertain")

    clin_df = None
    if clin:
        parts = {}
        for c in clin:
            s, ok = conv.get(c, _to_numeric(df[c])[:2])
            parts[c] = (s if ok else pd.Series(pd.factorize(df[c])[0], index=df.index)
                        .replace(-1, np.nan).astype(float)).to_numpy()[order]
        clin_df = pd.DataFrame(parts)

    return PreparedData(table=df, order=order, X=X, features=list(feats), log2=use_log,
                        id_col=id_col or None, time_col=time_col or None, patient=patient,
                        time=time, clinical_cols=clin, clinical=clin_df, survival=surv,
                        longitudinal=longitudinal, excluded_features=excluded,
                        ignored_columns=ignored, empty_rows=n_empty, notes=notes, warnings=warns)
