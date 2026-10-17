"""Moving between long tables and analysis matrices, and matching patient
identifiers across tables, with pandas, polars or DuckDB (SQL).

Long format: one row per patient x visit x feature (id, time, feature, value).
Wide format: one row per patient x visit, one column per feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINES = ("pandas", "polars", "duckdb")


def _con(con):
    if con is not None:
        return con, False
    import duckdb
    return duckdb.connect(), True


def _close(con, own, *views):
    for v in views:
        try:
            con.unregister(v)
        except Exception:
            pass
    if own:
        con.close()


def engine_available(engine: str) -> bool:
    import importlib.util
    return engine == "pandas" or importlib.util.find_spec(engine) is not None


# ---------------------------------------------------------------- pandas
def long_to_wide(df, id_col="patient_id", time_col="visit", feature_col="protein",
                 value_col="value", engine="pandas", con=None):
    """Pivot a long table to one row per (id, time) and one column per
    feature. Missing (id, time, feature) combinations become NaN. Returns a
    pandas DataFrame sorted by id and time."""
    if engine == "pandas":
        w = df.pivot_table(index=[id_col, time_col], columns=feature_col,
                           values=value_col, aggfunc="first", observed=True)
        w.columns = [str(c) for c in w.columns]
        return w.reset_index().sort_values([id_col, time_col]).reset_index(drop=True)
    if engine == "polars":
        import polars as pl
        d = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        w = d.pivot(on=feature_col, index=[id_col, time_col], values=value_col,
                    aggregate_function="first").sort([id_col, time_col])
        return w
    if engine == "duckdb":
        con, own = _con(con)
        con.register("long_t", df)
        w = con.execute(f"SELECT * FROM (PIVOT long_t ON {feature_col} USING first({value_col}) "
                        f"GROUP BY {id_col}, {time_col}) ORDER BY {id_col}, {time_col}").df()
        _close(con, own, "long_t")
        return w
    raise ValueError(f"unknown engine {engine!r}")


def wide_to_long(df, id_col="patient_id", time_col="visit", feature_col="protein",
                 value_col="value", engine="pandas", dropna=True, con=None):
    """Melt a wide table to long format, dropping missing values by default."""
    if engine == "pandas":
        lg = df.melt(id_vars=[id_col, time_col], var_name=feature_col, value_name=value_col)
        return lg.dropna(subset=[value_col]) if dropna else lg
    if engine == "polars":
        import polars as pl
        d = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        lg = d.unpivot(index=[id_col, time_col], variable_name=feature_col, value_name=value_col)
        return lg.drop_nulls(value_col).filter(pl.col(value_col).is_not_nan()) if dropna else lg
    if engine == "duckdb":
        con, own = _con(con)
        con.register("wide_t", df)
        feats = [c for c in df.columns if c not in (id_col, time_col)]
        flist = ", ".join(f'"{f}"' for f in feats)
        where = f"WHERE {value_col} IS NOT NULL AND NOT isnan({value_col})" if dropna else ""
        lg = con.execute(f"SELECT * FROM (UNPIVOT wide_t ON {flist} INTO NAME {feature_col} "
                         f"VALUE {value_col}) {where}").df()
        _close(con, own, "wide_t")
        return lg
    raise ValueError(f"unknown engine {engine!r}")


def match_ids(left, right, on="patient_id", engine="pandas", con=None):
    """Inner-join two tables on a patient identifier and report what did not
    match. Returns (joined table, report dict)."""
    if engine == "pandas":
        lk, rk = pd.Index(left[on].unique()), pd.Index(right[on].unique())
        rep = {"left_ids": len(lk), "right_ids": len(rk),
               "matched_ids": int(lk.isin(rk).sum()),
               "left_only": int((~lk.isin(rk)).sum()), "right_only": int((~rk.isin(lk)).sum()),
               "right_duplicates": int(right[on].duplicated().sum())}
        return left.merge(right, on=on, how="inner"), rep
    if engine == "polars":
        import polars as pl
        L = left if isinstance(left, pl.DataFrame) else pl.from_pandas(left)
        R = right if isinstance(right, pl.DataFrame) else pl.from_pandas(right)
        lk, rk = L.select(pl.col(on).unique()), R.select(pl.col(on).unique())
        matched = lk.join(rk, on=on, how="inner").height
        rep = {"left_ids": lk.height, "right_ids": rk.height, "matched_ids": matched,
               "left_only": lk.height - matched, "right_only": rk.height - matched,
               "right_duplicates": R.height - rk.height}
        return L.join(R, on=on, how="inner"), rep
    if engine == "duckdb":
        con, own = _con(con)
        con.register("L", left)
        con.register("R", right)
        q = lambda s: con.execute(s).fetchone()[0]
        rep = {"left_ids": q(f"SELECT COUNT(DISTINCT {on}) FROM L"),
               "right_ids": q(f"SELECT COUNT(DISTINCT {on}) FROM R"),
               "matched_ids": q(f"SELECT COUNT(*) FROM (SELECT DISTINCT {on} FROM L) "
                                f"SEMI JOIN (SELECT DISTINCT {on} FROM R) USING ({on})")}
        rep["left_only"] = rep["left_ids"] - rep["matched_ids"]
        rep["right_only"] = rep["right_ids"] - rep["matched_ids"]
        rep["right_duplicates"] = q(f"SELECT COUNT(*) - COUNT(DISTINCT {on}) FROM R")
        joined = con.execute(f"SELECT * FROM L JOIN R USING ({on})").df()
        _close(con, own, "L", "R")
        return joined, rep
    raise ValueError(f"unknown engine {engine!r}")


def to_matrix(wide, id_col="patient_id", time_col="visit", features=None):
    """Split a wide table into (X float array, patient array, time array,
    feature names), rows sorted by patient and time."""
    if not isinstance(wide, pd.DataFrame):
        wide = wide.to_pandas()
    wide = wide.sort_values([id_col, time_col]).reset_index(drop=True)
    features = features or [c for c in wide.columns if c not in (id_col, time_col)]
    X = wide[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return X, wide[id_col].to_numpy(), wide[time_col].to_numpy(), list(features)


def from_matrix(X, patient, time, features, id_col="patient_id", time_col="visit"):
    d = pd.DataFrame(np.asarray(X), columns=list(features))
    d.insert(0, time_col, np.asarray(time))
    d.insert(0, id_col, np.asarray(patient))
    return d
