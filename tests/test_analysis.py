"""analyze(), data checks, the example cohort and the command line."""
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mipy
from mipy.data import DataError, prepare_data, read_table


@pytest.fixture(scope="module")
def small():
    return mipy.example_cohort(n_patients=30, n_proteins=30, seed=5)


def test_example_cohort_layout():
    df = mipy.example_cohort()
    assert list(df.columns[:2]) == ["Patient_ID", "sampling_point"]
    assert {"RFS", "event", "trg_score", "ypn_stage", "ypt_stage", "batch"} <= set(df.columns)
    prot = df.filter(like="Prot_")
    assert prot.shape[1] == 120 and 0.05 < prot.isna().mean().mean() < 0.15
    assert set(df["sampling_point"]) <= {1, 2, 4, 5}
    assert (df.groupby("Patient_ID")["event"].nunique() == 1).all()
    assert df.equals(mipy.example_cohort())                 # reproducible


def test_prepare_finds_columns(small):
    p = prepare_data(small)
    assert p.id_col == "Patient_ID" and p.time_col == "sampling_point"
    assert p.survival == ("RFS", "event")
    assert set(p.clinical_cols) == {"trg_score", "ypn_stage", "ypt_stage", "batch"}
    assert len(p.features) == 30 and p.log2 and p.longitudinal
    # analysis rows are sorted by patient and visit
    key = pd.DataFrame({"p": p.patient, "t": p.time})
    assert key.equals(key.sort_values(["p", "t"]).reset_index(drop=True))
    # completed_table puts values back in the original order and scale
    back = p.completed_table(p.X)
    feats = p.features
    np.testing.assert_allclose(back[feats].to_numpy(), small[feats].to_numpy(), rtol=1e-9, atol=1e-9)


def test_prepare_messy_values():
    rs = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({"subject": np.repeat(np.arange(10), 4),
                       "Visit": np.tile(["baseline", "V2", "V3", "V4"], 10)})
    for k in range(6):
        v = rs.normal(10, 2, n).round(3).astype(str)
        v[rs.random(n) < 0.1] = "<LOD"
        v[rs.random(n) < 0.05] = "n/a"
        df[f"P{k}"] = v
    df["P0"] = df["P0"].str.replace(".", ",", regex=False).where(df["P0"] != "<LOD", "<LOD")
    df["notes"] = [f"free text {i}" for i in range(n)]
    p = prepare_data(df)
    assert p.id_col == "subject" and p.time_col == "Visit"
    assert sorted(set(p.time)) == [0.0, 2.0, 3.0, 4.0]
    assert len(p.features) == 6 and "notes" in p.ignored_columns
    assert np.isnan(p.X).mean() > 0.05
    assert any("treated as missing" in w for w in p.warnings)
    assert not p.log2


def test_prepare_errors(small):
    with pytest.raises(DataError, match="not found"):
        prepare_data(small, id_col="nope")
    with pytest.raises(DataError, match="at least 10"):
        prepare_data(small.head(6))
    with pytest.raises(DataError, match="no numeric feature"):
        prepare_data(pd.DataFrame({"id": list("abcdefghijkl"), "x": list("abcdefghijkl")}))
    with pytest.raises(DataError, match="file not found"):
        read_table("does_not_exist.csv")


def test_read_table_formats(tmp_path, small):
    small.to_csv(tmp_path / "a.csv", sep=";", index=False)
    assert read_table(tmp_path / "a.csv").shape == small.shape
    small.to_csv(tmp_path / "b.tsv", sep="\t", index=False)
    assert read_table(tmp_path / "b.tsv").shape == small.shape
    small.to_excel(tmp_path / "c.xlsx", index=False)
    assert read_table(tmp_path / "c.xlsx").shape == small.shape
    small.rename(columns={"Prot_001": "Protéine_1"}).to_csv(tmp_path / "d.csv", index=False, encoding="latin-1")
    assert "Protéine_1" in read_table(tmp_path / "d.csv").columns


def test_long_layout(small):
    lg = small.melt(id_vars=["Patient_ID", "sampling_point"], value_vars=[f"Prot_{k:03d}" for k in range(1, 31)],
                    var_name="protein", value_name="intensity")
    p = prepare_data(lg, id_col="Patient_ID", time_col="sampling_point", layout="long",
                     feature_col="protein", value_col="intensity")
    assert len(p.features) == 30 and len(p.X) == len(small)


def _check_outputs(res, df):
    out = Path(res.out_dir)
    assert res.report_path and Path(res.report_path).is_file()
    html = Path(res.report_path).read_text(encoding="utf-8")
    assert "data:image/png;base64" in html and "Key results" in html
    for name in ("missingness", "masking", "accuracy", "nrmse_per_feature", "runtime"):
        assert Path(res.figures[name]).is_file() and Path(res.figures[name]).with_suffix(".pdf").is_file()
    for t in ("accuracy_summary", "decision", "all_runs", "per_feature_nrmse"):
        assert (out / "tables" / f"{t}.csv").is_file()
    assert (out / "README.txt").is_file() and (out / "run_log.txt").is_file()
    comp = res.completed
    assert comp.shape == df.shape and list(comp.columns) == list(df.columns)
    feats = res.prepared.features
    assert not comp[feats].isna().any().any()
    obs = df[feats].notna().to_numpy()
    np.testing.assert_allclose(comp[feats].to_numpy()[obs], df[feats].to_numpy(dtype=float)[obs], rtol=1e-8)


def test_analyze_end_to_end(tmp_path, small):
    res = mipy.analyze(small, out=tmp_path / "r", methods="quick", repeats=2,
                       mechanisms=("MCAR", "MNAR"), show=False, verbose=False)
    assert res.warnings == []
    _check_outputs(res, small)
    assert set(res.summary["mechanism"]) == {"MCAR", "MNAR"}
    assert res.summary["nRMSE"].notna().all() and res.summary["MCSE"].notna().all()
    assert set(res.recommendation["best"]) == {"MCAR", "MNAR"}
    assert res.mi_method == "MICE-Long" and len(res.completed_draws) == 5
    assert "coverage" in res.figures and "sd_ratio" in res.figures
    assert (tmp_path / "r" / "tables" / "mipy_results.xlsx").is_file()
    assert "MIPY analysis" in res._repr_html_() and "MIPY analysis" in repr(res)
    res.show()                                              # text mode, must not fail
    # a second call reuses the saved runs and gives the same numbers
    res2 = mipy.analyze(small, out=tmp_path / "r", methods="quick", repeats=2,
                        mechanisms=("MCAR", "MNAR"), show=False, verbose=False)
    assert all("saved earlier" in l for l in res2.log_lines if "repeat" in l and "]" in l)
    pd.testing.assert_frame_equal(res.summary, res2.summary)


def test_analyze_survival_and_file_input(tmp_path):
    df = mipy.example_cohort(n_patients=60, n_proteins=25, seed=3)
    f = tmp_path / "cohort.csv"
    df.to_csv(f, index=False)
    res = mipy.analyze(f, out=tmp_path / "s", methods=["Mean", "softImpute", "MICE-Bayes"],
                       repeats=1, mechanisms="MCAR,MAR", show=False, verbose=False)
    assert res.warnings == []
    assert res.survival is not None and {"feature", "HR per SD", "p", "q"} <= set(res.survival.columns)
    assert "survival" in res.figures
    assert res.mi_method == "MICE-Bayes"


def test_analyze_cross_sectional(tmp_path):
    rs = np.random.default_rng(2)
    X = rs.normal(size=(60, 3)) @ rs.normal(size=(3, 12)) + rs.normal(scale=0.3, size=(60, 12))
    X[rs.random(X.shape) < 0.1] = np.nan
    df = pd.DataFrame(X, columns=[f"g{k}" for k in range(12)])
    res = mipy.analyze(df, out=tmp_path / "c", methods="standard", repeats=1, mechanisms="MCAR",
                       show=False, verbose=False)
    assert res.prepared.id_col is None and not res.prepared.longitudinal
    assert {"LOCF", "MICE-Long", "MICE-LongMNAR"} <= set(res.skipped)
    assert "sd_ratio" not in res.figures
    assert res.completed is not None and not res.completed.isna().any().any()


def test_failing_method_does_not_stop(tmp_path, small, monkeypatch):
    from mipy import imputers

    def broken(X, ctx):
        raise RuntimeError("deliberate failure")
    monkeypatch.setitem(imputers.METHODS, "KNN", dataclasses.replace(imputers.METHODS["KNN"], fn=broken))
    res = mipy.analyze(small, out=tmp_path / "f", methods=["Mean", "KNN", "softImpute"], repeats=1,
                       mechanisms="MCAR", show=False, verbose=False, resume=False)
    assert any("KNN failed" in w and "deliberate failure" in w for w in res.warnings)
    assert res.completed_method in ("Mean", "softImpute")
    assert Path(res.report_path).is_file()


def test_bad_arguments(tmp_path, small):
    out = tmp_path / "never"
    with pytest.raises(ValueError, match="unknown method"):
        mipy.analyze(small, out=out, methods=["NoSuchMethod"], show=False, verbose=False)
    with pytest.raises(ValueError, match="unknown mechanism"):
        mipy.analyze(small, out=out, mechanisms=("MCAR", "XYZ"), show=False, verbose=False)
    with pytest.raises(ValueError, match="unknown setting"):
        mipy.analyze(small, out=out, settings={"bogus": 1}, show=False, verbose=False)
    with pytest.raises(ValueError, match="export: unknown method"):
        mipy.analyze(small, out=out, export="nothing", show=False, verbose=False)
    with pytest.raises(DataError):
        mipy.analyze(small, out=out, id_col="nope", show=False, verbose=False)
    assert not out.exists()                                 # no empty result folders left behind


def test_cli(tmp_path, capsys):
    from mipy.cli import main
    f = tmp_path / "ex.csv"
    assert main(["example", str(f), "--patients", "25", "--proteins", "20"]) == 0
    assert f.is_file()
    assert main(["check", str(f)]) == 0
    assert "Patient_ID" in capsys.readouterr().out
    assert main(["impute", str(f), "--method", "softImpute", "--output", str(tmp_path / "done.csv")]) == 0
    assert not pd.read_csv(tmp_path / "done.csv").filter(like="Prot_").isna().any().any()
    assert main(["analyze", str(f), "--out", str(tmp_path / "cli"), "--methods", "Mean,softImpute",
                 "--repeats", "1", "--mechanisms", "MCAR", "--quiet"]) == 0
    assert (tmp_path / "cli" / "report.html").is_file()
    assert main(["methods"]) == 0
    assert main(["check", str(tmp_path / "missing.csv")]) == 2          # clear error, no traceback
    assert "file not found" in capsys.readouterr().err


def test_id_and_visit_found_by_position(small):
    d = small.rename(columns={"Patient_ID": "Sample", "sampling_point": "Tp"})
    p = prepare_data(d)
    assert p.id_col == "Sample" and p.time_col == "Tp" and p.longitudinal
    assert "Tp" not in p.clinical_cols
    # a first column of measurements is not taken for an ID
    q = prepare_data(small.drop(columns=["Patient_ID", "sampling_point"]))
    assert q.id_col is None


def test_tiny_table_still_completes(tmp_path, small):
    d = small.head(12)[["Patient_ID", "sampling_point", "Prot_001", "Prot_002", "Prot_003"]]
    res = mipy.analyze(d, out=tmp_path / "t", methods="quick", repeats=1, mechanisms="MCAR",
                       show=False, verbose=False)
    assert any("could not be scored" in w for w in res.warnings)
    assert res.completed_method == "softImpute"
    assert not res.completed[res.prepared.features].isna().any().any()
    assert "robust default" in Path(res.report_path).read_text(encoding="utf-8")


def test_export_choice(tmp_path, small):
    res = mipy.analyze(small, out=tmp_path / "e", methods=["Mean", "softImpute"], repeats=1,
                       mechanisms="MCAR", export="MICE-LongMNAR", show=False, verbose=False)
    assert res.completed_method == "MICE-LongMNAR" and res.mi_method == "MICE-LongMNAR"
    assert "chosen with export" in Path(res.report_path).read_text(encoding="utf-8")


def test_row_order_does_not_matter(tmp_path, small):
    kw = dict(methods=["Mean", "KNN", "softImpute"], repeats=1, mechanisms="MCAR,MNAR",
              show=False, verbose=False)
    a = mipy.analyze(small, out=tmp_path / "a", **kw)
    b = mipy.analyze(small.sample(frac=1, random_state=3), out=tmp_path / "b", **kw)
    pd.testing.assert_frame_equal(a.summary.drop(columns="seconds"), b.summary.drop(columns="seconds"))
    feats = a.prepared.features
    shuffled_back = b.completed.loc[small.sample(frac=1, random_state=3).index.argsort()]
    np.testing.assert_allclose(shuffled_back[feats].to_numpy(), a.completed[feats].to_numpy(), rtol=1e-10)
