"""Tests for the dataset readers and the audit, on tiny fake files (no real data needed)."""

import csv
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from src.data import audit, parallel, readers


class _RunsCode:
    def __reduce__(self):
        return (os.system, ("echo this-must-never-run",))


def _runner_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "nr. sessions": [1, 0, 1, 1],
        "perceived exertion": [0.1, -0.01, 0.2, 0.3],
        "nr. sessions.1": [0, 1, 1, 0],
        "perceived exertion.1": [-0.01, 0.2, 0.1, 0.2],
        "Athlete ID": [0, 0, 1, 1],
        "injury": [0, 1, 0, 0],
        "Date": [0, 1, 0, 1],
    })


# ---------------------------------------------------------------- readers


def test_wesad_reader_refuses_pickles_that_run_code(tmp_path):
    path = tmp_path / "S99.pkl"
    path.write_bytes(pickle.dumps(_RunsCode(), protocol=2))
    with pytest.raises(pickle.UnpicklingError, match="refused"):
        readers.read_wesad_pickle(path)


def test_wesad_reader_loads_numpy_arrays(tmp_path):
    data = {
        "subject": "S99",
        "label": np.array([0, 1, 1, 2], dtype=np.int32),
        "signal": {"chest": {"ECG": np.zeros((4, 1))}, "wrist": {"BVP": np.ones((2, 1))}},
    }
    path = tmp_path / "S99.pkl"
    # Protocol 4, not 2: Python 3 at protocol 2 routes bytes through _codecs.encode, which the
    # real WESAD pickles (written by Python 2) never use, so the reader rightly refuses it.
    path.write_bytes(pickle.dumps(data, protocol=4))
    out = readers.read_wesad_pickle(path)
    assert out["subject"] == "S99"
    np.testing.assert_array_equal(out["label"], data["label"])
    assert out["signal"]["wrist"]["BVP"].shape == (2, 1)


def test_pamap2_reader_names_54_columns_and_rejects_other_widths(tmp_path):
    row = " ".join(["0.01", "1", "NaN"] + ["0.5"] * 51)
    good = tmp_path / "subject101.dat"
    good.write_text(f"{row}\n{row}\n")
    df = readers.read_pamap2_dat(good)
    assert list(df.columns) == readers.PAMAP2_COLUMNS
    assert df["heart_rate"].isna().all()

    bad = tmp_path / "subject102.dat"
    bad.write_text("0.01 1 70\n")
    with pytest.raises(ValueError, match="expected 54 columns"):
        readers.read_pamap2_dat(bad)


def test_trnote_parser_reads_timed_phases_in_both_spellings():
    text = (
        "Resting : from 00:00:00 to 00:18:30\n"
        "Linear_speed: from 00:26:20 to 00:29:30.5\n"
        "KARDIA1 : at 00:02:29\n"
        "1)before warm-up\n"
    )
    assert readers.parse_trnote(text) == [
        ("Resting", 0.0, 1110.0),
        ("Linear_speed", 1580.0, 1770.5),
    ]


def test_dem_parser_handles_mixed_separators_aliases_and_na():
    text = "Sex\tAge\tSmoking\tDisease\n1\t22   0\tNA\n"
    assert readers.parse_sportdb_dem(text) == {"Sex": 1.0, "Age": 22.0, "Smoker": 0.0,
                                               "Disease": None}


# ---------------------------------------------------------------- audit facts


def test_runner_facts_counts_athletes_injuries_and_negative_codes():
    facts = audit.runner_facts(_runner_frame())
    assert facts["athletes"] == 2
    assert facts["injury_counts"] == {"0": 3, "1": 1}
    assert facts["injured_athletes"] == 1
    assert facts["steps"] == 2 and facts["features_per_step"] == [2]
    assert facts["negative_values_by_feature"] == {"perceived exertion": 2}
    assert facts["most_common_negative_values"] == [[-0.01, 2]]


def test_compare_tables_ignores_row_order_and_counts_one_changed_cell():
    a = _runner_frame()
    assert audit.compare_tables(a, a.iloc[::-1], keys=("Athlete ID", "Date"))["identical"]
    b = a.copy()
    b.loc[0, "nr. sessions"] = 5
    out = audit.compare_tables(a, b, keys=("Athlete ID", "Date"))
    assert out["identical"] is False and out["differing_cells"] == 1


def test_mirror_check_finds_csv_and_tab_copies_identical(tmp_path):
    original, mirror = tmp_path / "lovdal", tmp_path / "kaggle"
    original.mkdir()
    mirror.mkdir()
    df = _runner_frame()
    df.to_csv(original / "day_approach_maskedID_timeseries.tab", sep="\t", index=False)
    df.iloc[::-1].to_csv(mirror / "day_approach_maskedID_timeseries.csv", index=False)
    out = audit.mirror_check(original, mirror)
    assert out["day"]["same_sha256"] is False and out["day"]["identical"] is True
    assert out["week"]["identical"] is None


def test_label_rule_probe_detects_a_label_made_by_a_rule():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(200), "b": rng.random(200)})
    probe = audit.label_rule_probe(X, (X["a"] > 0.5).astype(int))
    assert probe["cv_accuracy_mean"] > 0.95
    assert probe["features_used"][0] == "a"


def test_uniformity_separates_even_spread_from_bell_shape():
    rng = np.random.default_rng(0)
    assert audit.uniformity(rng.uniform(0, 1, 600))["p"] > 0.05
    assert audit.uniformity(rng.normal(0, 1, 600))["p"] < 0.001


def test_empty_or_missing_folder_is_not_downloaded(tmp_path):
    assert audit.audit_dataset(tmp_path, "runners")["status"] == "not downloaded"
    assert audit.audit_dataset(tmp_path / "missing", "runners")["status"] == "not downloaded"


# ---------------------------------------------------------------- outputs


def test_table_row_has_ten_columns_and_reports_pending_mirror():
    cfg = {"number": 2, "origin": "unknown", "mirror_of": "lovdal", "licence": "[VERIFY]",
           "role": "pending"}
    row = audit.table_row(cfg, {"status": "not downloaded", "table": {}})
    assert row.count("|") == 11
    assert "not downloaded" in row and "mirror check pending" in row


def test_replace_table_block_only_touches_the_marked_block():
    text = f"intro\n{audit.TABLE_START}\nold\n{audit.TABLE_END}\nrules"
    assert audit.replace_table_block(text, "new") == (
        f"intro\n{audit.TABLE_START}\nnew\n{audit.TABLE_END}\nrules"
    )
    with pytest.raises(ValueError):
        audit.replace_table_block("no markers here", "new")


# ---------------------------------------------------------------- speed changes keep results equal


def _write_pamap2(path, timestamps):
    rows = [" ".join([str(t), "1", "70" if i % 2 else "NaN"] + [f"{i}.5"] * 51)
            for i, t in enumerate(timestamps)]
    path.write_text("\n".join(rows) + "\n")


def test_pamap2_pyarrow_parser_matches_classic_parser(tmp_path):
    path = tmp_path / "subject101.dat"
    _write_pamap2(path, [0.01, 0.02, 0.03, 0.04])
    pd.testing.assert_frame_equal(
        readers.read_pamap2_dat(path, engine="c"),
        readers.read_pamap2_dat(path, engine="pyarrow"),
        check_exact=True,
    )


def test_pamap2_file_facts_hashes_the_parsed_bytes_and_flags_bad_timestamps(tmp_path):
    path = tmp_path / "Protocol" / "subject101.dat"
    path.parent.mkdir()
    _write_pamap2(path, [0.01, 0.02, 0.02, 0.01])
    out = audit.pamap2_file_facts(path)
    assert out["sha256"] == audit.sha256_file(path)
    assert out["facts"]["duplicate_timestamps"] == 2
    assert out["facts"]["backward_time_steps"] == 1


def test_hashing_reader_gives_the_file_sha_while_unpickling(tmp_path):
    path = tmp_path / "S99.pkl"
    data = {"subject": "S99", "label": np.arange(10, dtype=np.int32),
            "signal": {"chest": {}, "wrist": {}}}
    path.write_bytes(pickle.dumps(data, protocol=4) + b"trailing bytes")
    with open(path, "rb") as f:
        reader = audit.HashingReader(f)
        out = readers.read_wesad_pickle(reader)
        sha = reader.finish()
    assert sha == audit.sha256_file(path)
    np.testing.assert_array_equal(out["label"], data["label"])


def test_parallel_map_runs_in_processes_and_keeps_input_order():
    items = ["01:00:00", "00:00:03", "00:01:00"]
    assert parallel.parallel_map(readers.hms_to_seconds, items, workers=2) == [3600.0, 3.0, 60.0]


def test_pick_workers_respects_task_count_cap_and_memory():
    assert parallel.pick_workers(3) <= 3
    assert parallel.pick_workers(100, max_workers=2) <= 2
    assert parallel.pick_workers(100, bytes_per_task=10**18) == 1


def test_update_manifest_replaces_only_audited_datasets(tmp_path):
    path = tmp_path / "MANIFEST.csv"

    def row(dataset: str, file: str) -> dict:
        return {"dataset": dataset, "file": file, "source_url": "u", "licence": "l",
                "access_date": "2026-10-13", "sha256": "0" * 64}

    audit.update_manifest({"a": [row("a", "a/1.csv")], "b": [row("b", "b/1.csv")]}, path)
    audit.update_manifest({"a": [row("a", "a/2.csv")]}, path)
    with open(path, newline="", encoding="utf-8") as f:
        files = [r["file"] for r in csv.DictReader(f)]
    assert files == ["a/2.csv", "b/1.csv"]
