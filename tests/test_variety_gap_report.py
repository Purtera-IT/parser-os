"""Variety gap report: train-vs-holdout label AND provenance.variety coverage
on a synthetic TrainingLog db (tools/variety_gap_report.py, pure stdlib).

Doctrine under test: variety-over-volume — the tool must surface exactly the
classes/varieties the deal-held-out split disagrees about (the chart-class
failure mode), never balanced coverage, and must stay read-only."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tools.variety_gap_report import (
    UNTAGGED,
    build_report,
    coverage,
    load_rows,
    main,
    rank_gaps,
)


def _db(tmp_path: Path, rows) -> Path:
    """rows: (relation, label, variety_or_None, split)"""
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE training_rows ("
        "id TEXT PRIMARY KEY, relation TEXT, label TEXT, "
        "provenance TEXT, split TEXT)"
    )
    for i, (relation, label, variety, split) in enumerate(rows):
        prov = json.dumps({"variety": variety}) if variety is not None else "{}"
        con.execute(
            "INSERT INTO training_rows VALUES (?,?,?,?,?)",
            (f"trn_{i}", relation, label, prov, split),
        )
    con.commit()
    con.close()
    return db


def test_load_rows_filters_relation_and_parses_variety(tmp_path):
    db = _db(tmp_path, [
        ("pdf_image_kind", "chart", "bar_chart", "holdout"),
        ("pdf_image_kind", "photo", None, "train"),       # no variety tag
        ("atom_type", "scope_item", "x", "train"),        # other relation: out
    ])
    rows = load_rows(db, "pdf_image_kind")
    assert len(rows) == 2
    assert {r["variety"] for r in rows} == {"bar_chart", UNTAGGED}
    assert {r["split"] for r in rows} == {"train", "holdout"}


def test_coverage_counts_both_axes(tmp_path):
    db = _db(tmp_path, [
        ("pdf_image_kind", "chart", "bar_chart", "holdout"),
        ("pdf_image_kind", "chart", "bar_chart", "holdout"),
        ("pdf_image_kind", "chart", "line_chart", "train"),
        ("pdf_image_kind", "photo", "rack", "train"),
    ])
    rows = load_rows(db, "pdf_image_kind")
    lab = coverage(rows, "label")
    assert lab["chart"] == {"train": 1, "holdout": 2}
    assert lab["photo"] == {"train": 1, "holdout": 0}
    var = coverage(rows, "variety")
    assert var["bar_chart"] == {"train": 0, "holdout": 2}
    assert var["line_chart"] == {"train": 1, "holdout": 0}


def test_rank_gaps_flags_thin_sides_and_skips_balanced():
    cov = {
        "bar_chart": {"train": 0, "holdout": 20},   # the chart-class shape
        "rack": {"train": 30, "holdout": 1},
        "screenshot": {"train": 5, "holdout": 5},   # balanced -> not a gap
        "label_plate": {"train": 2, "holdout": 2},  # balanced -> not a gap
    }
    gaps = rank_gaps(cov, "variety")
    assert [g["value"] for g in gaps] == ["bar_chart", "rack"]
    bar = gaps[0]
    assert bar["thin_side"] == "train"      # eval'd but untrained
    assert bar["gap_score"] == 20.0         # 20 / (1 + 0)
    assert gaps[1]["thin_side"] == "holdout"  # trained but uneval'd


def test_rank_gaps_orders_by_severity():
    cov = {
        "a": {"train": 1, "holdout": 10},   # 10/2 = 5.0
        "b": {"train": 0, "holdout": 30},   # 30/1 = 30.0
        "c": {"train": 2, "holdout": 4},    # 4/3 -> 1.333
    }
    gaps = rank_gaps(cov, "label")
    assert [g["value"] for g in gaps] == ["b", "a", "c"]


def test_build_report_names_grade_next_varieties(tmp_path):
    db = _db(tmp_path, [
        ("pdf_image_kind", "chart", "bar_chart", "holdout"),
        ("pdf_image_kind", "chart", "bar_chart", "holdout"),
        ("pdf_image_kind", "photo", "rack", "train"),
        ("pdf_image_kind", "photo", "rack", "train"),
        ("pdf_image_kind", "photo", "rack", "train"),
    ])
    rows = load_rows(db, "pdf_image_kind")
    report = build_report(rows, relation="pdf_image_kind", top=10)
    assert "GRADE THESE NEXT" in report
    assert "EVAL BLIND SPOTS" in report
    assert "bar_chart" in report   # holdout-only variety -> grade next
    assert "rack" in report        # train-only variety -> eval blind spot
    assert "5 rows: 3 train / 2 holdout" in report


def test_main_smoke_and_read_only(tmp_path, capsys):
    db = _db(tmp_path, [
        ("pdf_image_kind", "chart", "bar_chart", "holdout"),
        ("pdf_image_kind", "photo", "rack", "train"),
    ])
    before = db.read_bytes()
    assert main([str(db)]) == 0
    out = capsys.readouterr().out
    assert "VARIETY GAP REPORT" in out
    assert db.read_bytes() == before   # never writes the db


def test_main_errors_cleanly(tmp_path, capsys):
    assert main([str(tmp_path / "missing.db")]) == 1
    db = _db(tmp_path, [("atom_type", "scope_item", None, "train")])
    assert main([str(db)]) == 1        # no pdf_image_kind rows
