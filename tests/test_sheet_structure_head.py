"""PUR-21 / PUR-51 / PUR-52: fallthrough abstention, labelling census and the
structural sheet head. Synthetic fixtures only."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import sheet_structure_head as ssh
from app.core.sheet_fallthrough_census import CUSTOMER_PLACEHOLDER, run_census, write_outputs
from app.parsers.sheet_classifier import (
    MATCH_FALLTHROUGH,
    MATCH_HEAD,
    MATCH_POSITIVE,
    SheetDestination,
    SheetRole,
    classify_sheet,
)
from tests.fixtures.synthetic_sheet_kinds import make_rows, write_corpus

REPO = Path(__file__).resolve().parents[1]
_KIND_TO_LABEL = {"pricing": "pricing", "contact": "contact_list", "schedule": "schedule",
                  "junk": "junk", "scope": "scope"}


@pytest.fixture(autouse=True)
def _no_ambient_head(tmp_path, monkeypatch):
    monkeypatch.setenv(ssh.HEAD_ENV, str(tmp_path / "absent_head.json"))
    monkeypatch.delenv("SHEET_FALLTHROUGH_LEGACY_SCOPE", raising=False)
    ssh.clear_cache()
    yield
    ssh.clear_cache()


def _labelled(root: Path, customers=("cust_a", "cust_b", "cust_c", "cust_d")) -> list[dict]:
    write_corpus(root, customers=customers)
    res = run_census(root, customer_key_mode="parent_dir")
    for r in res.records:
        r["label"] = _KIND_TO_LABEL[r["workbook_file"].split("_")[1]]
    return res.records


# ── 1. explicit abstention ──────────────────────────────────────────


def test_customer_export_is_unclassified_not_scope():
    rows = [["Customer Code", "Order Date", "Status"]] + [["C1", "2029-01-01", "Open"]] * 50
    c = classify_sheet("Export", rows)
    assert c.role is SheetRole.UNCLASSIFIED
    assert c.destination is SheetDestination.REVIEW
    assert c.match == MATCH_FALLTHROUGH and c.is_fallthrough
    assert c.suppress is True


def test_positive_rules_record_positive_match():
    c = classify_sheet("x", [["Site", "Device", "Qty"], ["HQ", "Cam", "2"]])
    assert c.role is SheetRole.SCOPE and c.match == MATCH_POSITIVE
    c = classify_sheet("x", [[None], [""]])
    assert c.role is SheetRole.EMPTY and c.match == MATCH_POSITIVE


def test_legacy_env_restores_default_scope(monkeypatch):
    monkeypatch.setenv("SHEET_FALLTHROUGH_LEGACY_SCOPE", "1")
    c = classify_sheet("x", [["Foo", "Bar"], ["1", "2"], ["3", "4"]])
    assert c.role is SheetRole.SCOPE and c.reason == "default_scope" and c.is_fallthrough


def test_unclassified_xlsx_sheet_is_a_visible_marker_not_scope_items(tmp_path):
    import openpyxl

    from app.core.schemas import AtomType
    from app.parsers.xlsx_parser import XlsxParser

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Export"
    for r in make_rows("junk", 7):
        ws.append(r)
    ws2 = wb.create_sheet("Scope")
    for r in [["Site", "Device", "Qty"], ["HQ", "Camera", "4"], ["Annex", "Camera", "2"]]:
        ws2.append(r)
    p = tmp_path / "synthetic.xlsx"
    wb.save(p)
    atoms = XlsxParser().parse_artifact(project_id="T", artifact_id="art_syn", path=p)
    export_atoms = [a for a in atoms if any((s.locator or {}).get("sheet") == "Export" for s in a.source_refs)]
    assert [a.atom_type for a in export_atoms] == [AtomType.dropped_sheet]
    assert export_atoms[0].value["unclassified"] is True
    assert "sheet_unclassified" in export_atoms[0].review_flags
    assert any(a.atom_type != AtomType.dropped_sheet for a in atoms
               if any((s.locator or {}).get("sheet") == "Scope" for s in a.source_refs))


# ── 2. census / labelling set ───────────────────────────────────────


def test_census_writes_structure_only_labelling_records(tmp_path):
    write_corpus(tmp_path / "wbs", customers=("cust_a",), per_kind=1)
    (tmp_path / "wbs" / "cust_a" / "positive.csv").write_text("Site,Device,Qty\nHQ,Cam,2\n")
    res = run_census(tmp_path / "wbs")
    rep = res.report()
    assert rep["total_sheets"] == 6
    assert rep["fallthrough_rules_only"] == 5
    assert rep["fallthrough_rate_rules_only"] == round(5 / 6, 4)
    paths = write_outputs(res, tmp_path / "out")
    recs = [json.loads(x) for x in paths["jsonl"].read_text().splitlines()]
    assert len(recs) == 5
    for r in recs:
        assert r["label"] == "" and r["customer_key"] == CUSTOMER_PLACEHOLDER
        assert r["headers"] and r["column_types"] and r["row_count"] > 0
        assert "sample_rows" not in r
        assert set(r["label_options"].split("|")) == set(ssh.LABEL_CLASSES)
    assert "sheet_id" in paths["csv"].read_text().splitlines()[0]
    assert json.loads(paths["report"].read_text())["labelling_records"] == 5


def test_census_samples_are_opt_in(tmp_path):
    write_corpus(tmp_path / "wbs", customers=("c",), per_kind=1)
    res = run_census(tmp_path / "wbs", include_samples=2, customer_key_mode="parent_dir")
    assert all(len(r["sample_rows"]) == 2 and r["customer_key"] == "c" for r in res.records)


def test_labels_outside_class_set_are_rejected(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps({"sheet_id": "x", "label": "scopee"}) + "\n")
    with pytest.raises(ValueError):
        ssh.read_labels(p)
    p.write_text(json.dumps({"sheet_id": "x", "label": ""}) + "\n")
    assert ssh.read_labels(p) == []


# ── 3. structural head ──────────────────────────────────────────────


def test_no_head_means_abstention():
    assert ssh.load_head() is None
    rows = make_rows("contact_list", 3)
    assert classify_sheet("Contacts", rows).role is SheetRole.UNCLASSIFIED


def test_head_ignores_the_sheet_name(tmp_path, monkeypatch):
    head = ssh.train_from_records(_labelled(tmp_path / "wbs"))
    head.save(tmp_path / "head.json")
    monkeypatch.setenv(ssh.HEAD_ENV, str(tmp_path / "head.json"))
    ssh.clear_cache()
    rows = make_rows("junk", 999)
    a = classify_sheet("Customer Export", rows, use_learned_store=False)
    b = classify_sheet("Site Scope BOM", rows, use_learned_store=False)
    assert a.role == b.role == SheetRole.REFERENCE
    assert a.match == MATCH_HEAD and a.confidence == b.confidence


def test_head_routes_learned_kinds_and_abstains_below_threshold(tmp_path, monkeypatch):
    recs = _labelled(tmp_path / "wbs")
    head = ssh.train_from_records(recs)
    pred = head.predict_rows(make_rows("pricing", 4242))
    assert pred is not None and pred.label == "pricing" and pred.role == "catalog"
    strict = ssh.train_from_records(recs, threshold=0.999)
    assert strict.predict_rows([["Zeta", "Omega"], ["?", "?"]]) is None
    strict.save(tmp_path / "strict.json")
    monkeypatch.setenv(ssh.HEAD_ENV, str(tmp_path / "strict.json"))
    ssh.clear_cache()
    c = classify_sheet("x", [["Zeta", "Omega"], ["?", "?"], ["?", "?"]], use_learned_store=False)
    assert c.role is SheetRole.UNCLASSIFIED


def test_head_json_roundtrip_and_corrupt_head_abstains(tmp_path, monkeypatch):
    head = ssh.train_from_records(_labelled(tmp_path / "wbs", customers=("a", "b")))
    head.save(tmp_path / "h.json")
    again = ssh.SheetStructureHead.from_json(json.loads((tmp_path / "h.json").read_text()))
    rows = make_rows("schedule", 5)
    assert again.predict_rows(rows).label == head.predict_rows(rows).label
    (tmp_path / "bad.json").write_text("{not json")
    monkeypatch.setenv(ssh.HEAD_ENV, str(tmp_path / "bad.json"))
    ssh.clear_cache()
    assert ssh.load_head() is None


def test_customer_grouped_holdout_reports_per_class_and_fallthrough(tmp_path):
    recs = _labelled(tmp_path / "wbs")
    rep = ssh.evaluate_customer_holdout(recs, holdout_customers=["cust_d"])
    assert rep["n_test"] == 20
    assert rep["folds"][0]["n_train"] == 60  # cust_d never seen in training
    assert rep["fallthrough_before"] == 1.0
    assert rep["fallthrough_after"] < rep["fallthrough_before"]
    assert set(rep["per_class_accuracy"]) == {"scope", "pricing", "schedule", "contact_list", "junk"}
    assert rep["accuracy"] >= 0.8
    loco = ssh.evaluate_customer_holdout(recs)
    assert len(loco["folds"]) == 4 and loco["n_test"] == 80


def test_scripts_end_to_end_on_synthetic_corpus(tmp_path):
    write_corpus(tmp_path / "wbs", customers=("cust_a", "cust_b", "cust_c"))
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(REPO / "scripts/sheet_fallthrough_labels.py"), "--root",
                    str(tmp_path / "wbs"), "--out", str(out), "--customer-key", "parent_dir"],
                   check=True, capture_output=True, cwd=REPO)
    lines = [json.loads(x) for x in (out / "sheet_labels.jsonl").read_text().splitlines()]
    # No labels yet -> training refuses; the head stays absent (abstention).
    p = subprocess.run([sys.executable, str(REPO / "scripts/train_sheet_head.py"), "--labels",
                        str(out / "sheet_labels.jsonl")], capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 2
    for r in lines:
        r["label"] = _KIND_TO_LABEL[r["workbook_file"].split("_")[1]]
    (out / "labelled.jsonl").write_text("\n".join(json.dumps(r) for r in lines))
    p = subprocess.run([sys.executable, str(REPO / "scripts/train_sheet_head.py"), "--labels",
                        str(out / "labelled.jsonl"), "--holdout-customer", "cust_c",
                        "--report", str(out / "eval.json"), "--out", str(out / "head.json")],
                       capture_output=True, text=True, cwd=REPO)
    assert p.returncode == 0, p.stderr
    rep = json.loads((out / "eval.json").read_text())
    assert "per_class_accuracy" in rep and "fallthrough_after" in rep
    assert ssh.load_head(out / "head.json") is not None
