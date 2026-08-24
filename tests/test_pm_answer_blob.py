"""The answer ledger reaches the compile — and never breaks it when it can't."""
from __future__ import annotations

import json

import pytest

from app.core import pm_answer_blob as pab


def _row(**over):
    base = {
        "action": "answered",
        "deal_id": "d1",
        "rule_id": "mode.av_install.drywall_ownership",
        "question_text": "Who owns drywall patching?",
        "edited_text": "Customer's GC patches and paints.",
        "created_at": "2026-08-03T12:00:00Z",
    }
    base.update(over)
    return json.dumps(base)


def test_parse_ledger_survives_a_truncated_line():
    # One bad line must not cost every answer the PM has given on this deal.
    text = "\n".join([_row(), "{not json", "", _row(rule_id="r2")])
    rows = pab.parse_ledger(text)
    assert len(rows) == 2
    assert rows[0]["action"] == "answered"


def test_parse_ledger_ignores_non_object_rows():
    assert pab.parse_ledger('"a string"\n[1,2]\n' + _row()) == pab.parse_ledger(_row())


def test_disabled_by_default_so_normal_compiles_are_unchanged(monkeypatch):
    monkeypatch.delenv("SOWSMITH_FEEDBACK_BLOB", raising=False)
    assert pab._container_client() is None
    assert pab.load_answer_events("d1") == []
    assert pab.load_pm_answer_atoms(project_id="p1", deal_id="d1") == []


def test_enabled_without_a_connection_string_is_a_no_op(monkeypatch):
    monkeypatch.setenv("SOWSMITH_FEEDBACK_BLOB", "1")
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    assert pab.load_pm_answer_atoms(project_id="p1", deal_id="d1") == []


def test_blob_failure_never_breaks_the_compile(monkeypatch):
    class Boom:
        def download_blob(self, *a, **k):
            raise RuntimeError("network is down")

    monkeypatch.setattr(pab, "_container_client", lambda: Boom())
    assert pab.load_answer_events("d1") == []
    assert pab.load_pm_answer_atoms(project_id="p1", deal_id="d1") == []


def test_answers_become_atoms_the_compile_can_use(monkeypatch):
    class Fake:
        def download_blob(self, name):
            assert name == "deals/d1/orbitbrief/latest/question_feedback.jsonl"
            return self

        def readall(self):
            return ("\n".join([_row(), _row(action="dismiss", rule_id="r9")])).encode("utf-8")

    monkeypatch.setattr(pab, "_container_client", lambda: Fake())
    atoms = pab.load_pm_answer_atoms(project_id="p1", deal_id="d1")
    assert len(atoms) == 1, "only `answered` rows are evidence"
    assert "patches and paints" in atoms[0].raw_text
    assert atoms[0].authority_class.value == "pm_confirmed"


def test_atoms_are_stable_across_recompiles(monkeypatch):
    class Fake:
        def download_blob(self, name):
            return self

        def readall(self):
            return _row().encode("utf-8")

    monkeypatch.setattr(pab, "_container_client", lambda: Fake())
    first = pab.load_pm_answer_atoms(project_id="p1", deal_id="d1")
    second = pab.load_pm_answer_atoms(project_id="p1", deal_id="d1")
    assert [a.id for a in first] == [a.id for a in second]


@pytest.mark.parametrize("deal_id", ["", None])
def test_no_deal_id_is_a_no_op(deal_id):
    assert pab.load_answer_events(deal_id or "") == []
