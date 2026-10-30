"""The delivery read (owner 2026-10-29/30): the project's Delivery sources page.

It reads the whole deal, from the first line ("start at 0") with no quoting cut,
asks the delivery heads, and carries their answers -- never re-decided -- in
``c3_brief.delivery``. The quoting side keeps its pre-quote cut.
"""
from __future__ import annotations

import json

import pytest

from app.core import c3_type_base as tb
from app.core.orbitbrief_envelope import PARSER_MANIFEST_SIDECAR, _load_manifest_read


@pytest.fixture(autouse=True)
def _clean_deal():
    yield
    tb._DEAL.clear()


def _deal(read=None, as_of=None):
    tb.set_deal([{"doc_id": "d1", "kind": "email"}], {}, as_of=as_of, read=read)


def test_flag_and_view(monkeypatch):
    monkeypatch.delenv("SOWSMITH_C3_VIEW", raising=False)
    _deal("delivery", as_of="2026-01-01T00:00:00Z")
    assert tb.delivery_read()
    assert tb.view_name() == "planning"
    assert tb._DEAL["as_of"] is None
    _deal(None, as_of="2026-01-01T00:00:00Z")
    assert not tb.delivery_read() and tb.view_name() == "quoting"
    _deal(None)
    assert not tb.delivery_read() and tb.view_name() == "planning"


def test_no_quoting_cut_on_a_delivery_read(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_QUOTE_PREQUOTE_ONLY", "1")
    _deal("delivery")
    tb._DEAL["quote_out_why"] = {"a1": "after_cut"}
    assert tb.out_of_quote() == {}
    _deal(None)
    tb._DEAL["quote_out_why"] = {"a1": "after_cut"}
    assert tb.out_of_quote() == {"a1": "after_cut"}


def test_delivery_heads_are_asked_only_on_a_delivery_read(monkeypatch):
    monkeypatch.delenv("SOWSMITH_C3_VIEW", raising=False)
    _deal("delivery")
    asked = tb.brief_line_heads()
    for h in ("read:start_date", "read:triggered_task", "read:chase", "read:task_owner_role"):
        assert h in asked
    _deal(None)
    assert "read:start_date" not in tb.brief_line_heads()


def test_payload_carries_answers_as_read():
    _deal("delivery")
    tb._DEAL["reads"] = {"a1": {"read:start_date": ("2026-11-03", 0.81), "read:labor_hours": ("4", 0.9)},
                         "a2": {"read:noise_class": (None, 0.0)}}
    p = tb.delivery_payload()
    assert p["read"] == "delivery"
    assert p["lines"] == {"a1": {"read:start_date": ["2026-11-03", 0.81]}}
    assert p["trust"]["read:start_date"] in ("unproven", "weak", "thin", "one_sided", "ready", "live", "no_labels")
    assert set(p["trusted"]) <= set(p["heads"])


def test_manifest_read(tmp_path):
    (tmp_path / PARSER_MANIFEST_SIDECAR).write_text(json.dumps({"context": {"read": "Delivery"}}))
    assert _load_manifest_read(tmp_path) == "delivery"
    (tmp_path / PARSER_MANIFEST_SIDECAR).write_text(json.dumps({"context": {}}))
    assert _load_manifest_read(tmp_path) is None


def test_set_valued_and_odd_readings():
    assert tb._delivery_answer(["present", 0.8]) == ["present", 0.8]
    assert tb._delivery_answer([["site_contact", "access_note"], {"site_contact": 0.9, "access_note": 0.6, "x": 0.1}]) == [
        "site_contact, access_note", 0.6]
    assert tb._delivery_answer([[], {"a": 0.2}]) is None
    assert tb._delivery_answer([None, 0.9]) is None
    assert tb._delivery_answer(["x", {"bad": 1}]) is None
    _deal("delivery")
    tb._DEAL["reads"] = {"a1": {"read:needs_artifact": [["present"], {"present": 0.7}]}}
    assert tb.delivery_payload()["lines"] == {"a1": {"read:needs_artifact": ["present", 0.7]}}


def test_delivery_starts_at_the_signed_sow():
    from c3_record.view import delivery_start

    docs = [{"kind": "sow_draft", "arrival": 100.0}, {"kind": "purchase_order", "arrival": 200.0},
            {"kind": "sow_signed", "arrival": 300.0}, {"kind": "sow_signed", "arrival": 250.0}]
    assert delivery_start(docs, 90.0) == (250.0, "signed SOW")
    assert delivery_start(docs[:2], 90.0) == (200.0, "purchase order")
    assert delivery_start(docs[:1], 90.0) == (90.0, "quoting cut")
    assert delivery_start([], float("inf"))[0] == float("inf")
