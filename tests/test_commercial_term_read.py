"""The Terms tab's `term` head: its grammar, and how the mirror trains it.

Synthetic lines only.
"""

from __future__ import annotations

import pytest

from app.core.commercial_term_read import (
    KIND_KEYS,
    KINDS,
    RELATION,
    canonical_term_verdict,
    encode_term_verdict,
    parse_term_verdict,
)
from app.core.pm_feedback import HEAD_REGISTRY, _threshold_for
from app.learning.human_labels import IngestReport, _judgment_rows


# ── grammar ──────────────────────────────────────────────────────────────────

def test_every_kind_has_its_keys_and_no_money_key():
    assert set(KINDS) == set(KIND_KEYS)
    for keys in KIND_KEYS.values():
        for k in keys:
            assert not any(m in k for m in ("price", "amount", "cost", "money")), k
            assert k != "rate"


@pytest.mark.parametrize("kind, values, want", [
    ("payment_term", {"net_days": 30, "timing": "arrears", "tax": "excluded"},
     "kind=payment_term;net_days=30;timing=arrears;tax=excluded"),
    ("contract_term", {"term_months": 36, "renewal": "auto", "notice_days": 60},
     "kind=contract_term;term_months=36;renewal=auto;notice_days=60"),
    ("pricing_assumption", {"min_hours": 2, "increment_min": 15, "after_hours_multiplier": 1.5,
                            "rate_lock_months": 12},
     "kind=pricing_assumption;min_hours=2;increment_min=15;after_hours_multiplier=1.5;rate_lock_months=12"),
    ("change_order_rule", {"notice_days": 5, "approval": "written"},
     "kind=change_order_rule;notice_days=5;approval=written"),
    ("commercial_total", {}, "kind=commercial_total"),
    ("not_a_term", {}, "kind=not_a_term"),
])
def test_encode_parse_round_trip(kind, values, want):
    enc = encode_term_verdict(kind, **values)
    assert enc == want
    parsed = parse_term_verdict(enc)
    assert parsed == {"kind": kind, **values}
    assert encode_term_verdict(**parsed) == enc


def test_keys_another_kind_takes_are_dropped():
    # net_days is a payment-term key; a contract term does not carry it.
    got = parse_term_verdict("kind=contract_term;term_months=12;net_days=30;approval=written")
    assert got == {"kind": "contract_term", "term_months": 12}
    assert encode_term_verdict("payment_term", net_days=15, renewal="auto") == "kind=payment_term;net_days=15"


def test_unknown_keys_and_money_are_dropped():
    got = parse_term_verdict("kind=pricing_assumption;min_hours=4;hourly_rate=95;price=100;foo=bar")
    assert got == {"kind": "pricing_assumption", "min_hours": 4}


def test_invalid_values_are_dropped_not_the_whole_answer():
    got = parse_term_verdict("kind=contract_term;term_months=0;renewal=sometimes;notice_days=-1")
    assert got == {"kind": "contract_term"}
    got = parse_term_verdict("kind=payment_term;net_days=abc;timing=ADVANCE;tax=at_invoice")
    assert got == {"kind": "payment_term", "timing": "advance", "tax": "at_invoice"}
    # notice_days may be zero; term_months may not.
    assert parse_term_verdict("kind=change_order_rule;notice_days=0") == {
        "kind": "change_order_rule", "notice_days": 0}


def test_kind_alone_is_valid():
    for kind in KINDS:
        assert parse_term_verdict(f"kind={kind}") == {"kind": kind}
    assert parse_term_verdict(" kind = not_a_term ") == {"kind": "not_a_term"}


def test_keys_on_a_keyless_kind_are_dropped():
    assert parse_term_verdict("kind=not_a_term;net_days=30") == {"kind": "not_a_term"}
    assert parse_term_verdict("kind=commercial_total;term_months=12") == {"kind": "commercial_total"}


@pytest.mark.parametrize("verdict", [
    "", None, "payment_term", "net_days=30", "kind=", "kind=invoice", "kind=task",
    "billing=fixed;pm_hours=2",
])
def test_no_known_kind_is_none(verdict):
    assert parse_term_verdict(verdict) is None
    assert canonical_term_verdict(verdict) is None


def test_encode_refuses_an_unknown_kind():
    with pytest.raises(ValueError):
        encode_term_verdict("invoice")


def test_canonical_orders_and_normalises():
    assert canonical_term_verdict("tax=excluded ; net_days=30.0;KIND=Payment_Term") == (
        "kind=payment_term;net_days=30;tax=excluded")


# ── registry ─────────────────────────────────────────────────────────────────

def test_term_head_is_its_own_extract_head():
    spec = HEAD_REGISTRY["term"]
    assert spec.relation == RELATION == "commercial_term_read"
    assert spec.kind == "atom"
    assert spec.mode == "extract"
    # commercial stays the kit shape.
    assert HEAD_REGISTRY["commercial"].relation == "commercial_terms"
    for scope in ("deal", "global"):
        assert _threshold_for("term", scope) == _threshold_for("commercial", scope)


def test_term_sits_in_a_universal_head_and_commercial_in_the_company_one():
    """`term` is what a line says (universal); `commercial` is OUR kit shape,
    which #394 moved to the company head conduct.constants."""
    import json
    from pathlib import Path

    heads = json.loads((Path(__file__).resolve().parents[1] / "app/core/label_heads.json").read_text())
    owner = [h for h in heads["heads"] if "term" in h.get("judgments", [])]
    assert len(owner) == 1
    assert owner[0]["key"] == "content.claims"
    assert owner[0]["judgments"] == ["norm", "term", "terminology"]
    assert owner[0]["layer"] == "universal"
    kit = [h for h in heads["heads"] if "commercial" in h.get("judgments", [])]
    assert [h["key"] for h in kit] == ["conduct.constants"]
    assert kit[0]["layer"] == "company"


# ── mirror ───────────────────────────────────────────────────────────────────

def _j(head, verdict, atom_type, text="Synthetic line for a test of the mirror."):
    return {
        "head": head, "verdict": verdict, "text": text, "target_key": "atom:a1",
        "target": {"head": head, "targetKey": "atom:a1", "text": text,
                   "a": {"atomId": "a1", "atomType": atom_type, "text": text}},
        "labeler": "person@example.com", "judged_at": "2026-10-07T00:00:00Z",
    }


def _rows(*judgments):
    report = IngestReport()
    rows = _judgment_rows({"judgments": list(judgments)}, "deal-x", "train", report)
    return rows, report


def test_term_judgment_trains_canonical_verdict():
    rows, report = _rows(_j("term", "timing=arrears;net_days=45;kind=payment_term;rate=90", "payment_term"))
    [row] = [r for r in rows if r["relation"] == "commercial_term_read"]
    assert row["label"] == "kind=payment_term;net_days=45;timing=arrears"
    assert not report.skipped


def test_unparseable_term_judgment_is_skipped_with_a_reason():
    rows, report = _rows(_j("term", "net 30", "payment_term"))
    assert rows == []
    [(why, n)] = report.skipped.items()
    assert n == 1 and why.startswith("term verdict does not parse")


def test_commercial_on_a_term_line_is_skipped():
    for atom_type in ("payment_term", "contract_term", "change_order_rule", "commercial_total"):
        rows, report = _rows(_j("commercial", "billing=fixed;pm_hours=2", atom_type))
        assert rows == [], atom_type
        assert report.skipped == {"commercial verdict on a term line: re-answer under term": 1}


def test_commercial_on_a_task_line_still_trains():
    rows, report = _rows(_j("commercial", "billing=fixed;pm_hours=2", "task"))
    [row] = [r for r in rows if r["relation"] == "commercial_terms"]
    assert row["label"] == "billing=fixed;pm_hours=2"
    assert not report.skipped


def test_commercial_without_an_atom_type_is_not_guessed_from_text():
    j = _j("commercial", "billing=fixed", "", text="Payment terms are net thirty days from invoice.")
    j["target"] = {}
    rows, _ = _rows(j)
    assert [r["relation"] for r in rows] == ["commercial_terms"]
