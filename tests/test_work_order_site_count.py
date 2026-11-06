"""PUR-10: which document shapes leave the work order's site_count null, and why.

All shapes are synthetic reproductions of what a two-device, no-onsite-hands
quote (deal 010095) could plausibly look like. No real documents are read.
"""
from __future__ import annotations

import json

import pytest

from app.core import work_order
from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.work_order_site_count import backfill_work_order_site_count, resolve_site_count


def _atom(text, *, aid=None, art="art-quote", fn="quote.pdf", atom_type=AtomType.scope_item, value=None):
    return EvidenceAtom(
        id=aid or f"a-{abs(hash((text, art))) % 10**9}",
        project_id="p1",
        artifact_id=art,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        authority_class=AuthorityClass.customer_current_authored,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version="1",
        source_refs=[SourceRef(id=f"sr-{art}", artifact_id=art, artifact_type="txt", filename=fn,
                               locator={"line": 1}, extraction_method="t", parser_version="1")],
        value=value or {},
    )


DEVICE_LINES = [
    _atom("Qty 2  Secure console server, 16-port  $2,450.00"),
    _atom("Remote configuration and staging of console servers, customer racks devices"),
]


# ── shapes that genuinely name no site: null, with a reason, never 0 ──────────

def test_device_lines_only_no_address_is_unknown_not_zero():
    r = resolve_site_count(DEVICE_LINES, no_onsite_hands=True)
    assert r["site_count"] is None
    assert r["status"] == "unknown"
    assert "no site" in r["reason"] and "remote" in r["reason"]


def test_bill_to_and_ship_to_differ_is_ambiguous_not_guessed():
    atoms = DEVICE_LINES + [
        _atom("Bill To: Acme Corp, 100 Main St, Irvine, CA 92618"),
        _atom("Ship To: Acme Lab, 5 Oak Ave, Austin, TX 78701"),
    ]
    r = resolve_site_count(atoms)
    assert r["site_count"] is None and r["status"] == "unknown"
    assert "2 distinct addresses" in r["reason"]
    assert len(r["evidence"]) == 2


# ── shapes the old code left null although the documents carry a site ────────

def test_single_header_address_is_one_site_with_provenance():
    hdr = _atom("Quote for Acme Corp, 100 Main St, Irvine, CA 92618", aid="hdr")
    r = resolve_site_count(DEVICE_LINES + [hdr], no_onsite_hands=True)
    assert r["site_count"] == 1
    assert r["status"] == "single_address"
    assert r["evidence"][0]["atom_id"] == "hdr"


def test_same_address_as_bill_to_and_ship_to_is_one_site():
    atoms = DEVICE_LINES + [
        _atom("Bill To: Acme Corp, 100 Main St, Irvine, CA 92618"),
        _atom("Ship To: Acme Corp, 100 Main St, Irvine, CA 92618-1234"),
    ]
    assert resolve_site_count(atoms)["site_count"] == 1


def test_resolved_physical_site_counts():
    site = _atom("100 Main St, Irvine, CA 92618", atom_type=AtomType.physical_site,
                 value={"city": "Irvine", "state": "CA", "zip": "92618", "street_address": "100 Main St"})
    r = resolve_site_count(DEVICE_LINES + [site])
    assert (r["site_count"], r["status"]) == (1, "resolved_sites")


def test_stated_count_in_text_wins_over_addresses():
    r = resolve_site_count(DEVICE_LINES + [_atom("Install at both 2 locations")])
    assert (r["site_count"], r["status"]) == (2, "stated_in_text")


def test_structured_site_count_field():
    hdr = _atom("Qty of Sites 1", value={"kind": "deal_header", "fields": {"site_count": "1"}})
    r = resolve_site_count(DEVICE_LINES + [hdr])
    assert (r["site_count"], r["status"]) == (1, "declared_field")


def test_llm_number_is_kept():
    assert resolve_site_count(DEVICE_LINES, llm_site_count=3)["status"] == "work_order_llm"


def test_vendor_letterhead_address_is_not_a_site():
    from app.core.vendor_site_ban import _PURTERA_ADDRESS_MARKERS

    marker = sorted(_PURTERA_ADDRESS_MARKERS)[0]
    r = resolve_site_count(DEVICE_LINES + [_atom(f"{marker} Alpharetta, GA 30009")])
    assert r["site_count"] is None


# ── end to end through apply_work_order ─────────────────────────────────────

@pytest.fixture
def stage(monkeypatch):
    class D:
        verdict, source = "about_this_job", "store"

    monkeypatch.setattr("app.core.decide.decide", lambda *a, **k: D())
    wo = {
        "work_lines": [{"work": "remotely configure two console servers", "object": "console server",
                        "count": 2, "unit": "device"}],
        "site_count": None, "after_hours": False, "no_onsite_hands": True,
        "customer_supplies_equipment": False, "one_line_summary": "Configure 2 console servers remotely",
    }
    monkeypatch.setattr("app.core.llm_client.complete", lambda *a, **k: json.dumps(wo))


def _minted(atoms):
    return [a for a in atoms if (a.value or {}).get("backfill_reason") == "work_order"]


def test_apply_work_order_fills_null_from_single_address(stage):
    atoms = DEVICE_LINES + [_atom("Ship To: Acme Corp, 100 Main St, Irvine, CA 92618")]
    atoms, n, report = work_order.apply_work_order(list(atoms), project_id="p1", deal_name="d")
    assert n == 1
    assert report["summary"]["site_count"] == 1
    assert report["summary"]["site_count_status"] == "single_address"
    assert _minted(atoms)[0].value["site_count_evidence"]


def test_apply_work_order_keeps_null_with_reason_when_no_site(stage):
    atoms, _, report = work_order.apply_work_order(list(DEVICE_LINES), project_id="p1", deal_name="d")
    assert report["summary"]["site_count"] is None
    assert report["summary"]["site_count_status"] == "unknown"
    assert _minted(atoms)[0].value["site_count_reason"]


def test_site_found_after_geo_fallback_is_backfilled(stage):
    """The ordering miss: physical_site minted after work_order froze None."""
    atoms, _, _ = work_order.apply_work_order(list(DEVICE_LINES), project_id="p1", deal_name="d")
    assert _minted(atoms)[0].value["site_count"] is None
    atoms.append(_atom("Irvine, CA 92618", atom_type=AtomType.physical_site, art="art-notes",
                       value={"city": "Irvine", "state": "CA", "zip": "92618"}))
    assert backfill_work_order_site_count(atoms) == 1
    v = _minted(atoms)[0].value
    assert (v["site_count"], v["site_count_status"]) == (1, "resolved_sites")
