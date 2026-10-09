"""PUR-62: site count carries its receipt (synthetic fixtures only)."""

from __future__ import annotations

from app.core.site_count_receipt import build_site_count_receipt
from app.core.site_count_reconcile import reconcile_site_count


def atom(i, text, doc="doc-1", name="synthetic_sow.pdf", page=2):
    return {
        "id": f"atom-{i}",
        "raw_text": text,
        "artifact_id": doc,
        "source_refs": [{"artifact_id": doc, "filename": name, "locator": {"page": page}}],
    }


ATOMS = [
    atom(1, "Install badge readers at 4 sites across the region.", page=1),
    atom(2, "Pricing: $120 per site x 4 sites = $480.", page=3),
    atom(3, "Survey all four locations before kickoff.", name="synthetic_email.eml", page=None),
    atom(4, "Unrelated sentence about cabling."),
]


def test_read_receipt_has_facts_with_document_page_and_span():
    rec = reconcile_site_count(ATOMS, resolved_sites=4)
    r = build_site_count_receipt(ATOMS, rec)
    assert r["status"] == "read" and r["value"] == 4 and r["rule"]
    assert 1 <= len(r["facts"]) <= 3
    f = r["facts"][0]
    assert f["atom_id"] == "atom-1" and f["document_name"] == "synthetic_sow.pdf" and f["page"] == 1
    assert f["quote"] == f["text"][f["span"]["start"]:f["span"]["end"]]
    assert "4" in f["quote"]
    assert r["abstention_needs"] == []


def test_dollar_amounts_are_not_facts():
    r = build_site_count_receipt(ATOMS, reconcile_site_count(ATOMS, 4))
    assert all("$480" not in (f["quote"] or "") for f in r["facts"])


def test_disagreement_is_said_in_reasoning():
    r = build_site_count_receipt(ATOMS, reconcile_site_count(ATOMS, 2))
    assert r["status"] == "read" and "does not match" in r["reasoning"]


def test_derived_when_only_roster_exists():
    sites = [atom(9, "Site A - 1 Main St"), atom(10, "Site B - 2 Oak Ave")]
    r = build_site_count_receipt([], reconcile_site_count([], 2), site_atoms=sites)
    assert r["status"] == "derived" and r["value"] == 2
    assert r["rule"] == "resolved_site_roster_count"
    assert r["facts"][0]["atom_id"] == "atom-9"


def test_abstains_and_says_what_it_needs():
    r = build_site_count_receipt([atom(1, "No counts here.")], reconcile_site_count([], 0))
    assert r["status"] == "abstained" and r["value"] is None
    assert r["abstention_needs"] and r["facts"] == []


def test_works_with_attribute_atoms():
    class A:
        id = "x1"
        raw_text = "Deploy to 3 schools."
        artifact_id = "d"
        source_refs = []

    r = build_site_count_receipt([A()], reconcile_site_count([A()], 3))
    assert r["facts"][0]["atom_id"] == "x1" and r["facts"][0]["page"] is None
