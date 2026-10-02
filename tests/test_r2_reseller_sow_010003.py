"""Round-2 fixes from re-parsing a reseller display-install deal (010003):
a reseller SOW PDF (BOM table, PMO steps, exclusions, page footers), a
Gantt workbook, e-sign notices and seller emails with long quoted history.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EdgeType,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _mk(aid: str, atype: AtomType, text: str, keys: list[str]) -> EvidenceAtom:
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id="sow", atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value={"text": text}, entity_keys=keys,
        source_refs=[SourceRef(id="s" + aid, artifact_id="sow", artifact_type=ArtifactType.pdf,
                               filename="sow.pdf", extraction_method="x", parser_version="x", locator={})],
        authority_class=AuthorityClass.contractual_scope, confidence=0.85,
        review_status=ReviewStatus.needs_review, parser_version="x",
    )


# ── 3: one generic exclusion clause must not conflict with every scope line ──

def test_generic_exclusion_clause_does_not_conflict_with_every_scope_line() -> None:
    from app.core.graph_builder import build_edges

    clauses = [
        "Electrical work of any kind",
        "Drywall cutting or patching",
        "Furniture movement",
        "Troubleshooting or remediation of Customer network issues",
    ]
    # Every line of the SOW carries the document's one site.
    atoms = [_mk(f"ex{i}", AtomType.exclusion, t, ["site:nyc_office"]) for i, t in enumerate(clauses)]
    atoms += [_mk(f"sc{i:02d}", AtomType.scope_item, f"Mount display {i} on the wall",
                  ["site:nyc_office", "device:display"]) for i in range(37)]
    edges = build_edges("p", atoms, [])
    assert not [e for e in edges if e.edge_type == EdgeType.excludes]


def test_one_clause_is_capped_and_a_specific_exclusion_still_conflicts() -> None:
    from app.core.graph_builder import _MAX_EXCLUDES_PER_CLAUSE, build_edges

    ex = _mk("ex", AtomType.exclusion, "Display mounting above 12 ft is not included",
             ["site:nyc_office", "device:display"])
    scope = [_mk(f"sc{i:02d}", AtomType.scope_item, f"Mount display {i}", ["site:nyc_office", "device:display"])
             for i in range(30)]
    edges = [e for e in build_edges("p", [ex] + scope, []) if e.edge_type == EdgeType.excludes]
    assert 1 <= len(edges) <= _MAX_EXCLUDES_PER_CLAUSE

    # A whole-site exclusion names the site and still reaches its lines.
    site_ex = _mk("ex_site", AtomType.exclusion, "NYC Office is removed from scope", ["site:nyc_office"])
    edges = [e for e in build_edges("p", [site_ex] + scope[:3], []) if e.edge_type == EdgeType.excludes]
    assert len(edges) == 3


# ── 9: quoted signature blocks collapse to the message that authored them ──

_SIG_P = "Patrick Kelly\nSenior Account Manager\n770.769.7311\n"
_SIG_S = "Sarah Halpern\nFacilities Director\n212.555.0199\n"


def _write_long_chain(d: Path, *, with_address: bool) -> None:
    chain = "Hi Sarah,\n\nThe TVs ship next week.\n\nThanks,\n" + _SIG_P
    for i in range(8):
        sarah = i % 2 == 0
        body = ("Hi Patrick,\n\nPlease confirm the mount count.\n\n" + _SIG_S) if sarah else \
            ("Hi Sarah,\n\nConfirmed, 12 mounts.\n\n" + _SIG_P)
        prev = ("Patrick Kelly", "patrick.kelly@purtera-it.com") if sarah else ("Sarah Halpern", "sarah@acme.com")
        frm = f"{prev[0]} <{prev[1]}>" if with_address else prev[0]
        chain = (body + "\n________________________________\n"
                 f"From: {frm}\nSent: Monday, July {i + 6}, 2026 9:0{i} AM\nTo: x\nSubject: TVs\n\n" + chain)
        if i >= 4:
            me = ("Sarah Halpern", "sarah@acme.com") if sarah else ("Patrick Kelly", "patrick.kelly@purtera-it.com")
            (d / f"m{i}.eml").write_text(
                f"From: {me[0]} <{me[1]}>\nTo: x <x@acme.com>\nSubject: RE: TVs\n"
                f"Date: Mon, {i + 7} Jul 2026 10:0{i}:00 -0400\nMessage-ID: <m{i}@x>\n"
                "Content-Type: text/plain; charset=utf-8\n\n" + chain, encoding="utf-8")


@pytest.mark.parametrize("with_address", [True, False])
def test_quoted_signature_lines_are_one_atom_per_authored_message(tmp_path: Path, with_address: bool) -> None:
    from app.core.compiler import compile_project

    _write_long_chain(tmp_path, with_address=with_address)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    # Patrick authored two of the four emails in the deal, Sarah two.
    for line in ("Patrick Kelly", "770.769.7311", "Sarah Halpern", "212.555.0199", "Facilities Director"):
        hits = [a for a in r.atoms if a.raw_text == line]
        assert len(hits) == 2, (line, len(hits))
        assert all("chatter" in a.review_flags for a in hits)
        assert all(not (a.value or {}).get("quoted") for a in hits), line
    # The copies are recorded, not lost.
    assert [a for a in r.suppressed_atoms if a.raw_text == "770.769.7311"]


# ── 8: an e-sign robot's lines are chatter, never instructions or people ──

_ADOBE_BODY = (
    "Sarah Halpern has signed CDW_SOW_TV_Install.\n\n"
    "To ensure that you continue receiving our emails, please add "
    "echosign@echosign.com to your address book or safe list.\n"
)


def test_adobe_sign_email_chrome_is_chatter(tmp_path: Path) -> None:
    from app.parsers.email_parser import EmailParser

    p = tmp_path / "adobe.eml"
    p.write_text("From: Adobe Sign <echosign@echosign.com>\nTo: Sarah Halpern <sarah@acme.com>\n"
                 "Subject: Signed: CDW_SOW_TV_Install\nDate: Tue, 14 Jul 2026 10:00:00 -0400\n"
                 "Content-Type: text/plain; charset=utf-8\n\n" + _ADOBE_BODY, encoding="utf-8")
    atoms = EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=p).atoms
    add = [a for a in atoms if "add echosign@echosign.com" in a.raw_text]
    assert len(add) == 1
    assert add[0].atom_type == AtomType.deal_metadata and "chatter" in add[0].review_flags
    assert not [a for a in atoms if a.atom_type in (AtomType.customer_instruction, AtomType.stakeholder)]
    signed = [a for a in atoms if a.raw_text == "Sarah Halpern has signed CDW_SOW_TV_Install."]
    assert signed and "chatter" not in signed[0].review_flags


def test_pasted_adobe_sign_notice_in_a_note_is_chatter(tmp_path: Path) -> None:
    from app.parsers.hubspot_note_parser import HubspotNoteParser

    p = tmp_path / "010003-hs-note-7.txt"
    p.write_text("HubSpot Note: fwd\nHubSpot Note ID: 7\nDate: 2026-07-20T18:58:14.007Z\n"
                 "Author: Patrick Kelly\n\nFrom: Adobe Sign <echosign@echosign.com>\n"
                 "Sent: Tuesday, July 14, 2026 10:00 AM\nTo: Patrick Kelly\nSubject: Signed: CDW_SOW\n\n"
                 + _ADOBE_BODY, encoding="utf-8")
    atoms = HubspotNoteParser().parse_artifact("p", "a", p)
    robot = [a for a in atoms if "echosign" in a.raw_text.lower()]
    assert len(robot) == 2, [a.raw_text for a in robot]
    for a in robot:
        assert a.atom_type == AtomType.deal_metadata and "chatter" in a.review_flags, a.raw_text
    assert [a for a in atoms if a.raw_text.startswith("Sarah Halpern has signed")]


# ── 7: the TV-delivery dependency is split and never chatter ──

_TV = "We are waiting for the TVs to arrive, once they are delivered we will schedule the install."
_TV_PARTS = ["We are waiting for the TVs to arrive", "Once they are delivered we will schedule the install."]


def test_trigger_clause_splits_only_a_spliced_main_clause() -> None:
    from app.core.sentences import split_trigger_clause

    assert split_trigger_clause(_TV) == _TV_PARTS
    for whole in ("We will install once the TVs arrive.", "Install the displays, once they arrive.",
                  "Please call me, once you arrive I will open the door."):
        assert split_trigger_clause(whole) == [whole]


@pytest.mark.parametrize("opener", ["Hi Sarah,\n\n", "Thank you!\n\n", "Hi Sarah,\n\nThanks!\n\n"])
def test_email_dependency_is_split_and_not_chatter(tmp_path: Path, opener: str) -> None:
    from app.parsers.email_parser import EmailParser

    p = tmp_path / "tv.eml"
    p.write_text("From: Patrick Kelly <patrick.kelly@purtera-it.com>\nTo: Sarah <sarah@acme.com>\n"
                 "Subject: TVs\nDate: Mon, 06 Jul 2026 10:00:00 -0400\n"
                 "Content-Type: text/plain; charset=utf-8\n\n"
                 f"{opener}{_TV}\n\nThanks,\nPatrick Kelly\n770.769.7311\n", encoding="utf-8")
    atoms = EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=p).atoms
    by = {a.raw_text: a for a in atoms}
    for part in _TV_PARTS:
        assert part in by, [a.raw_text for a in atoms]
        assert "chatter" not in by[part].review_flags and by[part].atom_type != AtomType.deal_metadata
    # The signature under a real sign-off is still chrome.
    assert "chatter" in by["770.769.7311"].review_flags


def test_reply_that_is_only_a_signature_stays_chrome(tmp_path: Path) -> None:
    from app.parsers.email_parser import EmailParser

    p = tmp_path / "sig.eml"
    p.write_text("From: Patrick Kelly <patrick.kelly@purtera-it.com>\nTo: Sarah <sarah@acme.com>\n"
                 "Subject: TVs\nDate: Mon, 06 Jul 2026 10:00:00 -0400\n"
                 "Content-Type: text/plain; charset=utf-8\n\n"
                 "Thanks,\nPatrick Kelly\nSenior Account Manager\nCDW | 200 N Milwaukee Ave\n", encoding="utf-8")
    atoms = EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=p).atoms
    row = [a for a in atoms if a.raw_text == "CDW | 200 N Milwaukee Ave"]
    assert row and "chatter" in row[0].review_flags


def test_note_and_call_dependency_is_not_small_talk(tmp_path: Path) -> None:
    from app.core.compiler import compile_project
    from app.core.hybrid_summary_transcript import classify_transcript_turn_role

    assert classify_transcript_turn_role("We are waiting for the TVs to arrive") == "deal"
    assert classify_transcript_turn_role("We're waiting for Bob to join") != "deal"
    (tmp_path / "010003-hs-note-1.txt").write_text(
        "HubSpot Note: TV status\nHubSpot Note ID: 1\nDate: 2026-07-20T18:58:14.007Z\n"
        f"Author: Patrick Kelly\n\n{_TV}\n", encoding="utf-8")
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    by = {a.raw_text: a for a in r.atoms}
    for part in _TV_PARTS:
        assert part in by, list(by)
        assert "chatter" not in by[part].review_flags
        assert by[part].atom_type != AtomType.deal_metadata, (part, by[part].atom_type)


# ── 2: a site address whose city/ZIP line wraps under its street ──

def _pdf_atoms(path: Path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse(path)
    return list(getattr(out, "atoms", out))


def _site_table_pdf(path: Path, *, table: bool) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((36, 60), "Statement of Work", fontsize=12, fontname="hebo")
    p.insert_text((36, 80), "This Statement of Work is made between CDW Direct, LLC and Customer.", fontsize=10)
    p.insert_text((36, 104), "Project Locations", fontsize=12, fontname="hebo")
    if table:
        p.insert_text((36, 124), "Site Name", fontsize=9, fontname="hebo")
        p.insert_text((200, 124), "Address", fontsize=9, fontname="hebo")
        p.insert_text((36, 138), "NYC Office", fontsize=9)
        p.insert_text((200, 138), "40 10th Ave Fl 4", fontsize=9)
        p.insert_text((200, 150), "NEW YORK, NY 10014", fontsize=9)
    else:
        for i, line in enumerate(["NYC Office", "40 10th Ave Fl 4", "NEW YORK, NY 10014"]):
            p.insert_text((36, 124 + 13 * i), line, fontsize=10)
    p.insert_text((36, 190), "Project Management", fontsize=12, fontname="hebo")
    p.insert_text((36, 210), "CDW will provide project management for the duration of the project.", fontsize=10)
    doc.save(str(path))
    doc.close()


def test_site_table_address_keeps_its_wrapped_city_line(tmp_path: Path) -> None:
    pdf = tmp_path / "sow.pdf"
    _site_table_pdf(pdf, table=True)
    atoms = _pdf_atoms(pdf)
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1, [a.raw_text for a in atoms]
    site = sites[0]
    assert "NEW YORK, NY 10014" in site.raw_text and "40 10th Ave Fl 4" in site.raw_text
    assert (site.value["city"], site.value["state"], site.value["zip"]) == ("NEW YORK", "NY", "10014")
    for a in atoms:
        loc = a.source_refs[0].locator if a.source_refs else {}
        assert "NEW YORK, NY 10014" not in (loc.get("section_path") or []), a.raw_text


def test_an_all_caps_city_line_is_not_a_heading(tmp_path: Path) -> None:
    pdf = tmp_path / "sow.pdf"
    _site_table_pdf(pdf, table=False)
    atoms = _pdf_atoms(pdf)
    assert any("NEW YORK, NY 10014" in a.raw_text for a in atoms), [a.raw_text for a in atoms]
    for a in atoms:
        loc = a.source_refs[0].locator if a.source_refs else {}
        assert "NEW YORK, NY 10014" not in (loc.get("section_path") or []), a.raw_text


# ── 11: "After a Change" / "Order that requires..." is one sentence ──

def test_change_order_cut_by_a_line_break_is_not_a_heading() -> None:
    from app.parsers.orbitbrief_pdf import _text_rich_sections

    text = ("Change Management\n\nAfter a Change\n\n"
            "Order that requires additional work is signed, CDW will schedule the work.\n")
    sections = _text_rich_sections(text)
    blocks = [b.get("text") for s in sections for b in s["blocks"]]
    assert "After a Change Order that requires additional work is signed, CDW will schedule the work." in blocks
    assert "After a Change" not in blocks
    assert not any(s.get("heading") == "After a Change" for s in sections)
    # A real heading over a capitalised sentence stays a heading.
    sections = _text_rich_sections("Customer Responsibilities\n\nCustomer will provide power at each location.\n")
    assert [b.get("text") for s in sections for b in s["blocks"]] == ["Customer will provide power at each location."]


def test_change_order_cut_across_a_layout_gap_in_a_pdf(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "sow.pdf"
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((36, 60), "Change Management", fontsize=12, fontname="hebo")
    p.insert_text((36, 84), "Either party may request changes to this SOW through the Change Order process.", fontsize=10)
    p.insert_text((36, 110), "After a Change", fontsize=10)
    p.insert_text((36, 140), "Order that requires additional work is signed, CDW will schedule the work.", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    texts = [a.raw_text for a in _pdf_atoms(pdf)]
    assert any(t.startswith("After a Change Order that requires") for t in texts), texts
    assert not any(t.startswith("Order that requires") for t in texts), texts
