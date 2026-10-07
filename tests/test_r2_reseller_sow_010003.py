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
    from app.core.email_threading import _message_day

    _write_long_chain(tmp_path, with_address=with_address)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    # Patrick authored two of the four emails in the deal, Sarah two. Each
    # authored email keeps its own signature; of the quoted copies only the
    # earliest-dated survives (Patrick's July 6 message, Sarah's July 7), and
    # every later quoted copy folds onto it.
    for line, day in (("Patrick Kelly", "07-06"), ("770.769.7311", "07-06"), ("Sarah Halpern", "07-07"),
                      ("212.555.0199", "07-07"), ("Facilities Director", "07-07")):
        hits = [a for a in r.atoms if a.raw_text == line]
        assert len(hits) == 3, (line, len(hits))
        assert all("chatter" in a.review_flags for a in hits)
        quoted = [a for a in hits if (a.value or {}).get("quoted")]
        assert len(quoted) == 1, line
        assert str(_message_day(quoted[0])).endswith(day), line
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


# ── 10: page footers are boilerplate chatter and never a conflict ──

def test_footer_lines_never_make_a_cross_document_conflict() -> None:
    from app.core.cross_document_conflicts import find_cross_document_conflicts

    a = _mk("f1", AtomType.scope_item, "2026 CDW LLC. All rights reserved. | 800.800.4239 | CDW.com", [])
    b = _mk("f2", AtomType.scope_item, "CDW LLC. All rights reserved. | CDW.com | Terms and conditions apply", [])
    b.artifact_id = "quote"
    # Phone numbers are contacts, not clause figures.
    c = _mk("c1", AtomType.scope_item, "Call the CDW install desk at 800.800.4239 to book the crew visit", [])
    d = _mk("c2", AtomType.scope_item, "Call the CDW install desk at 877.555.0100 to book the crew visit", [])
    d.artifact_id = "quote"
    assert find_cross_document_conflicts([a, b, c, d], project_id="p") == []
    # A real clause still conflicts.
    e = _mk("e1", AtomType.scope_item, "A cancellation fee of $500 applies to visits cancelled late", [])
    f = _mk("e2", AtomType.scope_item, "A cancellation fee of $300 applies to visits cancelled late", [])
    f.artifact_id = "quote"
    assert len(find_cross_document_conflicts([e, f], project_id="p")) == 1


def test_pdf_footer_band_is_a_chatter_atom(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "sow.pdf"
    doc = fitz.open()
    for n in (1, 2):
        p = doc.new_page(width=612, height=792)
        p.insert_text((36, 60), "Statement of Work" if n == 1 else "Invoicing Procedures", fontsize=12, fontname="hebo")
        p.insert_text((36, 84), f"CDW will install the displays listed in section {n} of this SOW.", fontsize=10)
        p.insert_text((36, 770), "© 2026 CDW LLC. All rights reserved. | 800.800.4239 | CDW.com", fontsize=7)
        p.insert_text((540, 770), f"Page {n} of 2", fontsize=7)
    doc.save(str(pdf))
    doc.close()
    atoms = _pdf_atoms(pdf)
    band = [a for a in atoms if "All rights reserved" in a.raw_text]
    assert band, [a.raw_text for a in atoms]
    for a in band + [a for a in atoms if a.raw_text.startswith("Page ")]:
        assert a.atom_type == AtomType.deal_metadata and "chatter" in a.review_flags, a.raw_text


# ── 12: a BOM table row is one atom, cells never apart ──

_BOM_ROWS = [["7506872", "QM75C", ["Samsung QM75C QMC Series - 75\" LED-backlit", "LCD display - 4K - for digital signage"],
              "12", "$1,077.30", "$12,927.60"],
             ["5502114", "WMN6575SE", ["Samsung Slim Fit Wall Mount WMN6575SE -", "mounting kit - for LCD display"],
              "12", "$89.99", "$1,079.88"]]


@pytest.mark.parametrize("ruled", [True, False])
def test_bom_rows_are_one_atom_each(tmp_path: Path, ruled: bool) -> None:
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "bom.pdf"
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((36, 60), "Product Details", fontsize=12, fontname="hebo")
    cols = [36, 96, 170, 400, 445, 520, 590]
    y = 90
    for x, c in zip(cols, ["CDW#", "Mfg#", "Description", "Qty", "Unit Price", "Ext Price"]):
        p.insert_text((x + 2, y), c, fontsize=8, fontname="hebo")
    y += 16
    for r in _BOM_ROWS:
        for x, c in zip(cols, r):
            for k, line in enumerate(c if isinstance(c, list) else [c]):
                p.insert_text((x + 2, y + 10 * k), line, fontsize=8)
        y += 30
    if ruled:
        for yy in (80, 96, 126, 156):
            p.draw_line((36, yy), (590, yy))
        for x in cols:
            p.draw_line((x, 80), (x, 156))
    doc.save(str(pdf))
    doc.close()
    atoms = _pdf_atoms(pdf)
    texts = [a.raw_text for a in atoms]
    for r in _BOM_ROWS:
        hits = [t for t in texts if r[0] in t]
        assert len(hits) == 1, (r[0], texts)
        row = hits[0]
        for cell in (r[4], r[5], r[2][1]):
            assert cell in row, (cell, row)
        assert "CDW#" in row, row  # key: value, the header names the cells
    for t in texts:
        assert t.strip() not in {"7506872", "$1,077.30", "CDW#", "mounting kit - for LCD display"}, t
        assert not t.startswith("CDW# Mfg# Description | Qty"), t


# ── 1 / 4: SOW exclusions under a lead-in, PMO steps, the PO footnote ──

_EXCLUDED = [
    "Electrical work, including installation of new outlets or circuits",
    "Drywall cutting, patching or painting",
    "Furniture movement or removal of existing equipment",
]
_PMO = [
    "Conduct a remote kickoff meeting with Customer",
    "Develop schedule for installation activities and share with Customer",
    "Complete billing tasks",
]


def _sow_deal(d: Path) -> None:
    fitz = pytest.importorskip("fitz")
    d.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    y = 60

    def h(t):
        nonlocal y
        p.insert_text((36, y), t, fontsize=12, fontname="hebo")
        y += 20

    def line(t, x=36):
        nonlocal y
        p.insert_text((x, y), t, fontsize=10)
        y += 13

    def bullets(items):
        nonlocal y
        for b in items:
            p.insert_text((50, y), "•", fontsize=10)
            line(b, x=62)
        y += 8

    h("Statement of Work")
    line('This Statement of Work ("SOW") is made between CDW Direct, LLC and Customer.')
    y += 8
    h("Project Management")
    line("CDW will provide project management for the duration of the project. The CDW PM will:")
    bullets(_PMO)
    h("Invoicing Procedures")
    line("Services are billed on a time and materials basis. Consultant timesheets are submitted every")
    line("Monday for the prior week and Customer will be invoiced monthly.")
    y += 8
    h("Customer Responsibilities")
    line("The following are not included in this SOW:")
    bullets(_EXCLUDED)
    p.insert_text((36, 740), "Note: CDW PO's are not transferrable.", fontsize=7, fontname="heit")
    doc.save(str(d / "CDW_SOW.pdf"))
    doc.close()


def test_sow_lead_in_exclusions_pmo_steps_and_po_note_survive_compile(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    _sow_deal(tmp_path / "deal")
    r = compile_project(tmp_path / "deal", project_id="p", allow_errors=True, use_cache=False)
    kept = {" ".join(a.raw_text.split()): a for a in r.atoms}
    recorded = {" ".join(a.raw_text.split()) for a in r.suppressed_atoms}
    for t in _EXCLUDED:
        assert t in kept and kept[t].atom_type == AtomType.exclusion, (t, sorted(kept))
    for t in _PMO:  # an atom, or at least a recorded suppression
        assert t in kept or t in recorded, (t, sorted(kept), sorted(recorded))
    assert any("timesheets are submitted every Monday" in t for t in kept), sorted(kept)
    note = kept.get("Note: CDW PO's are not transferrable.")
    assert note is not None, sorted(kept)
    assert note.atom_type != AtomType.exclusion


def test_lead_in_sentence_opens_an_exclusions_list() -> None:
    from app.parsers.sow_sections import is_exclusion_heading

    for t in ("The following are not included in this SOW:",
              "The following items are excluded from the scope of work",
              "Not included in this quote:", "Out of Scope"):
        assert is_exclusion_heading(t), t
    for t in ("Inclusions and Exclusions", "Electrical work is not included in this SOW unless quoted separately.",
              "The following are included in this SOW:", "The following is out of scope."):
        assert not is_exclusion_heading(t), t


def test_apostrophe_abbreviation_plural_is_readable_text() -> None:
    from app.core.text_quality import is_unreadable

    assert not is_unreadable("Note: CDW PO's are not transferrable.")


# ── 6: schedule rows -- milestone, PO reference, total ──

def test_schedule_rows_are_typed_by_their_name_cell() -> None:
    from app.core.atom_type_sanity import retype_schedule_reference_rows

    rows = [
        ("m1", AtomType.vendor_line_item, "Milestone - Install complete | 2026-08-21 | 2026-08-21 | 0"),
        ("m2", AtomType.task, "4 | Milestone: Installation Complete | 0 days | 2026-08-21 | 3"),
        ("po", AtomType.commercial_total, "PO # 4500123 | $18,207.48"),
        ("tt", AtomType.task, "Total | 9 | $4,450.00"),
        ("t1", AtomType.task, "2 | TV Delivery to NYC Office | 3 days | 2026-08-10 | 1"),
        ("t2", AtomType.task, "Install displays | 2026-08-17 | 2026-08-21 | 5 | $4,200.00"),
        # prose is never judged as a row
        ("pr", AtomType.scope_item, "Milestone payments are invoiced when each phase is accepted."),
    ]
    atoms = [_mk(i, t, x, []) for i, t, x in rows]
    assert retype_schedule_reference_rows(atoms) == 4
    got = {a.id: a.atom_type for a in atoms}
    assert got == {
        "m1": AtomType.milestone_phase, "m2": AtomType.milestone_phase,
        "po": AtomType.deal_metadata, "tt": AtomType.commercial_total,
        "t1": AtomType.task, "t2": AtomType.task, "pr": AtomType.scope_item,
    }


def test_gantt_workbook_milestone_and_po_rows_after_compile(tmp_path: Path) -> None:
    import datetime as dt

    openpyxl = pytest.importorskip("openpyxl")
    from app.core.compiler import compile_project

    d = tmp_path / "deal"
    d.mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Gantt"
    ws.append(["Task", "Start Date", "End Date", "Days", "Cost", "Status"])
    ws.append(["Project kickoff", dt.date(2026, 8, 3), dt.date(2026, 8, 3), 1, 250, "Complete"])
    ws.append(["Install displays", dt.date(2026, 8, 17), dt.date(2026, 8, 21), 5, 4200, "Not Started"])
    ws.append(["Milestone - Install complete", dt.date(2026, 8, 21), dt.date(2026, 8, 21), 0, None, None])
    ws.append(["PO # 4500123", None, None, None, 18207.48, None])
    wb.save(d / "Gantt.xlsx")
    r = compile_project(d, project_id="p", allow_errors=True, use_cache=False)
    by = {a.raw_text.split(" | ")[0]: a.atom_type for a in r.atoms}
    assert by.get("Milestone - Install complete") == AtomType.milestone_phase, by
    assert by.get("PO # 4500123") == AtomType.deal_metadata, by
    assert by.get("Install displays") not in (AtomType.commercial_total, None), by


# ── 9 (Gmail): nested ">" quotes, greetings and "On ... wrote:" attribution ──

_PEOPLE = {"p": ("Patrick Kelly", "patrick.kelly@purtera-it.com"), "s": ("Sarah Halpern", "sarah@acme.com")}


def _gmail_message(k: int) -> tuple[str, str]:
    """Message k of the thread: (author key, body). 0 is the root."""
    if k == 0:
        return "p", "Hi Sarah,\n\nThe TVs ship next week.\n\nThanks,\n" + _SIG_P
    if k % 2:
        return "s", "Hi Patrick,\n\nPlease confirm the mount count.\n\n" + _SIG_S
    return "p", "Hi Sarah,\n\nConfirmed, 12 mounts.\n\n" + _SIG_P


def _write_gmail_chain(d: Path, *, first_file: int = 5, last: int = 8) -> None:
    chain = ""
    for k in range(last + 1):
        who, body = _gmail_message(k)
        if chain:
            pw, _ = _gmail_message(k - 1)
            name, addr = _PEOPLE[pw]
            quoted = "\n".join(("> " + ln).rstrip() for ln in chain.splitlines())
            chain = f"{body}\nOn Mon, Jul {k + 5}, 2026 at 9:0{k - 1} AM {name} <{addr}> wrote:\n{quoted}\n"
        else:
            chain = body
        if k >= first_file:
            name, addr = _PEOPLE[who]
            (d / f"m{k}.eml").write_text(
                f"From: {name} <{addr}>\nTo: x <x@acme.com>\nSubject: RE: TVs\n"
                f"Date: Mon, {k + 6} Jul 2026 09:0{k}:00 -0400\nMessage-ID: <m{k}@x>\n"
                "Content-Type: text/plain; charset=utf-8\n\n" + chain, encoding="utf-8")


def test_gmail_nested_quote_greetings_are_one_atom_per_message(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    _write_gmail_chain(tmp_path)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    # Nine messages: Patrick wrote five ("Hi Sarah,"), Sarah four ("Hi Patrick,").
    for line, n in (("Hi Sarah,", 5), ("Hi Patrick,", 4)):
        hits = [a for a in r.atoms if a.raw_text == line]
        assert len(hits) == n, (line, len(hits))
        assert all("chatter" in a.review_flags for a in hits)
    # The four messages the deal holds keep their own (unquoted) greeting.
    assert len([a for a in r.atoms if a.raw_text.startswith("Hi ") and not (a.value or {}).get("quoted")]) == 4
    assert [a for a in r.suppressed_atoms if a.raw_text == "Hi Sarah,"]  # recorded, not lost


def test_gmail_attribution_line_is_chrome_never_a_stakeholder(tmp_path: Path) -> None:
    from app.core.compiler import compile_project
    from app.parsers.email_parser import EmailParser

    _write_gmail_chain(tmp_path, first_file=4, last=4)
    atoms = EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=tmp_path / "m4.eml").atoms
    people = [a for a in atoms if a.atom_type == AtomType.stakeholder]
    assert people and all(not a.raw_text.startswith(("AM ", "PM ")) for a in people), [a.raw_text for a in people]
    assert all(" wrote:" not in a.raw_text for a in people)
    attr = [a for a in atoms if a.raw_text.endswith(" wrote:")]
    assert len(attr) == 4, [a.raw_text for a in atoms]
    for a in attr:
        assert a.atom_type == AtomType.deal_metadata and "chatter" in a.review_flags, a.raw_text
        # credited to the author it names, as is the quoted message it opens
        assert a.value["author"].split(" <")[0] in a.raw_text, (a.value["author"], a.raw_text)
    sarah_ask = [a for a in atoms if a.raw_text == "Please confirm the mount count."]
    assert sarah_ask and all("sarah@acme.com" in str(a.value.get("author")) for a in sarah_ask)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    for a in r.atoms:
        assert not (a.atom_type == AtomType.stakeholder and a.raw_text.startswith(("AM ", "PM "))), a.raw_text
        if " wrote:" in a.raw_text:
            assert a.atom_type == AtomType.deal_metadata and "chatter" in a.review_flags, a.raw_text
    assert not [p for p in r.packets if " wrote:" in str(getattr(p, "reason", "") or "")]
