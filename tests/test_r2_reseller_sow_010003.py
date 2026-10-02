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
