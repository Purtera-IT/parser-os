"""A company's legal footer is a reject with a reason, never scope or a clause.

Live 010003: CDW's mail footer ("CDW Trust Center", "Copyright 2026 CDW
LLC. All rights reserved.", its street address) became scope_items and a job
site, and a copy of the footer with the year beside one without raised the
open_question "Documents disagree on one clause: 2026 vs (no figure)".
"""
from __future__ import annotations

from pathlib import Path

from app.core.cross_document_conflicts import find_cross_document_conflicts
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.parsers.email_parser import EmailParser

MAIL = """From: Patrick Kelly <patrick.kelly@cdw.com>
To: Victor Lee <victor@purtera-it.com>
Subject: SOW signed
Date: Tue, 7 Jul 2026 18:35:00 -0400
Message-ID: <p1@cdw.com>
Content-Type: text/plain; charset=utf-8

Victor,
Signed SOW attached, please schedule the install of 12 TVs.

Patrick Kelly
Senior Account Manager

CDW Trust Center: https://www.cdw.com/trust-center.html
Copyright 2026 CDW LLC. All rights reserved. 200 N. Milwaukee Avenue, Vernon Hills, IL 60061
"""


def test_footer_lines_are_rejects_with_a_reason(tmp_path: Path):
    p = tmp_path / "m.eml"
    p.write_text(MAIL, encoding="utf-8")
    atoms = EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=p).atoms
    for words in ("Trust Center", "Copyright 2026", "All rights reserved", "Milwaukee Avenue"):
        hits = [a for a in atoms if words in a.raw_text]
        assert hits, words
        for a in hits:
            assert a.value.get("kind") == "admission_reject" and a.value.get("reason") == "footer", (words, a.atom_type, a.value)
    assert not any(a.atom_type == AtomType.physical_site for a in atoms)
    assert any(a.raw_text.startswith("Signed SOW attached") and a.value.get("kind") != "admission_reject" for a in atoms)


def _atom(aid: str, text: str) -> EvidenceAtom:
    src = SourceRef(id=f"src_{aid}", artifact_id=aid, artifact_type=ArtifactType.email, filename=f"{aid}.eml",
                    locator={"line_start": 3}, extraction_method="x", parser_version="t")
    return EvidenceAtom(id=f"atm_{aid}", project_id="p", artifact_id=aid, atom_type=AtomType.scope_item,
                        raw_text=text, normalized_text=text.lower(), value={"kind": "email_body_line"},
                        source_refs=[src], authority_class=AuthorityClass.customer_current_authored,
                        confidence=0.8, review_status=ReviewStatus.needs_review, parser_version="t")


def test_a_footer_never_feeds_the_clause_conflict_check():
    a = _atom("d1", "CDW LLC 2026, 200 N. Milwaukee Avenue Vernon Hills IL 60061 trust center privacy terms of use")
    b = _atom("d2", "CDW LLC, 200 N. Milwaukee Avenue Vernon Hills IL 60061 trust center privacy terms of use")
    assert find_cross_document_conflicts([a, b], project_id="p") == []
    # A real clause still conflicts.
    c = _atom("d1", "Provider requires a minimum of two (2) weeks notice before any scheduled site visit")
    d = _atom("d2", "Provider requires a minimum of five (5) weeks notice before any scheduled site visit")
    assert len(find_cross_document_conflicts([c, d], project_id="p")) == 1
