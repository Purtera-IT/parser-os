"""A recall miss leaves no trace: the only way anyone found out was a PM
reading the source beside the output. So every text artifact is diffed
against its own atoms and the unread lines are reported."""
from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.text_coverage import build_text_coverage, coverage_for_artifact

BODY = """Hey AJ,

Here are the details for the small job.

The riser room is locked after 6pm and the super has the only key.

Provided by us:

-Relay

Thanks,

Alec Burns
Cell: 281-840-3437 | Email: alec@vendor.example
The contents of this email are intended only for the recipient(s) listed above.
"""


def _atom(text: str, value: dict | None = None, artifact: str = "art_m"):
    return EvidenceAtom(
        id=f"atm_{abs(hash(text)) % 10**8}", project_id="p", artifact_id=artifact,
        atom_type=AtomType.scope_item, raw_text=text, normalized_text=text.lower(),
        value=value or {"kind": "email_body_line"}, entity_keys=[],
        source_refs=[SourceRef(id="s1", artifact_id=artifact, artifact_type=ArtifactType.txt, filename="m.eml",
                               locator={}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t",
    )


def _eml(tmp_path: Path) -> Path:
    m = EmailMessage()
    m["From"] = "alec@vendor.example"
    m["To"] = "aj@purtera-it.com"
    m["Subject"] = "Access Control"
    m.set_content(BODY)
    p = tmp_path / "m.eml"
    p.write_bytes(bytes(m))
    return p


def test_a_paragraph_that_produced_nothing_is_reported(tmp_path):
    kept = [_atom("Here are the details for the small job."),
            _atom("Relay", {"list_item": True, "list_label": "Provided by us:"})]
    cov = coverage_for_artifact(_eml(tmp_path), "art_m", kept)
    unread = [x["text"] for x in cov["unclaimed"] if x["state"] == "unread"]
    # the access constraint nobody read is named, with its line number
    assert any("riser room is locked" in t for t in unread)
    assert all(x["line"] > 0 for x in cov["unclaimed"])
    # a label consumed into its items was read; chrome is chrome, not a miss
    assert not any("Provided by us" in t for t in unread)
    assert not any("Cell: 281-840-3437" in t for t in unread)
    assert not any("contents of this email" in t.lower() for t in unread)
    assert cov["lines_claimed"] >= 2 and cov["unread_count"] == len(unread)


def test_a_dropped_atom_says_dropped_not_never_read(tmp_path):
    kept = [_atom("Here are the details for the small job.")]
    dropped = [_atom("The riser room is locked after 6pm and the super has the only key.")]
    cov = coverage_for_artifact(_eml(tmp_path), "art_m", kept, dropped)
    states = {x["text"]: x["state"] for x in cov["unclaimed"]}
    assert any(s == "suppressed" for t, s in states.items() if "riser room" in t)


def test_binaries_are_skipped_and_nothing_can_fail_a_compile(tmp_path):
    pdf = tmp_path / "quote.pdf"
    pdf.write_bytes(b"%PDF-1.4 junk")
    rows = build_text_coverage({"art_pdf": pdf, "art_missing": tmp_path / "gone.eml"}, [])
    assert rows == []


def test_coverage_survives_the_legacy_shape_validator():
    """CompileResult's before-validator rebuilds the payload from a fixed key
    list when a compatibility field is present -- which silently dropped
    text_coverage on every real compile while the INFO warning still fired."""
    from app.core.schemas import CompileResult

    r = CompileResult(
        project_id="p", atoms=[], entities=[], edges=[], packets=[], warnings=[],
        text_coverage=[{"artifact_id": "art_m", "lines_total": 3, "lines_claimed": 1,
                        "unclaimed": [{"line": 2, "text": "unread line", "state": "unread"}],
                        "unread_count": 1}],
        project_dir="/tmp/p", ranked_atoms=[], entity_edges=[], compile_id="c",
    )
    assert r.text_coverage and r.text_coverage[0]["unread_count"] == 1
