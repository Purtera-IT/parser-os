"""Lines the email admission regexes refuse still come out, as rejects.

"Hi Trent,", "Thank you," and the name under it are correctly not deal
content. Until now they were cut with a bare ``continue`` before they were
atoms, so the labeling page could not show one and the admission head never
saw a negative. They are now emitted pre-suppressed
(``suppressed:admission_regex``); the compiler diverts them into
``suppressed_atoms`` and text coverage marks their lines ``suppressed``.
Nothing downstream may change: the kept atoms are exactly what they were.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.compiler import compile_project
from app.parsers.email_parser import ADMISSION_REJECT_FLAG, EmailParser

BODY = (
    "Hi Trent,\n\n"
    "Hope you had a great 4th of July! By the way I have CC’d Sean to this email, "
    "Sean and I work closely together here at Summit 360, he schedules all of our "
    "logistics needs so just wanted to introduce him quickly!\n\n"
    "Thank you,\n"
    "Stephanie Hechsel\n"
)


def _write(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    raw = (
        "From: Stephanie Hechsel <stephanie@summit360.com>\n"
        "To: Trent <trent@purtera-it.com>\n"
        "Subject: Intro\n"
        "Date: Mon, 07 Jul 2026 09:00:00 -0400\n"
        "Content-Type: text/plain; charset=utf-8\n\n" + BODY
    )
    p = directory / "intro.eml"
    p.write_bytes(raw.encode("utf-8"))
    return p


def _rejects(atoms):
    return {
        a.raw_text: a.value["_suppression"]
        for a in atoms
        if ADMISSION_REJECT_FLAG in (a.review_flags or [])
    }


def test_greeting_signoff_and_signature_come_out_as_admission_rejects(tmp_path):
    out = EmailParser().parse_artifact_full(
        project_id="p", artifact_id="a", path=_write(tmp_path)
    )
    rejects = _rejects(out.atoms)
    assert rejects["Hi Trent,"] == {"stage": "admission_regex", "reason": "greeting"}
    assert rejects["Thank you,"] == {"stage": "admission_regex", "reason": "signoff"}
    assert rejects["Stephanie Hechsel"] == {"stage": "admission_regex", "reason": "signature"}
    # The list API is the kept atoms only.
    kept = EmailParser().parse_artifact("p", "a", _write(tmp_path))
    assert not _rejects(kept)
    assert "Hi Trent," not in [a.raw_text for a in kept]


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _kept_signature(result):
    return sorted(
        (a.id, getattr(a.atom_type, "value", a.atom_type), a.raw_text, a.review_status)
        for a in result.atoms
    )


def test_compile_diverts_rejects_and_leaves_kept_atoms_unchanged(tmp_path, monkeypatch, no_llm):
    _write(tmp_path / "deal")
    with_rejects = compile_project(tmp_path / "deal", project_id="deal", use_cache=False)

    # In suppressed_atoms with their stage and reason, never in the kept set.
    sup = _rejects(with_rejects.suppressed_atoms)
    assert sup["Hi Trent,"]["reason"] == "greeting"
    assert sup["Thank you,"]["reason"] == "signoff"
    assert not _rejects(with_rejects.atoms)

    # Coverage now shows those lines as suppressed, so the walk lists them.
    states = {
        line["text"]: line["state"]
        for row in with_rejects.text_coverage or []
        for line in row.get("unclaimed") or []
    }
    assert states.get("Hi Trent,") == "suppressed"
    assert states.get("Thank you,") == "suppressed"

    # The same compile without the rejects keeps exactly the same atoms.
    real = EmailParser.parse_artifact_full

    def without_rejects(self, *a, **kw):
        out = real(self, *a, **kw)
        out.atoms = [x for x in out.atoms if ADMISSION_REJECT_FLAG not in (x.review_flags or [])]
        return out

    monkeypatch.setattr(EmailParser, "parse_artifact_full", without_rejects)
    baseline = compile_project(tmp_path / "deal", project_id="deal", use_cache=False)
    assert _kept_signature(with_rejects) == _kept_signature(baseline)
    assert not _rejects(baseline.suppressed_atoms)
