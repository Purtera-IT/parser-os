"""Lines the email admission regexes refuse are KEPT atoms, flagged chatter.

"Hi Trent,", "Thank you," and the name under it are probably not deal
content -- but a regex is not the judge, the admission head is, and it learns
only from what a labeler sees. So each refused line is a real, kept atom
carrying the same ``chatter`` flag (and ``small_talk`` read) relationship talk
gets, plus ``value["admission_regex"] = <reason>``. The labeling walk reports
it ``chatter: true``; text coverage counts its line as claimed. The compiler
holds these atoms out of every head, so every other atom, entity, edge and
packet is exactly what it was without them.
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
    "We need 40 Cat6 drops on the second floor and two racks in the MDF.\n\n"
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
        a.raw_text: a.value["admission_regex"]
        for a in atoms
        if ADMISSION_REJECT_FLAG in (a.review_flags or [])
    }


def test_greeting_signoff_and_signature_come_out_as_chatter_atoms(tmp_path):
    out = EmailParser().parse_artifact_full(
        project_id="p", artifact_id="a", path=_write(tmp_path)
    )
    rejects = _rejects(out.atoms)
    # "Hope you had a great 4th of July!" is a pleasantry: the digit in "4th"
    # used to make it read as work, so it came out as deal_metadata context.
    assert rejects == {"Hi Trent,": "greeting", "Hope you had a great 4th of July!": "banter",
                       "Thank you,": "signoff", "Stephanie Hechsel": "signature"}
    for a in out.atoms:
        if a.raw_text in rejects:
            # The same chatter mark relationship talk gets, and the reason.
            assert "chatter" in a.review_flags
            assert not any(str(f).startswith("suppressed:") for f in a.review_flags)
            assert a.value["chatter"] is True
            assert a.value["provenance"] == f"admission_regex: {rejects[a.raw_text]}"
            assert any(r["key"] == "small_talk" for r in a.value["reads"])
            assert "_suppression" not in a.value
    # The list API is the content atoms only.
    kept = EmailParser().parse_artifact("p", "a", _write(tmp_path))
    assert not _rejects(kept)


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _signature(atoms):
    return sorted(
        (a.id, getattr(a.atom_type, "value", a.atom_type), a.raw_text, a.review_status,
         tuple(a.entity_keys or ()), tuple(sorted(map(str, a.review_flags or ()))))
        for a in atoms
    )


def test_compile_keeps_rejects_as_chatter_and_nothing_else_moves(tmp_path, monkeypatch, no_llm):
    _write(tmp_path / "deal")
    with_rejects = compile_project(tmp_path / "deal", project_id="deal", use_cache=False)

    # Kept, in the atom list, flagged chatter; never in suppressed_atoms.
    kept = _rejects(with_rejects.atoms)
    assert kept["Hi Trent,"] == "greeting"
    assert kept["Thank you,"] == "signoff"
    assert kept["Stephanie Hechsel"] == "signature"
    assert not _rejects(with_rejects.suppressed_atoms)
    for a in with_rejects.atoms:
        if a.raw_text in kept:
            assert "chatter" in a.review_flags and a.receipts

    # Coverage counts their lines as claimed: none is listed as unclaimed.
    unclaimed = {
        line["text"]
        for row in with_rejects.text_coverage or []
        for line in row.get("unclaimed") or []
    }
    assert "Hi Trent," not in unclaimed
    assert "Thank you," not in unclaimed

    # No packet, edge or entity cites one.
    reject_ids = {a.id for a in with_rejects.atoms if ADMISSION_REJECT_FLAG in a.review_flags}
    for p in with_rejects.packets:
        assert not reject_ids & set(p.supporting_atom_ids + p.contradicting_atom_ids)
    for e in with_rejects.edges:
        assert e.from_atom_id not in reject_ids and e.to_atom_id not in reject_ids

    # The same compile without the rejects: every non-chatter atom, entity,
    # edge and packet is identical.
    real = EmailParser.parse_artifact_full

    def without_rejects(self, *a, **kw):
        out = real(self, *a, **kw)
        out.atoms = [x for x in out.atoms if ADMISSION_REJECT_FLAG not in (x.review_flags or [])]
        return out

    monkeypatch.setattr(EmailParser, "parse_artifact_full", without_rejects)
    baseline = compile_project(tmp_path / "deal", project_id="deal", use_cache=False)
    assert not _rejects(baseline.atoms)
    non_chatter = [a for a in with_rejects.atoms if ADMISSION_REJECT_FLAG not in a.review_flags]
    assert _signature(non_chatter) == _signature(baseline.atoms)
    assert sorted(e.id for e in with_rejects.entities) == sorted(e.id for e in baseline.entities)
    assert sorted(e.id for e in with_rejects.edges) == sorted(e.id for e in baseline.edges)
    assert [p.id for p in with_rejects.packets] == [p.id for p in baseline.packets]
    assert with_rejects.quality.atom_count == baseline.quality.atom_count
    assert with_rejects.quality.entity_resolution_rate == baseline.quality.entity_resolution_rate


THREAD_1 = """From: Carl Painter <carl@acmehealth.com>
To: Trent <trent@purtera-it.com>
Subject: Clinic cabling
Date: Mon, 07 Jul 2026 09:00:00 -0400
Message-ID: <a1@acme>
Content-Type: text/plain; charset=utf-8

Hi Trent,

Hope you are well!

We need 40 Cat6 drops at 1200 Main Street, Springfield, IL 62701 on the second floor.
Please quote two 42U racks in the MDF and remove the West Wing from scope.
Budget is $18,500 and install must finish before August 15.
https://safelinks.protection.outlook.com/?url=https%3A%2F%2Fpurtera-it.com

Thanks,
Carl Painter
Facilities Manager
Acme Health
404-555-1212
carl@acmehealth.com
"""

THREAD_2 = """From: Trent <trent@purtera-it.com>
To: Carl Painter <carl@acmehealth.com>
Subject: RE: Clinic cabling
Date: Tue, 08 Jul 2026 10:00:00 -0400
Message-ID: <a2@purtera>
In-Reply-To: <a1@acme>
Content-Type: text/plain; charset=utf-8

Hello Carl,

Got it. Can you confirm whether the ceiling is drop tile or hard lid?
We will need escort access after 5pm.

Best regards,
Trent

-----Original Message-----
From: Carl Painter <carl@acmehealth.com>
Sent: Monday, July 7, 2026 9:00 AM
To: Trent <trent@purtera-it.com>
Subject: Clinic cabling

Hi Trent,

We need 40 Cat6 drops at 1200 Main Street, Springfield, IL 62701 on the second floor.

Thanks,
Carl Painter
"""

_VOLATILE = {"compile_id", "generated_at", "created_at", "output_signature", "duration_ms",
             "stage_durations_ms", "trace", "telemetry", "coverage",
             # process-wide "last run" model counters, not envelope content
             "deflect_counts"}


def _envelope_without(env, drop_ids):
    import json

    env = json.loads(json.dumps(env, default=str))

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items() if k not in _VOLATILE}
        if isinstance(o, list):
            return [walk(x) for x in o
                    if not (isinstance(x, dict) and (x.get("id") in drop_ids or x.get("atom_id") in drop_ids))]
        return o

    return walk(env)


def test_envelope_is_unchanged_but_for_the_chatter_atoms(tmp_path, monkeypatch, no_llm):
    """Every envelope section -- scope summary, roster, sites, packets,
    indexes, dashboards -- is built without the chatter atoms; they appear
    only in `atoms`, in reading order, flagged chatter."""
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope

    def run(directory, strip):
        directory.mkdir(parents=True)
        (directory / "e1.eml").write_text(THREAD_1, encoding="utf-8")
        (directory / "e2.eml").write_text(THREAD_2, encoding="utf-8")
        real = EmailParser.parse_artifact_full
        if strip:
            def without_rejects(self, *a, **kw):
                out = real(self, *a, **kw)
                out.atoms = [x for x in out.atoms if ADMISSION_REJECT_FLAG not in (x.review_flags or [])]
                return out
            monkeypatch.setattr(EmailParser, "parse_artifact_full", without_rejects)
        result = compile_project(directory, project_id="deal", use_cache=False)
        env = build_orbitbrief_envelope(project_dir=directory, compile_result=result)
        monkeypatch.setattr(EmailParser, "parse_artifact_full", real)
        return result, env

    with_r, env_with = run(tmp_path / "with", strip=False)
    base_r, env_base = run(tmp_path / "base", strip=True)
    reasons = {a.raw_text: a.value["admission_regex"] for a in with_r.atoms
               if ADMISSION_REJECT_FLAG in a.review_flags}
    assert reasons["Hello Carl,"] == "greeting"
    assert reasons["Best regards,"] == "signoff"
    assert reasons["Hope you are well!"] == "banter"
    assert reasons["https://safelinks.protection.outlook.com/?url=https%3A%2F%2Fpurtera-it.com"] == "link_only"

    reject_ids = {a.id for a in with_r.atoms if ADMISSION_REJECT_FLAG in a.review_flags}
    in_env = [a for a in env_with["atoms"] if a["id"] in reject_ids]
    assert len(in_env) == len(reject_ids)
    assert all("chatter" in a["review_flags"] for a in in_env)
    # Reading order: the greeting comes before the body of its own message.
    order = [a["text"] for a in env_with["atoms"] if a["artifact_id"] == in_env[0]["artifact_id"]]
    assert order.index("Hi Trent,") < order.index(
        "We need 40 Cat6 drops at 1200 Main Street, Springfield, IL 62701 on the second floor.")

    assert _envelope_without(env_with, reject_ids) == _envelope_without(env_base, set())
