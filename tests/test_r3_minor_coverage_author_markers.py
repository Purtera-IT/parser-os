"""Three small things deal 000132 (compile 03510cad) showed, fixed for every deal.

(a) Text coverage counted a reply's quoted copy of the message it answers,
    its markdown-bold "*Sent:*" header rows and the vendor's legal footer as
    "unread" -- 56 and 38 lines on two mails that had almost nothing unread.
(b) An email document carried ``sender`` and no ``author``.
(c) A note bullet kept its "- " / "1. " in the dedup key, so the note's copy of
    an email line keyed apart from the line.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.text_coverage import build_text_coverage


# ── (a) coverage ────────────────────────────────────────────────────────────

ORIGINAL = """From: Carl Painter <carl@customer.com>
To: Trent <t@purtera-it.com>
Subject: Riser access
Date: Mon, 10 Aug 2026 09:00:00 -0500

Trent,

The riser room on the third floor is locked after six every evening.
Facilities holds the only key and needs a day of notice for access.

Carl
"""

REPLY = """From: Trent <t@purtera-it.com>
To: Carl Painter <carl@customer.com>
Subject: RE: Riser access
Date: Tue, 11 Aug 2026 10:00:00 -0500

Carl,

Our crew can only start the pull after the ceiling grid is closed up.

Trent

CDW Trust Center: learn how we protect your information at the CDW Trust Center.
This email and any attachments are intended solely for the named recipient.

*From:* Carl Painter <carl@customer.com>
*Sent:* Monday, August 10, 2026 9:00 AM
*To:* Trent <t@purtera-it.com>
*Subject:* Riser access

Trent,

The riser room on the third floor is locked after six every evening.
Facilities holds the only key and needs a day of notice for access.

Carl
"""


def _write(tmp_path: Path) -> dict[str, Path]:
    a = tmp_path / "original.eml"
    b = tmp_path / "reply.eml"
    a.write_text(ORIGINAL, encoding="utf-8")
    b.write_text(REPLY, encoding="utf-8")
    return {"art_orig": a, "art_reply": b}


def _row(rows, aid):
    return next(r for r in rows if r["artifact_id"] == aid)


def _unread(row):
    return {x["text"] for x in row["unclaimed"] if x["state"] == "unread"}


def test_quoted_copy_headers_and_footer_are_not_unread(tmp_path):
    rows = build_text_coverage(_write(tmp_path), atoms=[])
    reply = _row(rows, "art_reply")
    # The reply's own sentence nobody read is still a miss.
    assert _unread(reply) == {"Our crew can only start the pull after the ceiling grid is closed up."}
    assert reply["unread_count"] == 1
    states = {x["text"]: x["state"] for x in reply["unclaimed"]}
    assert states["*Sent:* Monday, August 10, 2026 9:00 AM"] == "chrome"
    assert states["*From:* Carl Painter <carl@customer.com>"] == "chrome"
    assert states["CDW Trust Center: learn how we protect your information at the CDW Trust Center."] == "chrome"
    # The quoted copy is listed (visible), flagged, and not counted.
    quoted = [x for x in reply["unclaimed"] if "riser room" in x["text"]]
    assert quoted and all(x["state"] == "copy" and x.get("quoted") for x in quoted)


def test_the_original_still_counts_its_own_unread_lines(tmp_path):
    rows = build_text_coverage(_write(tmp_path), atoms=[])
    orig = _row(rows, "art_orig")
    assert "The riser room on the third floor is locked after six every evening." in _unread(orig)
    assert orig["unread_count"] == 2


def test_a_message_that_exists_only_as_a_quote_is_counted_once(tmp_path):
    # The original never reached the deal as a file: its lines are real
    # content nobody read, so they count -- once, not per quoting reply.
    paths = _write(tmp_path)
    paths.pop("art_orig")
    second = tmp_path / "reply2.eml"
    second.write_text(REPLY.replace("Our crew can only start", "Second reply: we can only start"), encoding="utf-8")
    paths["art_reply2"] = second
    rows = build_text_coverage(paths, atoms=[])
    total = sum(r["unread_count"] for r in rows)
    # 1 own line per reply + the two quoted original lines counted once.
    assert total == 4


def test_a_line_wrapped_across_two_atoms_is_claimed(tmp_path):
    p = tmp_path / "m.eml"
    p.write_text(
        "From: a@x.com\nTo: b@y.com\nSubject: s\n\n"
        "We need forty drops in the lobby. Also the riser\n"
        "needs a new ladder rack before the pull.\n",
        encoding="utf-8",
    )
    atoms = [
        SimpleNamespace(artifact_id="m", raw_text="We need forty drops in the lobby.", normalized_text="", value={}, locator=None),
        SimpleNamespace(artifact_id="m", raw_text="Also the riser needs a new ladder rack before the pull.",
                        normalized_text="", value={}, locator=None),
    ]
    row = build_text_coverage({"m": p}, atoms=atoms)[0]
    assert row["unread_count"] == 0


# ── (b) author on an email document ─────────────────────────────────────────

def test_email_author_prefers_display_name():
    from app.core.orbitbrief_envelope import _email_author

    assert _email_author({"sender": "Quinton James <Quinton.James@cdw.com>"}) == {
        "author": "Quinton James", "author_email": "quinton.james@cdw.com"}
    assert _email_author({"sender": '"James, Quinton" <qj@cdw.com>'})["author"] == "James, Quinton"
    assert _email_author({"sender": "qj@cdw.com"}) == {"author": "qj@cdw.com", "author_email": "qj@cdw.com"}
    assert _email_author(None, "qj@cdw.com")["author"] == "qj@cdw.com"
    assert _email_author({"sender": "unknown"}) == {}
    assert _email_author(None) == {}


def test_envelope_email_document_carries_its_author(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope

    deal = tmp_path / "deal"
    deal.mkdir()
    (deal / "original.eml").write_text(ORIGINAL, encoding="utf-8")
    result = compile_project(deal, project_id="deal", use_cache=False)
    env = build_orbitbrief_envelope(project_dir=deal, compile_result=result)
    doc = next(d for d in env["documents"] if d["filename"] == "original.eml")
    assert (doc.get("email_thread") or {}).get("sender")
    assert doc["author"] == "Carl Painter"
    assert doc["author_email"] == "carl@customer.com"


# ── (c) list markers in dedup keys ──────────────────────────────────────────

def _atom(text, aid="a", **value):
    return SimpleNamespace(raw_text=text, normalized_text=text, artifact_id=aid, value=dict(value),
                           atom_type=SimpleNamespace(value="scope_item"), source_refs=[])


@pytest.mark.parametrize("marked", [
    "- Install four APs in the lobby",
    "* Install four APs in the lobby",
    "• Install four APs in the lobby",
    "1. Install four APs in the lobby",
    "12) Install four APs in the lobby",
    "- 2. Install four APs in the lobby",
])
def test_copy_keys_ignore_leading_list_markers(marked):
    from app.core.cross_doc_copies import _text_key
    from app.core.pasted_note_dedup import _key

    plain = "Install four APs in the lobby"
    assert _key(_atom(marked)) == _key(_atom(plain))
    assert _text_key(_atom(marked)) == _text_key(_atom(plain))


def test_semantic_description_key_ignores_leading_list_marker():
    from app.core.semantic_dedup import _value_key

    long = "Install four ceiling mounted access points in the main lobby area"
    a = _atom("1. " + long, description="1. " + long)
    b = _atom(long, description=long)
    a.atom_type = b.atom_type = SimpleNamespace(value="requirement")
    assert _value_key(b) is not None
    assert _value_key(a) == _value_key(b)


def test_marker_strip_leaves_real_text_alone():
    from app.core.cross_doc_copies import strip_list_marker

    assert strip_list_marker("1.5 hours on site") == "1.5 hours on site"
    assert strip_list_marker("-48V DC plant") == "-48V DC plant"
    assert strip_list_marker("2 APs per floor") == "2 APs per floor"
    assert strip_list_marker("- ") == "- "
