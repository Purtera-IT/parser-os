"""One list folds as a unit, or not at all.

Live 000132 (shape only): a reply quoted an earlier message that carried a
request twice -- once as one line of " - "-separated items, once one item per
line -- and HubSpot notes held the same items. Each item was matched on its
own, so some items of each list folded onto a note and the rest stayed: the
quoted message showed a list with holes in it, and the missing items could
only be found in the suppression ledger.

A list whose items partly fold onto another document keeps them all, the
folded ones as its copies of the note's lines. A list that folds whole stays
folded, each item naming its survivor.
"""
from __future__ import annotations

from pathlib import Path

from app.core.cross_doc_copies import is_cross_doc_copy
from app.core.list_whole import keep_lists_whole
from app.core.suppression_ledger import SURVIVOR_KEY

LEAD = "Managed services for the branch offices"
ITEMS = [
    "Patch the core switches",
    "Replace the edge firewalls",
    "Back up the file servers",
    "Monitor the wireless controllers",
    "Weekend call-out for a technician",
]
#: In the mail only: the note never says it.
MAIL_ONLY = "Coordinate with the site managers"

NOTE = """HubSpot Note: Managed services
HubSpot Note ID: 900000000233
Date: 2026-05-29T18:58:14.007Z
Author: Pat Planner
Author-Email: pat@integrator.example

Managed services

{body}
"""

MAIL = """From: Casey Customer <casey@customer.example>
To: Pat Planner <pat@integrator.example>
Subject: RE: Managed services
Date: Mon, 8 Jun 2026 10:00:00 -0400
Message-ID: <reply@customer.example>
Content-Type: text/plain; charset=utf-8

Hi Pat,

Thanks, we will review the quote and come back to you this week.

Casey

From: Pat Planner <pat@integrator.example>
Sent: Thursday, June 4, 2026 6:07 PM
To: Casey Customer <casey@customer.example>
Subject: Managed services

Hi Casey,

Below is the original request from the branch team.

{body}

Thanks,
Pat
"""


def _compile(tmp_path: Path, note_body: str, mail_body: str):
    from app.core.compiler import compile_project

    (tmp_path / "000001-hs-note-900000000233.txt").write_text(NOTE.format(body=note_body), encoding="utf-8")
    (tmp_path / "000001-hs-email-900000000750.eml").write_text(MAIL.format(body=mail_body), encoding="utf-8")
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    mail_id = next(a.artifact_id for a in r.atoms + r.suppressed_atoms
                   if a.source_refs and a.source_refs[0].filename.endswith(".eml"))
    return r, mail_id


def _bare(t: str) -> str:
    t = " ".join((t or "").split())
    return t[2:] if t.startswith("- ") else t


def _assert_list_whole(r, mail_id: str, lines: list[str]) -> None:
    shown = {_bare(a.raw_text) for a in r.atoms if a.artifact_id == mail_id}
    for line in lines:
        assert line in shown, f"the quoted list lost {line!r}"
    by_id = {a.id: a for a in r.atoms}
    for a in r.atoms:
        if a.artifact_id == mail_id and is_cross_doc_copy(a):
            canon = by_id.get(a.value["duplicate_of"]["atom_id"])
            assert canon is not None and canon.artifact_id != mail_id, "a copy names a standing atom elsewhere"
    # Every item is accounted for: shown, or suppressed naming a standing atom.
    for s in r.suppressed_atoms:
        if s.artifact_id == mail_id and _bare(s.raw_text) in lines:
            assert (s.value.get(SURVIVOR_KEY) or {}).get("atom_id") in by_id


def test_a_partly_folded_inline_list_keeps_every_item(tmp_path: Path):
    note_line = " - ".join([LEAD, *ITEMS])
    mail_line = " - ".join([LEAD, *ITEMS, MAIL_ONLY])
    r, mail_id = _compile(tmp_path, note_line, mail_line)
    _assert_list_whole(r, mail_id, [LEAD, *ITEMS, MAIL_ONLY])
    copies = [a for a in r.atoms if a.artifact_id == mail_id and is_cross_doc_copy(a)]
    assert {_bare(a.raw_text) for a in copies} >= set(ITEMS), "the note's items stay the note's"


def test_a_partly_folded_line_per_item_list_keeps_every_item(tmp_path: Path):
    note_body = LEAD + "\n" + "\n".join(f"- {i}" for i in ITEMS)
    mail_body = LEAD + "\n" + "\n".join(f"- {i}" for i in [*ITEMS, MAIL_ONLY])
    r, mail_id = _compile(tmp_path, note_body, mail_body)
    _assert_list_whole(r, mail_id, [*ITEMS, MAIL_ONLY])


def test_a_list_that_folds_whole_stays_folded(tmp_path: Path):
    line = " - ".join([LEAD, *ITEMS])
    r, mail_id = _compile(tmp_path, line, line)
    shown = {_bare(a.raw_text) for a in r.atoms if a.artifact_id == mail_id}
    assert not (shown & set(ITEMS)), "a list that is a copy as a whole folds as a unit"
    by_id = {a.id: a for a in r.atoms}
    folded = [s for s in r.suppressed_atoms if s.artifact_id == mail_id and _bare(s.raw_text) in ITEMS]
    assert len(folded) == len(ITEMS)
    for s in folded:
        assert by_id[(s.value.get(SURVIVOR_KEY) or {})["atom_id"]].artifact_id != mail_id


# ---------------------------------------------------------------------------
# The pass on its own
# ---------------------------------------------------------------------------

class _Ref:
    def __init__(self, line: int, seq: int | None = None):
        self.locator = {"line_start": line, "message_index": 1}
        if seq is not None:
            self.locator["sentence_index"] = seq
        self.artifact_id = "mail"
        self.filename = "m.eml"


class _Atom:
    def __init__(self, aid: str, text: str, doc: str, line: int, seq: int | None = None, survivor: str = ""):
        self.id = aid
        self.raw_text = text
        self.artifact_id = doc
        self.source_refs = [_Ref(line, seq)]
        self.review_flags = []
        self.value = {"kind": "email_body_line", "message_index": 1, "quoted": True}
        if doc == "note":
            self.value = {"kind": "hubspot_note_body"}
            self.review_flags = ["hubspot_note_parser"]
        if survivor:
            self.value["_suppression"] = {"stage": "pasted_note_dedup", "kind": "fold"}
            self.value[SURVIVOR_KEY] = {"atom_id": survivor, "artifact_id": "note"}
            self.review_flags = self.review_flags + ["suppressed:pasted_note_dedup"]


def test_only_folds_onto_another_document_come_back():
    note = [_Atom(f"n{i}", t, "note", 9) for i, t in enumerate(ITEMS)]
    mail = [
        _Atom("m0", ITEMS[0], "mail", 18, 0, survivor="n0"),
        _Atom("m1", ITEMS[1], "mail", 18, 1, survivor="n1"),
        _Atom("m2", ITEMS[2], "mail", 18, 2),
    ]
    # Folded inside its own document: the document still shows the line.
    own = _Atom("m3", ITEMS[3], "mail", 18, 3, survivor="m2")
    own.value[SURVIVOR_KEY]["artifact_id"] = "mail"
    atoms, suppressed, back = keep_lists_whole(note + [mail[2]], [mail[0], mail[1], own])
    assert [a.id for a in back] == ["m0", "m1"]
    assert [a.id for a in suppressed] == ["m3"]
    assert [a.id for a in atoms if a.artifact_id == "mail"] == ["m0", "m1", "m2"], "the list keeps its order"
    for a in back:
        assert "_suppression" not in a.value and SURVIVOR_KEY not in a.value
        assert is_cross_doc_copy(a) and a.value["duplicate_of"]["artifact_id"] == "note"
        assert not any(f.startswith("suppressed:") for f in a.review_flags)


def test_a_quote_of_another_email_stays_that_messages_line():
    # 010003: a reply's quote of another email belongs in that message's section.
    other = [_Atom(f"o{i}", t, "other_mail", 4) for i, t in enumerate(ITEMS)]
    mail = [
        _Atom("m0", ITEMS[0], "mail", 18, 0, survivor="o0"),
        _Atom("m1", ITEMS[1], "mail", 18, 1),
        _Atom("m2", ITEMS[2], "mail", 18, 2),
    ]
    atoms, suppressed, back = keep_lists_whole(other + mail[1:], [mail[0]])
    assert back == [] and [a.id for a in suppressed] == ["m0"]


def test_prose_sentences_of_one_line_are_not_a_list():
    note = [_Atom("n0", "We will patch the switches on Friday.", "note", 9)]
    mail = [
        _Atom("m0", "We will patch the switches on Friday.", "mail", 18, 0, survivor="n0"),
        _Atom("m1", "The firewall order shipped yesterday and should arrive soon.", "mail", 18, 1),
        _Atom("m2", "Let me know if anything changes on your side before then.", "mail", 18, 2),
    ]
    atoms, suppressed, back = keep_lists_whole(note + mail[1:], [mail[0]])
    assert back == [] and [a.id for a in suppressed] == ["m0"]
