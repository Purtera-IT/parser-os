"""A later quote folds onto the earliest quote's line even when that line
was already folded onto a note.

One quoted message holds a request list and a restated list of it; two
earlier notes hold the two lists, and several restated items repeat request
items word for word. The pasted-note dedup takes the earliest email's quoted
copies out (they repeat the notes) before the quoted-history dedup runs. That
stage then never saw them, so:

* a later reply's copy of a list became a second survivor beside the earliest
  email's copy, and the next reply folded onto that later copy; and
* a later reply's request item, counted as the first line with its words in
  its message, folded onto the restated list's item with the same words.

The lines the pasted-note dedup took out still count as lines of their
message: each later copy folds onto its own list's earliest original.

Synthetic text only.
"""
from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

from app.core.suppression_ledger import SURVIVOR_KEY

REQ_HEAD = "Upkeep of the office network, servers and storage for the client"
REQ = ["Support for wired and wireless network gear", "Support for file servers and storage arrays",
       "Help with the hypervisor cluster at the head office", "Response to outages and service tickets"]
REST_HEAD = "Ongoing care of their network, server and storage estate"
REST = ["Support for wired and wireless network gear", "Support for file servers and storage arrays",
        "Care of their virtual machine hosts and clusters", "Response to outages and service tickets"]
SHARED = set(REQ) & set(REST)

QUOTED = "\n".join([
    "________________________________",
    "From: Dana Seller <dana@seller.example>",
    "Sent: Thursday, June 4, 2026 6:07 PM",
    "To: Lee Buyer <lee@buyer.example>",
    "Subject: Support proposal",
    "",
    "Thank you both for your time today and for the opportunity.",
    "Below is the original request from the client.",
    "",
    REQ_HEAD, *["- " + i for i in REQ],
    "",
    "Sites",
    "Springfield",
    "",
    REST_HEAD, *["- " + i for i in REST],
    "",
])


def _note(tmp: Path, name: str, nid: str, date: str, head: str, items: list[str]) -> None:
    body = "\n".join([head] + ["- " + i for i in items])
    (tmp / name).write_text(
        f"HubSpot Note: {head}\nHubSpot Note ID: {nid}\nDate: {date}\nAuthor: Dana Seller\n"
        f"Author-Email: dana@seller.example\n\n{head}\n\n{body}\n", encoding="utf-8")


def _eml(tmp: Path, name: str, opener: str, date: str, msgid: str, reply_to: str | None = None,
         frm: str = "Dana Seller <dana@seller.example>") -> None:
    m = EmailMessage()
    m["From"] = frm
    m["To"] = "lee@buyer.example"
    m["Subject"] = "RE: Support proposal"
    m["Date"] = date
    m["Message-ID"] = msgid
    if reply_to:
        m["In-Reply-To"] = reply_to
        m["References"] = reply_to
    m.set_content(opener + "\n\n" + QUOTED)
    (tmp / name).write_bytes(bytes(m))


def _compile(tmp: Path):
    from app.core.compiler import compile_project

    _note(tmp, "deal-hs-note-1001.txt", "1001", "2026-05-29T18:58:14.007Z", REQ_HEAD, REQ)
    _note(tmp, "deal-hs-note-1002.txt", "1002", "2026-05-29T18:58:26.171Z", REST_HEAD, REST)
    _eml(tmp, "deal-hs-email-1.eml", "Checking in on the proposal below, any update from the client side?",
         "Tue, 22 Sep 2026 16:29:56 +0000", "<m1@x>")
    _eml(tmp, "deal-hs-email-2.eml", "I reached out today and should hear back from them tomorrow morning.",
         "Tue, 22 Sep 2026 21:20:19 +0000", "<m2@x>", "<m1@x>", frm="Lee Buyer <lee@buyer.example>")
    _eml(tmp, "deal-hs-email-3.eml", "Thank you for the update on the client, talk again soon about it.",
         "Wed, 23 Sep 2026 01:04:18 +0000", "<m3@x>", "<m2@x>")
    return compile_project(tmp, project_id="p", allow_errors=True, use_cache=False)


def _file_of(r, atom) -> str:
    return next((ref.filename for ref in atom.source_refs or [] if ref.artifact_id == atom.artifact_id), "")


def _line(atom) -> int:
    return int((atom.source_refs[0].locator or {}).get("line_start") or 0)


def test_later_quotes_fold_onto_their_own_lists_earliest_original(tmp_path):
    r = _compile(tmp_path)
    list_words = set(REQ) | set(REST) | {REQ_HEAD, REST_HEAD}
    by_id = {a.id: a for a in list(r.atoms) + list(r.suppressed_atoms)}

    def in_later_reply(a) -> bool:
        return _file_of(r, a).endswith(("email-2.eml", "email-3.eml")) and a.raw_text.strip() in list_words

    # No later reply keeps its own copy of either list beside the earliest one.
    assert [a.raw_text for a in r.atoms if in_later_reply(a)] == []

    folded = [a for a in r.suppressed_atoms if in_later_reply(a)]
    assert len(folded) == 2 * (len(REQ) + len(REST) + 2)
    for a in folded:
        surv = by_id.get(((a.value or {}).get(SURVIVOR_KEY) or {}).get("atom_id"))
        assert surv is not None, a.raw_text
        surv_file = _file_of(r, surv)
        # Never a later reply's copy: the note, or the earliest email's line.
        assert surv_file.endswith(("note-1001.txt", "note-1002.txt", "email-1.eml")), (a.raw_text, surv_file)
        if a.raw_text.strip() in SHARED:
            # Which list this copy sits in: the request comes first.
            mine = sorted(_line(b) for b in folded if b.raw_text == a.raw_text and _file_of(r, b) == _file_of(r, a))
            in_request = _line(a) == mine[0]
            if surv_file.endswith("email-1.eml"):
                firsts = sorted(_line(b) for b in by_id.values()
                                if b.raw_text == a.raw_text and _file_of(r, b).endswith("email-1.eml"))
                assert (_line(surv) == firsts[0]) == in_request, (a.raw_text, in_request, _line(surv))
            else:
                assert surv_file.endswith("note-1001.txt" if in_request else "note-1002.txt"), (a.raw_text, surv_file)
