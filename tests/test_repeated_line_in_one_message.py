"""The same words twice in one email message are two lines, not a repeat.

A forwarded call-slot list puts "5:00-6:00 PM EST" under three different
days. Every reply quotes it again. The quoted-history dedup and the
intra-document collapse each folded the second and third slot onto the
first, so only one day kept its evening slot. Identical text under a different
parent is its own fact. A later reply's n-th copy still folds onto the n-th
line of the earliest quote.

Synthetic mail only.
"""
from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

from app.core.email_threading import dedup_quoted_history, thread_emails
from app.core.entity_resolution import collapse_duplicate_atoms
from app.parsers.email_parser import EmailParser

SLOT = "5:00–6:00 PM EST"

QUOTED = f"""________________________________
From: Dana Seller <dana@seller.example>
Sent: Thursday, June 4, 2026 6:07 PM
To: Lee Buyer <lee@buyer.example>
Subject: Kickoff call

Here are the open slots for the kickoff call next week.

Tuesday, June 9
1:30–2:00 PM EST
{SLOT}
Wednesday, June 10
2:00–2:30 PM EST
{SLOT}
Thursday, June 11
1:00–3:00 PM EST
{SLOT}

Please pick the one that suits the onsite team best.
"""


def _eml(tmp: Path, name: str, body: str, *, date: str, msgid: str, reply_to: str | None = None) -> Path:
    m = EmailMessage()
    m["From"] = "Dana Seller <dana@seller.example>"
    m["To"] = "lee@buyer.example"
    m["Subject"] = "Fw: Kickoff call"
    m["Date"] = date
    m["Message-ID"] = msgid
    if reply_to:
        m["In-Reply-To"] = reply_to
        m["References"] = reply_to
    m.set_content(body)
    p = tmp / name
    p.write_bytes(bytes(m))
    return p


def _thread(tmp_path):
    first = _eml(tmp_path, "1.eml", "Forwarding the slots again for the team.\n\n" + QUOTED,
                 date="Tue, 22 Sep 2026 16:29:56 +0000", msgid="<m1@x>")
    second = _eml(tmp_path, "2.eml", "Adding the onsite lead to this thread now.\n\n" + QUOTED,
                  date="Tue, 22 Sep 2026 21:20:19 +0000", msgid="<m2@x>", reply_to="<m1@x>")
    atoms = EmailParser().parse(first) + EmailParser().parse(second)
    atoms, _ = thread_emails(atoms, project_id="p")
    return atoms


def _slots(atoms, artifact_id=None):
    return [a for a in atoms if a.raw_text.strip() == SLOT
            and (artifact_id is None or a.artifact_id == artifact_id)]


def test_quoted_history_keeps_each_days_slot_and_folds_the_reply_line_for_line(tmp_path):
    from app.core.suppression_ledger import take_folds

    atoms = _thread(tmp_path)
    assert len(_slots(atoms)) == 6, "fixture: three slots in each of two emails"
    take_folds()
    kept, dropped = dedup_quoted_history(atoms, project_id="p")
    folds = take_folds()

    first_file = _slots(atoms)[0].artifact_id
    kept_slots = _slots(kept)
    assert len(kept_slots) == 3 and {a.artifact_id for a in kept_slots} == {first_file}
    assert len(_slots(dropped)) == 3
    # The reply's n-th slot folds onto the first email's n-th slot.
    order = {a.id: i for i, a in enumerate(sorted(kept_slots, key=lambda a: a.source_refs[0].locator["line_start"]))}
    folded = sorted(_slots(dropped), key=lambda a: a.source_refs[0].locator["line_start"])
    assert [order[folds[id(a)][1].id] for a in folded] == [0, 1, 2]


def test_intra_document_collapse_keeps_each_line_of_one_message(tmp_path):
    atoms = _thread(tmp_path)
    kept, _ = dedup_quoted_history(atoms, project_id="p")
    out = collapse_duplicate_atoms(kept)
    assert len(_slots(out)) == 3


def test_intra_document_collapse_still_folds_one_line_read_twice(tmp_path):
    atoms = _thread(tmp_path)
    kept, _ = dedup_quoted_history(atoms, project_id="p")
    twin = _slots(kept)[0].model_copy(deep=True)
    twin.id = "atm_twin_same_line"
    out = collapse_duplicate_atoms(kept + [twin])
    assert len(_slots(out)) == 3
