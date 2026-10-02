"""Quoted history and repeated signatures under later emails.

Live 010003 (Gmail thread): each reply kept the "On <date> <name> wrote:"
line of a message the deal holds as its own email -- a one-line section of
quoted history under the reply -- and Patrick Kelly's signature was a fresh
set of rejects under every email he sent. A signature is a reject with its
reason; its repeat in a later email is a copy of the first.
"""
from __future__ import annotations

from pathlib import Path

from app.core.admission_chatter import is_admission_chatter
from app.core.email_threading import dedup_quoted_chatter, mark_repeated_signature_copies, thread_emails
from app.parsers.email_parser import EmailParser

M1 = """From: Patrick Kelly <patrick.kelly@cdw.com>
To: Sarah Halpern <sarah@acme.com>
Subject: TV install
Date: Mon, 6 Jul 2026 09:00:00 -0400
Message-ID: <g1@cdw.com>
Content-Type: text/plain; charset=utf-8

Hi Sarah,

We are waiting for the TVs to arrive, once they are delivered we will schedule the install.

Thanks,
Patrick Kelly
Senior Account Manager
770.769.7311
"""

M2 = """From: Sarah Halpern <sarah@acme.com>
To: Patrick Kelly <patrick.kelly@cdw.com>
Subject: Re: TV install
Date: Tue, 7 Jul 2026 10:00:00 -0400
Message-ID: <g2@acme.com>
In-Reply-To: <g1@cdw.com>
References: <g1@cdw.com>
Content-Type: text/plain; charset=utf-8

Hi Patrick,

You guys are the best! Thank you so much for all of this work. The PO number is PO 4500123.

Sarah Halpern
Facilities Director

On Mon, Jul 6, 2026 at 9:00 AM Patrick Kelly <patrick.kelly@cdw.com> wrote:
> Hi Sarah,
>
> We are waiting for the TVs to arrive, once they are delivered we will schedule the install.
>
> Thanks,
> Patrick Kelly
> Senior Account Manager
> 770.769.7311
"""

M3 = """From: Patrick Kelly <patrick.kelly@cdw.com>
To: Sarah Halpern <sarah@acme.com>
Subject: Re: TV install
Date: Wed, 8 Jul 2026 10:00:00 -0400
Message-ID: <g3@cdw.com>
In-Reply-To: <g2@acme.com>
References: <g1@cdw.com> <g2@acme.com>
Content-Type: text/plain; charset=utf-8

Hi Sarah,

Thanks, we will install the week of July 20.

Thanks,
Patrick Kelly
Senior Account Manager
770.769.7311

On Tue, Jul 7, 2026 at 10:00 AM Sarah Halpern <sarah@acme.com> wrote:
> Hi Patrick,
>
> You guys are the best! Thank you so much for all of this work. The PO number is PO 4500123.
>
> Sarah Halpern
> Facilities Director
>
> On Mon, Jul 6, 2026 at 9:00 AM Patrick Kelly <patrick.kelly@cdw.com> wrote:
>> Hi Sarah,
>>
>> We are waiting for the TVs to arrive, once they are delivered we will schedule the install.
>>
>> Thanks,
>> Patrick Kelly
>> Senior Account Manager
>> 770.769.7311
"""


def _thread(tmp_path: Path):
    atoms = []
    for i, text in enumerate((M1, M2, M3), start=1):
        p = tmp_path / f"m{i}.eml"
        p.write_text(text, encoding="utf-8")
        atoms += EmailParser().parse_artifact_full(project_id="p", artifact_id=f"art_{i}", path=p).atoms
    thread_emails(atoms, project_id="p")
    chatter = [a for a in atoms if is_admission_chatter(a)]
    rest = [a for a in atoms if not is_admission_chatter(a)]
    return chatter, rest


def test_the_quote_line_of_a_message_the_deal_holds_is_dropped(tmp_path: Path):
    chatter, rest = _thread(tmp_path)
    kept, dropped = dedup_quoted_chatter(chatter, context=rest)
    assert not [a for a in kept if a.value.get("reason") == "quote_attribution"]
    assert any(a.value.get("reason") == "quote_attribution" for a in dropped)


def test_an_authors_signature_in_a_later_email_is_a_copy(tmp_path: Path):
    chatter, rest = _thread(tmp_path)
    kept, _ = dedup_quoted_chatter(chatter, context=rest)
    assert mark_repeated_signature_copies(kept) == 3
    sig = {(a.artifact_id, a.raw_text): a for a in kept if a.value.get("reason") == "signature"}
    first, later = sig[("art_1", "Patrick Kelly")], sig[("art_3", "Patrick Kelly")]
    assert "cross_doc_copy" not in (first.review_flags or [])
    assert "cross_doc_copy" in later.review_flags
    assert later.value["duplicate_of"]["atom_id"] == first.id
    # Greetings are not signatures: each email keeps its own.
    hi = [a for a in kept if a.raw_text == "Hi Sarah,"]
    assert len(hi) == 2 and not any("cross_doc_copy" in (a.review_flags or []) for a in hi)
