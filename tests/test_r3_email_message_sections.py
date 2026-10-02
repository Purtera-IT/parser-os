"""Every atom of an email belongs to one message, in its place; a quoted
message belongs to the earliest file that quotes it.

Live 000132 / 010003 / 010087: header and people atoms with no message and no
line sorted after the whole file; atoms later stages derived from a line lost
their message; a later reply kept quoted copies of earlier mail because the
quoted-history dedup walked files in list order; docx paragraphs and sheet
rows were read back to front.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

from app.core.email_threading import dedup_quoted_history
from app.core.orbitbrief_envelope import _in_reading_order, _inherit_message_stamps
from app.parsers.email_parser import EmailParser

GMAIL = """From: Saga Ops <saga@customer.com>
To: Victor Lee <victor@purtera-it.com>
Subject: Re: Rollout sites
Date: Wed, 12 Aug 2026 10:00:00 -0400
Message-ID: <g3@customer.com>
Content-Type: text/plain; charset=utf-8

Victor, Dallas is not ready. Please swap to Plano.

On Tue, Aug 11, 2026 at 4:00 PM Victor Lee <victor@purtera-it.com> wrote:
> Can you confirm the Dallas install date?
>
> On Mon, Aug 10, 2026 at 9:00 AM Saga Ops <saga@customer.com> wrote:
>> We need 12 access points installed at Dallas and Plano.
>> Sites:
>> Dallas, TX
>> Plano, TX
"""

FORWARD = """From: Christopher Picchietti <christopher.picchietti@cdw.com>
To: Trent Torrence <t@purtera-it.com>
Subject: Fw: 000132 - Multi site technical IT support
Date: Fri, 29 May 2026 14:58:00 -0400
Message-ID: <fw1@cdw.com>
Content-Type: text/plain; charset=utf-8

Hi Trent,

Can you quote the below?

From: Heather Rosenthal <heatros@cdw.com>
Sent: Friday, May 29, 2026 1:00 PM
To: Christopher Picchietti <christopher.picchietti@cdw.com>
Subject: 000132 - Multi site technical IT support

Maintenance and support of the technical IT infrastructure.
"""


def _parse(tmp_path: Path, text: str, name: str = "m.eml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return EmailParser().parse_artifact("p", "art_mail", p)


def test_the_header_atom_sits_on_the_files_own_message_above_line_one(tmp_path: Path):
    atoms = _parse(tmp_path, GMAIL)
    hdr = next(a for a in atoms if a.value.get("kind") == "email_header")
    loc = hdr.source_refs[0].locator
    assert loc["message_index"] == 0 and loc["line_start"] == 0 and loc["quoted"] is False
    assert "saga@customer.com" in loc["sender"] and "12 Aug 2026" in loc["sent_at"]
    assert hdr.value["message_index"] == 0


def test_a_person_read_from_a_quoted_header_keeps_its_line_and_message(tmp_path: Path):
    atoms = _parse(tmp_path, FORWARD)
    heather = next(a for a in atoms if a.raw_text.startswith("Heather Rosenthal |"))
    loc = heather.source_refs[0].locator
    lines = FORWARD.split("\n\n", 1)[1].splitlines()
    assert lines[loc["line_start"] - 1].startswith("From: Heather Rosenthal")
    assert loc["message_index"] == 1 and loc["quoted"] is True
    assert heather.value["message_index"] == 1


def test_every_atom_of_a_quoted_gmail_message_names_that_message(tmp_path: Path):
    atoms = _parse(tmp_path, GMAIL)
    body = [a for a in atoms if a.value.get("kind") == "email_body_line"]
    by_text = {a.raw_text: a.source_refs[0].locator for a in body}
    assert by_text["Can you confirm the Dallas install date?"]["message_index"] == 1
    assert "Victor Lee" in by_text["Can you confirm the Dallas install date?"]["sender"]
    # The quoted city list is a list of sites, in its own (oldest) message.
    assert by_text["Dallas, TX"]["message_index"] == 2
    assert "Saga Ops" in by_text["Dallas, TX"]["sender"]
    dallas = next(a for a in body if a.raw_text == "Dallas, TX")
    assert dallas.atom_type.value == "physical_site" and dallas.entity_keys[-1] == "site:dallas_tx"


def _atom(aid, text, *, line=None, value=None, loc=None):
    loc = dict(loc or {})
    if line is not None:
        loc["line_start"] = line
    return NS(id=f"atm_{aid}_{text[:6]}", artifact_id=aid, raw_text=text,
              value=dict(value or {}), source_refs=[NS(locator=loc, artifact_id=aid)])


def test_a_derived_atom_inherits_the_message_of_its_line_and_notes_never_carry_one():
    stamp = {"thread_id": "t", "position_in_file": 1, "message": {"index": 2, "author": "Saga"}}
    own = {"thread_id": "t", "position_in_file": 3, "message": {"index": 0, "author": "Saga"}}
    body = _atom("mail", "We need 12 APs", line=7, value={"email_thread": stamp})
    first = _atom("mail", "Victor, Dallas is not ready.", line=1, value={"email_thread": own})
    derived = _atom("mail", "We need 12 APs", line=7)  # a bom_line typed later
    header = _atom("mail", "From: Saga | To: Victor")  # no line
    note = _atom("note", "Delphos, OH", line=11, value={"email_thread": stamp})  # took it in a merge
    _inherit_message_stamps([body, first, derived, header, note], mail_files={"mail"})
    assert derived.value["email_thread"]["message"]["index"] == 2
    assert header.value["email_thread"]["message"]["index"] == 0
    assert "email_thread" not in note.value
    order = [a.raw_text for a in _in_reading_order([first, derived, header, body], [])]
    # Oldest quoted message first; the derived atom stays with its line.
    assert order.index("We need 12 APs") < order.index("Victor, Dallas is not ready.")
    assert order[-2:] == ["From: Saga | To: Victor", "Victor, Dallas is not ready."]


def test_docx_paragraphs_and_sheet_rows_read_in_source_order():
    paras = [_atom("doc", f"p{i}", loc={"paragraph_index": i}) for i in (3, 2, 1, 0)]
    assert [a.raw_text for a in _in_reading_order(paras, [])] == ["p0", "p1", "p2", "p3"]
    rows = [_atom("x", f"r{r}{k}", loc={"sheet": "s", "row": r}) for r, k in ((2, "a"), (3, "a"), (2, "b"), (4, "a"), (3, "b"))]
    assert [a.raw_text for a in _in_reading_order(rows, [])] == ["r2a", "r2b", "r3a", "r3b", "r4a"]


def test_the_earliest_file_that_quotes_a_message_owns_it():
    def q(aid, ti, text):
        return NS(artifact_id=aid, raw_text=text, normalized_text=text.lower(), atom_type="scope_item",
                  value={"quoted": True, "kind": "email_body_line",
                         "email_thread": {"thread_id": "t", "thread_index": ti}})

    line = "You guys are the best! Thank you so much for all of this work."
    late = q("late_reply", 5, line)
    early = q("first_reply", 2, line)
    kept, dropped = dedup_quoted_history([late, early])
    assert kept == [early] and dropped == [late]


ORIGINAL = """From: Saga Ops <saga@customer.com>
To: Victor Lee <victor@purtera-it.com>
Subject: Rollout sites
Date: Mon, 10 Aug 2026 09:00:00 -0400
Message-ID: <g1@customer.com>
Content-Type: text/plain; charset=utf-8

We need 12 access points installed at Dallas and Plano.
"""


def test_every_message_of_a_thread_gets_one_chronological_number(tmp_path: Path):
    """010003: messages that exist only as quotes had no number, so the thread
    looked like it started partway through; file numbers ran 1,2,3,1,1,..."""
    from app.core.email_threading import thread_emails

    a = tmp_path / "a.eml"
    a.write_text(ORIGINAL, encoding="utf-8")
    b = tmp_path / "b.eml"
    b.write_text(GMAIL.replace("Message-ID: <g3@customer.com>",
                               "Message-ID: <g3@customer.com>\nIn-Reply-To: <g1@customer.com>"), encoding="utf-8")
    atoms = EmailParser().parse_artifact("p", "art_a", a) + EmailParser().parse_artifact("p", "art_b", b)
    thread_emails(atoms, project_id="p")

    def pos(aid, text):
        at = next(x for x in atoms if x.artifact_id == aid and x.raw_text.startswith(text))
        return at.value["email_thread"]["message"]["thread_position"], at.value["email_thread"]["message"]["thread_message_count"]

    assert pos("art_a", "We need 12") == (1, 3)
    assert pos("art_b", "Can you confirm") == (2, 3)  # Victor's mail: only a quote
    assert pos("art_b", "Victor, Dallas") == (3, 3)
    assert pos("art_b", "We need 12") == (1, 3)  # B's quote of A is A's message
    files = {x.artifact_id: x.value["email_thread"]["thread_index"] for x in atoms}
    assert files == {"art_a": 1, "art_b": 2}


def test_a_logo_image_is_signature_chatter_with_no_borrowed_heading():
    """010087: AMTIVO / ASCDI logo alt text typed scope_item under "Equipment list"."""
    from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
    from app.parsers.email_parser import _logo_images_are_chatter

    def cid(text):
        src = SourceRef(id=f"src_{text}", artifact_id="a", artifact_type=ArtifactType.email, filename="m.eml",
                        locator={"kind": "email_cid_inline", "lead_in": ["Equipment list"], "line_start": 3},
                        extraction_method="x", parser_version="t")
        return EvidenceAtom(id=f"atm_{text[:5]}", project_id="p", artifact_id="a", atom_type=AtomType.scope_item,
                            raw_text=text, normalized_text=text.lower(),
                            value={"kind": "email_cid_inline_body", "lead_in": ["Equipment list"], "intro": "Equipment list"},
                            source_refs=[src], authority_class=AuthorityClass.customer_current_authored,
                            confidence=0.7, review_status=ReviewStatus.needs_review, parser_version="t")

    logo, table = cid("AMTIVO"), cid("Cisco C9300-48P switch qty 4 rack mounted in MDF")
    _logo_images_are_chatter([logo, table])
    assert logo.value.get("chatter") and "lead_in" not in logo.value
    assert "lead_in" not in logo.source_refs[0].locator
    assert not table.value.get("chatter") and table.value["lead_in"] == ["Equipment list"]


OUTLOOK_REPLY = """From: Trent Torrence <t@purtera-it.com>
To: Stephanie Hechsel <stephanie.hechsel@amtivo.com>
Subject: RE: Equipment list
Date: Wed, 8 Jul 2026 10:00:00 -0400
Message-ID: <o2@purtera-it.com>
Content-Type: text/plain; charset=utf-8

Thanks, we will review the list and come back with pricing.

T

________________________________
{F}From:{F} Stephanie Hechsel <stephanie.hechsel@amtivo.com>
{F}Sent:{F} Tuesday, July 7, 2026 3:12 PM
{F}To:{F} Trent Torrence <t@purtera-it.com>

Please find the equipment list below.
- 4 switches
- 2 firewalls
"""


def test_an_outlook_quoted_header_splits_at_any_quote_depth_or_in_bold(tmp_path: Path):
    """010087: Stephanie's quoted 7/7 mail stayed inside Trent's 7/8 reply,
    credited to "T", when its From:/Sent: rows were quoted or bolded."""
    for name, text in {
        "plain": OUTLOOK_REPLY.replace("{F}", ""),
        "bold": OUTLOOK_REPLY.replace("{F}", "*"),
        "quoted": OUTLOOK_REPLY.replace("{F}", "").replace("\n________", "\n> ________")
            .replace("\nFrom: Steph", "\n> From: Steph").replace("\nSent:", "\n> Sent:").replace("\nTo: Trent", "\n> To: Trent")
            .replace("\nPlease", "\n> Please").replace("\n- ", "\n> - "),
    }.items():
        atoms = _parse(tmp_path, text, f"{name}.eml")
        sw = next(a for a in atoms if a.raw_text.endswith("4 switches"))
        loc = sw.source_refs[0].locator
        assert loc["message_index"] == 1, name
        assert "Stephanie Hechsel" in loc["sender"] and "July 7, 2026" in loc["sent_at"], name


def test_each_body_atom_names_its_message(tmp_path: Path):
    atoms = _parse(tmp_path, OUTLOOK_REPLY.replace("{F}", ""), "label.eml")
    sw = next(a for a in atoms if a.raw_text.endswith("4 switches"))
    assert sw.source_refs[0].locator["message_label"] == "Stephanie Hechsel · Tuesday, July 7, 2026 3:12 PM"
    hdr = next(a for a in atoms if a.value.get("kind") == "email_header")
    assert hdr.source_refs[0].locator["message_label"].startswith("Trent Torrence · ")
