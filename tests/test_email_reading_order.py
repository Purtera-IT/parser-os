"""Email atoms must read like the conversation happened.

Live 010300: the audit view showed the newest reply first, every quoted
message's atoms were stamped with the FILE's top sender, nothing said which
message answered which, and a sign-off ("Thanks,") was stamped as the
addressee. These tests pin the structural fixes: per-block greeting from the
opening lines only, per-message author/reply stamps, and reading order.
"""

from __future__ import annotations

from pathlib import Path

from app.core.email_threading import thread_emails
from app.core.orbitbrief_envelope import _in_reading_order, _parse_loose_datetime
from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.parsers.email_parser import EmailParser

_THREAD = "\n".join(
    [
        "From: Patrick Kelly <patrick@purtera-it.com>",
        "To: team@purtera-it.com",
        "Subject: FW: Yealink install",
        "Date: Thu, 03 Sep 2026 13:20:00 -0400",
        "Message-ID: <top@purtera-it.com>",
        "",
        "Hello Team-",
        "",
        "Please see Carl's requirements below and start the site list review today.",
        "",
        "Thanks,",
        "",
        "Patrick Kelly",
        "Account Executive",
        "",
        "From: Carl Painter Jr <carlpai@cdw.com>",
        "Sent: Thursday, September 3, 2026 11:44 AM",
        "To: Trent Torrence <t@purtera-it.com>",
        "Subject: RE: Yealink install",
        "",
        "Hello Trent-",
        "",
        "It was good talking with you today about the Yealink rollout.",
        "We need 4 units installed at the Atlanta HQ site before the cutover.",
        "",
        "Thanks,",
        "Carl",
        "",
        "From: Trent Torrence <t@purtera-it.com>",
        "Sent: Thursday, September 3, 2026 9:02 AM",
        "To: Carl Painter Jr <carlpai@cdw.com>",
        "Subject: Yealink install",
        "",
        "Hi Carl,",
        "",
        "Following up on the Yealink phones for the dental offices we discussed.",
        "Can you send the site list and unit counts when you have them?",
        "",
        "Best,",
        "Trent",
    ]
)


def _parse(tmp_path: Path):
    p = tmp_path / "010300-hs-email-1.eml"
    p.write_text(_THREAD, encoding="utf-8")
    return EmailParser().parse_artifact("p", "art_email", p)


def _by_message(atoms):
    out: dict[int, list] = {}
    for a in atoms:
        v = a.value if isinstance(a.value, dict) else {}
        if v.get("message_index") is not None and str(v.get("kind") or "").startswith("email_body"):
            out.setdefault(int(v["message_index"]), []).append(a)
    return out


def test_sign_off_is_never_the_addressee(tmp_path: Path) -> None:
    atoms = _parse(tmp_path)
    names = {str((a.value or {}).get("addressee") or "") for a in atoms}
    assert "Thanks" not in names and "Best" not in names, names


def test_each_message_block_carries_its_own_greeting(tmp_path: Path) -> None:
    atoms = _parse(tmp_path)
    per = _by_message(atoms)
    assert set(per) >= {0, 1, 2}, sorted(per)
    top = {(a.value or {}).get("addressee") for a in per[0]}
    carl = {(a.value or {}).get("addressee") for a in per[1]}
    trent = {(a.value or {}).get("addressee") for a in per[2]}
    assert top == {"Team"}, top
    assert carl == {"Trent"}, carl
    assert trent == {"Carl"}, trent


def test_quoted_message_atoms_say_who_wrote_them_and_what_they_answer(tmp_path: Path) -> None:
    atoms = _parse(tmp_path)
    threaded, _ = thread_emails(atoms, project_id="p")
    per = _by_message(threaded)
    carl = per[1][0].value["email_thread"]
    assert "carlpai@cdw.com" in carl["message"]["author"].lower()
    assert carl["message"]["quoted"] is True
    assert "t@purtera-it.com" in carl["in_reply_to"]["author"].lower()
    assert "yealink phones" in carl["in_reply_to"]["gist"].lower()
    assert "patrick@purtera-it.com" in carl["answered_by"]["author"].lower()
    assert carl["position_in_file"] == 2
    oldest = per[2][0].value["email_thread"]
    assert oldest["position_in_file"] == 1 and "in_reply_to" not in oldest
    top = per[0][0].value["email_thread"]
    assert top["position_in_file"] == 3 and top["message"]["quoted"] is False


def _atom(aid: str, art: str, **value):
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id=art, atom_type=AtomType.scope_item,
        raw_text=aid, normalized_text=aid, value=value,
        authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.auto_accepted, entity_keys=[], parser_version="t",
        source_refs=[SourceRef(id=f"sr_{aid}", artifact_id=art, artifact_type="pdf", filename=f"{art}.pdf", extraction_method="t", parser_version="t", locator={"page": value.pop("_page", 1), "line": value.pop("_line", 1)})],
    )


def test_parse_loose_datetime_reads_client_date_shapes() -> None:
    assert _parse_loose_datetime("Thursday, September 3, 2026 11:44 AM") == "2026-09-03T11:44:00"
    assert _parse_loose_datetime("Thu, 03 Sep 2026 13:20:00 -0400").startswith("2026-09-03T13:20")
    assert _parse_loose_datetime("2026-09-03T15:00:00.000Z").startswith("2026-09-03T15:00:00")
    assert _parse_loose_datetime("garbage") == ""


def test_reading_order_is_oldest_message_first_then_page_and_line() -> None:
    docs = [
        {"artifact_id": "psow", "authored_at": "2026-08-20T00:00:00"},
        {"artifact_id": "eml", "authored_at": "2026-09-03T13:20:00"},
    ]
    atoms = [
        _atom("z_top_reply", "eml", message_index=0, email_thread={"position_in_file": 3}, _line=3),
        _atom("y_carl", "eml", message_index=1, email_thread={"position_in_file": 2}, _line=20),
        # A quoted header row without a clock of its own still sits with its message.
        _atom("x_trent", "eml", message_index=2, email_thread={"position_in_file": 1}, _line=30),
        _atom("w_psow_p2", "psow", _page=2, _line=1),
        _atom("v_psow_p1", "psow", _page=1, _line=5),
    ]
    ordered = [a.id for a in _in_reading_order(atoms, docs)]
    assert ordered == ["v_psow_p1", "w_psow_p2", "x_trent", "y_carl", "z_top_reply"], ordered
