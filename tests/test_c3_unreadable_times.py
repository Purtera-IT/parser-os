"""Document times the deal record cannot read ("end of year", "6/3") on a page export: no crash, the
unknown-time rule (the deal's latest arrival) and a time_unknown mark; RFC 2822 mail dates are real times."""
from ml.c3.data import _time
from ml.c3.export import TIME_UNKNOWN, from_page_export


def page_atom(key, art, text, sent_at=None):
    a = {"labelKey": key, "atomId": f"at-{key}", "artifactId": art, "text": text, "docType": "",
         "section": [], "saidBy": {"role": "internal", "side": "ours"}}
    if sent_at is not None:
        a["docOrigin"] = {"sent_at": sent_at}
    return a


def page():
    return {
        "dealId": "uuid-t", "compileId": "c", "myLinks": [], "judgments": [], "labels": [],
        "documents": [
            {"artifactId": "mail", "kind": "email", "origin": {"sent_at": "Mon, 08 Jun 2026 22:00:21 +0000"}},
            {"artifactId": "note", "kind": "note", "authoredAt": "2026-06-01T00:00:00Z",
             "origin": {"sent_at": "end of year"}},
            {"artifactId": "late", "kind": "email", "origin": {"sent_at": "2026-07-01T00:00:00Z"}},
        ],
        "atoms": [page_atom("a", "mail", "Mail.", "Mon, 08 Jun 2026 22:00:21 +0000"),
                  page_atom("b", "note", "Note.", "end of year"),
                  page_atom("c", "late", "Later.", "2026-07-01T00:00:00Z")],
    }


def test_rfc_mail_dates_read_and_text_is_unknown():
    assert _time("Mon, 08 Jun 2026 22:00:21 +0000") == _time("2026-06-08T22:00:21Z")
    assert _time("end of year") == 0.0 and _time("6/3") == 0.0 and _time("") == 0.0
    assert _time("2026-06-08T22:00:21Z") > 0


def test_unreadable_document_time_takes_the_latest_arrival_and_is_marked():
    atoms = {a["label_key"]: a for a in from_page_export(page(), "010999")["atoms"]}
    # The RFC date stands as the line's arrival; nothing marked.
    assert _time(atoms["a"]["entered_at"]) == _time("2026-06-08T22:00:21Z")
    assert not atoms["a"].get(TIME_UNKNOWN) and not atoms["c"].get(TIME_UNKNOWN)
    # "end of year" is no time: the line arrives with the deal's latest line (never early), marked.
    assert atoms["b"][TIME_UNKNOWN] is True
    assert _time(atoms["b"]["entered_at"]) == _time("2026-07-01T00:00:00Z")
    assert _time(atoms["b"]["entered_at"]) >= max(_time(x["entered_at"]) for x in atoms.values())
