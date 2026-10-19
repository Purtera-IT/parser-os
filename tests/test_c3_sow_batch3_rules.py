"""SOW writer rules, batch 3: the drop guard's holes (010217), exclusions that
are not requests (010284, 010147), third-party supply and terms (010376),
cited counts and the after-hours basis (010083), pack-and-move jobs (010147)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False}


def test_guard_restores_a_line_whose_other_cite_is_a_quoted_copy():
    """010217: tidy dropped the 86" install; one of its cites was the email's
    quoted copy (noise), so the guard let the whole line go."""
    line = {"text": "Install two (2) new 86 inch displays in training room 250.", "cites": ["note", "quoted"], "kind": "fact"}
    content = {"_noise_atoms": ["quoted"],
               "scope": [{"text": "Training Room 250", "cites": [], "kind": "heading"}, line,
                         {"text": "Room 247", "cites": [], "kind": "heading"},
                         {"text": "Install the 98 inch display.", "cites": ["r247"], "kind": "fact"},
                         {"text": "Closeout", "cites": [], "kind": "heading"}]}
    m._traced(content, "tidy", lambda: content.__setitem__("scope", [x for x in content["scope"] if x is not line]))
    texts = [x["text"] for x in content["scope"]]
    # back under its own heading, not before Closeout
    assert texts.index(line["text"]) == texts.index("Training Room 250") + 1
    assert content["rule_decisions"][0]["guard"] == "restored"


def test_a_fact_only_the_summary_cites_is_not_in_the_task_list():
    """010217: the 98" install's atom lived on in the overview only."""
    line = {"text": "Install one (1) new 98 inch display in room 247 with its wall mount.", "cites": ["a98"], "kind": "fact"}
    content = {"overview": [{"text": "Provider will install a 98-inch display in room 247.", "cites": ["a98"], "kind": "fact"}],
               "scope": [line, {"text": "Closeout", "cites": [], "kind": "heading"}]}
    m._traced(content, "tidy", lambda: content.__setitem__("scope", content["scope"][1:]))
    assert content["scope"][0] is line


def test_a_summary_line_leaning_on_a_quoted_line_is_still_droppable():
    content = {"_noise_atoms": ["q"],
               "overview": [{"text": "Provider will manage receipt, bench and disposal.", "cites": ["q", "r"], "kind": "fact"}]}
    m._traced(content, "grounding", lambda: content.__setitem__("overview", []))
    assert content["overview"] == []


def test_an_exclusion_citing_unrelated_chatter_stays_an_exclusion():
    """010284: "Lift rental, boom-lift services ..." cited pole talk and became
    "Provide lift rental ..." in the task list."""
    content = {"overview": [], "scope": [],
               "out_of_scope": [{"text": "Lift rental, boom-lift services, and other specialty access equipment.",
                                 "cites": ["p"], "kind": "fact"}]}
    m._grounding(content, [_fact("p", "That could be on a 60 foot pole and we have to bring out a boom lift.")])
    assert not any("lift" in x["text"].lower() for x in content["scope"])
    assert content["out_of_scope"][0]["text"].startswith("Lift rental")
    assert content["out_of_scope"][0]["cites"] == []


def test_a_requested_service_flipped_into_an_exclusion_is_still_scope():
    content = {"overview": [], "scope": [],
               "out_of_scope": [{"text": "End-user support for O365 clients is excluded from the service scope.",
                                 "cites": ["s"], "kind": "fact"}]}
    m._grounding(content, [_fact("s", "End-user support for O365 clients")])
    assert content["scope"] and content["scope"][0]["text"].startswith("Provide end-user support")


def test_a_zero_quantity_row_is_an_exclusion_never_a_task():
    """010147: "6U-10U servers x 0" became "Provide 6U-10U servers are not included"."""
    content = {"overview": [], "scope": [],
               "out_of_scope": [{"text": "6U-10U servers are not included.", "cites": ["z"], "kind": "fact"}]}
    m._grounding(content, [_fact("z", "6U-10U servers x 0")])
    assert not content["scope"]
    assert content["out_of_scope"][0]["text"].startswith("6U-10U servers")
