"""Writer rule fixes from the pass Z regression map (retrain session, 2026-10-19)."""
from app.core import c3_sow as m


def _env(*atoms):
    return {"atoms": [{"id": i, "text": t, "atom_type": ty} for i, t, ty in atoms]}


def test_an_unworded_exclusion_restating_a_request_becomes_a_task():
    # 000132 pass Z: the requested end-user support was written under Out of Scope.
    content = {"scope": [], "out_of_scope": [{"text": "End-user support, including O365/M365 client support.",
                                              "cites": ["a1"], "kind": "method"}]}
    fs = [{"id": "a1", "text": "End-user support including O365/M365 clients", "type": "deal_metadata",
           "section": "assumptions"}]
    m._source_polarity(content, fs, _env(("a1", "End-user support including O365/M365 clients", "deal_metadata")))
    assert content["out_of_scope"] == []
    assert any("end-user support" in str(x.get("text")).lower() for x in content["scope"])
    assert "End-user support, including O365/M365 client support." in content["_polarity_rejected"]


def test_an_unworded_exclusion_citing_a_caution_stays():
    # 010264: the drywall exclusion cites a remark about repainting; it is no request.
    line = {"text": "Drywall repair, patching, painting, or finish restoration.", "cites": ["a1"], "kind": "fact"}
    content = {"scope": [], "out_of_scope": [line]}
    src = "You may need to repaint part of the wall to hide areas formerly covered by the time clock you replaced."
    m._source_polarity(content, [{"id": "a1", "text": src, "type": "scope_item"}], _env(("a1", src, "scope_item")))
    assert content["out_of_scope"] == [line]


def test_merged_back_never_restores_a_polarity_rejection():
    line = {"text": "End-user support, including O365/M365 client support.", "cites": ["a1"], "kind": "method"}
    content = {"out_of_scope": [], "_polarity_rejected": [line["text"]],
               "_merged": [{"list": "out_of_scope", "target": "Gone target line.", "line": line}]}
    m._merged_back(content, {"out_of_scope": []}, "tidy")
    assert content["out_of_scope"] == []


def test_a_dated_milestone_window_survives_weak_commitments():
    # 010209 pass Z: the install window was dropped as merely mentioned.
    keep = {"text": "Project installs will start on 10/13 and continue through 11/19.", "cites": ["m1"]}
    oncall = {"text": "Work Window: 24/7 on-call availability.", "cites": ["w1"]}
    c = {"schedule": [keep, oncall], "overview": []}
    m._weak_commitments(c, {"m1", "w1"}, [{"id": "m1", "atom_type": "milestone_phase"},
                                         {"id": "w1", "atom_type": "milestone_phase"}])
    assert c["schedule"] == [keep]


def test_a_dated_line_is_never_recited_by_number():
    # 010209 pass Z: "Requested Completion Date: 11/19/2026" took an email header as its cite.
    c = {"schedule": [{"text": "Requested Completion Date: 11/19/2026", "cites": [], "kind": "fact"}]}
    m.check(c, {"hdr": "Sent: 11 Aug 2026 19:40 | Due 8/17/2026"})
    assert all(not x.get("cites") for x in c.get("schedule") or [])


def test_rule_regexes_from_the_regression_map():
    import inspect
    import re
    src = inspect.getsource(m)
    # 4: a comparison with what is installed is not existing gear (010217 TR250 backing board)
    assert "(?<!than )\\bcurrently (have|has|installed)\\b" in src
    pat = re.compile(r"(?i)\b(we|they) (already |currently )?have\b|\balready (have|installed|in place)\b|(?<!than )\bcurrently (have|has|installed)\b")
    assert not pat.search("the new boards must be moved up higher than currently installed")
    assert pat.search("we currently have a Logitech Rally Plus")
    # 5: "Standard Cat6 cable" is not a standards document (010082)
    doc = re.compile(r"(?i)\bstandards?\b(?!\s+(?:[\w-]+\s+){0,2}(?:power|cords?|cables?|mounts?|cat\s?\d\w*|rj-?45|patch))")
    assert not doc.search("Provide Standard Cat6 cable, not plenum-rated cable.")
    assert doc.search("Provide the cabling standards for the naming convention.")
