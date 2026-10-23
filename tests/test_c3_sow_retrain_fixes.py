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


# ---- step 0: a parser atom_type never vetoes or decides; the type head does.

def _typed(i, text, parser_type, head_type=None, **extra):
    st = dict(extra.pop("structured", {}))
    if head_type:
        st["c3_label"] = {"type": head_type, "p": 0.9}
    return {"id": i, "text": text, "atom_type": parser_type, "structured": st, **extra}


def test_program_atoms_ignore_the_parser_chatter_type():
    # c3_sow :588 -- the parser's small_talk no longer hides a program line.
    note = {"source": "hubspot_note"}
    env = {"atoms": [_typed("a1", "Leverage self-installation where feasible for the remote sites", "small_talk",
                            structured=note),
                     _typed("a2", "Self-install kits ship to each site before the go-live week", "scope_item",
                            "small_talk", structured=note)]}
    assert [x["id"] for x in m._program_atoms(env)] == ["a1"]


def test_standards_gate_reads_head_type_only():
    # c3_sow :1734 -- a parser physical_site/small_talk line still feeds the gates.
    env = {"atoms": [_typed("a1", "Mount the gateway in the IDF closet", "physical_site", artifact_id="d1"),
                     _typed("a2", "Gateway Road, Springfield", "scope_item", "physical_site", artifact_id="d1")]}
    assert m._gate_atoms(env, set(), {"d1"}) == ["Mount the gateway in the IDF closet"]


def test_chatter_cites_are_the_head_labels_only():
    # c3_sow :1762 -- the parser's signature type does not make a cite chatter.
    env = {"atoms": [_typed("a1", "This work is for a new unit opening", "signature"),
                     _typed("a2", "Thanks so much!", "scope_item", "small_talk")]}
    assert m._chatter_ids(env) == {"a2"}


def test_noise_ignores_the_parser_chatter_type():
    # c3_sow :3426
    env = {"atoms": [_typed("a1", "Install four cameras in the lobby", "small_talk"),
                     _typed("a2", "Have a great weekend everyone", "scope_item", "small_talk")]}
    ids, _ = m._noise(env, [])
    assert ids == ["a2"]


def test_recited_exclusion_may_cite_a_parser_small_talk_line():
    # c3_sow :7703 -- the boundary line typed small_talk by the parser only is cited.
    def run(head):
        c = {"out_of_scope": [], "dropped": [{"list": "out_of_scope", "reason": "check: cites",
                                              "text": "Extensive raceway installation and long cable routing."}]}
        atoms = [_typed("a1", "Work wire mold / basic cable cleanup into the install pricing", "small_talk", head)]
        m._recite_exclusions(c, atoms)
        return [x["cites"] for x in c["out_of_scope"]]
    assert run(None) == [["a1"]]
    assert run("small_talk") == []


def test_one_supplier_bar_for_brief_and_writer():
    # regex_retirement S3: c3_sow used .85, c3_brief .6; c3_brief's value is the one bar.
    from app.core import c3_brief

    assert m.SUPPLIER_BAR is c3_brief.SUPPLIER_BAR
    assert c3_brief.SUPPLIER_BAR == 0.6
