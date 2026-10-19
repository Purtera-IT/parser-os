"""SOW writer rules, batch 3: the drop guard's holes (010217), exclusions that
are not requests (010284, 010147), third-party supply and terms (010376),
cited counts and the after-hours basis (010083), pack-and-move jobs (010147)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


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


def _av_scope():
    return {"job_family": "av_conference_room",
            "scope": [{"text": "Training Room 250", "cites": [], "kind": "heading"},
                      {"text": "Remove the two (2) existing 55 inch displays for relocation to the NOC.", "cites": ["r"], "kind": "fact"},
                      {"text": "Room 247", "cites": [], "kind": "heading"},
                      {"text": "Install one (1) new 98 inch display with its wall mount.", "cites": ["n"], "kind": "fact"},
                      {"text": "Closeout", "cites": [], "kind": "heading"}]}


_FACTS_217 = ["They have (2) 55 inch Philips displays mounted on a backwall. They want to replace these with (2) larger displays.",
              "Those (2) 55 inch Philips displays would then need to be installed in their NOC (same location).",
              "The existing mounts should be to be reused as they support 55-100 inch displays.",
              "Lastly, they are buying (1) new 98 inch Philips display and need it installed in room 247. This will also need a backing board installed."]


def test_reused_mounts_do_not_rule_out_mounting_the_new_displays():
    content = _av_scope()
    m._apply_standards(content, "av_conference_room", _FACTS_217)
    texts = [x["text"] for x in content["scope"]]
    assert "Mount each display at the approved location and height." in texts
    assert "Adjust the mounts for alignment and spacing." in texts


def test_relocated_displays_are_not_decommissioned():
    """010217: "replace these" fired the removal/staging lines while the
    displays move to the NOC."""
    content = _av_scope()
    m._apply_standards(content, "av_conference_room", _FACTS_217)
    every = " ".join(x["text"] for k in ("scope", "customer_responsibilities") for x in content.get(k) or [])
    assert "designated for removal" not in every and "Stage removed equipment" not in every
    assert "approved for removal" not in every
    assert "Transport the relocated displays" in every
    # a real removal still decommissions
    content = _av_scope()
    m._apply_standards(content, "av_conference_room", ["Remove the existing projector and install a new 75 inch display."])
    assert any("designated for removal" in x["text"] for x in content["scope"])


def test_steps_for_every_room_get_their_own_heading():
    content = _av_scope()
    m._apply_standards(content, "av_conference_room", _FACTS_217)
    texts = [x["text"] for x in content["scope"]]
    room247 = texts.index("Room 247")
    dress = texts.index("Dress and secure accessible cabling associated with each display.")
    assert "All Work Areas" in texts[room247:dress]


def _env_376():
    atoms = [{"id": "t1", "artifact_id": "prop", "text": "Box Truck w/ Liftgate", "locator": {}},
             {"id": "t2", "artifact_id": "prop", "text": "ThinkTLS will provide dedicated vehicles for the secure transportation of equipment.", "locator": {}},
             {"id": "t3", "artifact_id": "prop", "text": "Recycling, certificates and ESG reporting are done offsite.", "locator": {}},
             {"id": "k1", "artifact_id": "mail1", "text": "The commodity value for UPS's is only $.15/pound, so your client would have to cut a PO.",
              "locator": {"sender": "kgrove@thinktls.com"}},
             {"id": "q1", "artifact_id": "quote", "text": "*Tax is estimated. Invoice will include the full and final tax due.", "locator": {}},
             {"id": "c1", "artifact_id": "mail2", "text": "Replace a 5K UPS and eWaste the old UPS.", "locator": {"sender": "fhasan@guidewire.com"}},
             {"id": "p1", "artifact_id": "mail3", "text": "Bring a step ladder.", "locator": {"sender": "aj@purtera-it.com"}}]
    docs = [{"artifact_id": "prop", "filename": "Exton PA - UPS Pickup Proposal Final.pdf"},
            {"artifact_id": "quote", "filename": "SHI Quote-28013789.pdf"}]
    return {"atoms": atoms, "documents": docs, "crm": {"account_name": "SHI", "deal_name": "Guidewire UPS"}}


def test_a_third_partys_supply_terms_and_work_are_never_providers():
    """010376: the recycler's truck, vehicles and packing supplies were
    Technician Provided Tools; its $/lb note and the hardware quote's tax line
    were PurTera assumptions; the overview had Provider eWaste the units."""
    env = _env_376()
    fs = [_fact(a["id"], a["text"], supplier=["us", 0.9]) for a in env["atoms"]]
    fs[0]["supplier"] = None
    content = {"end_customer": "Guidewire",
               "overview": [{"text": "Provider will replace the UPS and eWaste the removed UPS after the upgrade.", "cites": ["c1"], "kind": "fact"}],
               "scope": [{"text": "Remove the existing UPS for eWaste.", "cites": ["c1"], "kind": "fact"}],
               "provider_tools": [{"text": "A box truck with a liftgate for transport.", "cites": ["t1"], "kind": "fact"},
                                  {"text": "Dedicated vehicles for secure transportation.", "cites": ["t2"], "kind": "fact"},
                                  {"text": "A step ladder.", "cites": ["p1"], "kind": "fact"}],
               "provider_materials": [{"text": "Packing materials.", "cites": ["zz"], "kind": "fact"}],
               "assumptions": [{"text": "The UPS units have a commodity value of $0.15/lb.", "cites": ["k1"], "kind": "fact"},
                               {"text": "Tax is estimated; the invoice will include the full and final tax due.", "cites": ["q1"], "kind": "fact"}],
               "out_of_scope": []}
    m._traced(content, "third_party", m._third_party, content, env, fs)
    assert [x["text"] for x in content["provider_tools"]] == ["A step ladder."]
    assert content["provider_materials"] == []  # no supplier answer at all
    asm = [x["text"] for x in content["assumptions"]]
    assert "ThinkTLS provides a box truck with a liftgate for transport." in asm
    assert not any("commodity" in t or "Tax" in t for t in asm)
    assert any(t.startswith("ThinkTLS performs the offsite transport and recycling") for t in asm)
    assert "eWaste" not in content["overview"][0]["text"] and "eWaste" not in content["scope"][0]["text"]
    assert any("recycling" in x["text"] for x in content["out_of_scope"])
    assert any(d["rule"] == "third_party" for d in content["rule_decisions"])


def test_no_third_party_no_change():
    env = {"atoms": [{"id": "a", "artifact_id": "m", "text": "We will provide the cable.", "locator": {"sender": "x@cdw.com"}}],
           "crm": {"account_name": "CDW"}}
    content = {"provider_materials": [{"text": "Cat6 cable.", "cites": ["a"], "kind": "fact"}], "assumptions": []}
    m._third_party(content, env, [_fact("a", "We will provide the cable.", supplier=["us", 0.7])])
    assert content["provider_materials"][0]["text"] == "Cat6 cable."


def test_an_acronym_fact_the_writer_left_out_goes_to_the_must_place_pass(monkeypatch):
    """010376: "they will need PurTera's COI" was kept by the brief, silently dropped."""
    seen = {}
    monkeypatch.setattr(m, "_complete", lambda prompt, *a, **k: seen.setdefault("p", prompt) and '{"add": []}')
    content = {k: [] for k in m.LISTS}
    content["scope"] = [{"text": "Install the UPS.", "cites": ["u"], "kind": "fact"}]
    m._must_include(content, [_fact("coi", "Also, they will need PurTera's COI (certificate of insurance) if you are able to provide that.",
                                    section="customer_responsibilities")], None)
    assert "COI" in seen.get("p", "")


def test_ups_replacement_standards_gate_on_a_ups_swap_only():
    fixed = m.purtera_standards("smart_hands_general")
    ups = [t for t in fixed["method_steps"] if "UPS" in t]
    assert ups and all(m._standard_applies(t, ["Replace a 5K UPS (Power input connector type L6-30P)"])[0] for t in ups)
    assert not any(m._standard_applies(t, ["Action items: follow-ups by end of week"])[0] for t in ups)
