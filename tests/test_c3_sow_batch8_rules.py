"""SOW writer rules, batch 8 (pass W grades: 000132 C+)."""
from __future__ import annotations

import re

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


def _content():
    return {k: [] for k in m.LISTS}


def _texts(c, k):
    return [x["text"] for x in c.get(k) or [] if isinstance(x, dict)]


_NEG = re.compile(r"(?i)\b(not|no|never|exclud\w*)\b")


def _end_user_case():
    c = _content()
    c["scope"] = [{"text": "Network support", "cites": [], "kind": "heading"},
                  {"text": "Support physical network components (LAN, WLAN).", "cites": ["f2"], "kind": "fact"},
                  {"text": "Closeout", "cites": [], "kind": "heading"},
                  {"text": "Confirm completion with the onsite contact before leaving site.", "cites": [], "standard": True}]
    c["assumptions"] = [{"text": "Assume support does not include end-user support, including O365/M365 client support.",
                         "cites": ["f1"], "kind": "fact"}]
    fs = [_fact("f1", "End-user support including O365/M365 clients", type="scope_item"),
          _fact("f2", "Support for network (LAN/WLAN)", type="scope_item")]
    return c, fs


# --- 000132 C+: source_polarity never moves a negated line -------------------

def test_a_negated_line_moves_as_the_sources_positive_wording():
    c, fs = _end_user_case()
    m._traced(c, "source_polarity", m._source_polarity, c, fs, {"atoms": []})
    assert c["assumptions"] == []
    scope = _texts(c, "scope")
    assert "Provide end-user support including O365/M365 clients." in scope
    assert not any(_NEG.search(t) for t in scope), scope
    assert "End User Support" in scope and "Does Not Include End" not in scope


def test_a_negated_line_with_no_positive_wording_goes():
    c = _content()
    c["assumptions"] = [{"text": "Support does not include printer support.", "cites": ["f1"], "kind": "fact"}]
    # The source names work but has no clean positive form short enough.
    fs = [_fact("f1", " ".join(["printer support and"] * 12) + " more", type="scope_item")]
    m._source_polarity(c, fs, {"atoms": []})
    assert c["assumptions"] == [] and not any(_NEG.search(t) for t in _texts(c, "scope"))


def test_the_drop_guard_never_restores_what_source_polarity_rejected():
    c, fs = _end_user_case()
    m._traced(c, "source_polarity", m._source_polarity, c, fs, {"atoms": []})
    gone = "Assume support does not include end-user support, including O365/M365 client support."
    before = {("f9",): ("assumptions", gone)}
    lists_before = {"assumptions": [{"text": gone, "cites": ["f9"], "kind": "fact"}]}
    out = m._guard_drops(c, "tidy", before, lists_before)
    assert out == [] and gone not in _texts(c, "assumptions")
    assert any(d["reason"] == "source_polarity rejected it" for d in c.get("guard_declined") or [])


def test_the_summary_never_names_a_negated_line_as_a_service():
    c = _content()
    c["services"] = ["support"]
    c["overview"] = [{"text": "Provider will support physical network components.", "cites": [], "kind": "method"}]
    c["scope"] = [{"text": "Support does not include end-user support.", "cites": ["f1"], "kind": "fact"},
                  {"text": "Coordinate with local stakeholders.", "cites": ["f2"], "kind": "fact"}]
    m._summary_names_job(c)
    ov = " ".join(_texts(c, "overview"))
    assert "does not include" not in ov and "support for does not" not in ov
    assert "coordination with local stakeholders" in ov


# --- 000132 C+: a merge target later removed gives its merged lines back -----

def test_a_suppressed_merge_target_gives_the_merged_line_back():
    c = _content()
    std = {"text": "Coordinate visit priorities with local stakeholders before or during each visit.", "cites": [], "kind": "method", "standard": True}
    deal = {"text": "Coordinate with local stakeholders during each visit.", "cites": ["f1"], "kind": "fact"}
    other = {"text": "Coordinate visit priorities with the remote IT contact during each visit.", "cites": ["f2"], "kind": "fact"}
    c["scope"] = [{"text": "Onsite Support", "cites": [], "kind": "heading"}, std, deal, other]

    def drop_deal(c):  # a dedupe that takes the deal line as the standard's repeat
        c["scope"] = [x for x in c["scope"] if x is not deal]
    m._traced(c, "standards", drop_deal, c)
    assert deal["text"] not in _texts(c, "scope") and "f1" in std["cites"]  # merged

    def drop_std(c):  # a later rule suppresses the standard
        c["scope"] = [x for x in c["scope"] if x is not std]
    m._traced(c, "one_statement", drop_std, c)
    scope = _texts(c, "scope")
    assert deal["text"] in scope, scope
    assert scope.index(deal["text"]) == 1  # where its target stood
    assert any(d.get("guard") == "merged_back" and d["atom_ids"] == ["f1"] for d in c["rule_decisions"])


def test_a_merged_line_stays_merged_while_its_target_stands_or_is_reworded():
    c = _content()
    std = {"text": "Coordinate visit priorities with local stakeholders before or during each visit.", "cites": [], "kind": "method", "standard": True}
    deal = {"text": "Coordinate with local stakeholders during each visit.", "cites": ["f1"], "kind": "fact"}
    c["scope"] = [std, deal]
    m._traced(c, "standards", lambda c: c.__setitem__("scope", [std]), c)
    std["text"] = "Coordinate visit priorities with local stakeholders before or during each onsite visit."
    m._traced(c, "plain", lambda c: None, c)
    assert _texts(c, "scope") == [std["text"]]


# --- 010217 B: supply lines, counts, IT duty, pre-start step, onsite assumption, requester ---

_ATOMS_217 = [
    {"id": "a1", "text": "Those (2) 55” Philips displays would then need to be installed in their NOC (same location). They are ordering a 3rd 55” Philips display to be installed in a typical “3x1” design using articulating mounts."},
    {"id": "a2", "text": "Lastly, they are buying (1) new 98” Philips display and need it installed in room 247. This will also need a backing board installed to support the display and we’re providing a new wall mount for it as well."},
    {"id": "a3", "text": "(3) new 32-65 inch articulating TV mount"},
    {"id": "a4", "text": "(1) new 55-100 inch VESA TV mount"},
    {"id": "a5", "text": "IT will handle any connections outside of the display install."},
    {"id": "a6", "text": "The existing connections will need to be reconnected to the new displays (most likely HDMI cables)."},
] + [{"id": f"n{i}", "text": f"Filler line {i} about the wall and the project mount schedule."} for i in range(8)]


def test_a_supply_line_is_one_item_with_its_count_and_its_own_room():
    c = _content()
    c["customer_materials"] = [
        {"text": "Training Room 250: Customer will furnish (1) Chief Thinstall Large 25 Inch Extension Monitor Arm Wall Mount "
                 "(32-65 inch articulating TV mount) and (1) Chief Fusion X-Large Tilt Wall Mount XTM1U (55-100 inch VESA TV mount) for the install.",
         "cites": ["b1", "b2"], "kind": "fact"},
        {"text": "One (1) new 55-100 inch VESA TV mount.", "cites": ["a4"], "kind": "fact"}]
    m._supply_items(c, _ATOMS_217)
    assert _texts(c, "customer_materials") == [
        "Three (3) Chief Thinstall Large 25 Inch Extension Monitor Arm Wall Mounts (32-65 inch articulating TV mount) for the NOC.",
        "One (1) Chief Fusion X-Large Tilt Wall Mount XTM1U (55-100 inch VESA TV mount) for Room 247."]
    assert "a3" in c["customer_materials"][0]["cites"]


def test_an_inherited_room_label_with_no_own_sentence_is_left_off():
    c = _content()
    c["customer_materials"] = [{"text": "Training Room 250: (2) Philips 86 inch displays.", "cites": ["b1"], "kind": "fact"}]
    m._supply_items(c, [{"id": "x", "text": "Install the shelf in room 101."}])
    assert _texts(c, "customer_materials") == ["Two (2) Philips 86 inch displays."]


def test_a_cited_cable_type_stays_in_the_step():
    c = _content()
    c["scope"] = [{"text": "Reconnect the existing display cabling to the two new displays.", "cites": ["a6"], "kind": "fact"}]
    m._cited_connector(c, _ATOMS_217)
    assert _texts(c, "scope") == ["Reconnect the existing display cabling (most likely HDMI) to the two new displays."]


def test_the_backing_step_names_its_rooms_and_counts():
    c = _content()
    c["scope"] = [{"text": "Training Room 250", "kind": "heading", "cites": []},
                  {"text": "Install the required backing board and mount the two new 86-inch Philips displays higher on the back wall.", "cites": ["x"], "kind": "fact"},
                  {"text": "NOC", "kind": "heading", "cites": []},
                  {"text": "Install the new 55-inch display in the NOC using articulating mounts.", "cites": ["y"], "kind": "fact"},
                  {"text": "Room 247", "kind": "heading", "cites": []},
                  {"text": "Install the new 98-inch Philips display in room 247 with the required backing board and wall mount.", "cites": ["z"], "kind": "fact"},
                  {"text": "All Work Areas", "kind": "heading", "cites": []},
                  {"text": "Install three quarter (3/4) inch plywood backing, approximately twenty four (24) inches by forty eight (48) inches, as required to securely support each wall mount.",
                   "cites": [], "standard": True}]
    m._backing_counts(c)
    assert c["scope"][-1]["text"].endswith("to securely support the wall mounts: two (2) in Training Room 250 and one (1) in Room 247.")


def test_what_it_will_handle_is_a_customer_duty():
    c = _content()
    c["overview"] = [{"text": "Provider will reinstall the displays in the NOC.", "cites": ["a1", "a5"], "kind": "fact"}]
    m._it_handles_duty(c, _ATOMS_217)
    assert _texts(c, "customer_responsibilities") == ["Have Customer IT handle any connections outside of the display install."]


def test_a_before_start_step_leaves_closeout():
    c = _content()
    line = "If needed, obtain current room photos or conduct a site survey to confirm installation conditions before starting work."
    c["scope"] = [{"text": "Arrival", "kind": "heading", "cites": []}, {"text": "Check in with the onsite contact.", "cites": [], "kind": "method"},
                  {"text": "Room 247", "kind": "heading", "cites": []}, {"text": "Install the display.", "cites": ["z"], "kind": "fact"},
                  {"text": "Closeout", "kind": "heading", "cites": []}, {"text": line, "cites": ["q"], "kind": "fact"}]
    m._before_start_not_closeout(c)
    assert _texts(c, "scope").index(line) == 2


def test_an_onsite_assumption_names_only_customer_supplied_items():
    c = _content()
    c["customer_responsibilities"] = [{"text": "Provide all displays, hardware, cables and mounts.", "cites": [], "standard": True}]
    c["assumptions"] = [{"text": "All listed displays, mounts, and backing boards are onsite and ready for installation when Provider arrives.", "cites": [], "kind": "method"}]
    m._onsite_assumption_suppliers(c)
    assert _texts(c, "assumptions") == ["All listed displays and mounts are onsite and ready for installation when Provider arrives."]
    c["assumptions"].append({"text": "Customer-provided equipment is complete, compatible, undamaged and ready for installation.", "cites": [], "standard": True})
    c["assumptions"].insert(0, {"text": "All listed displays, mounts, and backing boards are onsite and ready.", "cites": [], "kind": "method"})
    m._onsite_assumption_suppliers(c)
    assert not any("backing" in t for t in _texts(c, "assumptions"))


def test_requested_by_reads_the_requesters_signature_card():
    env = {"crm": {"account_name": "CDW"}, "atoms": [
        {"id": "s1", "atom_type": "stakeholder", "text": "Ralph Smith III | Consulting Solutions Architect | CDW | 813.373.1644"},
        {"id": "e1", "atom_type": "scope_item", "text": "Hey Trent,", "locator": {"sender": "Ralph Smith III <ralpsmi@cdw.com>", "sent_at": "Wednesday, August 12, 2026 1:48 PM", "quoted": True},
         "structured": {"email_thread": {"to": ["ralpsmi@cdw.com"]}}}]}
    rb = m.requested_by(env)
    assert rb["name"] == "Ralph Smith III" and rb["phone"] == "813.373.1644"
