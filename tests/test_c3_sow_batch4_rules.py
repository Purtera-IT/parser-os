"""SOW writer rules, batch 4 (000132 pass T, C+): a line's meaning keeps its
source's polarity; the summary names the sites and every service; standard
steps sit under the service they support; the template's generic contact
line does not swallow a standard that says more."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "pictured": False, "quoted": False, "asked": False, "when": ""}


def _atom(i, text, atom_type="deal_metadata", was="scope_item"):
    return {"id": i, "text": text, "atom_type": atom_type,
            "structured": {"alt_atom_types": [was], "c3_label": {"type": atom_type, "was": was}}}


def test_an_exclusion_its_source_does_not_state_turns_into_the_requested_service():
    """000132: the requested "End-user support including O365/M365 clients"
    (a type-head relabel to deal_metadata, placed under assumptions) was
    written "End-user support, including O365/M365 clients, is excluded."."""
    src = "End-user support including O365/M365 clients"
    content = {"overview": [],
               "scope": [{"text": "Network Support", "cites": [], "kind": "heading"},
                         {"text": "Support physical network components.", "cites": ["n"], "kind": "fact"},
                         {"text": "Closeout", "cites": [], "kind": "heading"},
                         {"text": "Confirm completion with the onsite contact before leaving site.", "cites": [], "kind": "method",
                          "standard": True}],
               "assumptions": [{"text": "End-user support, including O365/M365 clients, is excluded.", "cites": ["e"], "kind": "fact"}]}
    fs = [_fact("e", src, section="assumptions", type="deal_metadata"), _fact("n", "Support for physical network components (LAN, WLAN)")]
    m._traced(content, "source_polarity", m._source_polarity, content, fs, {"atoms": [_atom("e", src)]})
    texts = [x["text"] for x in content["scope"]]
    assert content["assumptions"] == []
    assert "Provide end-user support, including O365/M365 clients." in texts
    assert texts.index("End User Support") + 1 == texts.index("Provide end-user support, including O365/M365 clients.")
    assert texts.index("End User Support") < texts.index("Closeout")
    d = content["rule_decisions"][0]
    assert (d["rule"], d["head"], d["was"], d["value"]) == ("source_polarity", "sow_section", "assumptions", "scope_of_work")


def test_an_exclusion_with_a_negating_or_limiting_source_stays():
    content = {"scope": [], "assumptions": [
        {"text": "Replacement parts are not included.", "cites": ["p"], "kind": "fact"},
        {"text": "Cabling is not included; Customer's team will handle it.", "cites": ["c"], "kind": "fact"}]}
    fs = [_fact("p", "we dont provide parts, engineering support only"), _fact("c", "Their IT team will handle the cabling")]
    m._source_polarity(content, fs, {"atoms": []})
    assert len(content["assumptions"]) == 2 and content["scope"] == []


def test_a_reversed_line_whose_source_is_not_work_goes():
    content = {"scope": [], "customer_responsibilities": [
        {"text": "Parking will not be provided by Customer.", "cites": ["g"], "kind": "fact"}]}
    m._source_polarity(content, [_fact("g", "Parking garage on Elm Street", section="job_details", type="site_condition")],
                       {"atoms": []})
    assert content["customer_responsibilities"] == [] and content["scope"] == []


def test_a_bound_or_a_site_condition_is_not_an_exclusion():
    content = {"scope": [], "assumptions": [
        {"text": "The longest cable run does not exceed two hundred (200) feet.", "cites": ["r"], "kind": "fact"},
        {"text": "No concealed utilities or structural obstructions are present in the work areas.", "cites": ["w"], "kind": "fact"}]}
    m._source_polarity(content, [_fact("r", "Run to IDF: 200 ft"), _fact("w", "Open ceiling in the work areas")], {"atoms": []})
    assert len(content["assumptions"]) == 2


def test_standards_and_uncited_lines_are_the_templates():
    content = {"scope": [], "assumptions": [
        {"text": "All work is performed on site; remote IT support is not included.", "cites": [], "kind": "method", "standard": True}]}
    m._source_polarity(content, [], {"atoms": []})
    assert len(content["assumptions"]) == 1
