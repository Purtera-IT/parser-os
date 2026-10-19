"""The SOW writer's deterministic rules: a rule may not drop a cited deal
line that the summary or a standard step only mentions in passing, and the
customer's ship-out list stays the customer's (000123, 010198, 010279, 010085)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": None, "pictured": False, "quoted": False, "asked": False}


def test_grounding_keeps_a_task_the_summary_only_mentions():
    content = {"overview": [{"text": "Provider will install one (1) Square register and one (1) kitchen printer at "
                                     "7021 Roosevelt Rd, Berwyn, IL 60402.", "cites": ["a1", "a2"], "kind": "fact"}],
               "scope": [{"text": "Install one (1) Square register at 7021 Roosevelt Rd, Berwyn, IL 60402.",
                          "cites": ["a1"], "kind": "fact"},
                         {"text": "Confirm completion with the onsite contact.", "cites": [], "kind": "method", "standard": True}]}
    m._grounding(content, [_fact("a1", "We will be setting just 1 Square register and 1 kitchen printer.")])
    assert any("Square register" in x["text"] for x in content["scope"])


def test_grounding_drops_a_line_that_restates_the_summary_whole():
    content = {"overview": [{"text": "Provider will install POS systems at 161 locations.", "cites": ["a1"], "kind": "fact"}],
               "scope": [{"text": "Install POS systems at 161 locations.", "cites": ["a1"], "kind": "fact"},
                         {"text": "Confirm completion with the onsite contact.", "cites": [], "kind": "method", "standard": True}]}
    m._grounding(content, [_fact("a1", "Installation of POS systems at 161 locations")])
    assert not any("POS systems" in x["text"] for x in content["scope"])


def test_tidy_keeps_a_cited_install_the_pooled_standard_words_cover():
    content = {"job_family": "pos_install_refresh",
               "scope": [{"text": "Install the printer at each location.", "cites": ["a1"], "kind": "fact"},
                         {"text": "Install the tablet and stand at each register lane.", "cites": [], "kind": "method", "standard": True},
                         {"text": "Connect each printer to power and perform a test print.", "cites": [], "kind": "method", "standard": True}]}
    m._tidy(content)
    assert any(x["text"] == "Install the printer at each location." for x in content["scope"])


def test_ship_out_list_siblings_follow_their_anchor():
    atoms = [{"id": "t", "artifact_id": "doc", "text": "Samsung Tablet + Stand", "locator": {"line_start": 15}},
             {"id": "p", "artifact_id": "doc", "text": "Printer", "locator": {"line_start": 16}},
             {"id": "r", "artifact_id": "doc", "text": "Reuse barcode scanner and cash drawers for each site.", "locator": {"line_start": 17}},
             {"id": "far", "artifact_id": "doc", "text": "Spare printer", "locator": {"line_start": 60}}]
    content = {"customer_materials": [{"text": "Samsung Tablet + Stand", "cites": ["t"], "kind": "fact"}]}
    m._ship_list_siblings(content, {"atoms": atoms}, [])
    got = [x["text"] for x in content["customer_materials"]]
    assert "Printer" in got
    assert not any("Reuse" in t for t in got)  # a task line beside the list is not an item
    assert "Spare printer" not in got           # far from the list


def test_lead_in_joins_a_list_label():
    atoms = [{"id": "h", "artifact_id": "d", "text": "Overall we'd ship out:", "locator": {"line_start": 31}},
             {"id": "c", "artifact_id": "d", "text": "Cords for the above", "locator": {"line_start": 36}}]
    fs = [{"id": "c", "under": "Customer notes:", "text": "Cords for the above"}]
    m._lead_ins(fs, atoms)
    assert "ship out" in fs[0]["under"]
