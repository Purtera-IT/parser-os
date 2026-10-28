"""SOW rules that ran over a sure head answer on dev (2026-10-28 end-to-end check):
source_polarity reversed exclusions the type head read as exclusions (010253), and
the ship-out list rule made an item the supplier head read as ours customer-provided
(010309)."""
from __future__ import annotations

from app.core import c3_sow as m


def _fact(i, text, section="scope_of_work", **kw):
    return {"id": i, "section": section, "text": text, "under": kw.get("under", ""), "type": kw.get("type", ""),
            "supplier": kw.get("supplier"), "owner": kw.get("owner"), "pictured": False, "quoted": False,
            "asked": False, "when": ""}


_TRENCH = "Trenching, excavation, concrete work, welding, fabrication, roof penetration, conduit installation, permanent electrical work, or utility service."


def test_an_exclusion_the_type_head_read_is_never_turned_into_a_task():
    """010253: the line sat under OUT OF SCOPE, typed exclusion at 1.0, and the
    brief filed it in scope_of_work; source_polarity wrote "Provide trenching ..."."""
    content = {"scope": [], "assumptions": [], "customer_responsibilities": [], "overview": [],
               "out_of_scope": [{"text": _TRENCH, "cites": ["x"], "kind": "fact"}]}
    fs = [_fact("x", _TRENCH, type="exclusion")]
    m._source_polarity(content, fs, {"atoms": []})
    assert not content["scope"]
    assert content["out_of_scope"][0]["text"] == _TRENCH


def test_a_line_under_an_exclusions_heading_is_never_turned_into_a_task():
    content = {"scope": [], "assumptions": [], "customer_responsibilities": [], "overview": [],
               "out_of_scope": [{"text": "Network design and firmware development.", "cites": ["n"], "kind": "fact"}]}
    fs = [_fact("n", "Network design and firmware development support", type="scope_item", under="OUT OF SCOPE")]
    m._source_polarity(content, fs, {"atoms": []})
    assert not content["scope"]


def test_a_requested_service_written_as_excluded_still_turns_into_a_task():
    """000132 (unchanged): a request with no exclusion type or heading keeps its own polarity."""
    content = {"scope": [], "assumptions": [], "customer_responsibilities": [], "overview": [],
               "out_of_scope": [{"text": "End-user support for O365 clients is excluded.", "cites": ["s"], "kind": "fact"}]}
    m._source_polarity(content, [_fact("s", "End-user support for O365 clients")], {"atoms": []})
    assert content["scope"] and content["scope"][0]["text"].startswith("Provide end-user support")


def test_a_list_sibling_the_supplier_head_reads_as_ours_stays_out_of_customer_materials():
    """010309: "Coiled sensor cable attached to the Sensor Interface Box" (col:supplier us 0.93)
    sat beside the customer's "2\" Port" and was listed as customer-provided."""
    atoms = [{"id": "port", "artifact_id": "doc", "text": "2\" Port", "locator": {"line_start": 10}},
             {"id": "coil", "artifact_id": "doc", "text": "Coiled sensor cable attached to the box", "locator": {"line_start": 11}},
             {"id": "cord", "artifact_id": "doc", "text": "Spare sensor cables", "locator": {"line_start": 12}}]
    fs = [_fact("coil", "Coiled sensor cable attached to the box", supplier=["us", 0.93]),
          _fact("cord", "Spare sensor cables", supplier=["us", 0.5])]
    content = {"customer_materials": [{"text": "2\" Port", "cites": ["port"], "kind": "fact"}]}
    m._ship_list_siblings(content, {"atoms": atoms}, fs)
    got = [x["text"] for x in content["customer_materials"]]
    assert "Coiled sensor cable attached to the box" not in got
    assert "Spare sensor cables" in got  # the head is unsure there: the list rule still fills in
