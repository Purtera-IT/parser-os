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
