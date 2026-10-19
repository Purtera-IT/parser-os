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
