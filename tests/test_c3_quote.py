"""Quote heads client: off by default, and a dead server never breaks a compile."""
import types

from app.core import c3_type_base as c


def _atoms():
    # A compile's atoms hold their words in raw_text (the quote saw no text when it read .text).
    return [types.SimpleNamespace(id="atm_1", raw_text="Install 4 TVs in the office", atom_type="deal_summary")]


def test_quote_off_by_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_C3_QUOTE", raising=False)
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://127.0.0.1:9")
    assert c.quote(_atoms(), {"deal": {"deal_name": "x"}}) == {}


def test_quote_server_down_is_a_visible_degraded_quote(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SOWSMITH_C3_QUOTE_S", "1")
    monkeypatch.delenv("SOWSMITH_C3_FALLBACK", raising=False)
    q = c.quote(_atoms(), {"deal": {"deal_name": "x"}, "sections": {"scope_of_work": [{"id": "atm_1"}]}})
    assert q["degraded"] is True and q["by"] == "none" and q["lines"] == [] and q["error"]


def test_quote_sends_atoms_and_brief(monkeypatch):
    seen = {}
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")

    def fake(path, body, timeout):
        seen.update(path=path, body=body)
        return {"job_type": "av", "lines": [{"techs": 2}]}

    monkeypatch.setattr(c, "_request", fake)
    out = c.quote(_atoms(), {"deal": {"deal_name": "CheckOut 4 TV Install", "account_name": "CDW"},
                             "sections": {"scope_of_work": [{"id": "atm_1", "text": "t"}]}, "job_details": []}, deal_id="d1")
    assert out["job_type"] == "av" and seen["path"] == "/v1/quote"
    deal = seen["body"]["deal"]
    assert deal["deal_name"] == "CheckOut 4 TV Install" and deal["account"] == "CDW"
    assert deal["atoms"][0] == {"id": "atm_1", "text": "Install 4 TVs in the office", "type": "deal_summary"}
    assert deal["site_count"] == 0 and deal["sites"] == []
    assert deal["c3_brief"]["sections"]["scope_of_work"] == [{"id": "atm_1"}]


# ---------------------------------------------------------------- the heads' readings ride with the quote

def _fake(seen):
    def fake(path, body, timeout):
        seen.update(body=body)
        return {"job_type": "av", "lines": [{"techs": 2}]}
    return fake


def test_quote_sends_the_heads_readings_kit_and_left_out_lines_and_site_roster(monkeypatch):
    seen = {}
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    monkeypatch.setattr(c, "_request", _fake(seen))
    import app.core.c3_brief as cb
    monkeypatch.setattr(cb, "_deal_kit", lambda a: a.id == "kit_1")
    monkeypatch.setattr(c, "_DEAL", {
        "reads": {"atm_1": {"read:labor_hours": ["present", 0.93, 6.0], "read:crew_size": ["present", 0.88],
                            "read:co_action": ["keep", 0.99]},
                  "s1": {"jdg:site_role": ["job_site", 0.97]},
                  "s3": {"jdg:site_role": ["vendor", 0.98]}},
        "deal": {"jdg:billing_type": ["fixed", 0.91], "jdg:travel_scope": ["none", 0.8], "other": ["x", 1.0]},
        "view": "quoting"})
    S = types.SimpleNamespace
    atoms = [S(id="atm_1", raw_text="Two techs, about 6 hours", atom_type="task"),
             S(id="kit_1", raw_text="101 stores x 4h", atom_type="task"),
             S(id="rej_1", raw_text="CDW quote 12 cameras", atom_type="vendor_line_item"),
             S(id="s1", raw_text="12 Main St, Austin", atom_type="physical_site", entity_keys=["site:a"]),
             S(id="s2", raw_text="Austin store", atom_type="physical_site", entity_keys=["site:a"]),
             S(id="s4", raw_text="9 Elm St, Dallas", atom_type="physical_site", entity_keys=["site:b"]),
             S(id="s3", raw_text="CDW, 200 N Milwaukee", atom_type="physical_site", entity_keys=["site:cdw"])]
    brief = {"deal": {"deal_name": "D"}, "sections": {}, "left_out": {"ids": ["rej_1"]}}
    c.quote(atoms, brief, deal_id="d1")
    deal = seen["body"]["deal"]
    by = {r["id"]: r for r in deal["atoms"]}
    # only the quote heads' readings, with a number head's value when the read had one
    assert by["atm_1"]["reads"] == {"read:labor_hours": ["present", 0.93, 6.0], "read:crew_size": ["present", 0.88]}
    assert by["kit_1"]["deal_kit"] is True and by["rej_1"]["left_out"] is True
    assert deal["c3_brief"]["left_out"] == ["rej_1"]
    # the heads' site roster, not 0; a sure vendor address is not a job site
    assert deal["site_count"] == 2
    assert [s["site"] for s in deal["sites"]] == ["site:a", "site:b"]
    assert deal["sites"][0]["atom_ids"] == ["s1", "s2"] and deal["sites"][0]["by"] == "heads:physical_site"
    assert deal["heads"]["asked"] is True and deal["heads"]["view"] == "quoting"
    assert deal["heads"]["deal"] == {"jdg:billing_type": ["fixed", 0.91], "jdg:travel_scope": ["none", 0.8]}


def test_quote_heads_are_asked_in_the_read_when_the_quote_is_on(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_BRIEF", "on")
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setattr(c, "_DEAL", {"view": "quoting"})
    monkeypatch.setattr(c, "brief_questions", lambda: [])
    heads = c._brief_body({})["line_heads"]
    for h in ("read:labor_hours", "read:crew_size", "read:tech_level", "read:co_hours_estimate", "jdg:commercial"):
        assert h in heads
    assert len(heads) == len(set(heads))
    # a planning read never asks them (they would read the answer off our own SOW)
    monkeypatch.setattr(c, "_DEAL", {"view": "planning"})
    assert "read:labor_hours" not in c._brief_body({})["line_heads"]


def test_quote_without_a_url_is_degraded_or_raises_under_never(monkeypatch):
    import pytest
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.delenv("SOWSMITH_C3_URL", raising=False)
    monkeypatch.setattr(c, "_url", lambda: "")
    monkeypatch.delenv("SOWSMITH_C3_FALLBACK", raising=False)
    assert c.quote(_atoms(), {})["degraded"] is True
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    with pytest.raises(c.C3HeadsUnavailable):
        c.quote(_atoms(), {})
