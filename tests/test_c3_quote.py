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


def test_quote_server_down_returns_empty(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SOWSMITH_C3_QUOTE_S", "1")
    assert c.quote(_atoms(), {"deal": {"deal_name": "x"}, "sections": {"scope_of_work": [{"id": "atm_1"}]}}) == {}


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
    assert deal["c3_brief"]["sections"]["scope_of_work"] == [{"id": "atm_1"}]


def test_quote_sends_the_site_list_count(monkeypatch):
    # The request hard-coded site_count 0, so the heads' "N places: check the site list" flag never fired (010358).
    from app.core import declared_scope

    seen = {}
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    monkeypatch.setattr(declared_scope, "_found_site_count", lambda atoms: 3)
    monkeypatch.setattr(c, "_request", lambda path, body, timeout: seen.update(body=body) or {"lines": []})
    c.quote(_atoms(), {"deal": {"deal_name": "x"}})
    assert seen["body"]["deal"]["site_count"] == 3


def test_quote_site_count_is_the_site_list(monkeypatch):
    from tests.test_declared_scope import _site

    seen = {}
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    monkeypatch.setattr(c, "_request", lambda path, body, timeout: seen.update(body=body) or {"lines": []})
    c.quote([_site("s1", "houston_77070"), _site("s2", "spring_77380")], {"deal": {"deal_name": "x"}})
    assert seen["body"]["deal"]["site_count"] == 2
