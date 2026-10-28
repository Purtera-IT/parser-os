"""Run O planning-heads wiring (audit 2026-10-28, run 8). Synthetic deals only."""
from types import SimpleNamespace as NS

from app.core import c3_type_base


def test_the_quote_is_sent_the_service_head(monkeypatch):
    seen = {}
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    monkeypatch.setattr(c3_type_base, "_DEAL", {"reads": {}, "view": "planning",
                                                "deal": {"deal:primary_service": ["imac", 0.83],
                                                         "jdg:billing_type": ["t_and_m", 0.9]}})
    monkeypatch.setattr(c3_type_base, "_request", lambda p, b, timeout: seen.update(body=b) or {"lines": []})
    atoms = [NS(id="w1", atom_type="task", review_flags=[], value={}, source_refs=[],
                raw_text="Unbox and connect 40 laptops at the clinic")]
    c3_type_base.quote(atoms, {"deal": {}, "sections": {"scope_of_work": [{"id": "w1"}]}})
    assert seen["body"]["deal"]["heads"]["deal"]["deal:primary_service"] == ["imac", 0.83]
