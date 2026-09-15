"""resolve(neural_head=False) answers from exemplar similarity alone."""
from app.core import feedback_store as fs
from app.core.decide import DecisionScope


def test_the_head_is_never_consulted_when_the_caller_declines_it(monkeypatch):
    st = fs.FeedbackStore(":memory:", embed_fn=None, reachable_fn=lambda: True)
    calls = []
    monkeypatch.setattr(st, "_relation_head", lambda *a, **k: calls.append(1) or None)
    st.resolve(relation="atom_type", text="x", candidates=["scope_item"], context="", scope=DecisionScope(),
               instruction="", relations=None, neural_head=False)
    assert calls == []
    st.resolve(relation="atom_type", text="x", candidates=["scope_item"], context="", scope=DecisionScope(),
               instruction="", relations=None)
    assert len(calls) <= 1  # the default path may consult it (only when corrections exist)
