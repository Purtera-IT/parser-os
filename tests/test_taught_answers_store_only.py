"""open_question_resolution asks the store only: a model's verdict was never
used, so it must never be asked (2026-10-15, 204 s median stage)."""
from types import SimpleNamespace

from app.core import decide as decide_mod
from app.core import semantic_role
from app.core import taught_answers


def test_an_open_question_never_reaches_the_model(monkeypatch):
    monkeypatch.setattr(semantic_role, "classify_role",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("asked the model")))
    seen = {}
    real = decide_mod.decide

    def spy(*a, **k):
        seen.setdefault("llm", []).append(k.get("llm", True))
        return real(*a, **k)

    monkeypatch.setattr(taught_answers, "decide", spy, raising=False)
    monkeypatch.setattr(decide_mod, "decide", spy)
    prev = decide_mod.get_store()
    decide_mod.set_store(SimpleNamespace(resolve=lambda **k: None))
    try:
        q = SimpleNamespace(id="q1", atom_type="open_question", raw_text="Who supplies the ladder?",
                            value={"question": "Who supplies the ladder?"}, review_flags=[])
        taught_answers.resolve_taught_answers([q], project_id="d")
    finally:
        decide_mod.set_store(prev)
    assert seen.get("llm") and not any(seen["llm"])
