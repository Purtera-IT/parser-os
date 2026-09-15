"""A task is written scope. What somebody said on a call is evidence, not a
unit of work the Deal Kit prices.

Live 000061 (2026-09-15, compile 9a6aacfc): eight transcript utterances --
"But I can check that during the site survey as well too", "I'll also include
a site survey as well too" -- came back from the typed classifier as tasks
beside the one survey line the kit priced. Same model, same prompt, on the
written recap line: a task, rightly.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.core import typed_atom_classifier as tac
from app.core.schemas import AtomType


def _atom(aid, text, value, refs=()):
    return SimpleNamespace(id=aid, atom_type=AtomType.scope_item, raw_text=text, value=value,
                           entity_keys=[], source_refs=list(refs), project_id="p")


def _spoken(aid, text):
    ref = SimpleNamespace(locator={"speaker": "Chase", "utterance_index": 41, "line_start": 90})
    return _atom(aid, text, {"speaker": "Chase", "speaker_role": "internal"}, [ref])


def _written(aid, text):
    ref = SimpleNamespace(locator={"line_start": 12})
    return _atom(aid, text, {"kind": "hubspot_note_body"}, [ref])


def _past_every_layer(monkeypatch, verdicts):
    monkeypatch.delenv("SOWSMITH_DISABLE_LLM", raising=False)
    monkeypatch.delenv("SOWSMITH_TYPED_CLASSIFIER_DISABLE", raising=False)
    monkeypatch.setattr(tac, "_atom_type_deflect_enabled", lambda: False)
    monkeypatch.setattr(tac, "_typed_student_enabled", lambda: False)
    monkeypatch.setattr(tac, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(tac, "_apply_taught_types", lambda atoms: {})
    try:
        from app.core import rubric_gate
        monkeypatch.setattr(rubric_gate, "keep_deflect_flags", lambda texts: [False] * len(texts))
    except Exception:
        pass
    monkeypatch.setattr(tac, "_classify_batch", lambda batch: {
        a.id: {"atom_type": verdicts.get(a.id, "_keep"), "value": {}} for a in batch})


def test_a_spoken_line_the_model_calls_a_task_stays_what_it_was(monkeypatch):
    said = _spoken("u1", "But I can check that during the site survey as well too.")
    wrote = _written("n1", "Conduct site survey during regular business hours to assess AP mounting")
    _past_every_layer(monkeypatch, {"u1": "task", "n1": "task"})
    promoted = tac.classify_atoms([said, wrote])
    assert said.atom_type == AtomType.scope_item
    assert wrote.atom_type == AtomType.task
    assert promoted == 1
    assert tac.get_last_deflect_stats()["deflected"]["speech_not_task"] == 1


def test_speech_may_still_become_any_other_type(monkeypatch):
    said = _spoken("u2", "We can only come in after 6 pm on weekdays.")
    _past_every_layer(monkeypatch, {"u2": "site_access_window"})
    tac.classify_atoms([said])
    assert said.atom_type == AtomType.site_access_window


def test_provenance_is_read_from_the_locator_or_the_value():
    assert tac._is_speech(_spoken("u3", "x"))
    assert tac._is_speech(_atom("u4", "x", {"speaker": "Min Lee"}))
    assert not tac._is_speech(_written("n2", "x"))
    assert not tac._is_speech(_atom("n3", "x", {}))
