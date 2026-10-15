"""The C3 base answers document_job_scope's question: it may keep a
conversation on its own word, never set one aside (run C, 2026-10-15)."""
from types import SimpleNamespace

import pytest

from app.core import c3_type_base
from app.core import decide as decide_mod
from app.core import semantic_role
from app.core.decide import Decision
from app.core.document_job_scope import c3_bundle_verdict, judge_documents
from ml.c3 import serve

KEY = c3_type_base.DOCUMENT_JOB_KEY
DEAL = "010162 - CDW- Sodexo SD-WAN Program"


def _atom(doc, text, subject=None):
    value = {"email_thread": {"subject": subject}} if subject else {}
    return SimpleNamespace(source_artifact_id=doc, artifact_id=doc, source_filename=f"{doc}.eml",
                           raw_text=text, value=value, review_flags=[], atom_type="scope_item")


SDWAN = [_atom("art_sdwan", "Statement of Work for a Standard SD-WAN Deployment", subject="Sodexo SDWAN")]
KIOSK = [_atom("art_kiosk", "Confirm ladder availability with Delta", subject="RE: Smart Hands Delta Admin")]


class _Store:
    def __init__(self, verdict_for):
        self.verdict_for = verdict_for

    def resolve(self, *, relation, text, candidates, **_):
        for needle, verdict in self.verdict_for.items():
            if needle in text:
                return Decision(verdict=verdict, confidence=0.95, source="store", correction_id="corr_pm")
        return None


@pytest.fixture
def judge(monkeypatch):
    """The model judge, counting its calls; no store."""
    calls = []

    def _clf(text, cands, **k):
        calls.append(text)
        return ("other_job", 0.95) if "Delta" in text else ("this_deal", 0.9)

    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    prev = decide_mod.get_store()
    decide_mod.set_store(None)
    yield calls
    decide_mod.set_store(prev)


def _base(monkeypatch, verdicts):
    monkeypatch.setattr(c3_type_base, "documents", lambda atoms, *a, **k: verdicts)


def test_off_by_default_the_base_is_never_asked(judge, monkeypatch):
    monkeypatch.delenv("SOWSMITH_C3_DOCUMENT_JOB", raising=False)
    monkeypatch.setattr(c3_type_base, "documents", lambda *a, **k: pytest.fail("asked the base"))
    kept, dropped, _ = judge_documents(SDWAN + KIOSK, deal_name=DEAL)
    assert len(judge) == 2 and [a.artifact_id for a in dropped] == ["art_kiosk"]


def test_a_sure_this_deal_keeps_the_conversation_without_the_model(judge, monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_DOCUMENT_JOB", "on")
    _base(monkeypatch, {"art_sdwan": {KEY: ("this_deal", 0.97)}, "art_kiosk": {KEY: ("other_job", 0.99)}})
    kept, dropped, verdicts = judge_documents(SDWAN + KIOSK, deal_name=DEAL)
    by = {v["filename"]: v for v in verdicts}
    assert by["art_sdwan.eml"]["source"] == "c3" and by["art_sdwan.eml"]["verdict"] == "this_deal"
    # The base's other_job is not acted on: the model judges that one, as before.
    assert len(judge) == 1 and "Delta" in judge[0]
    assert [a.artifact_id for a in dropped] == ["art_kiosk"]
    assert by["art_kiosk.eml"]["source"] != "c3" and by["art_kiosk.eml"]["c3_verdict"] == "other_job"


def test_the_base_never_sets_a_document_aside_on_its_own(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_DOCUMENT_JOB", "on")
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: (None, 0.0))
    prev = decide_mod.get_store()
    decide_mod.set_store(None)
    try:
        _base(monkeypatch, {"art_kiosk": {KEY: ("other_job", 0.999)}})
        kept, dropped, _ = judge_documents(SDWAN + KIOSK, deal_name=DEAL)
    finally:
        decide_mod.set_store(prev)
    assert not dropped and len(kept) == 2


def test_an_unsure_base_leaves_the_conversation_to_the_model(judge, monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_DOCUMENT_JOB", "on")
    _base(monkeypatch, {"art_sdwan": {KEY: ("this_deal", 0.6)}})
    _, _, verdicts = judge_documents(SDWAN, deal_name=DEAL)
    assert len(judge) == 1 and verdicts[0]["source"] != "c3"


def test_a_pm_lesson_still_comes_before_the_base(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_DOCUMENT_JOB", "on")
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: pytest.fail("asked the model"))
    prev = decide_mod.get_store()
    decide_mod.set_store(_Store({"Delta Admin": "other_job"}))
    try:
        _base(monkeypatch, {"art_kiosk": {KEY: ("this_deal", 0.99)}})
        _, dropped, verdicts = judge_documents(KIOSK, deal_name=DEAL)
    finally:
        decide_mod.set_store(prev)
    assert [a.artifact_id for a in dropped] == ["art_kiosk"] and verdicts[0]["source"] == "store"


def test_shadow_records_the_base_and_changes_nothing(judge, monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_DOCUMENT_JOB", "shadow")
    _base(monkeypatch, {"art_sdwan": {KEY: ("this_deal", 0.97)}})
    _, _, verdicts = judge_documents(SDWAN, deal_name=DEAL)
    assert len(judge) == 1 and verdicts[0]["source"] != "c3"
    assert verdicts[0]["c3_verdict"] == "this_deal" and verdicts[0]["c3_confidence"] == 0.97


def test_a_conversation_is_this_deal_only_when_every_document_read_says_so():
    a, b = [_atom("a", "x")], [_atom("b", "y")]
    assert c3_bundle_verdict([a, b], {"a": {KEY: ("this_deal", 0.9)}, "b": {KEY: ("this_deal", 0.85)}}, 0.8) \
        == ("this_deal", 0.85)
    assert c3_bundle_verdict([a, b], {"a": {KEY: ("this_deal", 0.9)}, "b": {KEY: ("this_deal", 0.7)}}, 0.8)[0] is None
    assert c3_bundle_verdict([a, b], {"a": {KEY: ("this_deal", 0.9)}, "b": {KEY: ("other_job", 0.6)}}, 0.8)[0] \
        == "other_job"
    assert c3_bundle_verdict([a], {}, 0.8) == (None, 0.0)


def test_the_service_returns_document_judgments_from_the_same_read():
    seen = {}

    def read(model, deal, schema, **kw):
        seen.update(kw)
        return {"lines": {"k1": {"reads": {serve.TYPE_KEY: {"answer": "task", "p": 0.91},
                                           "read:sow_section": {"answer": "scope", "p": 0.8}}}},
                "documents": {"doc1": {KEY: {"answer": "this_deal", "p": 0.93}}}}

    base = serve.Base(load=lambda: (object(), object(), "cpu"), read=read)
    base.load()
    deal = {"deal_id": "d", "company": "purtera", "atoms": [
        {"key": "k1", "text": "t", "entered_at": 0, "doc_id": "doc1", "doc_kind": "email", "section": "",
         "order": 0, "speaker_role": "", "speaker_side": "", "atom_id": "k1"}]}
    out = base.read(deal, 30.0, line_heads=("read:sow_section",), document_heads=(KEY,))
    assert out["types"] == {"k1": ["task", 0.91]}
    assert out["reads"] == {"k1": {"read:sow_section": ["scope", 0.8]}}
    assert out["documents"] == {"doc1": {KEY: ["this_deal", 0.93]}}
    assert seen["document_heads"] == (KEY,) and "read:sow_section" in seen["heads"]
    plain = base.read(deal, 30.0)
    assert set(plain) == {"types", "read_ms", "lines"}


def test_a_request_names_heads_as_a_list_of_strings():
    assert serve._asked(None) == ()
    assert serve._asked([KEY, KEY, " "]) == (KEY,)
    with pytest.raises(ValueError):
        serve._asked(KEY)



def test_a_full_read_returns_links_and_whole_deal_answers():
    seen = {}

    def read(model, deal, schema, **kw):
        seen.update(kw)
        return {"lines": {"k1": {"reads": {serve.TYPE_KEY: {"answer": "task", "p": 0.9}}}},
                "links": [{"src": "k1", "dst": "k2", "relation": "answers", "p": 0.81}],
                "deal": {"deal:primary_service": {"answer": "low_voltage_cabling", "p": 0.7},
                         "deal:declared_site_count": {"value": 3.0}}}

    base = serve.Base(load=lambda: (object(), object(), "cpu"), read=read)
    base.load()
    deal = {"deal_id": "d", "company": "purtera", "atoms": [
        {"key": "k1", "text": "t", "entered_at": 0, "doc_id": "doc1", "doc_kind": "email", "section": "",
         "order": 0, "speaker_role": "", "speaker_side": "", "atom_id": "k1"}]}
    out = base.read(deal, 30.0, full=True)
    assert seen["lines_only"] is False
    assert out["links"] == [{"src": "k1", "dst": "k2", "relation": "answers", "p": 0.81}]
    assert out["deal"]["deal:primary_service"] == ["low_voltage_cabling", 0.7]
    assert out["deal"]["deal:declared_site_count"] == [3.0, None]
    assert base.read(deal, 30.0)["types"] and seen["lines_only"] is True
