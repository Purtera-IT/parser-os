"""A document about another job for the same customer is set aside, losslessly (live 010162)."""
from types import SimpleNamespace

import pytest

from app.core import decide as decide_mod
from app.core import semantic_role
from app.core.decide import Decision
from app.core.document_job_scope import document_text, judge_documents


def _atom(doc, text, kind=None, subject=None, atom_type="scope_item"):
    value = {}
    if kind:
        value["kind"] = kind
    if subject:
        value["email_thread"] = {"subject": subject}
    return SimpleNamespace(source_artifact_id=doc, source_filename=f"{doc}.eml", raw_text=text,
                           value=value, review_flags=[], atom_type=atom_type)


SDWAN = [
    _atom("art_sdwan", "Can you work up a Statement of Work for a Standard SD-WAN Deployment", subject="Sodexo SDWAN"),
    _atom("art_sdwan", "a single fixed nationwide price for this standard deployment"),
]
KIOSK = [
    _atom("art_kiosk", "Send two technician names, emails, and phone numbers to Dyiesha via email for badge processing",
          subject="RE: CDW Smart Hands SOW Delta Admin 70598001"),
    _atom("art_kiosk", "Confirm ladder availability and network closet access with Delta"),
]
DEAL = "010162 - CDW- Sodexo SD-WAN Program"


class _Store:
    def __init__(self, verdict_for):
        self.verdict_for = verdict_for

    def resolve(self, *, relation, text, candidates, **_):
        if relation != "document_job":
            return None
        for needle, verdict in self.verdict_for.items():
            if needle in text:
                return Decision(verdict=verdict, confidence=0.95, source="store", correction_id="corr_pm")
        return None


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: (None, 0.0))
    prev = decide_mod.get_store()
    yield
    decide_mod.set_store(prev)


def test_a_taught_other_job_sets_the_document_aside(no_llm):
    decide_mod.set_store(_Store({"Delta Admin": "other_job"}))
    kept, dropped, verdicts = judge_documents(SDWAN + KIOSK, deal_name=DEAL, project_id="deal")
    assert [a.source_artifact_id for a in dropped] == ["art_kiosk", "art_kiosk"]
    assert all(a.source_artifact_id == "art_sdwan" for a in kept)
    assert all("other_job" in a.review_flags for a in dropped)
    assert {v["filename"]: v["verdict"] for v in verdicts} == {"art_sdwan.eml": "this_deal", "art_kiosk.eml": "other_job"}


def test_the_model_needs_confidence_and_doubt_keeps_the_document(no_llm, monkeypatch):
    decide_mod.set_store(None)
    monkeypatch.setattr(semantic_role, "classify_role",
                        lambda text, cands, **k: ("other_job", 0.6) if "Delta" in text else ("this_deal", 0.9))
    kept, dropped, _ = judge_documents(SDWAN + KIOSK, deal_name=DEAL)
    assert not dropped and len(kept) == 4
    monkeypatch.setattr(semantic_role, "classify_role",
                        lambda text, cands, **k: ("other_job", 0.92) if "Delta" in text else ("this_deal", 0.9))
    kept, dropped, _ = judge_documents(SDWAN + KIOSK, deal_name=DEAL)
    assert len(dropped) == 2 and len(kept) == 2


def test_nothing_moves_without_a_deal_name_or_any_judge(no_llm):
    decide_mod.set_store(_Store({"Delta Admin": "other_job"}))
    kept, dropped, verdicts = judge_documents(SDWAN + KIOSK, deal_name="")
    assert len(kept) == 4 and not dropped and verdicts == []
    decide_mod.set_store(None)
    kept, dropped, _ = judge_documents(SDWAN + KIOSK, deal_name=DEAL)
    assert len(kept) == 4 and not dropped


def test_document_text_leads_with_the_threads_subject_and_skips_meta():
    meta = _atom("art_kiosk", "note_id=1 | author=x"); meta.value["field_name"] = "hubspot_note_meta"
    atoms = KIOSK + [meta]
    text = document_text(atoms, "art_kiosk.eml")
    assert text.startswith("DOCUMENT: RE: CDW Smart Hands SOW Delta Admin 70598001\n- Send two technician names")
    assert "note_id=1" not in text


def test_the_judge_reads_the_documents_own_words_and_skips_deal_wide_chrome():
    from app.core.document_job_scope import common_lines

    banner = "External sender Check the sender and the content are safe before clicking links or open attachments."
    a = [_atom("art_a", banner), _atom("art_a", "Quoted: earlier message about SD-WAN pricing"),
         _atom("art_a", "This is received. Please see below tech information for badge processing", subject="RE: CDW Smart Hands SOW Delta Admin 70598001")]
    a[1].value["quoted"] = True
    a[2].value["quoted"] = False
    b = [_atom("art_b", banner + " | " + banner), _atom("art_b", "Standard SD-WAN Deployment SOW")]
    common = common_lines(a + b)
    assert banner.lower()[:80] in common
    text = document_text(a, "art_a.eml", common)
    lines = text.split("\n")
    assert lines[0] == "DOCUMENT: RE: CDW Smart Hands SOW Delta Admin 70598001"
    assert lines[1].startswith("- This is received"), lines
    assert "External sender" not in text
