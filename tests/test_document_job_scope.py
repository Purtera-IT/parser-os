"""A document about another job for the same customer is set aside, losslessly (live 010162)."""
from types import SimpleNamespace

import pytest

from app.core import decide as decide_mod
from app.core import semantic_role
from app.core.decide import Decision
from app.core.document_job_scope import (
    bundle_documents, bundle_text, document_text, judge_documents, thread_key, verdict_note,
)


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


# ── conversations, not documents ─────────────────────────────────────────────
#
# Live 010198 (2026-09-15): judged one message at a time, three models called
# the "010179 POS Installation 8/2" replies another job because the number in
# front of the subject differed from the deal's; they were the same customer
# confirming the same date. Live 010162: the kiosk packing list judged alone is
# counts and box sizes; beside the smart-hands thread it arrived with, it is
# plainly the close-down.


def _doc(doc, *texts, filename=None, subject=None):
    out = []
    for i, t in enumerate(texts):
        a = _atom(doc, t, subject=subject if i == 0 else None)
        if filename:
            a.source_filename = filename
        out.append(a)
    return out


def _judge_recording(monkeypatch, verdict_for):
    """Stub the model: verdict by a needle in the prompt; records every text asked."""
    asked: list[dict] = []

    def _clf(text, candidates, *, instruction, context="", timeout=None, model=None):
        asked.append({"text": text, "context": context, "model": model, "timeout": timeout})
        for needle, (verdict, conf) in verdict_for.items():
            if needle in text:
                return (verdict, conf)
        return ("this_deal", 0.95)

    monkeypatch.setattr(semantic_role, "classify_role", _clf)
    decide_mod.set_store(None)
    return asked


def test_reply_markers_and_reference_numbers_are_not_the_subject():
    assert thread_key("Re: 010198  Fw: POS Installation 8/2") == "pos installation 8/2"
    assert thread_key("Fw: 010179 POS Installation 8/2") == "pos installation 8/2"
    assert thread_key("RE: CDW Smart Hands SOW Delta Admin 70598001") == "cdw smart hands sow delta admin 70598001"
    assert thread_key("[EXT] 3 Verkada cameras") == "[ext] 3 verkada cameras"
    assert thread_key("70598001") == "70598001"  # a subject that is only a number stays itself
    assert thread_key("") == ""


def test_one_thread_is_judged_once_whatever_number_someone_typed_in_front(no_llm, monkeypatch):
    asked = _judge_recording(monkeypatch, {})
    ours = _doc("art_a", "We will be setting just 1 Square register and 1 kitchen printer.",
                subject="010198 Fw: POS Installation 8/2")
    theirs = _doc("art_b", "Yes, we can hit that date! Looking forward to getting this thing rolling for you.",
                  subject="Re: 010179 POS Installation 8/2")
    plain = _doc("art_c", "What are the next steps to lock in that date?", subject="Re: POS Installation 8/2")
    kept, dropped, verdicts = judge_documents(ours + theirs + plain, deal_name="010198 - Square POS Install Bridgewave")
    assert len(asked) == 1 and len(verdicts) == 1
    assert sorted(verdicts[0]["filenames"]) == ["art_a.eml", "art_b.eml", "art_c.eml"]
    assert asked[0]["text"] == "DOCUMENT: POS Installation 8/2"  # the thread, not what someone typed in front of it
    # The model reads the conversation's lines as context; the text it is
    # asked about -- and a taught verdict is matched on -- is the deal and the subject.
    assert "Yes, we can hit that date!" in asked[0]["context"] and "kitchen printer" in asked[0]["context"]
    assert "kitchen printer" not in asked[0]["text"]
    assert dropped == [] and len(kept) == 3


def test_a_file_joins_the_message_that_carried_it_and_shares_its_verdict(no_llm, monkeypatch):
    asked = _judge_recording(monkeypatch, {"Smart Hands": ("other_job", 0.9)})
    thread = _doc("art_mail", "Confirm ladder availability and network closet access with Delta",
                  filename="010162-hs-email-113968718225.eml", subject="RE: CDW Smart Hands SOW Delta Admin 70598001")
    packing = _doc("art_pdf", "Count: 5 | Size Ordered: 24 x 16 x 16 | Equipment Type: EvD Kiosk Counter",
                   filename="Delta Close Down.pdf")
    index = {
        "010162-hs-email-113968718225.eml": {"subject": "RE: CDW Smart Hands SOW Delta Admin 70598001",
                                             "attachment_ids": ["218364543862"], "external_id": "hs-email:113968718225",
                                             "authored_at": "2026-07-30T16:16:40Z", "source": "email"},
        "Delta Close Down.pdf": {"subject": "", "attachment_ids": [], "external_id": "hs-file:218364543862",
                                 "authored_at": "2026-07-30T16:17:16.895Z", "source": "hubspot"},
    }
    kept, dropped, verdicts = judge_documents(SDWAN + thread + packing, deal_name=DEAL, index=index)
    assert len(asked) == 2  # the SD-WAN mail, and the smart-hands conversation with its file
    v = next(v for v in verdicts if v["verdict"] == "other_job")
    assert v["links"] == {"010162-hs-email-113968718225.eml": "thread", "Delta Close Down.pdf": "attachment"}
    assert "EvD Kiosk Counter" in next(a["context"] for a in asked if "Smart Hands" in a["text"])
    assert {a.source_artifact_id for a in dropped} == {"art_mail", "art_pdf"}
    assert [a.source_artifact_id for a in kept] == ["art_sdwan", "art_sdwan"]


def test_a_mirrored_file_beside_one_message_with_attachments_arrived_with_it():
    thread = _doc("art_mail", "Confirm ladder availability", filename="mail.eml", subject="RE: Smart Hands")
    packing = _doc("art_pdf", "Count: 5 | EvD Kiosk Counter", filename="Delta Close Down.pdf")
    by_doc = {"art_mail": thread, "art_pdf": packing}
    index = {
        "mail.eml": {"subject": "RE: Smart Hands", "attachment_ids": ["1"], "external_id": "hs-email:1",
                     "authored_at": "2026-07-30T16:16:40Z", "source": "email"},
        "Delta Close Down.pdf": {"subject": "", "attachment_ids": [], "external_id": "",
                                 "authored_at": "2026-07-30T16:17:16Z", "source": "hubspot"},
    }
    bundles = bundle_documents(by_doc, index)
    assert len(bundles) == 1
    assert bundles["thread:smart hands"]["links"] == {"art_mail": "thread", "art_pdf": "arrived_with"}
    # Two candidate messages, or a message that carried nothing: the file stays its own.
    index["Delta Close Down.pdf"]["authored_at"] = "2026-07-30T19:00:00Z"
    assert len(bundle_documents(by_doc, index)) == 2
    index["mail.eml"]["attachment_ids"] = []
    index["Delta Close Down.pdf"]["authored_at"] = "2026-07-30T16:17:16Z"
    assert bundle_documents(by_doc, index)["art_pdf"]["links"] == {"art_pdf": "alone"}


def test_the_conversation_text_leads_with_the_opener_then_takes_each_document_in_turn():
    mail = _doc("m", *[f"message line {i}" for i in range(20)], filename="m.eml", subject="RE: Smart Hands")
    pdf = _doc("p", "Count: 5 | EvD Kiosk Counter", "Count: 2 | Digi Scales", filename="p.pdf")
    text = bundle_text("RE: Smart Hands", [mail, pdf])
    lines = text.split("\n")[1:]
    # The opener's first four lines carry the ask; then one line per document in turn.
    assert lines[:4] == [f"- message line {i}" for i in range(4)]
    assert lines[4] == "- Count: 5 | EvD Kiosk Counter" and lines[5] == "- Count: 2 | Digi Scales"
    assert lines[6] == "- message line 4"
    assert len(lines) == 14


def test_every_conversation_gets_a_verdict_and_a_trace_line(no_llm, monkeypatch):
    _judge_recording(monkeypatch, {"Smart Hands": ("other_job", 0.9)})
    kept, dropped, verdicts = judge_documents(SDWAN + KIOSK, deal_name=DEAL)
    assert [v["verdict"] for v in verdicts] == ["this_deal", "other_job"]
    notes = [verdict_note(v) for v in verdicts]
    assert notes[0].startswith("INFO: document_job_scope kept Sodexo SDWAN (2 atoms, 1 document(s)); this_deal 0.95 llm")
    assert notes[1].startswith("INFO: document_job_scope set aside RE: CDW Smart Hands SOW Delta Admin 70598001 (2 atoms, 1 document(s)); other_job 0.90 llm")


def test_the_judge_names_its_model_and_gives_it_time(no_llm, monkeypatch):
    asked = _judge_recording(monkeypatch, {})
    monkeypatch.delenv("SOWSMITH_DOCUMENT_JOB_MODEL", raising=False)
    monkeypatch.delenv("SOWSMITH_DOCUMENT_JOB_TIMEOUT", raising=False)
    judge_documents(SDWAN, deal_name=DEAL)
    assert asked[0]["model"] == "ollama:qwen3:32b" and asked[0]["timeout"] == 120
    monkeypatch.setenv("SOWSMITH_DOCUMENT_JOB_MODEL", "default")
    monkeypatch.setenv("SOWSMITH_DOCUMENT_JOB_TIMEOUT", "30")
    judge_documents(SDWAN, deal_name=DEAL)
    assert asked[1]["model"] is None and asked[1]["timeout"] == 30


def test_the_floor_is_a_setting_and_a_taught_verdict_ignores_it(no_llm, monkeypatch):
    _judge_recording(monkeypatch, {"Smart Hands": ("other_job", 0.82)})
    monkeypatch.setenv("SOWSMITH_DOCUMENT_JOB_MIN_CONF", "0.9")
    kept, dropped, verdicts = judge_documents(SDWAN + KIOSK, deal_name=DEAL)
    assert dropped == [] and verdicts[1]["model_verdict"] == "other_job" and verdicts[1]["verdict"] == "this_deal"
    monkeypatch.setenv("SOWSMITH_DOCUMENT_JOB_MIN_CONF", "0.8")
    kept, dropped, verdicts = judge_documents(SDWAN + KIOSK, deal_name=DEAL)
    assert len(dropped) == 2


def test_the_judge_asks_for_judgments_only_and_reads_documents_in_written_order(no_llm, monkeypatch):
    seen = {}

    def _decide(relation, text, candidates, **kw):
        seen.update(kw); seen["text"] = text
        return None

    import app.core.decide as decide_mod_
    monkeypatch.setattr(decide_mod_, "decide", _decide)
    later = _doc("art_b", "Yes, we can hit that date!", filename="b.eml", subject="Re: POS Installation 8/2")
    opener = _doc("art_a", "We will be setting just 1 Square register and 1 kitchen printer.",
                  filename="a.eml", subject="POS Installation 8/2")
    index = {"b.eml": {"subject": "Re: POS Installation 8/2", "attachment_ids": [], "external_id": "", "authored_at": "2026-08-07T16:33:23Z", "source": "email"},
             "a.eml": {"subject": "POS Installation 8/2", "attachment_ids": [], "external_id": "", "authored_at": "2026-08-06T15:49:00Z", "source": "email"}}
    judge_documents(later + opener, deal_name="010198 - Square POS Install Bridgewave", index=index)
    assert seen["exclude_created_by"] == ("teacher",)
    assert seen["context"].startswith("DEAL: 010198 - Square POS Install Bridgewave\n- We will be setting just 1 Square register")
    assert seen["text"] == "DOCUMENT: POS Installation 8/2"


def test_a_real_atom_names_its_file_on_the_source_ref(no_llm, monkeypatch):
    """EvidenceAtom has no source_filename attribute; the file is on the source ref."""
    asked = _judge_recording(monkeypatch, {"Smart Hands": ("other_job", 0.9)})

    def _real(doc, text, filename, subject=None):
        a = SimpleNamespace(source_artifact_id=doc, raw_text=text, value={}, review_flags=[], atom_type="scope_item",
                            source_refs=[SimpleNamespace(filename=filename, locator={})])
        if subject:
            a.value["email_thread"] = {"subject": subject}
        return a

    mail = _real("art_mail", "Confirm ladder availability with Delta", "010162-hs-email-113968718225.eml", "RE: CDW Smart Hands SOW Delta Admin 70598001")
    packing = _real("art_pdf", "Count: 5 | EvD Kiosk Counter", "Delta Close Down.pdf")
    index = {
        "010162-hs-email-113968718225.eml": {"subject": "RE: CDW Smart Hands SOW Delta Admin 70598001", "attachment_ids": ["1"],
                                             "external_id": "hs-email:113968718225", "authored_at": "2026-07-30T16:16:40Z", "source": "email"},
        "Delta Close Down.pdf": {"subject": "", "attachment_ids": [], "external_id": "", "authored_at": "2026-07-30T16:17:16.895Z", "source": "hubspot"},
    }
    kept, dropped, verdicts = judge_documents([mail, packing], deal_name=DEAL, index=index)
    assert len(asked) == 1  # one conversation: the message and the file that arrived with it
    assert verdicts[0]["links"] == {"010162-hs-email-113968718225.eml": "thread", "Delta Close Down.pdf": "arrived_with"}
    assert {a.source_artifact_id for a in dropped} == {"art_mail", "art_pdf"} and kept == []


def test_a_document_lesson_is_keyed_on_the_document_not_the_deal(no_llm):
    """Dev 2026-09-16: the judge's text led with the DEAL line, so a deal-scoped
    `other_job` lesson for the Delta thread matched every bundle of 010162 and
    the compile kept 12 atoms. The store compares the text alone: the text is
    the document; the deal the model reads lives in the context."""
    seen = []

    class _Exact:
        def resolve(self, *, relation, text, candidates, context="", **_):
            if relation != "document_job":
                return None
            seen.append((text, context))
            if text == "DOCUMENT: CDW Smart Hands SOW Delta Admin 70598001":
                return Decision(verdict="other_job", confidence=0.95, source="store", correction_id="corr_pm")
            return None

    decide_mod.set_store(_Exact())
    kept, dropped, verdicts = judge_documents(SDWAN + KIOSK, deal_name=DEAL, project_id="deal")
    assert [a.source_artifact_id for a in dropped] == ["art_kiosk", "art_kiosk"]
    assert all(a.source_artifact_id == "art_sdwan" for a in kept)
    assert all(t.startswith("DOCUMENT: ") and not t.startswith("DOCUMENT: RE:") for t, _ in seen)
    assert all(c.startswith(f"DEAL: {DEAL}\n") for _, c in seen), "the model still reads the deal, from the context"
