"""Wave-4 wiring gaps on the fixed dev build (audit 2026-10-28). Synthetic deals only."""
import io
import urllib.error
from types import SimpleNamespace as NS

import pytest

from app.core import c3_brief, c3_type_base, decide as decide_mod, semantic_role
from app.core.document_job_scope import judge_documents
from app.core.schemas import AtomType

KEY = c3_type_base.DOCUMENT_JOB_KEY
TIMELINE = {"created_at": "2026-07-01T00:00:00Z", "transitions": [
    {"ts": "2026-07-01T00:00:00Z", "label": "Submitted for Quoting", "order": 1},
    {"ts": "2026-09-01T00:00:00Z", "label": "Closed Lost", "order": 4}]}


def _line(i, art, text, kind="scope_item", keys=(), flags=(), value=None):
    return NS(id=f"a{i}", atom_type=kind, raw_text=text, review_status="approved", artifact_id=art,
              entity_keys=list(keys), review_flags=list(flags), value=value or {}, source_refs=[])


# ---- 1. the document gate bars the documents it says do not feed

def test_a_deal_record_document_is_barred_by_its_doc_id(monkeypatch):
    """The deal record's documents (deal_documents) carry ``doc_id``, not ``artifact_id``."""
    monkeypatch.setitem(c3_type_base._DEAL, "types", {})
    monkeypatch.setitem(c3_type_base._DEAL, "stage_timeline", TIMELINE)
    monkeypatch.setitem(c3_type_base._DEAL, "document_job", [])
    monkeypatch.setitem(c3_type_base._DEAL, "documents", [
        {"doc_id": "art_req", "filename": "request.eml", "authored_at": "2026-07-02T00:00:00Z", "kind": "email"},
        {"doc_id": "art_late", "filename": "late-call.json", "authored_at": "2026-09-10T00:00:00Z", "kind": "meeting"}])
    atoms = [_line(1, "art_req", "Install two displays."), _line(2, "art_late", "Five access points in another city.")]
    reads = {a.id: {"col:admission": ["keep", 0.95]} for a in atoms}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    docs = {d["filename"]: d for d in b["documents"]}
    assert docs["late-call.json"]["artifact_id"] == "art_late" and docs["late-call.json"]["feeds"] is False
    assert [x["id"] for v in b["sections"].values() for x in v] == ["a1"]
    assert b["decided_by"]["keep"].get("document:after_decision") == 1


# ---- 2. the document_job verdict says what the head said, per document, with its bar

def _doc(art, text, subject):
    return NS(source_artifact_id=art, artifact_id=art, source_filename=f"{art}.eml", raw_text=text,
              value={"email_thread": {"subject": subject}}, review_flags=[], atom_type="scope_item")


def test_a_head_other_job_under_its_bar_is_kept_and_says_so(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_DOCUMENT_JOB", "on")
    monkeypatch.setenv("SOWSMITH_DOCUMENT_JOB_MODEL_JUDGE", "off")
    monkeypatch.setattr(semantic_role, "classify_role", lambda *a, **k: pytest.fail("asked the model"))
    prev = decide_mod.get_store()
    decide_mod.set_store(None)
    try:
        # One thread, two documents: the head reads the questionnaire as another job (under the bar).
        atoms = [_doc("art_mail", "Please quote the cabling at the data center.", "Data Center Cabling"),
                 _doc("art_form", "Questionnaire: ceiling height?", "Data Center Cabling")]
        monkeypatch.setattr(c3_type_base, "documents", lambda *a, **k: {
            "art_mail": {KEY: ("this_deal", 0.93)}, "art_form": {KEY: ("other_job", 0.896)}})
        kept, dropped, verdicts = judge_documents(atoms, deal_name="000001 - Data center cabling")
    finally:
        decide_mod.set_store(prev)
    assert not dropped and len(kept) == 2
    (v,) = verdicts
    assert v["verdict"] == "this_deal" and v["source"] == "c3_below_bar" and v["c3_bar"] == 0.98
    assert v["c3_docs"] == {"art_mail.eml": ["this_deal", 0.93], "art_form.eml": ["other_job", 0.896]}
    assert "under its set-aside bar 0.98" in v["reason"]
    # The brief's document rows: each document's own reading, the bar, and who kept it.
    monkeypatch.setitem(c3_type_base._DEAL, "document_job", [v])
    monkeypatch.setitem(c3_type_base._DEAL, "stage_timeline", None)
    monkeypatch.setitem(c3_type_base._DEAL, "documents", [{"doc_id": "art_mail", "filename": "art_mail.eml"},
                                                          {"doc_id": "art_form", "filename": "art_form.eml"}])
    rows = {d["filename"]: d["document_job"] for d in c3_brief.document_gate()}
    assert rows["art_mail.eml"]["head"] == ["this_deal", 0.93]
    assert rows["art_form.eml"]["head"] == ["other_job", 0.896]
    assert rows["art_form.eml"]["by"] == "head:jdg:document_job (below its bar)" and rows["art_form.eml"]["bar"] == 0.98


# ---- 3. every line's keep decision is in the brief, with the keep heads' readings

def test_line_decisions_tell_admission_from_co_action(monkeypatch):
    monkeypatch.setitem(c3_type_base._DEAL, "types", {})
    monkeypatch.setitem(c3_type_base._DEAL, "documents", [])
    atoms = [_line(1, "art", "Install two displays."), _line(2, "art", "We will also sell you a warranty."),
             _line(3, "art", "Thanks all, talk soon.")]
    reads = {"a1": {"col:admission": ["keep", 0.95], "read:co_action": ["keep", 0.9]},
             "a2": {"col:admission": ["keep", 0.9], "read:co_action": ["reject", 0.97]},
             "a3": {"col:admission": ["drop", 0.2], "read:co_action": ["keep", 0.6]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    ld = b["line_decisions"]
    rows = {r[0]: dict(zip(ld["cols"], r)) for r in ld["rows"]}
    assert rows["a1"]["outcome"] == "kept" and rows["a1"]["by"] == "base:col:admission"
    assert rows["a2"]["by"].endswith("read:co_action") and rows["a2"]["read:co_action"] == "reject:0.970"
    assert rows["a2"]["col:admission"] == "keep:0.900"
    assert rows["a3"]["outcome"] == "not_admitted" and rows["a3"]["by"] == "base:col:admission"
    assert rows["a3"]["p"] == 0.2


# ---- 4. the quote's sites are the heads' settled roster

@pytest.fixture
def settled(monkeypatch):
    monkeypatch.setattr("app.core.c3_brief._deal_kit", lambda a: False)
    monkeypatch.setitem(c3_type_base._DEAL, "view", "planning")
    monkeypatch.setitem(c3_type_base._DEAL, "reads", {})
    monkeypatch.setitem(c3_type_base._DEAL, "site_keys", {"by": "heads:physical_site", "kept": []})


def test_quote_sites_count_settled_keys_once_and_skip_keyless_and_left_out_lines(settled):
    atoms = [
        _line(1, "n", "Origin: 100 Main Street, Springfield, IL 62701", "physical_site",
              ["site:springfield_il_62701", "site:100_main_street_springfield_il_62701"]),
        _line(2, "n", "Springfield, IL 62701", "physical_site", ["site:springfield_il_62701"]),
        _line(3, "t", "RU: 13", "physical_site"),                               # a table field, no key
        _line(4, "s", "200 Oak Ave, Shelbyville, IL", "physical_site", ["site:200_oak_ave"]),  # left out
    ]
    sites = c3_type_base.quote_sites(atoms, left_out={"a4"})
    assert len(sites) == 1
    assert sites[0]["atom_ids"] == ["a1", "a2"] and len(sites[0]["keys"]) == 2


def test_quote_sites_without_a_heads_read_keep_the_old_line_fallback(monkeypatch):
    monkeypatch.setattr("app.core.c3_brief._deal_kit", lambda a: False)
    monkeypatch.setitem(c3_type_base._DEAL, "view", "planning")
    monkeypatch.setitem(c3_type_base._DEAL, "site_keys", None)
    assert [s["site"] for s in c3_type_base.quote_sites([_line(1, "n", "Warehouse", "physical_site")])] == ["site-line:a1"]


def test_a_fill_label_with_no_p_reads_the_heads_own_p():
    from app.core.site_keys_follow_heads import _head

    a = _line(1, "n", "Site", "physical_site", value={"c3_label": {"type": "physical_site", "act": "fill"},
                                                       "c3_base_says": {"type": "physical_site", "p": 0.577}})
    assert _head(a, {}) == ("physical_site", 0.577)


def test_a_filled_line_keeps_the_heads_p_on_its_label():
    from app.core.typed_atom_classifier import _mark_c3, _with_head_p

    a = NS(value={})
    _mark_c3(a, "physical_site", None, "fill")
    _with_head_p(a, ("physical_site", 0.8123))
    assert a.value["c3_label"] == {"type": "physical_site", "act": "fill", "p": 0.812}


# ---- 5. a planning read never asked the quote heads, and the quote says so

def test_quote_heads_are_not_asked_in_a_planning_read(monkeypatch):
    monkeypatch.setitem(c3_type_base._DEAL, "reads", {"a1": {"col:admission": ["keep", 0.9]}})
    monkeypatch.setitem(c3_type_base._DEAL, "view", "planning")
    assert c3_type_base.quote_heads_asked() is False
    monkeypatch.setitem(c3_type_base._DEAL, "view", "quoting")
    assert c3_type_base.quote_heads_asked() is True
    monkeypatch.setitem(c3_type_base._DEAL, "reads", {})
    assert c3_type_base.quote_heads_asked() is False


# ---- 6. an empty read is not asked; the same 500 from a ready service ends the wait

def test_a_read_with_no_lines_asks_nothing(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://heads.invalid")
    monkeypatch.setattr(c3_type_base, "_request_raw", lambda *a, **k: pytest.fail("asked the service"))
    assert c3_type_base._wait_for_heads("/read", {"deal": {"atoms": []}}, {}, "document read") == {}
    assert c3_type_base._request("/read", {"deal": {"atoms": []}}, timeout=1.0) == {}


def test_the_same_500_from_a_ready_service_fails_fast(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://heads.invalid")
    monkeypatch.setenv("SOWSMITH_C3_RETRY_S", "0")
    monkeypatch.setenv("SOWSMITH_C3_WAIT_MAX_S", "5")  # a regression fails in seconds, not 30 minutes
    monkeypatch.setattr(c3_type_base, "_sleep", lambda s: None)
    calls = []

    def _raw(path, body, timeout):
        if path == "/health":
            return {"ready": True}
        calls.append(path)
        raise urllib.error.HTTPError("http://heads.invalid/read", 500, "err", {},
                                     io.BytesIO(b'{"error": "read failed: RuntimeError: min()"}'))

    monkeypatch.setattr(c3_type_base, "_request_raw", _raw)
    with pytest.raises(c3_type_base.C3HeadsUnavailable, match="same error 3 times"):
        c3_type_base._wait_for_heads("/read", {"deal": {"atoms": [{"key": "k"}]}}, {}, "document read")
    assert len(calls) == 3


# ---- 7. dedup keeps different places apart; numbers are not cut at their commas

class _Site:
    atom_type = AtomType.physical_site
    source_refs: list = []
    artifact_id = "art1"
    confidence = 0.9

    def __init__(self, i, text, value):
        self.id, self.raw_text, self.value, self.entity_keys, self.review_flags = i, text, value, [], []


def test_a_street_and_a_town_no_site_names_are_not_folded_into_another_site():
    from app.core.semantic_dedup import _dedupe_physical_site_atoms

    main = _Site("m", "Destination: 100 Main Street, Springfield, IL 62701",
                 {"id": "SPRINGFIELD-IL-62701", "site_id": "SPRINGFIELD-IL-62701", "name": "Springfield",
                  "street_address": "100 Main Street", "address": "100 Main Street", "city": "Springfield",
                  "state": "IL", "zip": "62701"})
    origin = _Site("o", "Origin: 55 Elm Road, Peoria, IL 61602",
                   {"id": "PEORIA-IL-61602", "site_id": "PEORIA-IL-61602", "name": "Peoria",
                    "street_address": "55 Elm Road", "address": "55 Elm Road", "city": "Peoria", "state": "IL",
                    "zip": "61602"})
    street = _Site("s", "900 Springfield Pike", {"site_id": "900", "id": "900", "name": "Springfield Pike",
                                                 "city": "Springfield", "state": "IL"})
    town = _Site("t", "Decatur, IL", {"kind": "note_field_item", "field_name": "Primary Destination"})
    out = _dedupe_physical_site_atoms([main, street, origin, town])
    assert {a.raw_text for a in out} >= {"900 Springfield Pike", "Decatur, IL"}


def test_a_town_the_site_names_still_folds():
    from app.core.semantic_dedup import _place_no_site_states

    main = _Site("m", "100 Main Street, Springfield, IL 62701", {"city": "Springfield"})
    assert _place_no_site_states(_Site("t", "Springfield, IL", {}), [main]) == ""
    assert _place_no_site_states(_Site("t", "100 Main Street", {}), [main]) == ""


def test_a_note_list_keeps_numbers_whole():
    from app.parsers.hubspot_note_parser import _split_list_value

    assert _split_list_value("Expedited support costs $1,000-$2,000 to reduce delays, badge access, 1,500 hours") == [
        "Expedited support costs $1,000-$2,000 to reduce delays", "badge access", "1,500 hours"]
    assert _split_list_value("Dallas, Austin, Houston") == ["Dallas", "Austin", "Houston"]
