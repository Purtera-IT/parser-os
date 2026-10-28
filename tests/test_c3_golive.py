"""C3 go-live: heads final, no silent overrides, no time fallback, 1-2 reads per compile.

Hermetic: the heads service is a recorder (``_request``), the LLM a stub, and all text is made up."""
from __future__ import annotations

import io
import urllib.error
from types import SimpleNamespace as N

import pytest

import app.core.c3_type_base as c3
import app.core.typed_atom_classifier as tac
from app.core.schemas import AtomType
from tests.test_c3_type_base import _Atom, _atoms

URL = "http://heads.invalid"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("SOWSMITH_C3_FALLBACK", "SOWSMITH_C3_WAIT_MAX_S", "SOWSMITH_C3_DOC_KIND", "SOWSMITH_C3_TYPES",
                 "SOWSMITH_C3_RULES_YIELD", "SOWSMITH_C3_RULES_YIELD_EXEMPT", "SOWSMITH_C3_SURE",
                 "SOWSMITH_C3_SURE_BY_TYPE", "SOWSMITH_C3_UNSURE_KEEP", "SOWSMITH_TYPED_CLASSIFIER_DISABLE",
                 "SOWSMITH_DISABLE_LLM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SOWSMITH_C3_URL", URL)
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "on")
    monkeypatch.setenv("SOWSMITH_C3_AUTHORITY", "on")
    monkeypatch.setattr(c3, "_STATUS", {"state": "off"})
    monkeypatch.setattr(c3, "_DOC_STATUS", {"state": "off"})
    monkeypatch.setattr(c3, "_sleep", lambda s: None)
    monkeypatch.setattr(tac, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(c3, "_MODEL", None)
    monkeypatch.setattr(c3, "_TRIED", False)
    c3._DEAL.clear()
    yield
    c3._DEAL.clear()


def _http(code):
    return urllib.error.HTTPError(URL + "/read", code, f"HTTP {code}", {}, io.BytesIO(b""))


# ------------------------------------------------------- 1. doc_kind under never

def _doc_kind_rows():
    _, rows, _ = c3._rows(_atoms())
    return rows


@pytest.mark.parametrize("failure", [_http(400), ValueError("bad json")])
def test_never_doc_kind_read_raises_instead_of_keeping_the_compiles_kinds(monkeypatch, failure):
    rows = _doc_kind_rows()
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    steps = [{"ready": True}, failure]

    def _req(path, body, timeout):
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(c3, "_request", _req)
    with pytest.raises(c3.C3HeadsUnavailable):
        c3.doc_kind_reads("deal-test", rows)


def test_llm_mode_doc_kind_failure_still_keeps_the_compiles_kinds(monkeypatch):
    rows = _doc_kind_rows()
    steps = [{"ready": True}, _http(500)]

    def _req(path, body, timeout):
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(c3, "_request", _req)
    assert c3.doc_kind_reads("deal-test", rows) == {}


def test_never_any_error_in_the_typing_layer_fails_the_compile(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_FALLBACK", "never")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "task")

    def _boom(atoms, shas=None):
        raise ValueError("schema mismatch")

    monkeypatch.setattr(c3, "classify", _boom)
    monkeypatch.setattr(tac, "_call_ollama", lambda *a, **k: pytest.fail("the LLM must not type these lines"))
    with pytest.raises(ValueError):
        tac.classify_atoms(_atoms())


# ------------------------------------------------------- 2. view cut: reads per compile

def _compile_reads(monkeypatch, *, doc_kind: bool, late_line: bool) -> int:
    monkeypatch.setenv("SOWSMITH_C3_DOC_KIND", "1" if doc_kind else "0")
    calls: list[str] = []

    def _req(path, body, timeout):
        calls.append(path)
        if path == "/health":
            return {"ready": True}
        rows = body["deal"]["atoms"]
        got = {"types": {r["key"]: ["task", 0.95] for r in rows}, "documents": {}}
        if body.get("document_heads") == [c3.DOC_KIND_KEY]:
            got["documents"] = {r["doc_id"]: {c3.DOC_KIND_KEY: ["email", 0.9]} for r in rows}
        return got

    monkeypatch.setattr(c3, "_request", _req)
    # The view leaves out every third line (an old thread before the cut).
    monkeypatch.setattr(c3, "_view_rows",
                        lambda rows, docs, m, s, kinds=None: [r for r in rows if not r["text"].startswith("OUT")])
    atoms = [_Atom(f"a{i}", ("OUT old line %d" if i % 3 == 0 else "Mount display number %d on the wall.") % i)
             for i in range(9)]
    c3.set_deal([])
    c3._DEAL["view"] = "planning"
    c3.documents(atoms)          # the document-job read
    c3.classify(atoms)           # typing: shares that read
    if late_line:
        atoms.append(_Atom("late1", "Signature split line minted late."))
    c3.reread_if_new(atoms)      # end of compile: only lines the read truly missed
    return calls.count("/read")


@pytest.mark.parametrize("doc_kind,late,reads", [
    (True, False, 2),   # doc_kind + full        (was 4)
    (True, True, 3),    # + one full re-read     (was 5)
    (False, False, 1),  # full                   (was 3)
    (False, True, 2),   # + one full re-read     (was 3)
])
def test_a_view_cut_costs_no_extra_reads(monkeypatch, doc_kind, late, reads):
    assert _compile_reads(monkeypatch, doc_kind=doc_kind, late_line=late) == reads


def test_cut_lines_are_recorded_and_never_count_as_new(monkeypatch):
    _compile_reads(monkeypatch, doc_kind=False, late_line=False)
    assert c3._DEAL["view_out"] == {"a0", "a3", "a6"}


# ------------------------------------------------------- 3. failed fill keeps the head's type

def test_a_failed_fill_batch_keeps_the_heads_type_without_values(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "stakeholder")
    atoms = _atoms()
    monkeypatch.setattr(c3, "classify", lambda a, shas=None: {id(x): ("stakeholder", 0.97) for x in a
                                                                if x.id == "a_heavy"})
    prompts: list[str] = []

    def _fake(prompt, *, max_tokens=4096):
        prompts.append(prompt)
        if "already has its type" in prompt:
            raise TimeoutError("fill endpoint timed out")
        return '{"results": []}'

    monkeypatch.setattr(tac, "_call_ollama", _fake)
    tac.classify_atoms(atoms)
    heavy = next(a for a in atoms if a.id == "a_heavy")
    assert str(getattr(heavy.atom_type, "value", heavy.atom_type)) == "stakeholder"
    assert heavy.value["c3_label"] == {"type": "stakeholder", "act": "assign", "p": 0.97}
    assert not any("a_heavy" in p for p in prompts if "TAXONOMY" in p)


def test_settle_keeps_the_bases_type_when_its_fill_fails(monkeypatch):
    monkeypatch.setattr(tac, "_c3_fill_types", lambda: {"stakeholder"})
    monkeypatch.setattr(tac, "_fill_values", lambda fills: (0, [a for a, _ in fills]))
    c3.set_deal([], {})
    a = N(id="s1", raw_text="Primary contact is the facilities office.", atom_type=AtomType("deal_metadata"),
          value={"c3_base_says": {"type": "stakeholder", "p": 0.97}}, source_refs=[], artifact_id="x")
    monkeypatch.setattr(c3, "skipped", lambda atom, sha=None: False)
    assigned, filled = c3.settle_base_answers([a])
    assert (assigned, filled) == (1, 0)
    assert a.atom_type == AtomType("stakeholder") and a.value["c3_label"]["act"] == "authority"


# ------------------------------------------------------- 4. restore before entity_resolution

def test_restore_order_head_type_back_before_entity_resolution_party_veto_spared():
    src = open("app/core/compiler.py").read()
    pre = src.index("C3 base restored {_pre} line(s)")
    er = src.index('telemetry.stage("entity_resolution"')
    pk = src.index("_c3_final.apply_authority(atoms, final=True)")
    assert pre < er < pk  # the early restore, then the roster; the end-of-compile restore stays

    c3.set_deal([], {})
    retyped = N(id="r1", atom_type=AtomType("deal_metadata"), review_flags=[],
                value={"c3_label": {"type": "physical_site", "act": "fill"}})
    vetoed = N(id="v1", atom_type=AtomType("deal_metadata"), review_flags=["party_address"],
               value={"c3_label": {"type": "physical_site", "act": "fill"}, "kind": "party_address"})
    spare = lambda a: "party_address" in (a.review_flags or [])  # noqa: E731 - as the compiler passes it
    assert c3.apply_authority([retyped, vetoed], final=True, spare=spare) == 1
    assert retyped.atom_type == AtomType("physical_site")
    assert vetoed.atom_type == AtomType("deal_metadata")


# ------------------------------------------------------- 5. rule drops skip base-decided lines

def _gate_atom(aid, text, typ, label=None):
    a = N(id=aid, raw_text=text, text=text, atom_type=AtomType(typ), value=dict(label or {}), review_flags=[],
          entity_keys=[], source_refs=[], artifact_id="art1", project_id="d", review_status=None,
          confidence=0.8, receipts=[])
    return a


def test_substance_gate_leaves_a_head_typed_bare_name_and_counts_it(monkeypatch):
    from app.core import atom_substance_gate as g

    c3.set_deal([], {})
    head = _gate_atom("h1", "Jane Doe", "stakeholder", {"c3_label": {"type": "stakeholder", "act": "assign"}})
    rule = _gate_atom("r1", "John Roe", "stakeholder")
    kept, dropped = g.drop_contextless_stakeholders([head, rule])
    assert head not in kept and rule not in kept  # the rule alone drops both bare names
    kept, dropped = c3.yield_split("drop_contextless_stakeholders", g.drop_contextless_stakeholders)([head, rule])
    assert head in kept and head not in dropped
    assert rule not in kept
    assert c3.rules_yielded()["drop_contextless_stakeholders"]["kept"] == 1


def test_an_unsure_keep_is_not_a_head_decision(monkeypatch):
    from app.core import atom_substance_gate as g

    monkeypatch.setenv("SOWSMITH_C3_SURE", "0.5")
    c3.set_deal([], {})
    unsure = _gate_atom("u1", "Jane Doe", "stakeholder", {"c3_label": {"type": "_keep", "act": "keep", "p": 0.3}})
    kept, _ = c3.yield_split("drop_contextless_stakeholders", g.drop_contextless_stakeholders)([unsure])
    assert unsure not in kept


def test_type_sanity_drops_skip_head_lines_in_place(monkeypatch):
    c3.set_deal([], {})
    head = _gate_atom("h1", "Project Name:", "scope_item", {"c3_label": {"type": "scope_item", "act": "keep", "p": 0.9}})
    rule = _gate_atom("r1", "Site Name:", "scope_item")
    tail = _gate_atom("t1", "Install four access points.", "scope_item")
    from app.core.atom_type_sanity import strip_document_chrome

    atoms = [head, rule, tail]
    c3.yield_inplace("strip_document_chrome", strip_document_chrome, atoms)
    assert atoms == [head, tail]
    assert c3.rules_yielded()["strip_document_chrome"]["kept"] == 1


def test_yield_off_or_exempt_is_the_rule_as_before(monkeypatch):
    from app.core.atom_type_sanity import strip_document_chrome

    c3.set_deal([], {})
    head = _gate_atom("h1", "Project Name:", "scope_item", {"c3_label": {"type": "scope_item", "act": "assign"}})
    monkeypatch.setenv("SOWSMITH_C3_RULES_YIELD_EXEMPT", "strip_document_chrome")
    atoms = [head]
    c3.yield_inplace("strip_document_chrome", strip_document_chrome, atoms)
    assert atoms == []
    assert c3.rules_yielded() == {}


def test_party_address_veto_still_retypes_a_head_site():
    from app.core.party_address_veto import veto_party_page_sites

    ref = N(locator={"page": 3}, filename="psow.pdf", artifact_id="art1")
    sig = N(id="s1", raw_text="By: Jane Doe | Title: CFO | 100 Main St", text=None, atom_type=AtomType("signatory"),
            value={}, review_flags=[], entity_keys=[], source_refs=[ref], artifact_id="art1", locator={"page": 3})
    site = N(id="p1", raw_text="100 Main St, Springfield, IL 62701", text=None, atom_type=AtomType("physical_site"),
             value={"c3_label": {"type": "physical_site", "act": "authority", "p": 0.99}}, review_flags=[],
             entity_keys=["site:main"], source_refs=[ref], artifact_id="art1", locator={"page": 3})
    assert c3.base_decided(site)
    assert veto_party_page_sites([sig, site]) >= 1
    assert site.atom_type == AtomType("deal_metadata") and "party_address" in site.review_flags


# ------------------------------------------------------- 6. quote beside the writer

def test_the_quote_snapshot_holds_what_quote_reads():
    brief = {"deal": {"deal_name": "D"}, "sections": {"scope": [{"id": "a1", "text": "x"}]},
             "job_details": [{"cites": ["a1"], "text": "x"}], "written": {"sections": {}}}
    snap = c3.quote_snapshot(brief)
    brief["sections"]["scope"].append({"id": "a2"})
    assert snap == {"deal": {"deal_name": "D"}, "sections": {"scope": [{"id": "a1", "text": "x"}]},
                    "job_details": [{"cites": ["a1"], "text": "x"}]}


def test_brief_calls_on_built_before_the_writer(monkeypatch):
    from app.core import c3_brief

    order: list[str] = []
    monkeypatch.setattr(c3_brief, "writing_enabled", lambda: True)
    monkeypatch.setattr(c3_brief, "write", lambda b: order.append("write") or {})
    monkeypatch.setattr(c3_brief, "build", lambda *a, **k: {"sections": {}, "also_mentioned": []})
    monkeypatch.setattr(c3_brief, "_crm_deal", lambda d: {})
    c3_brief.run([], [], on_built=lambda b: order.append("quote"))
    assert order == ["quote", "write"]
