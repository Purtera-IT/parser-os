"""The C3 base as a deflect layer on the atom-type stage (SOWSMITH_C3_TYPE).

Same contract as the other deflect layers: one-sided, guess-free, OFF by
default. A sure value-light answer is assigned and skips the LLM; a sure
``_keep`` drops the candidate with its type unchanged; an unsure or
value-heavy answer reaches the LLM untouched; a model that will not load
abstains. Hermetic: the reading is stubbed, the LLM is a recorder, and the
one real read uses a tiny hashed-encoder model. All text is made up.
"""
from __future__ import annotations

import logging

import pytest

import app.core.c3_type_base as c3
import app.core.typed_atom_classifier as tac


class _Type:
    def __init__(self, value):
        self.value = value


class _Ref:
    def __init__(self, filename="notes.txt", artifact_type="txt", locator=None, artifact_id="art1"):
        self.filename = filename
        self.artifact_type = _Type(artifact_type)
        self.locator = locator or {}
        self.artifact_id = artifact_id


class _Atom:
    def __init__(self, atom_id, raw_text, type_str="scope_item", ref=None):
        self.id = atom_id
        self.raw_text = raw_text
        self.atom_type = _Type(type_str)
        self.value = {}
        self.project_id = "deal-test"
        self.artifact_id = "art1"
        self.source_refs = [ref] if ref is not None else []


_TEXTS = {
    "a_task": "Contractor will mount the four lobby displays on the north wall.",
    "a_keep": "Thanks again for the walkthrough on Tuesday, talk to you soon everyone.",
    "a_unsure": "The customer would like the wiring closet relabelled before turnover.",
    "a_heavy": "Primary contact for the site is the facilities office on the second floor.",
}


def _atoms():
    return [_Atom(k, t) for k, t in _TEXTS.items()]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("SOWSMITH_C3_TYPE", "SOWSMITH_C3_SURE", "SOWSMITH_TYPED_STUDENT",
                 "SOWSMITH_ATOM_TYPE_DEFLECT", "SOWSMITH_TYPE_HEAD_GPU", "SOWSMITH_TYPE_HEAD_DEFLECT",
                 "SOWSMITH_CONTRASTIVE_TYPE", "SOWSMITH_RUBRIC_GATE",
                 "SOWSMITH_TYPED_CLASSIFIER_DISABLE", "SOWSMITH_DISABLE_LLM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tac, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(c3, "_MODEL", None)
    monkeypatch.setattr(c3, "_TRIED", False)
    yield


@pytest.fixture()
def llm(monkeypatch):
    """Records which atoms reached the LLM; it keeps every one."""
    seen: list[str] = []

    def _fake(prompt, *, max_tokens=4096):
        ids = [k for k in _TEXTS if k in prompt]
        seen.extend(ids)
        return '{"results": [' + ", ".join(
            f'{{"atom_id": "{k}", "atom_type": "_keep"}}' for k in ids) + "]}"

    monkeypatch.setattr(tac, "_call_ollama", _fake)
    return seen


def _stub_readings(monkeypatch, calls=None):
    def _classify(atoms, shas=None):
        if calls is not None:
            calls.append(len(atoms))
        by = {a.id: a for a in atoms}
        return {id(by["a_task"]): ("task", 0.95), id(by["a_keep"]): ("_keep", 0.9),
                id(by["a_unsure"]): ("requirement", 0.4), id(by["a_heavy"]): ("stakeholder", 0.99)}

    monkeypatch.setattr(c3, "classify", _classify)
    monkeypatch.setattr(c3, "_prepare", lambda atoms, shas=None: (lambda: _classify(atoms, shas)))


def test_flag_off_never_reads(monkeypatch, llm):
    def _tripwire(atoms, shas=None):
        raise AssertionError("the C3 layer must not run with the flag off")

    monkeypatch.setattr(c3, "classify", _tripwire)
    atoms = _atoms()
    assert tac.classify_atoms(atoms) == 0
    assert sorted(llm) == sorted(_TEXTS)
    assert all(a.atom_type.value == "scope_item" for a in atoms)
    assert tac.get_last_deflect_stats()["deflected"]["c3"] == 0


def test_flag_on_assigns_sure_light_drops_sure_keep_and_passes_the_rest(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "task,_keep")
    calls: list[int] = []
    _stub_readings(monkeypatch, calls)
    atoms = _atoms()
    extra = _Atom("a_context", "Context line from another document that is not promotable.",
                  type_str="risk")
    promoted = tac.classify_atoms(atoms + [extra])
    by = {a.id: a for a in atoms}
    # The whole compile is read once, not only the promotable atoms.
    assert calls == [5]
    assert by["a_task"].atom_type.value == "task"
    assert by["a_keep"].atom_type.value == "scope_item"
    assert sorted(llm) == ["a_heavy", "a_unsure"]
    assert by["a_unsure"].atom_type.value == "scope_item"
    assert by["a_heavy"].atom_type.value == "scope_item"
    assert promoted == 1
    assert tac.get_last_deflect_stats()["deflected"]["c3"] == 2


def test_sure_threshold_comes_from_the_environment(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "on")
    monkeypatch.setenv("SOWSMITH_C3_SURE", "0.3")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "task,requirement,_keep")
    _stub_readings(monkeypatch)
    atoms = _atoms()
    tac.classify_atoms(atoms)
    by = {a.id: a for a in atoms}
    assert by["a_unsure"].atom_type.value == "requirement"
    assert llm == ["a_heavy"]


def test_only_the_listed_answers_are_acted_on(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "_keep")
    _stub_readings(monkeypatch)
    atoms = _atoms()
    tac.classify_atoms(atoms)
    by = {a.id: a for a in atoms}
    # The sure task is not on the list, so the LLM types it; the keep drops.
    assert by["a_task"].atom_type.value == "scope_item"
    assert sorted(llm) == ["a_heavy", "a_task", "a_unsure"]
    assert tac.get_last_deflect_stats()["deflected"]["c3"] == 1


def test_the_list_parses_and_reads_unset_as_none(monkeypatch):
    monkeypatch.delenv("SOWSMITH_C3_TYPES", raising=False)
    assert c3.acts_on() is None
    monkeypatch.setenv("SOWSMITH_C3_TYPES", " task, _keep ,")
    assert c3.acts_on() == {"task", "_keep"}


def test_on_with_no_list_takes_nothing(monkeypatch, llm):
    # Flipping the layer on before a list clears the route report changes nothing.
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.delenv("SOWSMITH_C3_TYPES", raising=False)
    _stub_readings(monkeypatch)
    atoms = _atoms()
    assert tac.classify_atoms(atoms) == 0
    assert sorted(llm) == sorted(_TEXTS)
    assert all(a.atom_type.value == "scope_item" for a in atoms)
    assert tac.get_last_deflect_stats()["deflected"]["c3"] == 0


def test_a_deal_kit_line_is_never_acted_on(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    atom = _Atom("a_task", _TEXTS["a_task"], ref=_Ref(filename="Example Deal Kit v3.xlsx",
                                                      artifact_type="xlsx"))
    monkeypatch.setattr(c3, "classify", lambda atoms, shas=None: {id(atom): ("task", 0.99)})
    tac.classify_atoms([atom])
    assert atom.atom_type.value == "scope_item"
    assert llm == ["a_task"]


def test_shadow_changes_nothing_and_reports(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "shadow")
    _stub_readings(monkeypatch)
    atoms = _atoms()
    assert tac.classify_atoms(atoms) == 0
    assert sorted(llm) == sorted(_TEXTS)
    assert all(a.atom_type.value == "scope_item" for a in atoms)
    report = c3.get_last_shadow()
    assert report["would_assign"] == 1 and report["would_keep"] == 1
    # A sure value-carrying answer is one the base would take too (its type,
    # the LLM filling the fields).
    assert report["would_fill"] == 1
    # The LLM kept everything: the keep agrees; the assignment and the fill do not.
    assert report["agree"] == 1 and report["disagree"] == 2
    assert sum(report["disagree_pairs"].values()) == 2
    assert tac.get_last_deflect_stats()["deflected"]["c3"] == 0


def test_a_model_that_will_not_load_falls_through(monkeypatch, llm, caplog):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    tries = []

    def _boom():
        tries.append(1)
        raise FileNotFoundError("no checkpoint")

    monkeypatch.setattr(c3, "_load", _boom)
    with caplog.at_level(logging.WARNING, logger=c3.__name__):
        for _ in range(2):
            atoms = _atoms()
            assert tac.classify_atoms(atoms) == 0
            assert all(a.atom_type.value == "scope_item" for a in atoms)
    assert tries == [1]
    assert sum("C3 type base not loaded" in r.getMessage() for r in caplog.records) == 1
    assert len(llm) == 2 * len(_TEXTS)


def test_a_reading_that_raises_leaves_every_atom_alone(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")

    def _raise(atoms, shas=None):
        raise RuntimeError("read failed")

    monkeypatch.setattr(c3, "classify", _raise)
    atoms = _atoms()
    assert tac.classify_atoms(atoms) == 0
    assert sorted(llm) == sorted(_TEXTS)


# ------------------------------------------------------ atoms_to_deal


def _evidence_atoms():
    from app.core.schemas import (ArtifactType, AuthorityClass, EvidenceAtom, ReviewStatus,
                                  SourceRef)

    def make(i, text, artifact_id, kind, locator, value=None):
        ref = SourceRef(id=f"src{i}", artifact_id=artifact_id, artifact_type=kind,
                        filename=f"{artifact_id}.file", locator=locator,
                        extraction_method="test", parser_version="test")
        return EvidenceAtom(id=f"atm{i}", project_id="deal-test", artifact_id=artifact_id,
                            atom_type="scope_item", raw_text=text, normalized_text=text.lower(),
                            value=value or {}, source_refs=[ref],
                            authority_class=AuthorityClass.machine_extractor, confidence=0.5,
                            review_status=ReviewStatus.auto_accepted, parser_version="test")

    return [
        # A document with no time of its own comes after the email before it.
        make(1, "Install two access points in the break room.", "doc", ArtifactType.docx,
             {"section_path": ["Scope", "Wireless"]}),
        make(2, "Can you confirm the ceiling height in the break room?", "mail", ArtifactType.email,
             {"sent_at": "Tue, 02 Jun 2026 10:00:00 +0000"},
             {"said_by": {"role_guess": "reseller", "side": "theirs"}}),
        make(3, "Customer will provide a lift for the high ceilings.", "call", ArtifactType.transcript,
             {"speaker_role": "customer", "utterance_index": 4, "sent_at": "2026-06-01T09:00:00Z"}),
    ]


def test_atoms_to_deal_maps_the_export_fields():
    atoms = _evidence_atoms()
    deal = c3.atoms_to_deal(atoms)
    rows = {r.atom_id: r for r in deal.atoms}
    assert deal.deal_id == "deal-test"
    # Arrival order; the document with no time takes the first known one
    # (export.py's fill) and ties keep compile order.
    assert [r.atom_id for r in deal.atoms] == ["atm3", "atm1", "atm2"]
    assert rows["atm1"].section == "Scope > Wireless"
    # A file the lifecycle dataset does not know reads as training's "file".
    assert rows["atm1"].doc_kind == "file" and rows["atm1"].doc_id == "doc"
    assert rows["atm2"].doc_kind == "email"
    assert (rows["atm2"].speaker_role, rows["atm2"].speaker_side) == ("reseller", "theirs")
    # "customer" is no speaker role training saw: it reads as ''.
    assert rows["atm3"].doc_kind == "meeting" and rows["atm3"].speaker_role == ""
    assert rows["atm1"].entered_at == rows["atm2"].entered_at
    assert rows["atm3"].entered_at < rows["atm2"].entered_at


def test_atoms_to_deal_reads_through_a_tiny_model(monkeypatch):
    torch = pytest.importorskip("torch")
    from ml.c3.data import featurize
    from ml.c3.model import C3Config, C3Model
    from ml.c3.schema import load_schema

    schema = load_schema()
    atoms = _evidence_atoms()
    batch = featurize(c3.atoms_to_deal(atoms), schema)
    assert len(batch.texts) == 3

    cfg = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                   residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
    torch.manual_seed(0)
    model = C3Model(schema, cfg).eval()
    monkeypatch.setattr(c3, "_MODEL", (model, schema, "cpu"))
    monkeypatch.setattr(c3, "_TRIED", True)
    got = c3.classify(atoms)
    answers = {a.value for a in schema.by_key()["col:label_type"].answers}
    assert set(got) == {id(a) for a in atoms}
    for answer, p in got.values():
        assert answer in answers and 0.0 <= p <= 1.0


@pytest.fixture()
def lifecycle(monkeypatch):
    """The lifecycle dataset, as a dict of sha256 -> type."""
    import app.core.document_lifecycle.dataset as ds

    table: dict[str, str] = {}
    monkeypatch.setattr(ds, "lookup", lambda sha: {"type": table[sha]} if sha in table else None)
    return table


def test_a_file_takes_its_lifecycle_type_by_sha(lifecycle):
    lifecycle["sha-doc"] = "SOW_DRAFT"
    deal = c3.atoms_to_deal(_evidence_atoms(), {"doc": "sha-doc", "mail": "sha-mail"})
    kinds = {r.atom_id: r.doc_kind for r in deal.atoms}
    assert kinds == {"atm1": "sow_draft", "atm2": "email", "atm3": "meeting"}


def test_an_email_is_an_email_whatever_its_sha_says(lifecycle):
    lifecycle["sha-mail"] = "SOW_DRAFT"
    deal = c3.atoms_to_deal(_evidence_atoms(), {"mail": "sha-mail"})
    assert {r.atom_id: r.doc_kind for r in deal.atoms}["atm2"] == "email"


def test_a_type_training_never_saw_reads_as_file(lifecycle):
    lifecycle["sha-doc"] = "WIN_WIRE"
    deal = c3.atoms_to_deal(_evidence_atoms(), {"doc": "sha-doc"})
    assert {r.atom_id: r.doc_kind for r in deal.atoms}["atm1"] == "file"
    assert "win_wire" not in c3.TRAINED_KINDS


def test_speaker_values_outside_training_read_empty():
    a = _Atom("a1", "Made-up line from an email thread about the lobby displays.",
              ref=_Ref(artifact_type="email"))
    a.value = {"said_by": {"role_guess": "installer", "side": "OURS"}}
    row = c3.atoms_to_deal([a]).atoms[0]
    assert (row.speaker_role, row.speaker_side) == ("", "ours")


def test_a_deal_kit_found_by_sha_is_never_acted_on(monkeypatch, llm, lifecycle):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "task")
    lifecycle["sha-kit"] = "DEAL_KIT"
    # A plain name: only the dataset says it is our pricing workbook.
    atom = _Atom("a_task", _TEXTS["a_task"], ref=_Ref(filename="workbook.xlsx", artifact_type="xlsx",
                                                      artifact_id="art1"))
    monkeypatch.setattr(c3, "classify", lambda atoms, shas=None: {id(atom): ("task", 0.99)})
    tac.classify_atoms([atom], artifact_sha256={"art1": "sha-kit"})
    assert atom.atom_type.value == "scope_item"
    assert llm == ["a_task"]
    # The same line with no sha is acted on.
    llm.clear()
    tac.classify_atoms([atom])
    assert atom.atom_type.value == "task"
    assert llm == []


def test_the_live_read_uses_the_window_the_base_was_trained_and_scored_with():
    import re
    from pathlib import Path

    pod = Path(__file__).resolve().parents[1] / "ml" / "c3" / "pod"
    for script in ("runC.sh", "runD.sh"):
        got = re.search(r"ATOMS=\$\{ATOMS:-(\d+)\}", (pod / script).read_text())
        assert got and int(got.group(1)) == c3.MAX_ATOMS, script


def test_too_little_memory_abstains_before_loading(monkeypatch, tmp_path):
    ck = tmp_path / "base.pt"
    ck.write_bytes(b"x")
    monkeypatch.setenv("SOWSMITH_C3_CHECKPOINT", str(ck))
    monkeypatch.setenv("SOWSMITH_C3_ENCODER", "answerdotai/ModernBERT-large")
    monkeypatch.setattr(c3, "_free_mb", lambda: 900.0)
    assert c3.get_model() is None
    assert c3.status()["state"] == "no_memory" and "900 MB free" in c3.status()["reason"]


def test_a_read_past_its_budget_stops_and_sends_every_atom_to_the_llm(monkeypatch, llm):
    import ml.c3.predict as predict

    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_BUDGET_S", "5")
    seen = {}

    def _slow(model, deal, schema, **kw):
        seen.update(kw)
        raise TimeoutError("read passed its deadline at 2/5 lines")

    monkeypatch.setattr(predict, "read_deal", _slow)
    monkeypatch.setattr(c3, "_MODEL", (object(), object(), "cpu"))
    monkeypatch.setattr(c3, "_TRIED", True)
    monkeypatch.setattr(c3, "_build", lambda atoms, shas=None: (None, []))
    atoms = _atoms()
    tac.classify_atoms(atoms)
    assert sorted(llm) == sorted(_TEXTS)
    assert seen["lines_only"] is True and seen["deadline"] > 0
    assert c3.status()["state"] == "over_budget"


def test_the_compile_manifest_carries_the_layers_state(monkeypatch):
    from app.core.ml_capabilities import build_compile_capabilities

    monkeypatch.setenv("SOWSMITH_C3_TYPE", "shadow")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "task")
    monkeypatch.setattr(c3, "_STATUS", {"state": "read", "read_ms": 1200.0, "lines": 40})
    got = build_compile_capabilities()["deflect_counts"]["c3"]
    assert got["mode"] == "shadow" and got["types"] == ["task"] and got["state"] == "read"


def test_reading_only_the_type_head_gives_the_same_types():
    torch = pytest.importorskip("torch")
    from ml.c3.model import C3Config, C3Model
    from ml.c3.predict import read_deal
    from ml.c3.schema import load_schema

    schema = load_schema()
    deal = c3.atoms_to_deal(_evidence_atoms())
    cfg = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                   residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
    torch.manual_seed(0)
    model = C3Model(schema, cfg).eval()
    full = read_deal(model, deal, schema, lines_only=True)
    only = read_deal(model, deal, schema, lines_only=True, heads=("col:label_type",))
    for key, line in only["lines"].items():
        assert set(line["reads"]) == {"col:label_type"}
        assert line["reads"]["col:label_type"] == full["lines"][key]["reads"]["col:label_type"]


def test_a_read_scores_containment_in_slices_with_the_same_result(monkeypatch):
    torch = pytest.importorskip("torch")
    import ml.c3.model as m
    from ml.c3.model import C3Config, C3Model
    from ml.c3.schema import load_schema

    cfg = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8, d_head=16,
                   residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
    torch.manual_seed(0)
    model = C3Model(load_schema(), cfg).eval()
    r = torch.randn(10, cfg.d_r)
    with torch.no_grad():
        whole = model._box_containment(r)
        monkeypatch.setattr(m, "BOX_ROWS", 3)
        sliced = model._box_containment(r)
    assert torch.allclose(whole, sliced)


def test_a_shadow_read_runs_beside_the_llm(monkeypatch, llm):
    import threading

    monkeypatch.setenv("SOWSMITH_C3_TYPE", "shadow")
    llm_started = threading.Event()
    real = tac._call_ollama

    def _llm(prompt, *, max_tokens=4096):
        llm_started.set()
        return real(prompt, max_tokens=max_tokens)

    monkeypatch.setattr(tac, "_call_ollama", _llm)

    def _prepare(atoms, shas=None):
        by = {a.id: a for a in atoms}
        texts = {a.id: a.raw_text for a in atoms}

        def _run():
            # Finishes only once the LLM has started: a read in front of the
            # LLM would wait here until the timeout and report nothing.
            assert llm_started.wait(5)
            assert {a.id: a.raw_text for a in atoms} == texts
            return {id(by["a_task"]): ("task", 0.95)}
        return _run

    monkeypatch.setattr(c3, "_prepare", _prepare)
    atoms = _atoms()
    assert tac.classify_atoms(atoms) == 0
    assert sorted(llm) == sorted(_TEXTS)
    report = c3.get_last_shadow()
    assert report["would_assign"] == 1
    assert tac.get_last_deflect_stats()["layer_ms"]["c3"] < 5000


def test_a_slow_shadow_read_never_holds_the_stage(monkeypatch, llm):
    """Shadow changes nothing, so the stage waits at most SHADOW_WAIT_S for it."""
    import threading

    monkeypatch.setenv("SOWSMITH_C3_TYPE", "shadow")
    monkeypatch.setenv("SOWSMITH_C3_SHADOW_WAIT_S", "0.3")
    release = threading.Event()

    def _prepare(atoms, shas=None):
        def _run():
            release.wait(10)
            return {}
        return _run

    monkeypatch.setattr(c3, "_prepare", _prepare)
    try:
        tac.classify_atoms(_atoms())
        assert tac.get_last_deflect_stats()["layer_ms"]["c3"] < 2000
    finally:
        release.set()


def test_a_sure_small_talk_drops_like_keep_and_only_when_listed(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "small_talk")

    def _classify(atoms, shas=None):
        by = {a.id: a for a in atoms}
        return {id(by["a_keep"]): ("small_talk", 0.9), id(by["a_task"]): ("task", 0.95)}

    monkeypatch.setattr(c3, "classify", _classify)
    monkeypatch.setattr(c3, "_prepare", lambda atoms, shas=None: (lambda: _classify(atoms, shas)))
    atoms = _atoms()
    by = {a.id: a for a in atoms}
    tac.classify_atoms(atoms)
    # Dropped from the LLM with its type unchanged; task is not on the list.
    assert "a_keep" not in llm and "a_task" in llm
    assert by["a_keep"].atom_type.value == "scope_item"
    assert tac.get_last_deflect_stats()["deflected"]["c3"] == 1


def test_on_records_why_the_llm_still_gets_each_line(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "task,_keep")
    monkeypatch.setenv("SOWSMITH_C3_SURE", "0.85")
    atoms = _atoms()
    answers = {"a_task": ("task", 0.95), "a_keep": ("_keep", 0.9),
               "a_heavy": ("stakeholder", 0.97), "a_unsure": ("task", 0.5)}
    monkeypatch.setattr(c3, "classify", lambda a, shas=None: {id(x): answers[x.id] for x in a})
    tac.classify_atoms(atoms)
    why = c3.status()["typing"]
    assert (why["lines"], why["taken"], why["unsure"], why["no_reading"]) == (4, 2, 1, 0)
    assert why["sure_not_taken"] == {"stakeholder": 1}


def test_a_sure_value_carrying_type_is_the_bases_and_the_llm_only_fills_its_fields(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "stakeholder,task")
    atoms = _atoms()
    answers = {"a_heavy": ("stakeholder", 0.97), "a_task": ("task", 0.95)}
    monkeypatch.setattr(c3, "classify", lambda a, shas=None: {id(x): answers[x.id] for x in a if x.id in answers})
    prompts: list[str] = []

    def _fake(prompt, *, max_tokens=4096):
        prompts.append(prompt)
        if "already has its type" in prompt:
            return '{"results": [{"atom_id": "a_heavy", "atom_type": "stakeholder", "value": {"name": "facilities office"}}]}'
        return '{"results": []}'

    monkeypatch.setattr(tac, "_call_ollama", _fake)
    tac.classify_atoms(atoms)
    heavy = next(a for a in atoms if a.id == "a_heavy")
    assert str(getattr(heavy.atom_type, "value", heavy.atom_type)) == "stakeholder"
    assert heavy.value.get("name") == "facilities office"
    fill = [p for p in prompts if "already has its type" in p]
    assert len(fill) == 1 and "TAXONOMY" not in fill[0] and "a_heavy" in fill[0]
    # The full classifier never saw the lines the base decided.
    assert not any("a_heavy" in p or "a_task" in p for p in prompts if "TAXONOMY" in p)
    assert c3.status()["typing"]["filled"] == 1


def test_a_failed_fill_sends_the_line_back_to_the_full_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "stakeholder")
    atoms = _atoms()
    monkeypatch.setattr(c3, "classify", lambda a, shas=None: {id(x): ("stakeholder", 0.97) for x in a if x.id == "a_heavy"})
    prompts: list[str] = []

    def _fake(prompt, *, max_tokens=4096):
        prompts.append(prompt)
        return "" if "already has its type" in prompt else '{"results": []}'

    monkeypatch.setattr(tac, "_call_ollama", _fake)
    tac.classify_atoms(atoms)
    assert any("a_heavy" in p for p in prompts if "TAXONOMY" in p)
    assert c3.status()["typing"]["fill_back_to_llm"] == 1


def test_the_typing_layer_runs_without_ml_code(monkeypatch, llm):
    """The worker ships without ml/. 2026-10-15: skipped() imported
    ml.c3.data, so the first sure reading raised ImportError and the whole
    layer abstained on every live compile."""
    import builtins

    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "task,_keep")
    atoms = _atoms()
    answers = {"a_task": ("task", 0.95), "a_keep": ("_keep", 0.9)}
    monkeypatch.setattr(c3, "classify", lambda a, shas=None: {id(x): answers[x.id] for x in a if x.id in answers})
    real = builtins.__import__

    def _no_ml(name, *a, **kw):
        if name == "ml" or name.startswith("ml."):
            raise ImportError("the worker ships without ml/")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_ml)
    tac.classify_atoms(atoms)
    monkeypatch.setattr(builtins, "__import__", real)
    why = c3.status()["typing"]
    assert "error" not in why and why["taken"] == 2
    assert "a_task" not in llm and "a_keep" not in llm


def test_each_answer_can_carry_its_own_bar(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "task,_keep")
    monkeypatch.setenv("SOWSMITH_C3_SURE", "0.85")
    monkeypatch.setenv("SOWSMITH_C3_SURE_BY_TYPE", "task:0.5")
    atoms = _atoms()
    answers = {"a_task": ("task", 0.6), "a_keep": ("_keep", 0.6)}
    monkeypatch.setattr(c3, "classify", lambda a, shas=None: {id(x): answers[x.id] for x in a if x.id in answers})
    tac.classify_atoms(atoms)
    why = c3.status()["typing"]
    assert why["taken"] == 1 and why["unsure"] == 1     # task clears 0.5; _keep misses 0.85
    assert "a_task" not in llm and "a_keep" in llm


def test_a_kept_line_takes_the_bases_type_at_the_end_unless_a_stage_typed_it(monkeypatch, llm):
    """000002: gold small_talk / deal_state lines ended as scope_item because a
    keep left the type as it was."""
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    monkeypatch.setenv("SOWSMITH_C3_TYPES", "small_talk,deal_state")
    atoms = _atoms()
    answers = {"a_keep": ("small_talk", 0.95), "a_unsure": ("deal_state", 0.9)}
    monkeypatch.setattr(c3, "classify", lambda a, shas=None: {id(x): answers[x.id] for x in a if x.id in answers})
    tac.classify_atoms(atoms)
    keep = next(a for a in atoms if a.id == "a_keep")
    other = next(a for a in atoms if a.id == "a_unsure")
    assert keep.atom_type.value == "scope_item" and keep.value["c3_label"]["type"] == "small_talk"
    other.atom_type = _Type("task")              # a later stage typed it: leave it
    from app.core.schemas import AtomType

    n = c3.apply_final_labels([keep, other])
    assert n == 1
    assert str(getattr(keep.atom_type, "value", keep.atom_type)) == "small_talk"
    assert "scope_item" in keep.value["alt_atom_types"] and "c3_label" in keep.review_flags
    assert other.atom_type.value == "task"


def test_a_sure_base_reading_replaces_a_simple_rule_type_at_the_end(monkeypatch):
    from app.core.schemas import AtomType

    a = _Atom("r1", "Payment due within thirty days of invoice.", "acceptance_criterion")
    b = _Atom("r2", "Ann Lee, Facilities Manager, ann@example.com", "stakeholder")
    a.atom_type, b.atom_type = AtomType("acceptance_criterion"), AtomType("stakeholder")
    c3.set_deal([], {})
    c3._DEAL["types"].update({"r1": ["contract_term", 0.93], "r2": ["small_talk", 0.99]})
    assert c3.apply_final_labels([a, b]) == 1
    assert a.atom_type.value == "contract_term" and a.value["c3_label"]["was"] == "acceptance_criterion"
    assert b.atom_type.value == "stakeholder"          # carries values: never replaced
    c3._DEAL["types"]["r1"] = ["contract_term", 0.8]
    a.atom_type = AtomType("acceptance_criterion")
    assert c3.apply_final_labels([a]) == 0             # under 0.85
    c3.set_deal(None)


def test_the_signature_rule_leaves_a_line_the_base_typed():
    from app.core import deal_chatter
    from app.core.schemas import AtomType

    a = _Atom("s1", "Ann Lee | Facilities Manager | ann@example.com", "stakeholder")
    a.atom_type = AtomType("stakeholder")
    a.value = {"c3_label": {"type": "stakeholder", "act": "fill"}}
    assert deal_chatter._typed_by_base(a) is True
    a.atom_type = AtomType("task")                     # a stage re-typed it since
    assert deal_chatter._typed_by_base(a) is False


def test_typing_can_read_no_lessons_at_all(monkeypatch):
    from app.core import decide as decide_mod

    monkeypatch.setenv("SOWSMITH_TAUGHT_TYPES", "off")
    monkeypatch.setattr(decide_mod, "get_store", lambda: (_ for _ in ()).throw(AssertionError("read the store")))
    assert tac._apply_taught_types(_atoms()) == {}


def test_site_role_reading_decides_before_the_llm(monkeypatch):
    from app.core import site_geo_fallback as sgf

    c3._DEAL.clear()
    c3._DEAL.update(reads={"a1": {"jdg:site_role": ["job_site", 0.97]},
                           "a2": {"jdg:site_role": ["vendor_or_billing_address", 0.6]}})
    monkeypatch.setenv("SOWSMITH_C3_SITE_ROLE", "on")
    d = sgf._base_site_role("a1")
    assert d.verdict == "job_site" and d.source == "c3"
    assert sgf._base_site_role("a2") is None  # unsure: the LLM decides
    assert sgf._base_site_role("a3") is None  # not read
    monkeypatch.setenv("SOWSMITH_C3_SITE_ROLE", "off")
    assert sgf._base_site_role("a1") is None
    c3._DEAL.clear()


def test_authority_types_every_sure_line_and_survives_rules(monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.setenv("SOWSMITH_C3_AUTHORITY", "on")
    c3._DEAL.clear()
    c3._DEAL.update(types={"a1": ["small_talk", 0.97], "a2": ["physical_site", 0.95],
                           "a3": ["physical_site", 0.95], "a4": ["task", 0.6]})
    a1 = NS(id="a1", atom_type="scope_item", value={}, review_flags=[])
    a2 = NS(id="a2", atom_type="deal_metadata", value={"kind": "physical_site"}, review_flags=[])  # demoted site row
    a3 = NS(id="a3", atom_type="deal_metadata", value={}, review_flags=[])  # no site value: recorded, not typed
    a4 = NS(id="a4", atom_type="scope_item", value={}, review_flags=[])  # unsure: left alone
    assert c3.apply_authority([a1, a2, a3, a4]) == 2
    assert str(a1.atom_type.value) == "small_talk" and str(a2.atom_type.value) == "physical_site"
    assert a3.atom_type == "deal_metadata" and a3.value["c3_base_says"]["type"] == "physical_site"
    assert a4.atom_type == "scope_item"
    a2.atom_type = "deal_metadata"  # a later rule re-types it
    assert c3.apply_authority([a2], final=True) == 1 and str(a2.atom_type.value) == "physical_site"
    c3._DEAL.clear()
