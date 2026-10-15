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
    # The LLM kept everything: the keep agrees, the assignment does not.
    assert report["agree"] == 1 and report["disagree"] == 1
    assert sum(report["disagree_pairs"].values()) == 1
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
