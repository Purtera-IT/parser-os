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
    def _classify(atoms):
        if calls is not None:
            calls.append(len(atoms))
        by = {a.id: a for a in atoms}
        return {id(by["a_task"]): ("task", 0.95), id(by["a_keep"]): ("_keep", 0.9),
                id(by["a_unsure"]): ("requirement", 0.4), id(by["a_heavy"]): ("stakeholder", 0.99)}

    monkeypatch.setattr(c3, "classify", _classify)


def test_flag_off_never_reads(monkeypatch, llm):
    def _tripwire(atoms):
        raise AssertionError("the C3 layer must not run with the flag off")

    monkeypatch.setattr(c3, "classify", _tripwire)
    atoms = _atoms()
    assert tac.classify_atoms(atoms) == 0
    assert sorted(llm) == sorted(_TEXTS)
    assert all(a.atom_type.value == "scope_item" for a in atoms)
    assert tac.get_last_deflect_stats()["deflected"]["c3"] == 0


def test_flag_on_assigns_sure_light_drops_sure_keep_and_passes_the_rest(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
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
    _stub_readings(monkeypatch)
    atoms = _atoms()
    tac.classify_atoms(atoms)
    by = {a.id: a for a in atoms}
    assert by["a_unsure"].atom_type.value == "requirement"
    assert llm == ["a_heavy"]


def test_a_deal_kit_line_is_never_acted_on(monkeypatch, llm):
    monkeypatch.setenv("SOWSMITH_C3_TYPE", "1")
    atom = _Atom("a_task", _TEXTS["a_task"], ref=_Ref(filename="Example Deal Kit v3.xlsx",
                                                      artifact_type="xlsx"))
    monkeypatch.setattr(c3, "classify", lambda atoms: {id(atom): ("task", 0.99)})
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

    def _raise(atoms):
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
    assert rows["atm1"].doc_kind == "docx" and rows["atm1"].doc_id == "doc"
    assert rows["atm2"].doc_kind == "email"
    assert (rows["atm2"].speaker_role, rows["atm2"].speaker_side) == ("reseller", "theirs")
    assert rows["atm3"].doc_kind == "meeting" and rows["atm3"].speaker_role == "customer"
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
