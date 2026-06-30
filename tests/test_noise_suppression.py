"""Universal noise-suppression gate: rate-card / materials-catalog rows and
rate-label-as-person atoms are dropped from scope, real deal evidence is kept.

This is the learning-loop fix for the single biggest accuracy failure we see on
real deals: a Deal-Kit spreadsheet's global rate card + materials catalog being
ingested as hundreds of deal ``pricing_assumption`` atoms (and time-window /
rate labels becoming ``stakeholder`` people). The input texts below are the
shapes observed live on deal #010063.

The store matches by embedding kNN. In prod the embedder is qwen3-embedding:8b;
here we inject a deterministic stand-in that keys on the GENERIC features that
distinguish a reference/template row (per-country labor rate, OEM/part catalog
line, billing time-window label) from real deal scope. The seeded exemplars and
the test inputs are independent strings — the gate fires by *concept*, not by
literal match — so this proves the universal mechanism, not a lookup table.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core import decide as decide_mod
from app.core.feedback_store import FeedbackStore, seed_default_corrections
from app.core.noise_suppression import suppress_noise_atoms
from app.core.noise_suppression_seed import (
    NOISE_DROP_VERDICT,
    NOISE_KEEP_VERDICT,
    PERSON_NOISE_RELATION,
    PRICING_NOISE_RELATION,
    noise_gate_corrections,
)
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

# ── deterministic concept embedder (qwen3-embedding stand-in) ────────────────
# Four generic feature axes that separate reference/template content from real
# deal scope. Keys on shapes (country+labor-rate, catalog line, rate label, deal
# action), never on the exemplar strings themselves.
_FEATURES: list[tuple[int, tuple[str, ...]]] = [
    (0, ("country",)),                 # per-country rate-card row …
    (0, ("technician", "hr. min", "hour minimum", "l1", "l2 euc")),  # … + labor rate
    (1, ("material description", "oem", "qty in packaging", "cost $", "part:")),
    (2, ("business hours", "after hours", "stated rate", "local time")),
    (3, ("replace", "install", "provide", "survey", "access point", "lift",
         "payment terms", "net 30", "onsite", "field eng", "rf valid")),
]
_DIM = 5  # 4 concept axes + 1 "other" so unrelated text isn't a zero vector


def _embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        low = (t or "").lower()
        hit = False
        for dim, keys in _FEATURES:
            if any(k in low for k in keys):
                out[i, dim] = 1.0
                hit = True
        if not hit:
            out[i, 4] = 1.0
        n = float(np.linalg.norm(out[i]))
        if n > 0:
            out[i] /= n
    return out


def _store() -> FeedbackStore:
    store = FeedbackStore(
        ":memory:", embed_fn=_embed, reachable_fn=lambda: True
    )
    # Force the deterministic per-correction cosine path (the neural head is
    # exercised by its own tests; here we want a fixed, threshold-based proof).
    store._enable_head = False
    seed_default_corrections(store)
    return store


def _atom(atom_id: str, atom_type: AtomType, raw_text: str) -> EvidenceAtom:
    src = SourceRef(
        id=f"src_{atom_id}",
        artifact_id="art",
        artifact_type=ArtifactType.xlsx,
        filename="Deal Kit.xlsx",
        locator={"extraction": "test"},
        extraction_method="test",
        parser_version="test",
    )
    return EvidenceAtom(
        id=atom_id,
        project_id="010063",
        artifact_id="art",
        atom_type=atom_type,
        raw_text=raw_text,
        normalized_text=raw_text.lower(),
        value={},
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.vendor_quote,
        confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


# Real #010063 shapes — the noise we want gone …
_JUNK = [
    _atom("rc1", AtomType.pricing_assumption,
          "Country: Indonesia | Networking L1 Technician 2 hr. min: 73.5 | 4hr. Min.: 69"),
    _atom("rc2", AtomType.pricing_assumption,
          "Country: United Arab Emirates | Networking L1 Technician 8 hour minimum: 73.0"),
    _atom("cat1", AtomType.pricing_assumption,
          "ID #: 16 | Material Description: 24-Port CAT6 Patch Panel | OEM: CommScope | "
          "QTY in Packaging: 1 | USA Cost $ [Pre-Tax]"),
    _atom("cat2", AtomType.pricing_assumption,
          "ID #: 96 | Material Description: Fender Washers | QTY in Packaging: 100"),
    _atom("lbl1", AtomType.stakeholder,
          "Business Hours: 8:00 AM to 5:00 PM (17:00) local time at the Stated Rate."),
    _atom("lbl2", AtomType.stakeholder, "Stated Rate"),
]

# … and the real deal evidence that MUST survive.
_REAL = [
    _atom("scope1", AtomType.scope_item,
          "Completion of the one-for-one replacement of nine Ubiquiti access points."),
    _atom("scope2", AtomType.scope_item,
          "Provide and operate the required lift to access ceiling-mounted AP locations."),
    _atom("price1", AtomType.pricing_assumption,
          "Wifi Survey Tech onsite RF validation — fixed price 1,700."),
    # The #010065 root-cause guard: a REAL after-hours premium for THIS deal is
    # pricing, not the template "After Hours: 50% increase of Stated Rate" label.
    # It must survive (checked only against the rate-card/catalog gate).
    _atom("ahprice", AtomType.pricing_assumption,
          "After-hours work at the hospital sites billed at 1.5x the standard onsite rate."),
    _atom("person1", AtomType.stakeholder,
          "Dana Lee, Network Engineer, dana.lee@monument.org, 555-998-1212"),
]


def test_disabled_by_default_is_a_noop(monkeypatch):
    monkeypatch.delenv("SOWSMITH_NOISE_SUPPRESSION", raising=False)
    decide_mod.set_store(_store())
    try:
        atoms = _JUNK + _REAL
        kept, dropped = suppress_noise_atoms(atoms, project_id="010063")
        assert dropped == []
        assert kept == atoms
    finally:
        decide_mod.set_store(None)


def test_no_store_is_a_noop(monkeypatch):
    monkeypatch.setenv("SOWSMITH_NOISE_SUPPRESSION", "1")
    decide_mod.set_store(None)
    atoms = _JUNK + _REAL
    kept, dropped = suppress_noise_atoms(atoms, project_id="010063")
    assert dropped == []
    assert kept == atoms


def test_drops_reference_noise_keeps_real_evidence(monkeypatch):
    monkeypatch.setenv("SOWSMITH_NOISE_SUPPRESSION", "1")
    decide_mod.set_store(_store())
    try:
        atoms = _JUNK + _REAL
        kept, dropped = suppress_noise_atoms(atoms, project_id="010063")

        dropped_ids = {a.id for a in dropped}
        kept_ids = {a.id for a in kept}

        # Every rate-card / catalog / rate-label atom is gone …
        assert dropped_ids == {"rc1", "rc2", "cat1", "cat2", "lbl1", "lbl2"}
        # … and every real scope / price / person survived — INCLUDING the real
        # after-hours deal rate (the #010065 false-positive guard).
        assert {"scope1", "scope2", "price1", "ahprice", "person1"} <= kept_ids
        assert "ahprice" not in dropped_ids
        # Partition is total and lossless.
        assert len(kept) + len(dropped) == len(atoms)
    finally:
        decide_mod.set_store(None)


def test_real_after_hours_rate_is_kept_not_dropped(monkeypatch):
    # Focused regression for deal #010065: "after hours" as a deal pricing term
    # must never be suppressed, even though it shares wording with the template
    # rate label. The pricing gate never sees the rate-label concept.
    monkeypatch.setenv("SOWSMITH_NOISE_SUPPRESSION", "1")
    decide_mod.set_store(_store())
    try:
        real_rate = _atom(
            "ah_only", AtomType.pricing_assumption,
            "Weekend access requires an after-hours premium for this deal's four hospital sites.",
        )
        kept, dropped = suppress_noise_atoms([real_rate], project_id="010065")
        assert dropped == []
        assert kept == [real_rate]
    finally:
        decide_mod.set_store(None)


def test_time_window_as_stakeholder_is_dropped(monkeypatch):
    # The flip side: the SAME after-hours wording, when mis-typed as a PERSON,
    # is correctly dropped (a schedule is not a stakeholder).
    monkeypatch.setenv("SOWSMITH_NOISE_SUPPRESSION", "1")
    decide_mod.set_store(_store())
    try:
        fake_person = _atom(
            "win_person", AtomType.stakeholder,
            "After Hours: 5:00 PM to 8:00 AM: 50% increase of Stated Rate.",
        )
        kept, dropped = suppress_noise_atoms([fake_person], project_id="010065")
        assert [a.id for a in dropped] == ["win_person"]
        assert kept == []
    finally:
        decide_mod.set_store(None)


def test_only_noise_prone_types_are_examined(monkeypatch):
    # A scope_item that happens to read like a catalog line is NOT a candidate
    # type, so the gate never touches it (bounded blast radius).
    monkeypatch.setenv("SOWSMITH_NOISE_SUPPRESSION", "1")
    decide_mod.set_store(_store())
    try:
        disguised = _atom(
            "sc_catalogish", AtomType.scope_item,
            "ID #: 5 | Material Description: CAT6 Module | OEM: CommScope | QTY in Packaging: 1",
        )
        kept, dropped = suppress_noise_atoms([disguised], project_id="010063")
        assert dropped == []
        assert kept == [disguised]
    finally:
        decide_mod.set_store(None)


def test_seed_installs_both_gates():
    store = _store()
    for relation in (PRICING_NOISE_RELATION, PERSON_NOISE_RELATION):
        corrs = [c for c in store.all_corrections() if c.relation == relation]
        verdicts = {c.verdict for c in corrs}
        assert NOISE_DROP_VERDICT in verdicts, relation
        assert NOISE_KEEP_VERDICT in verdicts, relation


def test_noise_gate_corrections_shape():
    corrs = noise_gate_corrections()
    relations = {c.relation for c in corrs}
    assert relations == {PRICING_NOISE_RELATION, PERSON_NOISE_RELATION}
    drops = [c for c in corrs if c.verdict == NOISE_DROP_VERDICT]
    keeps = [c for c in corrs if c.verdict == NOISE_KEEP_VERDICT]
    # pricing: 2 drop concepts + 1 keep; person: 1 drop concept + 1 keep.
    assert len(drops) >= 3 and len(keeps) == 2
    assert all(c.exemplars for c in corrs)  # no empty exemplar sets
