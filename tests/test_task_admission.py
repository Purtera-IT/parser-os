"""A task a PM removed as "not this job" is not proposed again.

The Deal Kit teaches `admission → drop` on the task's own sentence at publish.
The compiler's task_admission stage asks the store on every task atom and drops
a confident learned `drop`; everything else is kept, and no store is a no-op.
"""

from __future__ import annotations

import numpy as np

from app.core import decide as decide_mod
from app.core.feedback_store import FeedbackStore
from app.core.pm_feedback import apply_pm_correction
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.task_admission import drop_taught_out_tasks

_KEYS = ["lift rental", "cable drop", "rack and stack", "survey"]


def _embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), len(_KEYS) + 1), dtype=np.float32)
    for i, t in enumerate(texts):
        low = (t or "").lower()
        hit = False
        for j, k in enumerate(_KEYS):
            if k in low:
                out[i, j] = 1.0
                hit = True
        if not hit:
            out[i, len(_KEYS)] = 1.0
        out[i] /= max(float(np.linalg.norm(out[i])), 1e-9)
    return out


def _atom(atom_id: str, raw_text: str, atom_type: AtomType = AtomType.task) -> EvidenceAtom:
    src = SourceRef(id=f"src_{atom_id}", artifact_id="art", artifact_type=ArtifactType.pdf, filename="SOW.pdf",
                    locator={"extraction": "test"}, extraction_method="test", parser_version="test")
    return EvidenceAtom(
        id=atom_id, project_id="deal-1", artifact_id="art", atom_type=atom_type, raw_text=raw_text,
        normalized_text=raw_text.lower(), value={}, entity_keys=[], source_refs=[src], receipts=[],
        authority_class=AuthorityClass.vendor_quote, confidence=0.8, review_status=ReviewStatus.auto_accepted,
        review_flags=[], parser_version="test",
    )


def _store() -> FeedbackStore:
    s = FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)
    s._enable_head = False
    return s


def test_no_store_keeps_everything():
    decide_mod.set_store(None)
    atoms = [_atom("a", "Lift rental for the ceiling APs")]
    assert drop_taught_out_tasks(atoms, project_id="deal-1") == (atoms, [])


def test_a_taught_out_task_is_dropped_and_the_rest_kept():
    store = _store()
    decide_mod.set_store(store)
    try:
        apply_pm_correction(store, {
            "head": "admission", "dealId": "deal-1", "compileId": "", "targetId": "task:lift rental",
            "text": "Lift rental for the ceiling APs", "oldValue": "keep", "newValue": "drop",
            "scope": "deal", "context": "", "rationale": "customer provides the lift",
            "relations": {"outcome": "correct"}, "pm": "pm@x", "candidates": ["keep", "drop"],
        })
        atoms = [
            _atom("lift", "Lift rental for the ceiling APs"),
            _atom("drops", "Install 24 Cat6 cable drops"),
            _atom("scope", "Lift rental for the ceiling APs", AtomType.scope_item),
        ]
        kept, dropped = drop_taught_out_tasks(atoms, project_id="deal-1")
        assert [a.id for a in dropped] == ["lift"]
        assert [a.id for a in kept] == ["drops", "scope"]
        # Taught at deal scope: another deal keeps its lift line.
        kept2, dropped2 = drop_taught_out_tasks([_atom("lift2", "Lift rental for the ceiling APs")], project_id="deal-2")
        assert dropped2 == [] and len(kept2) == 1
    finally:
        decide_mod.set_store(None)
