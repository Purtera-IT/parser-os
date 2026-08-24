"""Offline tests for the blob-mirror (no Azure needed).

Covers the two invariants that matter: (1) a Correction survives the
asdict→JSON→Correction(**) round-trip the mirror uses, and (2) the mirror is a
hard no-op when SOWSMITH_FEEDBACK_BLOB is off (so it can never touch a default
compile or a fix). Run: python _test_feedback_blob.py
"""
import dataclasses
import json
import os

from app.core.feedback_store import Correction
from app.core import feedback_blob as fb


def test_roundtrip():
    c = Correction(
        id="pm_type_abc123",
        relation="atom_type",
        verdict="rate_card",
        scope="deal",
        scope_key="deal-1",
        exemplars=["Field Technician | $98/hr | 55 hrs"],
        threshold=0.74,
        relations={"authoritative": "a"},
        instruction="PM Type: bom_line -> rate_card",
        created_by="pm",
    )
    back = Correction(**json.loads(json.dumps(dataclasses.asdict(c))))
    assert back == c, "round-trip changed the correction"
    print("  ok: asdict->json->Correction round-trip is lossless")


def test_gated_off_is_noop():
    os.environ.pop("SOWSMITH_FEEDBACK_BLOB", None)
    c = Correction(id="x", relation="atom_type", verdict="rate_card")

    class _Spy:
        added = 0

        def all_corrections(self, *, active_only=True):
            return []

        def add(self, _c):
            self.added += 1

    spy = _Spy()
    assert fb.upload_correction(c) is False, "upload should no-op when gated off"
    assert fb.sync_into_store(spy) == 0, "sync should no-op when gated off"
    assert spy.added == 0, "gated-off sync must not touch the store"
    print("  ok: gated-off mirror is a hard no-op (upload=False, sync=0)")


if __name__ == "__main__":
    test_roundtrip()
    test_gated_off_is_noop()
    print("PASS _test_feedback_blob")
