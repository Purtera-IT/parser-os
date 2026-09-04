"""What one replica learns, every replica must know.

The feedback store is per-process SQLite. The service runs behind replicas
that recycle, so a correction written by one is invisible to the next and to
every fresh container. The compiler already syncs the blob mirror before a
compile for exactly that reason; the service endpoints did not, and it showed:
16 corrections sat in blob while `GET /feedback/corrections` answered 0.

That is not a cosmetic gap. `/feedback/questions/screen` runs in the service,
so a question the PM taught us to hold back was being screened against an
empty store and shown again.
"""

from __future__ import annotations

import app.api.routes_feedback as rf


class FakeStore:
    def __init__(self):
        self.rows = []

    def all_corrections(self, active_only: bool = True):
        return list(self.rows)


def test_the_service_pulls_the_blob_mirror_into_its_store(monkeypatch) -> None:
    store = FakeStore()
    synced = {"count": 0}

    monkeypatch.setattr(rf, "get_store", lambda: store)

    class FakeBlob:
        @staticmethod
        def sync_into_store(s):
            assert s is store
            synced["count"] += 1
            return 3

    import app.core.feedback_blob as real_blob

    monkeypatch.setattr(real_blob, "sync_into_store", FakeBlob.sync_into_store)

    assert rf._active_store() is store
    assert synced["count"] == 1, "the mirror is pulled in on the way to the store"


def test_a_broken_mirror_never_fails_the_endpoint(monkeypatch) -> None:
    store = FakeStore()
    monkeypatch.setattr(rf, "get_store", lambda: store)

    import app.core.feedback_blob as real_blob

    def explode(_s):
        raise RuntimeError("blob unreachable")

    monkeypatch.setattr(real_blob, "sync_into_store", explode)
    assert rf._active_store() is store, "an offline mirror still yields the local store"


def test_no_store_configured_stays_none(monkeypatch) -> None:
    monkeypatch.setattr(rf, "get_store", lambda: None)
    monkeypatch.setattr(
        "app.core.compiler._maybe_wire_feedback_store", lambda: None, raising=False
    )
    assert rf._active_store() is None


def test_the_reason_survives_the_api_boundary() -> None:
    """The caller was already sending `rationale` and the request model had no
    such field, so every forwarded rejection stored "PM Gap: valid → invalid"
    and lost the sentence that made it teachable."""
    assert "rationale" in rf.PMCorrectionRequest.model_fields


def test_a_gold_row_is_mirrored_even_without_a_local_training_log(monkeypatch) -> None:
    """The service has no SOWSMITH_TRAINING_LOG_DB, so a correction made
    through the API wrote a correction blob and no gold row: instant learning,
    nothing the nightly retrain could ever see."""
    import app.core.feedback_blob as blob
    from app.core.pm_feedback import apply_pm_correction

    seen = {"rows": 0, "corrections": 0}
    monkeypatch.setattr(blob, "upload_correction", lambda c: seen.__setitem__("corrections", 1) or True)
    monkeypatch.setattr(
        blob, "upload_training_rows", lambda cid, rows: seen.__setitem__("rows", len(rows)) or True
    )
    monkeypatch.setattr("app.core.training_log.log_rows", lambda rows: (_ for _ in ()).throw(RuntimeError("no log")))

    class Store:
        def __init__(self):
            self.rows = {}

        def get(self, cid):
            return self.rows.get(cid)

        def add(self, c):
            self.rows[c.id] = c

    store = Store()
    cid = apply_pm_correction(
        store,
        {
            "head": "gap", "dealId": "d1", "targetId": "r1",
            "text": "Confirm install documentation is in the fixed fee",
            "oldValue": "invalid", "newValue": "valid", "scope": "deal",
            "rationale": "the PSOW includes a per-location report",
        },
    )
    assert seen["rows"] == 1, "a broken local log must not swallow the mirror"
    assert "per-location report" in store.get(cid).instruction


def test_a_pm_answer_joins_the_site_it_was_asked_about() -> None:
    """An answer with no entity keys is an island: it never reaches the site
    rollups, the scope truth or a conflict, so the answer that was supposed to
    settle a question cannot be found by the thing it settles. Live 010300: the
    cutover answer landed with `entity_keys: []` while the site it named
    carried 98 atoms."""
    from app.core.pm_answer_atoms import pm_answers_to_atoms
    from app.core.schemas import AtomType, AuthorityClass

    events = [
        {
            "action": "answered",
            "rule_id": "site.2970_brandywine_rd_ste_200.cutover_window",
            "question_text": "Confirm the approved cutover / maintenance window for 2970 brandywine rd ste 200.",
            "edited_text": (
                "Weekends and after hours only. Carl confirmed on the 3 Sep call that the "
                "dentistry sites have to be done Saturday and Sunday after hours."
            ),
            "created_at": "2026-09-04T22:52:32.501Z",
        },
        {
            "action": "answered",
            "rule_id": "mode.staff_aug.docs",
            "question_text": "Confirm install documentation / photos / completion report are in the fixed fee",
            "edited_text": "Yes, included. The PSOW has a per-location report and a customer Sign Off List.",
            "created_at": "2026-09-04T22:53:04.751Z",
        },
    ]
    atoms = pm_answers_to_atoms(events, project_id="p1")
    assert len(atoms) == 2
    site_answer, mode_answer = atoms

    assert site_answer.atom_type == AtomType.decision
    assert site_answer.authority_class == AuthorityClass.pm_confirmed
    assert "site:2970_brandywine_rd_ste_200" in site_answer.entity_keys, (
        "the answer joins the site the card was about"
    )
    # A question with no site and no entity in the answer stays unkeyed rather
    # than inventing one, and a person named mid-sentence is not a contact.
    assert not [k for k in mode_answer.entity_keys if k.startswith("stakeholder:")]

    # The claim reads as one sentence, not "…fee.? PM answer: …".
    assert "?? " not in site_answer.raw_text
    assert ".? PM answer" not in site_answer.raw_text
