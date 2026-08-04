"""Compile training rows survive the container, without workers clobbering
each other.

The bug these pin: the training log lives on ephemeral /tmp and was only ever
downloaded, never uploaded — so every row every compile produced was lost, the
shared log sat at 3 rows, and the eval gate had no holdout, so nothing could
ever be promoted.
"""
from __future__ import annotations

from app.core import training_row_blob as trb
from app.core.training_log import TEACHER_LLM, TEACHER_PM, TrainingRow


def _row(**over) -> TrainingRow:
    kw = dict(
        relation="atom_type",
        label="service_line",
        raw_text="Field Technician | $98.00 | Per Hour",
        teacher=TEACHER_LLM,
        deal_id="d1",
        project_id="p1",
    )
    kw.update(over)
    return TrainingRow(**kw)


def test_rows_round_trip_through_jsonl():
    rows = [_row(), _row(label="bom_line", teacher=TEACHER_PM)]
    back = trb.jsonl_to_rows(trb.rows_to_jsonl(rows))
    assert [r.id for r in back] == [r.id for r in rows], "ids must survive"
    assert [r.label for r in back] == ["service_line", "bom_line"]
    assert back[1].teacher == TEACHER_PM


def test_provenance_survives_as_a_dict():
    # to_row() JSON-strings provenance; the blob form must not, or the reload
    # yields a string where the trainer expects a mapping.
    r = _row(provenance={"model": "deepseek", "role_map": {"ORG": "PurTera"}})
    back = trb.jsonl_to_rows(trb.rows_to_jsonl([r]))[0]
    assert isinstance(back.provenance, dict)
    assert back.provenance["model"] == "deepseek"


def test_one_bad_line_does_not_cost_the_batch():
    good = trb.rows_to_jsonl([_row()])
    text = "\n".join([good, "{not json", "", '"a string"', good])
    assert len(trb.jsonl_to_rows(text)) == 2


def test_unknown_fields_are_dropped_not_fatal():
    # An older worker's blob must not crash a newer nightly.
    line = '{"relation":"atom_type","label":"risk","raw_text":"x","field_from_the_future":1}'
    rows = trb.jsonl_to_rows(line)
    assert len(rows) == 1 and rows[0].label == "risk"


def test_rows_without_a_label_are_skipped():
    assert trb.jsonl_to_rows('{"relation":"atom_type","raw_text":"x"}') == []


def test_disabled_by_default_so_normal_compiles_are_unchanged(monkeypatch):
    monkeypatch.delenv("SOWSMITH_FEEDBACK_BLOB", raising=False)
    assert trb._container_client() is None
    assert trb.upload_rows([_row()]) is False
    assert trb.sync_into_log() == 0


def test_upload_failure_never_breaks_a_compile(monkeypatch):
    class Boom:
        def upload_blob(self, **kw):
            raise RuntimeError("network is down")

    monkeypatch.setattr(trb, "_container_client", lambda: Boom())
    assert trb.upload_rows([_row()]) is False  # swallowed, not raised


def test_each_batch_gets_its_own_blob_so_workers_cannot_clobber(monkeypatch):
    """The reason this is per-batch rather than uploading the SQLite file:
    concurrent workers would otherwise silently erase each other's rows."""
    names: list[str] = []

    class Fake:
        def upload_blob(self, name, data, overwrite=False):
            names.append(name)

    monkeypatch.setattr(trb, "_container_client", lambda: Fake())
    assert trb.upload_rows([_row()]) is True
    assert trb.upload_rows([_row(deal_id="d2")]) is True
    assert len(set(names)) == 2, "two batches must not share a blob name"
    assert all(n.startswith("_training_rows/") for n in names)


def test_sync_merges_every_batch_and_is_idempotent(monkeypatch):
    batch_a = trb.rows_to_jsonl([_row(), _row(label="risk")])
    batch_b = trb.rows_to_jsonl([_row(deal_id="d2", label="stakeholder")])

    class Blob:
        def __init__(self, name):
            self.name = name

    class Fake:
        def list_blobs(self, name_starts_with=""):
            return [Blob("_training_rows/a.jsonl"), Blob("_training_rows/b.jsonl")]

        def download_blob(self, name):
            self._cur = batch_a if name.endswith("a.jsonl") else batch_b
            return self

        def readall(self):
            return self._cur.encode("utf-8")

    added: list[TrainingRow] = []

    class Log:
        def add_many(self, rows):
            added.extend(rows)
            return len(rows)

    monkeypatch.setattr(trb, "_container_client", lambda: Fake())
    log = Log()
    first = trb.sync_into_log(log)
    assert first == 3
    labels = {r.label for r in added}
    assert labels == {"service_line", "risk", "stakeholder"}

    # Re-running imports the same ids again; the log upserts on PRIMARY KEY, so
    # this is a no-op in the DB rather than duplicate training signal.
    added.clear()
    assert trb.sync_into_log(log) == 3
    assert {r.id for r in added} <= {r.id for r in log_ids(batch_a, batch_b)}


def log_ids(*batches):
    out = []
    for b in batches:
        out.extend(trb.jsonl_to_rows(b))
    return out


def test_one_unreadable_batch_does_not_stop_the_others(monkeypatch):
    good = trb.rows_to_jsonl([_row()])

    class Blob:
        def __init__(self, name):
            self.name = name

    class Fake:
        def list_blobs(self, name_starts_with=""):
            return [Blob("bad.jsonl"), Blob("good.jsonl")]

        def download_blob(self, name):
            if name == "bad.jsonl":
                raise RuntimeError("corrupt")
            self._cur = good
            return self

        def readall(self):
            return self._cur.encode("utf-8")

    class Log:
        def add_many(self, rows):
            return len(rows)

    monkeypatch.setattr(trb, "_container_client", lambda: Fake())
    assert trb.sync_into_log(Log()) == 1
