"""Blob sync refreshes a correction that was merged after this process loaded it."""
import dataclasses
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from app.core import feedback_blob
from app.core.feedback_store import Correction, FeedbackStore, SCOPE_GLOBAL


def _embed(texts):
    return np.ones((len(texts), 4), dtype=np.float32) / 2.0


class _Container:
    def __init__(self):
        self.blobs = {}
        self.downloads = 0

    def put(self, corr: Correction, when: float):
        name = f"_feedback/corrections/{corr.id}.json"
        self.blobs[name] = (json.dumps(dataclasses.asdict(corr)).encode(), when)

    def list_blobs(self, name_starts_with=""):
        return [SimpleNamespace(name=n, last_modified=datetime.fromtimestamp(ts, tz=timezone.utc))
                for n, (_, ts) in self.blobs.items() if n.startswith(name_starts_with)]

    def download_blob(self, name):
        self.downloads += 1
        data = self.blobs[name][0]
        return SimpleNamespace(readall=lambda: data)


def _corr(exemplars, updated_at):
    return Correction(id="pm_type_task", relation="atom_type", verdict="task", scope=SCOPE_GLOBAL,
                      scope_key="", exemplars=list(exemplars), created_at=1000.0, updated_at=updated_at)


def test_merged_correction_replaces_the_older_local_copy(monkeypatch):
    cc = _Container()
    monkeypatch.setattr(feedback_blob, "_container_client", lambda: cc)
    store = FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)

    cc.put(_corr(["line 1"], updated_at=2000.0), when=2001.0)
    assert feedback_blob.sync_into_store(store) == 1
    assert store.get("pm_type_task").exemplars == ["line 1"]

    # Unchanged blob: no download, nothing refreshed.
    before = cc.downloads
    assert feedback_blob.sync_into_store(store) == 0
    assert cc.downloads == before

    # The service merges a new exemplar into the same id.
    cc.put(_corr(["line 1", "000036 request"], updated_at=3000.0), when=3001.0)
    assert feedback_blob.sync_into_store(store) == 1
    assert store.get("pm_type_task").exemplars == ["line 1", "000036 request"]


def test_a_new_correction_is_still_added(monkeypatch):
    cc = _Container()
    monkeypatch.setattr(feedback_blob, "_container_client", lambda: cc)
    store = FeedbackStore(":memory:", embed_fn=_embed, reachable_fn=lambda: True)
    cc.put(_corr(["x"], updated_at=10.0), when=11.0)
    assert feedback_blob.sync_into_store(store) == 1
    assert store.get("pm_type_task") is not None
