"""The planning-heads contract (run O): the SOW placement, quote and primary-service heads train AND serve in
the planning view, switched together -- C3_PLANNING_HEADS=1 at training lifts the stored mask and records
``planning_heads`` in the checkpoint; serve answers them in the planning view only for such a checkpoint.

Hermetic: made-up lines, no GPU, the service read is a stub."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from c3_record import view as V

KEYS = ["col:label_type", "read:co_action", "read:sow_section", "read:sow_coverage", "read:sow_fill",
        "read:billing_type", "read:labor_hours", "read:crew_size", "read:rate", "jdg:billing_type",
        "jdg:billing_type:reason", "jdg:hours:qty", "jdg:hours:per", "deal:primary_service", "deal:site_count",
        "read:sow_element", "jdg:term", "read:task_owner_role"]
PLANNING_ON = ["read:sow_section", "read:sow_coverage", "read:sow_fill", "read:billing_type", "read:labor_hours",
               "read:crew_size", "read:rate", "jdg:billing_type", "jdg:billing_type:reason", "jdg:hours:qty",
               "jdg:hours:per", "deal:primary_service"]


@pytest.fixture(autouse=True)
def _off(monkeypatch):
    monkeypatch.delenv("C3_PLANNING_HEADS", raising=False)
    monkeypatch.delenv("SOWSMITH_C3_PLANNING_HEADS", raising=False)


def test_off_by_default_every_view_reads_and_trains_as_before():
    old_planning = [k for k in KEYS if not V.is_quote_head(k)]
    assert V.heads("planning", KEYS) == old_planning
    assert V.head_scale("planning", KEYS) == {k: 0.0 for k in KEYS if V.is_quote_head(k)}
    stored = V.head_scale("planning", KEYS)
    assert V.train_scale("planning", stored) == stored


def test_on_the_planning_heads_are_asked_and_trained_in_planning_only():
    on = V.heads("planning", KEYS, planning_heads=True)
    assert set(PLANNING_ON) <= set(on)
    # Not part of the contract: still masked in planning.
    assert {"deal:site_count", "read:sow_element", "jdg:term"}.isdisjoint(on)
    scale = V.head_scale("planning", KEYS, planning_heads=True)
    assert set(scale) == {"deal:site_count", "read:sow_element", "jdg:term"}
    # The quoting view is untouched either way.
    assert V.heads("quoting", KEYS, planning_heads=True) == V.heads("quoting", KEYS)
    assert V.head_scale("quoting", KEYS, planning_heads=True) == V.head_scale("quoting", KEYS)


def test_family_keys_cover_their_sub_heads_only():
    assert V.is_planning_head("jdg:hours:qty") and V.is_planning_head("jdg:billing_type:reason")
    assert not V.is_planning_head("jdg:hours_x") and not V.is_planning_head("jdg:term:tax")


def test_train_scale_lifts_only_the_stored_planning_masks(monkeypatch):
    stored = V.head_scale("planning", KEYS)               # what every v10 planning file stores
    stored["read:task_owner_role"] = 0.5                  # any other entry is kept exactly
    monkeypatch.setenv("C3_PLANNING_HEADS", "1")
    got = V.train_scale("planning", stored)
    assert all(k not in got for k in PLANNING_ON)
    assert got["deal:site_count"] == 0.0 and got["read:task_owner_role"] == 0.5
    quoting = {"read:task_owner_role": 0.25}
    assert V.train_scale("pre_quote", quoting) == quoting


def _blob(view, head_scale):
    return {"deal_id": "x", "view": view, "head_scale": head_scale,
            "atoms": [{"key": "a", "text": "We need 4 APs.", "entered_at": "2026-05-01T00:00:00Z", "doc_id": "d",
                       "doc_kind": "email", "stage_at": "scoping", "view": view}]}


def test_the_loader_applies_the_contract(monkeypatch):
    from ml.c3.data import DealExample

    stored = V.head_scale("planning", KEYS)
    assert DealExample.from_dict(_blob("planning", stored)).head_scale == stored
    monkeypatch.setenv("C3_PLANNING_HEADS", "1")
    lifted = DealExample.from_dict(_blob("planning", stored)).head_scale
    assert "read:sow_section" not in lifted and lifted["deal:site_count"] == 0.0
    q = V.head_scale("pre_quote", KEYS)
    assert DealExample.from_dict(_blob("pre_quote", q)).head_scale == q


class _Part(SimpleNamespace):
    def __len__(self):
        return len(self.view)


def test_training_refuses_planning_parts_that_still_mask_under_the_contract(monkeypatch):
    import ml.c3.loop as loop
    from ml.c3.schema import load_schema

    schema = load_schema()
    keys = [o.key for o in schema.opportunities]
    monkeypatch.setattr(loop, "replace", lambda cfg, **kw: SimpleNamespace(**{**vars(cfg), **kw}))
    model = SimpleNamespace(schema=schema, cfg=SimpleNamespace(planning_heads=False), encoder=SimpleNamespace())
    part = _Part(task_profile=None, view=["planning"], head_scale=V.head_scale("planning", keys),
                 deal_id="010001@planning")
    loop.configure_training_planning_heads(model, [part])          # off + old data: unchanged
    assert model.cfg.planning_heads is False
    monkeypatch.setenv("C3_PLANNING_HEADS", "1")
    with pytest.raises(ValueError, match="disagree"):
        loop.configure_training_planning_heads(model, [part])
    part.head_scale = V.head_scale("planning", keys, planning_heads=True)
    loop.configure_training_planning_heads(model, [part])
    assert model.cfg.planning_heads is True
    monkeypatch.setenv("C3_PLANNING_HEADS", "0")
    with pytest.raises(ValueError, match="resumed"):
        loop.configure_training_planning_heads(model, [part], resumed=True)


def test_the_checkpoint_carries_the_contract(tmp_path):
    from dataclasses import replace

    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.schema import load_schema

    schema = load_schema()
    m = C3Model(schema, replace(SMALL, tag_dropout=0.0))
    path = tmp_path / "old.pt"
    save_checkpoint(path, m, step=1, schema_sha256="s")
    fresh = C3Model(schema, replace(SMALL, tag_dropout=0.0))
    load_checkpoint(path, fresh, schema_sha256="s")
    assert fresh.cfg.planning_heads is False               # every older base: masked in planning, as trained
    m.cfg.planning_heads = True
    save_checkpoint(path, m, step=1, schema_sha256="s")
    load_checkpoint(path, fresh, schema_sha256="s")
    assert fresh.cfg.planning_heads is True


def _service(cfg):
    from ml.c3 import serve

    seen = {}

    def read(m, d, s, **kw):
        seen["heads"] = kw["heads"]
        return {"lines": {}, "deal": {"deal:primary_service": {"answer": "wireless", "p": 0.9},
                                      "deal:site_count": {"answer": "present", "p": 0.9},
                                      "jdg:task_tier": {"answer": "t1", "p": 0.9}}}

    model = SimpleNamespace(cfg=cfg)
    base = serve.Base(load=lambda: (model, object(), "cpu"), read=read)
    base.model = (model, object(), "cpu")
    base._loaded.set()
    payload = {"deal_id": "d", "atoms": [{"key": "a", "text": "We need 4 APs.", "entered_at": 1.0}]}
    out = base.read(payload, 5.0, line_heads=("read:sow_section", "read:billing_type", "read:task_owner_role"),
                    full=True, view="planning")
    return seen, out


def test_serve_follows_the_checkpoint_not_the_caller(monkeypatch):
    monkeypatch.setenv("C3_PLANNING_HEADS", "1")                     # the caller's switch does not decide serve
    seen, out = _service(SimpleNamespace(planning_heads=False))
    assert "read:sow_section" not in seen["heads"] and set(out["deal"]) == {"jdg:task_tier"}
    assert out["planning_heads"] is False
    seen, out = _service(SimpleNamespace(planning_heads=True))
    assert {"read:sow_section", "read:billing_type", "read:task_owner_role"} <= set(seen["heads"])
    assert set(out["deal"]) == {"jdg:task_tier", "deal:primary_service"}   # site_count is not in the contract
    monkeypatch.setenv("SOWSMITH_C3_PLANNING_HEADS", "0")                 # the kill switch
    seen, out = _service(SimpleNamespace(planning_heads=True))
    assert "read:sow_section" not in seen["heads"] and out["planning_heads"] is False


def test_the_anchor_leaves_planning_heads_free_on_planning_parts_only(monkeypatch):
    from ml.c3.anchor import _planning_free

    plan, quote = SimpleNamespace(view=["planning"] * 3), SimpleNamespace(view=["quoting"] * 3)
    assert not _planning_free(plan)("read:billing_type")
    monkeypatch.setenv("C3_PLANNING_HEADS", "1")
    assert _planning_free(plan)("read:billing_type") and not _planning_free(plan)("col:label_type")
    assert not _planning_free(quote)("read:billing_type")


def test_the_pick_sees_the_primary_service_at_its_own_minimum_and_weight():
    from ml.c3 import pick

    row = {"valid_col_admission": 0.8, "valid_col_admission_n": 500.0,
           "valid_read_sow_section": 0.4, "valid_read_sow_section_n": 900.0,
           "valid_deal_primary_service": 1.0, "valid_deal_primary_service_n": 4.0}
    heads = pick.composite_heads({"start": row}, heads=("col:admission", "read:sow_section", "deal:primary_service"))
    assert heads == ["col:admission", "read:sow_section", "deal:primary_service"]
    assert pick.composite(row, heads) == pytest.approx((0.8 + 0.4 + 0.5 * 1.0) / 2.5)
    row["valid_deal_primary_service_n"] = 1.0
    assert "deal:primary_service" not in pick.composite_heads({"start": row}, heads=pick.PICK_HEADS)
