"""View boundaries and wire/checkpoint contracts, without model downloads."""
from dataclasses import replace

import torch

from app.core.c3_type_base import deal_payload
from app.core import c3_type_base as serving
from c3_record import provenance as prov
from c3_record import view
from ml.c3.data import DealExample, featurize
from ml.c3.gates import load_checkpoint, save_checkpoint
from ml.c3.loop import SMALL, training_batches
from ml.c3.model import C3Model
from ml.c3.schema import load_schema


def test_provenance_kinds_match_the_final_head_registry():
    answers = {a.value for a in load_schema().by_key()[prov.DOC_KIND_HEAD].answers}
    assert set(prov.DOC_KINDS) == answers - {"_absent"}
    assert all(prov.head_kind(k) in answers for k in prov.DOC_KIND_SYNONYMS)


def test_quoted_history_cannot_enter_before_its_carrier_arrived():
    rows = [{"label_key": "request", "atom_id": "request", "doc_id": "early", "text": "Install 4 APs.",
             "entered_at": 10.0, "doc_kind": "email"},
            {"label_key": "copy", "atom_id": "copy", "doc_id": "late", "text": "Install 4 APs.",
             "entered_at": 30.0, "doc_kind": "email"},
            {"label_key": "quote", "atom_id": "quote", "doc_id": "quote", "text": "Total $400.",
             "entered_at": 20.0, "doc_kind": "quote"}]
    info = {"early": {"direction": "inbound", "kind": "email", "message": True},
            "late": {"direction": "inbound", "kind": "email", "message": True},
            "quote": {"direction": "internal", "kind": "quote"}}
    p = prov.decide(rows, info, {"copy": ("buyer@example.com", 10.0, True, 1)})
    assert p.cut == 20.0
    assert prov.is_input(p, rows[1])  # The old words alone are insufficient.
    assert prov.quoting_inputs(p, rows) == [0]


def test_cut_boundaries_and_other_job_never_become_inputs():
    rows = [{"label_key": k, "atom_id": k, "doc_id": k, "text": "Install APs.",
             "entered_at": t, "doc_kind": "email"} for k, t in [("before", 10), ("tie", 20), ("after", 30)]]
    info = {k: {"direction": "inbound", "kind": "email", "message": True} for k in ["before", "tie", "after"]}
    p = prov.decide(rows, info, {}, other_job={"before": "other job"})
    assert prov.quoting_inputs(p, rows, as_of=20.0) == [1]
    assert view.planning_inputs(rows, 20.0, other_job=p.other_job) == []


def test_view_survives_windows_and_wire_without_supervision():
    schema = load_schema()
    rows = [{"key": str(i), "atom_id": str(i), "text": f"Install AP {i}.", "entered_at": i + 1,
             "doc_id": "request", "stage_at": "scoping", "view": "quoting"} for i in range(6)]
    deal = DealExample.from_dict({"deal_id": "900001@pre_quote", "atoms": rows, "view": "pre_quote",
                                 "head_scale": {"read:task_owner_role": 0.25}})
    parts, _ = training_batches(deal, schema, max_atoms=3)
    assert len(parts) > 1
    assert all(set(b.inputs()["view"]) == {"quoting"} for b in parts)
    assert all(b.head_scale["read:task_owner_role"] == 0.25 for b in parts)
    wire = deal_payload(deal)
    clean = DealExample.from_dict(wire)
    clean.source_supervision = {"judgments": [{"verdict": "SECRET"}], "missed_history": [{"text": "SECRET"}]}
    left, right = featurize(deal, schema).inputs(), featurize(clean, schema).inputs()
    assert set(left) == set(right)
    assert all(torch.equal(left[k], right[k]) if isinstance(left[k], torch.Tensor) else left[k] == right[k]
               for k in left)


def test_legacy_checkpoint_disables_view_even_on_a_tag_enabled_model(tmp_path):
    schema = load_schema()
    old = C3Model(schema, replace(SMALL, view_tag=False))
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, old, step=0, schema_sha256="same")
    fresh = C3Model(schema, replace(SMALL, view_tag=True))
    load_checkpoint(path, fresh, schema_sha256="same")
    assert fresh.cfg.view_tag is False and fresh.encoder.cfg.view_tag is False


def test_new_read_cannot_retain_heads_or_lines_from_the_previous_view():
    serving.set_deal([], {})
    try:
        serving._keep_full({"reads": {"old-line": {"read:sow_section": ["scope", 1.0]}},
                            "deal": {"deal:primary_service": ["av", 1.0]}, "question_bar": 0.9})
        serving._keep_full({"reads": {"current-line": {"read:task_owner_role": ["pm", 1.0]}}})
        current = serving.full_read()
        assert set(current["reads"]) == {"current-line"}
        assert current["deal"] == {} and current["question_bar"] is None
    finally:
        serving.set_deal(None)


def test_read_cache_invalidates_for_every_wire_input():
    row = {"key": "line", "text": "Install APs.", "speaker_role": "customer", "image_pixels": [0.0],
           "table_row": 1, "view": "quoting"}
    before = serving.read_key([row])
    for field, value in [("speaker_role", "internal"), ("image_pixels", [1.0]),
                         ("table_row", 2), ("view", "planning")]:
        assert serving.read_key([dict(row, **{field: value})]) != before
