"""C3_COPY_OF: a line that repeats an earlier line is an input fact (copy:repeat).
C3_COPY_LABELS: a repeat follows one admission policy (drop), company reject or not."""
from dataclasses import replace


def _rows():
    return [{"key": "a", "text": "Estimated Duration: Approximately one week", "entered_at": "2026-05-01T00:00:00Z",
             "doc_id": "d1", "doc_kind": "email", "atom_id": "atm_a",
             "label": {"label_key": "a", "labeler": "pm", "label_type": "work_scope_item"}},
            # The same words quoted under a later email: a company reject (co_action) today keeps admission.
            {"key": "b", "text": "Estimated duration - approximately one week.", "entered_at": "2026-05-03T00:00:00Z",
             "doc_id": "d2", "doc_kind": "email", "atom_id": "atm_b",
             "label": {"label_key": "b", "labeler": "pm", "label_type": "work_scope_item", "rejected": "true",
                       "reads_set": {"co_action": "reject"}}},
            # A plain rejected repeat is drop either way.
            {"key": "c", "text": "Estimated Duration: Approximately one week", "entered_at": "2026-05-04T00:00:00Z",
             "doc_id": "d3", "doc_kind": "email", "atom_id": "atm_c",
             "label": {"label_key": "c", "labeler": "pm", "label_type": "work_scope_item", "rejected": "true"}},
            # Different words, but the parser names the line it copies (duplicate_of).
            {"key": "d", "text": "> Est. duration ~1 wk", "entered_at": "2026-05-05T00:00:00Z",
             "doc_id": "d4", "doc_kind": "email", "copy_of": "atm_a"},
            {"key": "e", "text": "Two engineers on site.", "entered_at": "2026-05-06T00:00:00Z",
             "doc_id": "d4", "doc_kind": "email", "copy_of": "atm_missing"}]


def test_copy_flags_mark_repeats_only():
    from ml.c3.data import DealExample, copy_flags
    deal = DealExample.from_dict({"deal_id": "x", "atoms": _rows()})
    assert copy_flags(deal.atoms) == ["", "repeat", "repeat", "repeat", ""]


def test_flag_off_is_unchanged_and_on_makes_repeats_consistent(monkeypatch):
    from ml.c3.data import DealExample, featurize
    from ml.c3.schema import load_schema
    schema = load_schema()
    adm = schema.opportunity("col:admission") if hasattr(schema, "opportunity") else \
        next(o for o in schema.opportunities if o.key == "col:admission")
    co = next(o.key for o in schema.opportunities if o.field == "co_action")
    monkeypatch.delenv("C3_COPY_LABELS", raising=False)
    off = featurize(DealExample.from_dict({"deal_id": "x", "atoms": _rows()}), schema)
    assert off.copy == ["", "repeat", "repeat", "repeat", ""]
    assert [off.targets[adm.key][i] for i in range(3)] == [adm.index("keep"), adm.index("keep"), adm.index("drop")]
    assert off.targets[co][1] == next(o for o in schema.opportunities if o.key == co).index("reject")
    monkeypatch.setenv("C3_COPY_LABELS", "1")
    on = featurize(DealExample.from_dict({"deal_id": "x", "atoms": _rows()}), schema)
    assert [on.targets[adm.key][i] for i in range(3)] == [adm.index("keep"), adm.index("drop"), adm.index("drop")]
    for k in off.targets:                            # nothing else moves (co_action included)
        if k != adm.key:
            assert off.targets[k] == on.targets[k], k


def test_copy_tag_is_an_input_only_for_a_base_trained_on_it(tmp_path):
    import torch
    from ml.c3.data import DealExample, featurize
    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.schema import load_schema
    schema = load_schema()
    batch = featurize(DealExample.from_dict({"deal_id": "x", "atoms": _rows()}), schema)
    torch.manual_seed(0)
    model = C3Model(schema, replace(SMALL, tag_dropout=0.0)).eval()
    inp = batch.inputs()
    with torch.no_grad():
        a = model.encoder.embed_atoms(inp)
        b = model.encoder.embed_atoms(dict(inp, copy=[""] * len(batch)))
    assert torch.equal(a, b)                         # default: the column is never read
    model.cfg.copy_tag = True
    with torch.no_grad():
        c = model.encoder.embed_atoms(inp)
    assert torch.equal(a[[0, 4]], c[[0, 4]]) and not torch.equal(a[1], c[1])
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, step=1, schema_sha256="s")
    fresh = C3Model(schema, replace(SMALL, tag_dropout=0.0))
    load_checkpoint(path, fresh, schema_sha256="s")
    assert fresh.cfg.copy_tag is True
