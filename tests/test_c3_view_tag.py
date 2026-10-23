"""The view:<quoting|planning> tag: an input like stage:, hashed only for a base trained on it."""


def test_the_view_tag_is_an_input_only_for_a_base_trained_on_it(tmp_path):
    import torch
    from ml.c3.data import DealExample, featurize
    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.schema import load_schema
    from dataclasses import replace

    schema = load_schema()
    rows = [{"key": "a", "text": "We need 4 APs.", "entered_at": "2026-05-01T00:00:00Z", "doc_id": "d",
             "doc_kind": "email", "stage_at": "scoping", "view": "quoting"},
            {"key": "b", "text": "Install by June.", "entered_at": "2026-05-02T00:00:00Z", "doc_id": "d",
             "doc_kind": "email", "stage_at": "scoping", "view": "quoting"}]
    batch = featurize(DealExample.from_dict({"deal_id": "x", "atoms": rows}), schema)
    assert batch.inputs()["view"] == ["quoting", "quoting"]
    torch.manual_seed(0)
    old = C3Model(schema, replace(SMALL, tag_dropout=0.0)).eval()
    inp = batch.inputs()
    with torch.no_grad():
        a = old.encoder.embed_atoms(inp)
        b = old.encoder.embed_atoms(dict(inp, view=["planning", "planning"]))
    assert torch.equal(a, b)                       # an older base never reads the tag
    old.cfg.view_tag = True
    with torch.no_grad():
        c = old.encoder.embed_atoms(dict(inp, view=["planning", "planning"]))
    assert not torch.equal(a, c)                   # hashed like stage: once trained on it
    path = tmp_path / "ck.pt"
    save_checkpoint(path, old, step=1, schema_sha256="s")
    fresh = C3Model(schema, replace(SMALL, tag_dropout=0.0))
    load_checkpoint(path, fresh, schema_sha256="s")
    assert fresh.cfg.view_tag is True
