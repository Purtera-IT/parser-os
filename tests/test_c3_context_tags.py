"""C3_CONTEXT_TAGS: a line's table header, sheet and section heading are input
facts (hdr:/sheet:/sec: tags), the same rule for a training row and a served one."""
from dataclasses import replace
from types import SimpleNamespace


def _rows():
    return [
        # A spreadsheet's bare header row, then its rows (build.py stamps the header on rows 2..).
        {"key": "h", "text": "City | Hours | Qty", "entered_at": "2026-05-01T00:00:00Z", "doc_id": "d1",
         "doc_kind": "file", "section": "12E Closet", "atom_id": "atm_h",
         "table_id": "d1:12E Closet", "table_row_index": 1, "sheet": "12E Closet"},
        {"key": "r", "text": "Dallas | 4 | 2", "entered_at": "2026-05-01T00:00:00Z", "doc_id": "d1",
         "doc_kind": "file", "section": "12E Closet", "atom_id": "atm_r", "order": 1,
         "table_id": "d1:12E Closet", "table_row_index": 2, "sheet": "12E Closet",
         "table_header": ["City", "Hours", "Qty"],
         "label": {"label_key": "r", "labeler": "pm", "label_type": "site_detail"}},
        # A SOW line under a heading.
        {"key": "s", "text": "Customer provides lift access.", "entered_at": "2026-05-02T00:00:00Z",
         "doc_id": "d2", "doc_kind": "sow_draft", "atom_id": "atm_s",
         "section": "Services Proposal > Project Scope > Project Assumptions"},
        # A meeting line: its section is the speaker, never a heading.
        {"key": "m", "text": "We need two techs.", "entered_at": "2026-05-03T00:00:00Z", "doc_id": "d3",
         "doc_kind": "meeting", "section": "Trent Torrence", "atom_id": "atm_m"}]


def test_tags_follow_the_row_fields():
    from ml.c3.data import DealExample, context_tags
    deal = DealExample.from_dict({"deal_id": "x", "atoms": _rows()})
    by = dict(zip([a.key for a in deal.atoms], context_tags(deal.atoms)))
    assert by["h"] == ["sheet:closet"]
    assert by["r"] == ["hdr:city", "hdr:hours", "hdr:qty", "sheet:closet"]
    assert by["s"] == ["sec:project", "sec:assumptions"]
    assert by["m"] == []


def test_flag_off_is_byte_identical_and_on_reads_the_tags(tmp_path):
    import torch
    from ml.c3.data import DealExample, featurize
    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.schema import load_schema
    schema = load_schema()
    with_ctx = featurize(DealExample.from_dict({"deal_id": "x", "atoms": _rows()}), schema)
    plain = featurize(DealExample.from_dict({"deal_id": "x", "atoms": [
        {k: v for k, v in r.items() if k not in ("sheet", "table_header", "table_id", "table_row_index")}
        for r in _rows()]}), schema)
    assert with_ctx.targets == plain.targets                       # an input only, never a target
    torch.manual_seed(0)
    model = C3Model(schema, replace(SMALL, tag_dropout=0.0)).eval()
    with torch.no_grad():
        a = model.encoder.embed_atoms(with_ctx.inputs())
        b = model.encoder.embed_atoms(dict(with_ctx.inputs(), ctx_tags=[[] for _ in range(len(with_ctx))]))
        c = model.encoder.embed_atoms(plain.inputs())
    assert torch.equal(a, b) and torch.equal(a, c)                 # default: the column is never read
    model.cfg.context_tags = True
    with torch.no_grad():
        on = model.encoder.embed_atoms(with_ctx.inputs())
    meeting = next(i for i, t in enumerate(with_ctx.texts) if t.startswith("We need"))
    assert torch.equal(a[meeting], on[meeting])
    assert sum(not torch.equal(a[i], on[i]) for i in range(len(with_ctx))) == 3
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, step=1, schema_sha256="s")
    fresh = C3Model(schema, replace(SMALL, tag_dropout=0.0))
    load_checkpoint(path, fresh, schema_sha256="s")
    assert fresh.cfg.context_tags is True


def test_env_flag_sets_the_config(monkeypatch):
    from ml.c3.loop import SMALL, configure_training_context_tags
    from ml.c3.model import C3Model
    from ml.c3.schema import load_schema
    model = C3Model(load_schema(), SMALL)
    monkeypatch.delenv("C3_CONTEXT_TAGS", raising=False)
    configure_training_context_tags(model)
    assert model.cfg.context_tags is False
    monkeypatch.setenv("C3_CONTEXT_TAGS", "1")
    configure_training_context_tags(model)
    assert model.cfg.context_tags is True and model.encoder.cfg.context_tags is True


def _atom(i, text, ref_loc, filename="survey.xlsx", art="d1"):
    loc = dict(ref_loc)
    return SimpleNamespace(id=f"atm_{i}", raw_text=text, artifact_id=art, project_id="x", value={},
                           source_refs=[SimpleNamespace(locator=loc, filename=filename, artifact_id=art,
                                                        artifact_type="xlsx")])


def test_serving_rows_give_the_training_tags():
    """The served rows of a sheet's table carry the fields the training build
    writes (sheet, table_header), so featurize gives the same tags."""
    from app.core import c3_type_base as tb
    from ml.c3.data import DealExample, context_tags
    atoms = [_atom(0, "City | Hours | Qty", {"sheet": "12E Closet", "row": 1}),
             _atom(1, "Dallas | 4 | 2", {"sheet": "12E Closet", "row": 2})]
    _id, rows, _keys = tb._rows(atoms, {}, documents=[])
    served = {r["atom_id"]: r for r in rows}
    assert served["atm_1"]["sheet"] == "12E Closet"
    assert served["atm_1"]["table_header"] == ["City", "Hours", "Qty"]
    assert "table_header" not in served["atm_0"]
    # Training's row for the same line (build.py: table_id, row index, sheet, header).
    train = dict(_rows()[1], section=served["atm_1"]["section"])
    deal_s = DealExample.from_dict({"deal_id": "x", "atoms": [served["atm_1"]]})
    deal_t = DealExample.from_dict({"deal_id": "x", "atoms": [train]})
    assert context_tags(deal_s.atoms) == context_tags(deal_t.atoms)
    assert context_tags(deal_s.atoms)[0][:3] == ["hdr:city", "hdr:hours", "hdr:qty"]
    # And deal_payload hands the remote server the same fields back.
    back = tb.deal_payload(deal_s)["atoms"][0]
    assert back["sheet"] == "12E Closet" and back["table_header"] == ["City", "Hours", "Qty"]
