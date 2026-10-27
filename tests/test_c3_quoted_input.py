"""quoted_depth: a line in quoted reply history is an input fact (quoted:1, one tag for every depth),
under the C3_CONTEXT_TAGS input contract; copy_of (the parser's duplicate_of) feeds copy_flags."""
from dataclasses import replace
from types import SimpleNamespace


def _rows():
    return [{"key": "a", "text": "We need two techs on site.", "entered_at": "2026-05-01T00:00:00Z",
             "doc_id": "d1", "doc_kind": "email", "atom_id": "atm_a"},
            {"key": "b", "text": "Can you quote the lift?", "entered_at": "2026-05-02T00:00:00Z",
             "doc_id": "d2", "doc_kind": "email", "atom_id": "atm_b", "quoted_depth": 1},
            {"key": "c", "text": "Original ask from March.", "entered_at": "2026-05-02T00:00:00Z",
             "doc_id": "d2", "doc_kind": "email", "atom_id": "atm_c", "quoted_depth": 3},
            # A semantic copy the parser named by atom id (duplicate_of), different words.
            {"key": "row_d", "text": "> two technicians onsite", "entered_at": "2026-05-03T00:00:00Z",
             "doc_id": "d3", "doc_kind": "email", "atom_id": "atm_d", "copy_of": "atm_a"}]


def test_quoted_depth_rule_and_tags():
    from c3_record.context import quoted_depth, quoted_tags
    assert quoted_depth(False, 2) == 0
    assert quoted_depth(True, 0) == 1 and quoted_depth(True, None) == 1 and quoted_depth("true", 1) == 1
    assert quoted_depth(True, 2) == 2
    assert quoted_depth(None, 3, SimpleNamespace(value="quoted_old_email")) == 3
    assert quoted_depth(None, None, "AuthorityClass.quoted_old_email") == 1
    assert quoted_tags(0) == [] and quoted_tags(1) == ["quoted:1"] and quoted_tags(5) == ["quoted:1"]


def test_quoted_tag_in_context_tags():
    from ml.c3.data import DealExample, context_tags
    deal = DealExample.from_dict({"deal_id": "x", "atoms": _rows()})
    by = dict(zip([a.key for a in deal.atoms], context_tags(deal.atoms)))
    assert by == {"a": [], "b": ["quoted:1"], "c": ["quoted:1"], "row_d": []}


def test_copy_of_names_an_atom_id():
    from ml.c3.data import DealExample, copy_flags
    deal = DealExample.from_dict({"deal_id": "x", "atoms": _rows()})
    assert copy_flags(deal.atoms) == ["", "", "", "repeat"]


def test_flag_off_is_byte_identical_and_quoted_exempt_from_dropout():
    import torch
    from ml.c3.data import DealExample, featurize
    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.schema import load_schema
    schema = load_schema()
    on_rows = featurize(DealExample.from_dict({"deal_id": "x", "atoms": _rows()}), schema)
    plain = featurize(DealExample.from_dict({"deal_id": "x", "atoms": [
        {k: v for k, v in r.items() if k != "quoted_depth"} for r in _rows()]}), schema)
    assert on_rows.targets == plain.targets
    torch.manual_seed(0)
    model = C3Model(schema, replace(SMALL, tag_dropout=0.0)).eval()
    with torch.no_grad():
        a = model.encoder.embed_atoms(on_rows.inputs())
        c = model.encoder.embed_atoms(plain.inputs())
    assert torch.equal(a, c)                                     # off: never read
    model.cfg.context_tags = True
    with torch.no_grad():
        on = model.encoder.embed_atoms(on_rows.inputs())
    assert torch.equal(a[0], on[0]) and torch.equal(a[3], on[3])
    assert not torch.equal(a[1], on[1]) and not torch.equal(a[2], on[2]) and not torch.equal(on[1], on[2])
    # Full tag dropout keeps the quoted tag.
    drop = C3Model(schema, replace(SMALL, tag_dropout=1.0, context_tags=True)).train()
    drop.load_state_dict(model.state_dict())
    torch.manual_seed(1)
    with torch.no_grad():
        x = drop.encoder.embed_atoms(on_rows.inputs())
        y = drop.encoder.embed_atoms(dict(on_rows.inputs(), ctx_tags=[[] for _ in range(len(on_rows))]))
    assert not torch.equal(x[1], y[1]) and torch.equal(x[0], y[0])


def test_atom_digest_unchanged_without_quoted():
    from ml.c3.data import DealExample
    from ml.c3.tasks import atom_input_digest
    rows = [{k: v for k, v in r.items() if k != "quoted_depth"} for r in _rows()]
    a = DealExample.from_dict({"deal_id": "x", "atoms": rows}).atoms[0]
    from dataclasses import asdict
    assert "quoted_depth" in asdict(a)
    assert atom_input_digest(a) == atom_input_digest(replace(a, quoted_depth=0))
    assert atom_input_digest(a) != atom_input_digest(replace(a, quoted_depth=1))


def _atom(i, text, loc, value=None, authority=None):
    return SimpleNamespace(id=f"atm_{i}", raw_text=text, artifact_id="mail1", project_id="x",
                           value=dict(value or {}), authority_class=authority,
                           source_refs=[SimpleNamespace(locator=dict(loc), filename="thread.eml",
                                                        artifact_id="mail1", artifact_type="email")])


def test_serving_rows_give_the_training_tags():
    from app.core import c3_type_base as tb
    from ml.c3.data import DealExample, context_tags, copy_flags
    atoms = [_atom(0, "We need two techs on site.", {"message_index": 0, "quoted": False}),
             _atom(1, "Can you quote the lift?", {"message_index": 1, "quoted": True}),
             _atom(2, "Original ask from March.", {"message_index": 3},
                   authority=SimpleNamespace(value="quoted_old_email")),
             _atom(3, "> two technicians onsite", {"message_index": 0, "quoted": True},
                   value={"duplicate_of": {"atom_id": "atm_0", "stage": "semantic_dedup"}})]
    _id, rows, _keys = tb._rows(atoms, {}, documents=[])
    served = {r["atom_id"]: r for r in rows}
    assert "quoted_depth" not in served["atm_0"] and "copy_of" not in served["atm_0"]
    assert served["atm_1"]["quoted_depth"] == 1 and served["atm_2"]["quoted_depth"] == 3
    assert served["atm_3"]["copy_of"] == "atm_0"
    deal_s = DealExample.from_dict({"deal_id": "x", "atoms": rows})
    tags_s = dict(zip([a.atom_id for a in deal_s.atoms], context_tags(deal_s.atoms)))
    copy_s = dict(zip([a.atom_id for a in deal_s.atoms], copy_flags(deal_s.atoms)))
    # Training's rows for the same lines (the views' quoted_depth / copy_of fields).
    train = [dict(served[f"atm_{i}"]) for i in range(4)]
    for t in train:
        t.pop("quoted_depth", None), t.pop("copy_of", None)
    train[1]["quoted_depth"], train[2]["quoted_depth"], train[3]["quoted_depth"] = 1, 3, 1
    train[3]["copy_of"] = "atm_0"
    deal_t = DealExample.from_dict({"deal_id": "x", "atoms": train})
    assert tags_s == dict(zip([a.atom_id for a in deal_t.atoms], context_tags(deal_t.atoms)))
    assert copy_s == dict(zip([a.atom_id for a in deal_t.atoms], copy_flags(deal_t.atoms)))
    assert tags_s["atm_1"] == ["quoted:1"] and tags_s["atm_2"] == ["quoted:1"] and copy_s["atm_3"] == "repeat"
    # deal_payload hands the remote server the same fields back.
    back = {r["atom_id"]: r for r in tb.deal_payload(deal_s)["atoms"]}
    assert back["atm_2"]["quoted_depth"] == 3 and back["atm_3"]["copy_of"] == "atm_0"
    assert "quoted_depth" not in back["atm_0"] and "copy_of" not in back["atm_0"]
