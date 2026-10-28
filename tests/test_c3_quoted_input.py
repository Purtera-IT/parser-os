"""C3 quoted-history and copy inputs, serving side: the worker's rows carry
quoted_depth (c3_record.context.quoted_depth) and copy_of (value.duplicate_of),
as the training views write them, so the base's quoted:1 / quoted:2+ tags and
its copy:repeat input read a served line as a training one. Ported from the
training branch (claude/c3-quoted-input, tests/test_c3_quoted_input.py)."""
from types import SimpleNamespace

from c3_record.context import context_tags, quoted_depth, quoted_tags


def _atom(i, text, loc, value=None, authority=None):
    return SimpleNamespace(id=f"atm_{i}", raw_text=text, artifact_id="mail1", project_id="x",
                           value=dict(value or {}), authority_class=authority,
                           source_refs=[SimpleNamespace(locator=dict(loc), filename="thread.eml",
                                                        artifact_id="mail1", artifact_type="email")])


def test_quoted_depth_rule_and_tags():
    assert quoted_depth(False, 2) == 0
    assert quoted_depth(True, 0) == 1 and quoted_depth(True, None) == 1
    assert quoted_depth(True, 2) == 2
    assert quoted_depth(None, 3, SimpleNamespace(value="quoted_old_email")) == 3
    assert quoted_tags(0) == [] and quoted_tags(1) == ["quoted:1"] and quoted_tags(4) == ["quoted:1"]
    assert context_tags("", "", (), "email", 2) == ["quoted:1"]


def test_serving_rows_carry_quoted_depth_and_copy_of():
    from app.core import c3_type_base as tb
    atoms = [_atom(0, "We need two techs on site.", {"message_index": 0, "quoted": False}),
             _atom(1, "Can you quote the lift?", {"message_index": 1, "quoted": True}),
             _atom(2, "Original ask from March.", {"message_index": 3},
                   authority=SimpleNamespace(value="quoted_old_email")),
             _atom(3, "> two technicians onsite", {"message_index": 0, "quoted": True},
                   value={"duplicate_of": {"atom_id": "atm_0", "stage": "semantic_dedup"}})]
    _id, rows, _keys = tb._rows(atoms, {}, documents=[])
    served = {r["atom_id"]: r for r in rows}
    assert "quoted_depth" not in served["atm_0"] and "copy_of" not in served["atm_0"]
    assert [served[f"atm_{i}"].get("quoted_depth") for i in range(4)] == [None, 1, 3, 1]
    assert served["atm_3"]["copy_of"] == "atm_0"
    tags = {k: context_tags(r.get("section", ""), r.get("sheet", ""), r.get("table_header") or (),
                            r.get("doc_kind", ""), r.get("quoted_depth", 0)) for k, r in served.items()}
    assert tags["atm_0"] == [] and tags["atm_1"] == ["quoted:1"] and tags["atm_2"] == ["quoted:1"]
    # deal_payload hands the remote server the same fields back.
    deal = SimpleNamespace(deal_id="x", company="purtera", atoms=[SimpleNamespace(**{
        "key": r["key"], "text": r["text"], "entered_at": r["entered_at"], "doc_id": r["doc_id"],
        "doc_kind": r["doc_kind"], "section": r["section"], "order": r["order"],
        "speaker_role": r["speaker_role"], "speaker_side": r["speaker_side"], "atom_id": r["atom_id"],
        "quoted_depth": r.get("quoted_depth", 0), "copy_of": r.get("copy_of", "")}) for r in rows])
    back = {r["atom_id"]: r for r in tb.deal_payload(deal)["atoms"]}
    assert back["atm_2"]["quoted_depth"] == 3 and back["atm_3"]["copy_of"] == "atm_0"
    assert "quoted_depth" not in back["atm_0"] and "copy_of" not in back["atm_0"]
