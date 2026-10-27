"""C3_CONTEXT_TAGS serving side: the worker's rows carry the table header and
sheet a training row carries, so the base's hdr:/sheet:/sec: tags
(c3_record.context.context_tags) are the same for a served line and a training
one. Ported from the training branch (30e96c45, tests/test_c3_context_tags.py)."""
from types import SimpleNamespace

from c3_record.context import context_tags


def _training_row():
    # build.py's (d) for a sheet's second row: table_id, row index, sheet, header.
    return {"key": "r", "text": "Dallas | 4 | 2", "entered_at": "2026-05-01T00:00:00Z", "doc_id": "d1",
            "doc_kind": "file", "section": "12E Closet", "atom_id": "atm_r", "order": 1,
            "table_id": "d1:12E Closet", "table_row_index": 2, "sheet": "12E Closet",
            "table_header": ["City", "Hours", "Qty"]}


def _tags(row):
    return context_tags(row.get("section", ""), row.get("sheet", ""), row.get("table_header") or (),
                        row.get("doc_kind", ""))


def test_tags_follow_the_row_fields():
    assert _tags(_training_row()) == ["hdr:city", "hdr:hours", "hdr:qty", "sheet:closet"]
    sow = {"section": "Services Proposal > Project Scope > Project Assumptions", "doc_kind": "sow_draft"}
    assert _tags(sow) == ["sec:project", "sec:assumptions"]
    assert _tags({"section": "Trent Torrence", "doc_kind": "meeting"}) == []


def _atom(i, text, ref_loc, filename="survey.xlsx", art="d1"):
    loc = dict(ref_loc)
    return SimpleNamespace(id=f"atm_{i}", raw_text=text, artifact_id=art, project_id="x", value={},
                           source_refs=[SimpleNamespace(locator=loc, filename=filename, artifact_id=art,
                                                        artifact_type="xlsx")])


def test_serving_rows_give_the_training_tags():
    """The served rows of a sheet's table carry the fields the training build
    writes (sheet, table_header), so the same tags follow."""
    from app.core import c3_type_base as tb
    atoms = [_atom(0, "City | Hours | Qty", {"sheet": "12E Closet", "row": 1}),
             _atom(1, "Dallas | 4 | 2", {"sheet": "12E Closet", "row": 2})]
    _id, rows, _keys = tb._rows(atoms, {}, documents=[])
    served = {r["atom_id"]: r for r in rows}
    assert served["atm_1"]["sheet"] == "12E Closet"
    assert served["atm_1"]["table_header"] == ["City", "Hours", "Qty"]
    assert "table_header" not in served["atm_0"]
    assert "sheet_role" not in served["atm_1"]                       # not in the training views
    train = dict(_training_row(), section=served["atm_1"]["section"], doc_kind=served["atm_1"]["doc_kind"])
    assert _tags(served["atm_1"]) == _tags(train)
    assert _tags(served["atm_1"])[:3] == ["hdr:city", "hdr:hours", "hdr:qty"]
    # And deal_payload hands the remote server the same fields back.
    line = SimpleNamespace(**{**served["atm_1"], "table_header": tuple(served["atm_1"]["table_header"])})
    back = tb.deal_payload(SimpleNamespace(deal_id="x", company="purtera", atoms=[line]))["atoms"][0]
    assert back["sheet"] == "12E Closet" and back["table_header"] == ["City", "Hours", "Qty"]
    assert _tags(back) == _tags(train)


def test_a_line_outside_a_table_carries_no_table_fields():
    from app.core import c3_type_base as tb
    atoms = [SimpleNamespace(id="atm_p", raw_text="Customer provides lift access.", artifact_id="d2",
                             project_id="x", value={}, source_refs=[])]
    _id, rows, _keys = tb._rows(atoms, {}, documents=[])
    assert not {"table_id", "table_row_index", "sheet", "table_header"} & set(rows[0])
