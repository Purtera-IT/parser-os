"""Train/serve input parity on the compile's side: the rows the base reads
carry the order, document kind, table row and sender the training export
writes for the same deal (c3_record.reading, c3_record.table_geometry). All
text is made up."""
from __future__ import annotations

from types import SimpleNamespace as NS

import app.core.c3_type_base as c3
from app.core.typed_atom_classifier import _atom_table_ref
from c3_record import table_geometry
from ml.c3.data import DealExample


class _Type:
    def __init__(self, value):
        self.value = value


def _atom(aid, text, art, *, filename="sheet.xlsx", artifact_type="xlsx", locator=None, value=None):
    ref = NS(locator=dict(locator or {}), artifact_type=_Type(artifact_type), filename=filename, artifact_id=art)
    return NS(id=aid, atom_id=aid, artifact_id=art, project_id="deal-test", raw_text=text,
              value=dict(value or {}), source_refs=[ref], atom_type=_Type("scope_item"))


def _page_table_ref(loc, value=None):
    """atom-labeling.js tableRef(), the string training's table_row is read from."""
    tid = loc.get("block_id") or loc.get("sheet") or None
    if tid is None and loc.get("table_index") is not None:
        tid = f"t{loc['table_index']}"
    if tid is not None:
        row = loc["row"] if loc.get("row") is not None else loc.get("row_index")
        return f"{str(tid)[:24]} r{row}" if row is not None else str(tid)[:24]
    v = value or {}
    return f"t{v['_table_idx']} r{v.get('_row_idx')}" if v.get("_table_idx") is not None else ""


def test_the_parser_table_ref_is_the_page_table_ref():
    for loc, value in (({"sheet": "Sites", "row": 3}, {}), ({"block_id": "p2_tbl1", "row": 0}, {}),
                       ({"table_index": 4, "row_index": 7}, {}), ({"page": 2}, {}),
                       ({}, {"_table_idx": 1, "_row_idx": 5}), ({"sheet": "Rates"}, {})):
        atom = _atom("a", "x", "art", locator=loc, value=value)
        assert _atom_table_ref(atom) == _page_table_ref(loc, value)


def test_a_table_row_line_sends_the_table_row_training_writes():
    rows = [_atom("r1", "Store 12 | Austin | TX", "wb", locator={"sheet": "Sites", "row": 4}),
            _atom("p1", "Please quote the survey.", "wb", locator={"page": 1, "line_start": 2})]
    docs = [{"doc_id": "wb", "kind": "file", "sent_at": "2026-01-01T00:00:00Z", "authored_at": None, "pos": 0}]
    _, lines, _ = c3._rows(rows, {}, docs)
    payload = c3.deal_payload(DealExample.from_dict({"deal_id": "d", "company": "purtera", "atoms": lines}))
    got = {a["atom_id"]: a.get("table_row") for a in payload["atoms"]}
    want = table_geometry(_page_table_ref({"sheet": "Sites", "row": 4}))["table_row"]
    assert got == {"r1": want, "p1": None} and want == 5
    assert all(set(a) - {"table_row"} == set(payload["atoms"][0]) - {"table_row"} for a in payload["atoms"])


def test_a_workbook_reads_one_sheet_after_another():
    atoms = [_atom(f"{sh}{r}", f"{sh} row {r}", "wb", locator={"sheet": sh, "row": r})
             for r in (1, 2) for sh in ("Sites", "Rates")]
    docs = [{"doc_id": "wb", "kind": "file", "sent_at": "2026-01-01T00:00:00Z", "authored_at": None, "pos": 0}]
    _, lines, _ = c3._rows(atoms, {}, docs)
    assert [ln["atom_id"] for ln in sorted(lines, key=lambda ln: ln["order"])] == ["Sites1", "Sites2",
                                                                                     "Rates1", "Rates2"]


def test_a_file_named_as_a_site_list_is_one_without_a_lifecycle_type(monkeypatch):
    monkeypatch.setattr(c3, "_lifecycle", lambda sha: {})
    atom = _atom("a", "x", "f", filename="Store List Q3.xlsx")
    assert c3._file_kind(atom, {"f": "abc"}) == "site_list"
    monkeypatch.setattr(c3, "_lifecycle", lambda sha: {"type": "PURCHASE_ORDER", "role": "purchase order"})
    assert c3._file_kind(atom, {"f": "abc"}) == "purchase_order"


def test_documents_carry_the_lifecycle_role(monkeypatch):
    monkeypatch.setattr(c3, "_lifecycle", lambda sha: {"type": "SOW_DRAFT", "role": "SOW drafted by the customer"}
                        if sha else {})
    atoms = [_atom("a", "Scope.", "f", filename="sow.docx", artifact_type="docx", locator={"paragraph_index": 1})]
    docs = c3.deal_documents(atoms, {"f": {"filename": "sow.docx", "sha256": "abc"}})
    assert docs[0]["role"] == "SOW drafted by the customer" and docs[0]["kind"] == "sow_draft"


def test_the_origin_names_a_sender_from_the_roster():
    """The page passes the deal roster to every document's origin; a carrier
    known only by address then has a name a quoted author can be matched to."""
    prov = {"note.txt": {"carried_by": {"kind": "note", "filename": "", "sender_email": "pat@purtera-it.com",
                                        "sent_at": "2026-01-02T00:00:00Z", "direction": "internal"}}}
    bare = c3._origin(prov["note.txt"], prov)
    named = c3._origin(prov["note.txt"], prov, names={"pat@purtera-it.com": "Pat Example"})
    assert bare.get("sender_name") is None and named["sender_name"] == "Pat Example"
