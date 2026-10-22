"""Train/serve input parity: the training export and the parser's compile give
the base the same lines, in the same order, with the same document kind,
arrival and table row (c3_record.reading, c3_record.table_geometry)."""
from c3_record import (build_rows, document_kind, document_times, file_kind, page_group, page_position,
                       reading_order, table_geometry)
from ml.c3.data import DealExample, featurize
from ml.c3.export import from_page_export
from ml.c3.schema import load_schema

PM = "reviewer@example.com"


def page_atom(key, art, text, position, **kw):
    return {"labelKey": key, "atomId": f"at-{key}", "artifactId": art, "text": text, "docType": "",
            "section": [], "position": position, "page": None, "tableRef": "", **kw}


def page(atoms, documents):
    return {"dealId": "uuid", "compileId": "c", "myLinks": [], "judgments": [], "labels": [],
            "documents": documents, "atoms": atoms}


MAIL = {"artifactId": "mail", "kind": "email", "authoredAt": "2026-02-01T00:00:00Z",
        "origin": {"sent_at": "2026-02-01T00:00:00Z", "via": "email"}}


def test_suppressed_and_chrome_lines_are_not_model_input():
    atoms = [page_atom("a", "mail", "Kept.", 0),
             page_atom("b", "mail", "Thanks, Pat", 1, suppressed={"kind": "chrome"}),
             page_atom("c", "mail", "Kept again.", 2, copyOf="a", suppressed={"kind": "copy"}),
             page_atom("d", "mail", "Last.", 3)]
    doc = from_page_export(page(atoms, [MAIL]), "010999")
    assert [a["label_key"] for a in doc["atoms"]] == ["a", "d"]
    assert [a["order"] for a in doc["atoms"]] == [0, 1]


def test_only_atom_rows_are_lines_by_default():
    """Missed rows, missed_history and Dropped-card copies are never sent by
    the compile: by default they add no line, so every kept line's rank
    feature (index / count) matches serving."""
    atoms = [{"label_key": k, "atom_id": f"at-{k}", "text": f"Line {k}.", "entered_at": "2026-01-01T00:00:00Z",
              "order": i} for i, k in enumerate("ab")]
    blob = {"labels": [{"label_key": "m", "labeler": PM, "origin": "labeler", "label_type": "task",
                        "text": "Missed line.", "reads_set": {}}],
            "missed_history": [{"labeler": PM, "origin": "labeler", "text": "Old miss.",
                                "reads_set": {"miss_cause": "table_cell"}}],
            "judgments": [{"head": "suppression", "target_key": "sup:a@line_start=4", "labeler": PM,
                           "verdict": "correctly_dropped", "text": "Line a.",
                           "target": {"isFold": True, "survivor": "Line a.", "text": "Line a."}}]}
    served = DealExample.from_training_blob(blob, atoms, deal_id="d")
    assert [a.key for a in served.atoms] == ["a", "b"]
    assert served.source_supervision["missed_history"] and served.source_supervision["labels"]
    bare = DealExample.from_training_blob({}, atoms, deal_id="d")
    schema = load_schema()
    got, want = featurize(served, schema).inputs(), featurize(bare, schema).inputs()
    assert all(repr(got[k]) == repr(want[k]) for k in ("texts", "times", "doc_kind", "same_doc", "geometry"))
    old = DealExample.from_training_blob(blob, atoms, deal_id="d", serve_lines_only=False)
    assert len(old.atoms) > len(served.atoms)


def test_a_workbook_reads_sheet_by_sheet_on_both_sides():
    """The page interleaves sheets by row; the record reads each sheet whole,
    from the page's own positions or from the parser's locators alike."""
    locs = [{"sheet": "Sites", "row": 1}, {"sheet": "Rates", "row": 1}, {"sheet": "Sites", "row": 2},
            {"sheet": "Rates", "row": 2}]
    docs = [{"doc_id": "x", "authored_at": None, "pos": 0}]
    serve = reading_order([{"doc_id": "x", "within": page_position(loc, {}, "", email=False),
                            "group": page_group(loc)} for loc in locs], docs)
    train = reading_order([{"doc_id": "x", "within": float(i), "group": loc["sheet"]}
                           for i, loc in enumerate(locs)], docs)
    assert serve == train == [0, 2, 1, 3]


def test_an_email_file_reads_its_deepest_quote_first():
    lines = [({"line_start": 1}, {"email_thread": {"message": {"index": 0}}}),
             ({"line_start": 9, "message_index": 2}, {}),
             ({"line_start": 5}, {"email_thread": {"message": {"index": 1}}})]
    keys = [page_position(loc, v, "", email=True) for loc, v in lines]
    assert sorted(range(3), key=lambda i: keys[i]) == [1, 2, 0]


def test_documents_read_by_authored_time_then_manifest_place():
    docs = [{"doc_id": "late", "authored_at": "2026-03-01", "pos": 0},
            {"doc_id": "undated", "authored_at": None, "pos": 1},
            {"doc_id": "early", "authored_at": "2026-01-01", "pos": 2}]
    lines = [{"doc_id": d, "within": 0.0, "group": None} for d in ("late", "undated", "early")]
    assert reading_order(lines, docs) == [1, 2, 0]


def test_a_missing_arrival_fills_from_the_document_before_it_in_reading_order():
    docs = [{"doc_id": "mail2", "kind": "email", "sent_at": "2026-05-01", "authored_at": "2026-05-01", "pos": 0},
            {"doc_id": "xlsx", "kind": "file", "authored_at": "2026-02-01", "pos": 1},
            {"doc_id": "mail1", "kind": "email", "sent_at": "2026-01-01", "authored_at": "2026-01-01", "pos": 2}]
    assert document_times(docs)["xlsx"] == "2026-01-01"
    # Order of the list does not matter, only the reading order.
    assert document_times(list(reversed(docs)))["xlsx"] == "2026-01-01"


def test_one_document_kind_rule():
    assert file_kind("Site List Project Bastion.xlsx") == "site_list"
    assert file_kind("x.xlsx", "PURCHASE_ORDER") == "purchase_order"
    assert file_kind("x.xlsx", "SOMETHING_NEW") == "file"
    assert document_kind("email", "PURCHASE_ORDER", "a.eml") == "email"
    assert document_kind("file", "", "010447 Deal Kit.xlsx") == "deal_kit"
    doc = from_page_export(page([page_atom("a", "f", "Row.", 0)],
                                [MAIL, {"artifactId": "f", "kind": "file", "docType": "",
                                        "filename": "Store List.xlsx"}]), "010999")
    assert doc["atoms"][0]["doc_kind"] == "site_list"


def test_table_row_from_the_page_and_the_parser_is_the_same_nonzero_row():
    # atom-labeling.js tableRef(): "<block_id|sheet|t<table_index>> r<row>".
    loc = {"sheet": "Sites", "row": 4}
    page_ref = f"{loc['sheet'][:24]} r{loc['row']}"
    doc = from_page_export(page([page_atom("a", "mail", "Row.", 0, tableRef=page_ref)], [MAIL]), "010999")
    row = doc["atoms"][0]
    assert row["table_row"] == 5 == table_geometry("Sites r4.0")["table_row"]
    built = build_rows([{"key": "a", "doc_id": "mail", "text": "Row.", **table_geometry(page_ref)}], [])
    assert built[0]["table_row"] == 5
    assert table_geometry("Sites") == {} and table_geometry("") == {}


def test_one_event_kind_rule():
    from c3_record import event_kind

    assert event_kind("txt", "010001-hs-email-1.eml") == "email"
    assert event_kind("docx", "notes.docx", [{"kind": "hubspot_note_meta"}]) == "note"
    assert event_kind("txt", "010001-hs-call-7.txt") == "meeting"
    assert event_kind("transcript", "call.vtt") == "meeting"
    assert event_kind("xlsx", "Store List.xlsx") == ""
