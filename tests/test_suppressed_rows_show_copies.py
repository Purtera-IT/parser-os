# -*- coding: utf-8 -*-
"""A folded line stays visible in the labeller as a copy beside its original.

Dedup still runs for the product; the envelope's ``suppressed`` rows carry
what the labelling walk needs to show each folded line where it sits:

* ``locator`` -- the atom's own ``source_refs[0].locator``, the same object a
  kept atom carries in ``env.atoms``;
* ``order`` -- how many atoms of ``env.atoms`` precede it in reading order;
* ``label_key`` -- the bare key the walk would give the atom if it were kept
  (Platform-infra ``buildWalk``: page = ``loc.page ?? loc.sheet``, filename =
  the envelope document's). A docx table row has ``page: None,
  table_index: 0``; the row used to key it under page "0", the walk under no
  page, so a verdict on the copy could never meet the original's;
* ``same_key_as_survivor`` -- the same words in the same file on the same page.

Synthetic shapes only; they mirror a two-version SOW docx whose table rows
collapse onto their twins in the same file.
"""
from __future__ import annotations

import pytest

from app.core import orbitbrief_envelope as env
from app.core.label_key import label_key


class _Ref:
    def __init__(self, filename: str, locator: dict) -> None:
        self.filename = filename
        self.locator = locator
        self.extraction_method = "docx_table_row"


class _Atom:
    def __init__(self, aid: str, text: str, artifact: str, filename: str, locator: dict,
                 stage: str | None = None, value: dict | None = None) -> None:
        self.id = aid
        self.artifact_id = artifact
        self.atom_type = "scope_item"
        self.raw_text = text
        self.text = text
        self.value = dict(value or {})
        self.entity_keys = []
        self.receipts = []
        self.source_refs = [_Ref(filename, locator)]
        self.review_flags = [f"suppressed:{stage}"] if stage else []


class _Result:
    def __init__(self, dropped, atoms=()) -> None:
        self.project_id = "deal_x"
        self.suppressed_atoms = list(dropped)
        self.atoms = list(atoms)


SOW = "art_sow_v2"
SOW_NAME = "Example SOW v2.docx"
DOCS = [{"artifact_id": SOW, "filename": SOW_NAME, "authored_at": "2026-08-28"}]


def _table_loc(line: int, row: int) -> dict:
    # A docx table row: no page, the table's index, the row within it.
    return {"page": None, "table_index": 0, "row": row, "block_index": 4,
            "line_start": line, "line_end": line, "section_path": ["Contacts"]}


@pytest.fixture(autouse=True)
def _ledger_on(monkeypatch):
    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")
    monkeypatch.delenv("SOWSMITH_SUPPRESSED_MAX_PER_DOC", raising=False)
    monkeypatch.delenv("SOWSMITH_SUPPRESSED_MAX", raising=False)


def _pair():
    text = "Pat Example | Project Manager | pat@customer.example"
    kept = [
        _Atom("atm_head", "Contacts", SOW, SOW_NAME, {"page": None, "block_index": 3, "line_start": 8}),
        _Atom("atm_row1", text, SOW, SOW_NAME, _table_loc(9, 1)),
        _Atom("atm_tail", "Assumptions", SOW, SOW_NAME, {"page": None, "block_index": 5, "line_start": 20}),
    ]
    twin = _Atom("atm_row1_twin", text, SOW, SOW_NAME, _table_loc(11, 2), stage="duplicate_atom_collapse")
    return kept, twin


def test_a_folded_row_carries_its_own_locator_verbatim():
    kept, twin = _pair()
    (row,) = env._suppressed_for_review(_Result([twin], kept), kept, DOCS)
    assert row["locator"] == twin.source_refs[0].locator
    assert row["locator"] is not twin.source_refs[0].locator


def test_the_row_is_keyed_as_the_walk_keys_a_kept_atom():
    kept, twin = _pair()
    (row,) = env._suppressed_for_review(_Result([twin], kept), kept, DOCS)
    # buildWalk: page = loc.page ?? loc.sheet -> null; never table_index.
    assert row["label_key"] == label_key("deal_x", SOW_NAME, None, twin.raw_text)
    assert row["label_key"] != label_key("deal_x", SOW_NAME, "0", twin.raw_text)


def test_a_sheet_is_the_page_and_an_unknown_document_keys_by_its_artifact_id():
    cell = _Atom("atm_c", "Rack 14 | 42U", "art_q", "q.xlsx", {"sheet": "Pricing", "row_index": 7},
                 stage="semantic_dedup")
    (row,) = env._suppressed_for_review(_Result([cell]), [], DOCS)
    # buildWalk: ``doc ? doc.filename : a.artifact_id``.
    assert row["label_key"] == label_key("deal_x", "art_q", "Pricing", "Rack 14 | 42U")


def test_same_words_same_file_same_page_is_flagged():
    kept, twin = _pair()
    (row,) = env._suppressed_for_review(_Result([twin], kept), kept, DOCS)
    assert row["survivor"]["id"] == "atm_row1"
    assert row["survivor"]["in_atoms"] is True
    assert row["same_key_as_survivor"] is True


def test_a_survivor_in_another_file_is_not_the_same_key():
    kept, twin = _pair()
    other = _Atom("atm_v1", twin.raw_text, "art_sow_v1", "Example SOW v1.docx", _table_loc(9, 1))
    docs = DOCS + [{"artifact_id": "art_sow_v1", "filename": "Example SOW v1.docx"}]
    kept = [k for k in kept if k.id != "atm_row1"] + [other]
    (row,) = env._suppressed_for_review(_Result([twin], kept), kept, docs)
    assert row["survivor"]["id"] == "atm_v1"
    assert row["same_key_as_survivor"] is False


def test_order_places_the_copy_between_the_kept_atoms_around_it():
    kept, twin = _pair()
    (row,) = env._suppressed_for_review(_Result([twin], kept), kept, DOCS)
    in_order = [a.id for a in env._in_reading_order(kept, DOCS)]
    # Line 11 sits after the original row (line 9) and before line 20.
    assert in_order[row["order"] - 1] == "atm_row1"
    assert in_order[row["order"]] == "atm_tail"


def test_order_is_null_without_the_document_list():
    kept, twin = _pair()
    (row,) = env._suppressed_for_review(_Result([twin], kept), kept)
    assert row["order"] is None


def test_a_survivor_outside_env_atoms_resolves_to_one_inside():
    kept, twin = _pair()
    original = kept[1]
    copy = _Atom("atm_copy", twin.raw_text, "art_mail", "mail.eml", {"line_start": 3},
                 value={"duplicate_of": {"atom_id": original.id, "artifact_id": SOW}})
    twin.value["_survivor"] = {"atom_id": copy.id}
    (row,) = env._suppressed_for_review(_Result([twin], kept + [copy]), kept, DOCS)
    assert row["survivor"]["id"] == "atm_copy"
    assert row["survivor"]["in_atoms"] is False
    assert row["survivor"]["copy_of"] == original.id


def test_chrome_rows_carry_locator_kind_and_drop_reason(monkeypatch):
    import app.core.email_chrome as chrome

    monkeypatch.setattr(chrome, "atom_chrome_reason", lambda a: "signature" if a.raw_text.startswith("--") else None)
    sig = _Atom("atm_sig", "-- Pat, 555-0100", "art_mail", "mail.eml", {"line_start": 30, "message_index": 0},
                stage="chrome")
    (row,) = env._suppressed_chrome_for_review(_Result([sig]))
    assert row["locator"] == {"line_start": 30, "message_index": 0}
    assert row["kind"] == "drop"
    assert "drop_reason" in row
    assert row["reason"] == "signature"


def test_the_cap_holds_a_real_deal_and_reads_the_environment_now(monkeypatch):
    assert env._suppressed_cap() == env._SUPPRESSED_MAX_DEFAULT == 5000
    dropped = [_Atom(f"a{i}", f"line {i}", SOW, SOW_NAME, {"line_start": i}, stage="semantic_dedup")
               for i in range(400)]
    result = _Result(dropped)
    assert len(env._suppressed_for_review(result, [], DOCS)) == 400
    assert env._suppressed_truncated(result)["count"] == 0
    monkeypatch.setenv("SOWSMITH_SUPPRESSED_MAX_PER_DOC", "150")
    assert len(env._suppressed_for_review(result, [], DOCS)) == 150
    trunc = env._suppressed_truncated(result)
    assert trunc == {"count": 250, "per_document_cap": 150, "by_artifact": {SOW: 250}}


def test_the_rows_stay_off_by_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", raising=False)
    kept, twin = _pair()
    assert env._suppressed_for_review(_Result([twin], kept), kept, DOCS) == []


def test_order_is_re_anchored_when_env_atoms_gains_rows_after_the_ledger():
    """Held chatter is merged into ``env.atoms`` after the ledger is built; a
    row's ``order`` must index the final list, not the one before the merge."""
    kept, twin = _pair()
    result = _Result([twin], kept)
    rows = env._suppressed_for_review(result, kept, DOCS)
    # Two rows that arrive later, both ahead of the copy.
    held = [
        _Atom("atm_hi", "Hi all", SOW, SOW_NAME, {"page": None, "block_index": 1, "line_start": 1}),
        _Atom("atm_ok", "Thanks", SOW, SOW_NAME, {"page": None, "block_index": 2, "line_start": 2}),
    ]
    final = env._in_reading_order(kept + held, DOCS)
    envelope = {"suppressed": rows, "atoms": [{"id": a.id} for a in final]}
    env._place_suppressed_rows(envelope, result, kept + held, DOCS)
    (row,) = envelope["suppressed"]
    ids = [a.id for a in final]
    assert ids[row["order"] - 1] == "atm_row1"
    assert ids[row["order"]] == "atm_tail"


def test_a_fractional_heading_slot_survives_in_locator_and_order():
    """A PDF heading takes the slot before its first line (block_index 40.5);
    the copy's locator keeps the float and its order still lands beside it."""
    kept = [
        _Atom("atm_a", "Line forty", SOW, SOW_NAME, {"page": 3, "block_index": 40, "line_start": 40}),
        _Atom("atm_b", "Line forty-one", SOW, SOW_NAME, {"page": 3, "block_index": 41, "line_start": 41}),
    ]
    head = _Atom("atm_h", "Site Survey", SOW, SOW_NAME, {"page": 3, "block_index": 40.5, "line_start": 40.5},
                 stage="semantic_dedup")
    (row,) = env._suppressed_for_review(_Result([head], kept), kept, DOCS)
    assert row["locator"]["block_index"] == 40.5 and isinstance(row["locator"]["block_index"], float)
    assert row["order"] == 1
    assert row["label_key"] == label_key("deal_x", SOW_NAME, 3, "Site Survey")
