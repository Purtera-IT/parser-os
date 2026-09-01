"""Scope: how wide a document is, and what may be read from it on THIS deal.

Fixtures are the real atoms Sodexo Breakdown.xlsx produced on deal 010215
(compile 2484a1fa, 2026-08-31). Its covering email says "the attached breakdown
for all Sodexo sites"; 010215's own PO is PO-00034965.
"""
from app.core.document_lifecycle import scope as S

BREAKDOWN = [
    {"atom_type": "deal_metadata", "text": "Oppty: PO# 00034150"},
    {"atom_type": "service_line", "text": "Unit tyoe: Technicial Dispatch | Unit Rate: 75 | Units: 2 | Total: 150"},
    {"atom_type": "scope_item", "text": "Total | Total: 4750"},
    {"atom_type": "scope_item", "text": "Oppty: PO# 00033068"},
    {"atom_type": "commercial_total", "text": "Total | Total: 535"},
]
MINE = ["010215", "34965"]


class TestDetectScope:
    def test_a_multi_po_breakdown_is_a_programme_document(self):
        d = S.detect_scope(texts=[a["text"] for a in BREAKDOWN], document_type="COST_BREAKDOWN", this_deal_keys=MINE)
        assert d["scope"] == S.SCOPE_PROGRAM
        assert set(d["foreign_keys"]) == {"33068", "34150"}

    def test_one_passing_mention_is_not_enough(self):
        # Documents name other deals in passing. A single foreign key without
        # scope language must not reclassify the whole document.
        d = S.detect_scope(texts=["see also 010199 for the switch RMA"], this_deal_keys=MINE)
        assert d["scope"] == S.SCOPE_DEAL

    def test_one_mention_plus_scope_language_is_enough(self):
        d = S.detect_scope(
            texts=["see also 010199"], this_deal_keys=MINE,
            delivering_text="breakdown for all Sodexo sites",
        )
        assert d["scope"] == S.SCOPE_PROGRAM

    def test_a_document_naming_only_this_deal_stays_deal_scoped(self):
        d = S.detect_scope(texts=["Oppty: PO# 00034965", "010215 Marion County"], this_deal_keys=MINE)
        assert d["scope"] == S.SCOPE_DEAL
        assert d["foreign_keys"] == []

    def test_customer_level_types_route_to_account(self):
        assert S.detect_scope(texts=["rates"], document_type="RATE_CARD")["scope"] == S.SCOPE_ACCOUNT

    def test_standing_references_route_to_global(self):
        assert S.detect_scope(texts=["step 1"], document_type="INSTALL_INSTRUCTIONS")["scope"] == S.SCOPE_GLOBAL


class TestAggregates:
    def test_a_bare_total_is_an_aggregate(self):
        assert S.is_aggregate({"atom_type": "scope_item", "text": "Total | Total: 4750"})
        assert S.is_aggregate({"atom_type": "commercial_total", "text": "Total | Total: 535"})

    def test_a_line_item_with_a_total_column_is_not(self):
        # "Total: 150" here is this row's extension, not a rollup.
        assert not S.is_aggregate(
            {"atom_type": "service_line", "text": "Unit tyoe: Technicial Dispatch | Unit Rate: 75 | Units: 2 | Total: 150"}
        )

    def test_aggregates_are_withheld_from_a_programme_document(self):
        v, why = S.admit_atom({"atom_type": "commercial_total", "text": "Total | Total: 535"},
                              scope=S.SCOPE_PROGRAM, this_deal_keys=MINE)
        assert v == "context"
        assert "rollup" in why

    def test_aggregates_are_fine_in_a_deal_scoped_document(self):
        # The rule is about documents that span deals, not about totals per se.
        v, _ = S.admit_atom({"atom_type": "commercial_total", "text": "Total | Total: 535"},
                            scope=S.SCOPE_DEAL, this_deal_keys=MINE)
        assert v == "admit"


class TestRowResolution:
    def test_a_row_naming_this_deal_is_admitted(self):
        v, why = S.admit_atom({"atom_type": "service_line", "text": "Oppty: PO# 00034965 | Total: 900"},
                              scope=S.SCOPE_PROGRAM, this_deal_keys=MINE)
        assert v == "admit"

    def test_a_row_naming_another_deal_is_demoted(self):
        v, why = S.admit_atom({"atom_type": "deal_metadata", "text": "Oppty: PO# 00034150"},
                              scope=S.SCOPE_PROGRAM, this_deal_keys=MINE)
        assert v == "context"
        assert "another deal" in why

    def test_an_unattributed_row_in_a_programme_document_is_demoted(self):
        # Silence in a multi-deal document is not neutral -- the same reason a
        # silent zero and a real zero must never look alike.
        v, why = S.admit_atom({"atom_type": "service_line", "text": "Unit Rate: 115 | Units: 2"},
                              scope=S.SCOPE_PROGRAM, this_deal_keys=MINE)
        assert v == "context"

    def test_an_unattributed_row_in_a_deal_document_is_admitted(self):
        v, _ = S.admit_atom({"atom_type": "service_line", "text": "Unit Rate: 115 | Units: 2"},
                            scope=S.SCOPE_DEAL, this_deal_keys=MINE)
        assert v == "admit"

    def test_account_and_global_documents_still_apply(self):
        for sc in (S.SCOPE_ACCOUNT, S.SCOPE_GLOBAL):
            v, _ = S.admit_atom({"atom_type": "service_line", "text": "Unit Rate: 115"},
                                scope=sc, this_deal_keys=MINE)
            assert v == "admit", sc


class TestSummary:
    def test_the_breakdown_contributes_nothing_to_this_deal(self):
        d = S.detect_scope(texts=[a["text"] for a in BREAKDOWN], this_deal_keys=MINE)
        out = S.summarise(BREAKDOWN, scope=d["scope"], this_deal_keys=MINE)
        assert out["scope"] == "program"
        assert out["atoms_admitted"] == 0
        assert out["aggregates_withheld"] == 2
        assert out["reasons"]


class TestAtomAttributeContract:
    """Scope reads atom text off EvidenceAtom, and the field is `raw_text`.

    Reading a non-existent `.text` returned empty strings for every atom, so no
    deal keys were ever found: the Sodexo Breakdown came back scope=deal with no
    foreign keys, while the very same atoms analysed directly gave scope=program
    with 33068 and 34150. Nothing raised -- the feature simply never fired.
    """

    def test_evidence_atom_carries_raw_text_not_text(self):
        from app.core.schemas import EvidenceAtom

        fields = set(EvidenceAtom.model_fields)
        assert "raw_text" in fields
        assert "text" not in fields, "if this ever gains `text`, scope detection must be revisited"

    def test_detection_finds_keys_when_given_raw_text(self):
        # The shape the envelope passes: whatever it reads off each atom.
        atoms = [{"raw_text": "Oppty: PO# 00034150"}, {"raw_text": "Total | Total: 4750"}]
        d = S.detect_scope(
            texts=[a["raw_text"] for a in atoms],
            this_deal_keys=["010215", "34965"],
        )
        assert d["foreign_keys"] == ["34150"]

    def test_detection_finds_nothing_when_handed_empty_strings(self):
        # What the bug looked like from the outside: a clean, confident "deal".
        d = S.detect_scope(texts=["", "", ""], this_deal_keys=["010215", "34965"])
        assert d["scope"] == S.SCOPE_DEAL
        assert d["foreign_keys"] == []


class TestRowNarrowing:
    """Part 2: resolve each row to the deal it actually belongs to.

    A multi-deal workbook is blocked by opportunity, and line items sit under
    their deal's header rather than repeating it. Judging each row on its own
    text calls every line item "unattributed" -- true, and useless.

    Fixtures are the real Sodexo Breakdown rows on deal 010215.
    """

    BREAKDOWN = [
        {"atom_type": "service_line", "text": "Oppty: 10131 | Site: Boston, MA", "locator": {"sheet": "Sheet1", "row": 2}},
        {"atom_type": "service_line", "text": "Unit tyoe: Additional Onsite Hourly | Unit Rate: 115", "locator": {"sheet": "Sheet1", "row": 3}},
        {"atom_type": "service_line", "text": "Oppty: 10082 | Site: Southern Farm", "locator": {"sheet": "Sheet1", "row": 18}},
        {"atom_type": "service_line", "text": "Unit tyoe: Cat6/5 Cable Drops", "locator": {"sheet": "Sheet1", "row": 20}},
        {"atom_type": "scope_item", "text": "Oppty: 10115 | Site: Decom Delta", "locator": {"sheet": "Sheet1", "row": 28}},
        {"atom_type": "scope_item", "text": "Total | Total: 4750", "locator": {"sheet": "Sheet1", "row": 30}},
    ]

    def test_a_bare_oppty_number_is_a_deal_key(self):
        # The workbook writes 010131 as "10131". Requiring the leading zero made
        # every section header invisible, so rows inherited whichever PO number
        # appeared earlier -- attributing Boston's rows to a Decom PO.
        assert S._keys_in("Oppty: 10131 | Site: Boston, MA") == {"10131"}

    def test_a_line_item_inherits_the_header_above_it(self):
        rows = S.narrow_rows(self.BREAKDOWN, scope=S.SCOPE_PROGRAM, this_deal_keys=["010215"])
        by_text = {r["text"][:24]: r for r in rows}
        assert by_text["Unit tyoe: Additional On"]["belongs_to"] == "10131"
        assert by_text["Unit tyoe: Additional On"]["inherited"] is True

    def test_rows_are_rescued_for_the_deal_that_owns_them(self):
        rows = S.narrow_rows(self.BREAKDOWN, scope=S.SCOPE_PROGRAM, this_deal_keys=["010082"])
        assert sum(1 for r in rows if r["verdict"] == "admit") == 2

    def test_a_deal_the_document_does_not_cover_gets_nothing(self):
        # The Breakdown genuinely contains nothing for 010215. "Nothing" is the
        # right answer, not a failure.
        rows = S.narrow_rows(self.BREAKDOWN, scope=S.SCOPE_PROGRAM, this_deal_keys=["010215"])
        assert all(r["verdict"] == "context" for r in rows)

    def test_an_aggregate_is_never_rescued(self):
        # "Total: 4750" sits under 10115's header, so narrowing knows whose it
        # is -- and it still must not become that deal's number from a
        # multi-deal document.
        rows = S.narrow_rows(self.BREAKDOWN, scope=S.SCOPE_PROGRAM, this_deal_keys=["010115"])
        total = next(r for r in rows if r["text"].startswith("Total"))
        assert total["verdict"] == "context"
        assert total["belongs_to"] == "10115"

    def test_a_row_before_any_header_stays_unattributed(self):
        # Inheriting backwards would be inventing a key.
        rows = S.narrow_rows(
            [{"atom_type": "service_line", "text": "Unit Rate: 115", "locator": {"sheet": "S", "row": 1}},
             {"atom_type": "service_line", "text": "Oppty: 10131", "locator": {"sheet": "S", "row": 2}}],
            scope=S.SCOPE_PROGRAM, this_deal_keys=["010131"],
        )
        assert rows[0]["belongs_to"] is None
        assert rows[0]["verdict"] == "context"

    def test_a_deal_scoped_document_is_untouched(self):
        rows = S.narrow_rows(self.BREAKDOWN, scope=S.SCOPE_DEAL, this_deal_keys=["010215"])
        assert all(r["verdict"] == "admit" for r in rows)
