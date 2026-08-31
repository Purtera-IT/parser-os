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
