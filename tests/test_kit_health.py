"""Is this Deal Kit actually this deal's kit?

Deal 010215's kit is a copy of 000131's. Its header still reads OPPTY # 131, it
is dated ten weeks before 010215 existed, its QTY column is empty, and every
summary formula reads #REF! because the line-items table those SUMIFs pointed at
was deleted when the file was repurposed.

The parser reported all of that honestly. It is a defect in the DOCUMENT -- and
the kind worth telling a PM: small, specific, fixable in minutes. A kit that
silently belongs to another deal is how one deal's pricing walks into another's
quote.
"""
from app.core.document_lifecycle.kit_health import check_deal_kit

REAL = [
    {"atom_type": "deal_metadata", "text": "OPPTY #: 131"},
    {"atom_type": "commercial_total", "text": "Total Deal Revenue: #REF!"},
    {"atom_type": "commercial_total", "text": "Total PMO Revenue: #REF!"},
]


class TestKitNamesAnotherDeal:
    def test_flags_a_kit_copied_from_another_deal(self):
        out = check_deal_kit(atoms=REAL, deal_number="010215")
        kinds = [f["kind"] for f in out["findings"]]
        assert "kit_names_another_deal" in kinds
        assert out["claimed_deal"] == "131"

    def test_leading_zeros_do_not_make_a_false_positive(self):
        # "010215" in the kit and 010215 the deal are the same deal.
        out = check_deal_kit(atoms=[{"atom_type": "deal_metadata", "text": "OPPTY #: 010215"}], deal_number="10215")
        assert not any(f["kind"] == "kit_names_another_deal" for f in out["findings"])

    def test_says_nothing_when_the_kit_names_no_deal(self):
        # Absent is not wrong. Only a CONTRADICTION is worth raising.
        out = check_deal_kit(atoms=[{"atom_type": "scope_item", "text": "Region: USA"}], deal_number="010215")
        assert not any(f["kind"] == "kit_names_another_deal" for f in out["findings"])

    def test_says_nothing_when_the_deal_number_is_unknown(self):
        out = check_deal_kit(atoms=REAL, deal_number=None)
        assert not any(f["kind"] == "kit_names_another_deal" for f in out["findings"])


class TestBrokenFormulas:
    def test_flags_excel_errors_in_the_totals(self):
        out = check_deal_kit(atoms=REAL, deal_number="010215")
        broken = next(f for f in out["findings"] if f["kind"] == "broken_formulas")
        assert "Total Deal Revenue" in broken["detail"]

    def test_catches_the_other_excel_errors_too(self):
        for err in ("#DIV/0!", "#VALUE!", "#N/A", "#NAME?"):
            out = check_deal_kit(atoms=[{"atom_type": "commercial_total", "text": f"Margin: {err}"}], deal_number="1")
            assert any(f["kind"] == "broken_formulas" for f in out["findings"]), err

    def test_a_healthy_kit_reports_nothing(self):
        out = check_deal_kit(
            atoms=[{"atom_type": "deal_metadata", "text": "OPPTY #: 010215"},
                   {"atom_type": "commercial_total", "text": "Total Deal Revenue: 3050"}],
            deal_number="010215",
        )
        assert out["healthy"] is True
        assert out["findings"] == []

    def test_every_finding_carries_a_fix(self):
        # A defect a PM cannot act on is noise they learn to ignore.
        out = check_deal_kit(atoms=REAL, deal_number="010215")
        assert all(f.get("fix") for f in out["findings"])
