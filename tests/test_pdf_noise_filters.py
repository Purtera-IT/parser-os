"""Regression tests for PDF parser noise filters.

Locks in the page-footer / fragment / form-field / fused-table-row
detectors added in Week 3 (PRODUCTION_GAPS.md P1.2, P1.3, P1.4, P1.7).
Each test is a real example drawn from the STRESS_* corpus.
"""
from __future__ import annotations

from app.parsers.orbitbrief_pdf import (
    _looks_like_form_field,
    _looks_like_fragment,
    _looks_like_fused_table_row,
    _looks_like_page_footer,
)


# ─── Page footer detection (P1.3) ───
class TestPageFooter:
    def test_natomas_footer(self) -> None:
        assert _looks_like_page_footer(
            "RFP 25-107 Wireless Equipment November 20, 2024 "
            "Technology Services Department Page 17 of 25"
        )

    def test_octa_footer(self) -> None:
        assert _looks_like_page_footer("RFP 4-2293 Page 5 of 30")

    def test_bare_page_pattern(self) -> None:
        assert _looks_like_page_footer("Page 12 of 30")

    def test_long_paragraph_not_footer(self) -> None:
        # Real Q&A content with "page" mentioned shouldn't trigger
        long_text = (
            "Q15. What page references the storage retention requirement? "
            "A15. The retention requirement is on Page 4 of the technical "
            "specifications. We expect 30-day retention as a baseline."
        )
        assert not _looks_like_page_footer(long_text)


# ─── Fragment / bullet-noise detection (P1.4) ───
class TestFragment:
    def test_natomas_cost_proposal(self) -> None:
        assert _looks_like_fragment("Cost Proposal")

    def test_natomas_addendums(self) -> None:
        assert _looks_like_fragment("Addendums")

    def test_natomas_project_description(self) -> None:
        assert _looks_like_fragment("Project Description")

    def test_natomas_equipment_service_installed(self) -> None:
        assert _looks_like_fragment("Equipment/Service Installed")

    def test_real_install_scope_not_fragment(self) -> None:
        # Has modal verb "shall" — must not be flagged
        assert not _looks_like_fragment(
            "Vendor shall provide all conduits"
        )

    def test_real_constraint_with_digits_not_fragment(self) -> None:
        # Has digits — must not be flagged
        assert not _looks_like_fragment("100 Mbps wireless")

    def test_real_device_keyword_not_fragment(self) -> None:
        # Has "camera" device hint — must not be flagged
        assert not _looks_like_fragment("Provide IP camera")

    def test_long_paragraph_not_fragment(self) -> None:
        assert not _looks_like_fragment(
            "Q1. I assume the parking garage is the first project. "
            "A1. Yes, Perry Street is the first project."
        )


# ─── Fused table row detection (P1.7) ───
class TestFusedTableRow:
    def test_natomas_fused_dna_skus(self) -> None:
        # The "AIR-DNA-E: AIR-DNA-NWSTACK-E | ... | 500: 500" pattern
        row = {
            "AIR-DNA-E": "AIR-DNA-NWSTACK-E",
            "Wireless Cisco DNA On-Prem Essential, Term Lic": (
                "Wireless DNA Perpetual Network Stack - Essentials"
            ),
            "500": "500",
        }
        assert _looks_like_fused_table_row(row)

    def test_natomas_real_bom_row_not_fused(self) -> None:
        row = {
            "Part Number": "CW9166I-B",
            "Description": "Catalyst 9166I AP (W6E, tri-band 4x4, XOR) w/Reg-B",
            "Qty": "136",
        }
        assert not _looks_like_fused_table_row(row)

    def test_camera_inventory_row_not_fused(self) -> None:
        # Mobile RFP camera inventory — col is "Site Name", val is "MMOA"
        row = {"Site Name": "MMOA", "Current": "75", "Total": "75"}
        assert not _looks_like_fused_table_row(row)


# ─── Form field detection (P1.2 — already added Week 2; extra coverage) ───
class TestFormField:
    def test_vt_cam_form_field(self) -> None:
        assert _looks_like_form_field(
            "FULL LEGAL NAME (PRINT) (Company name as it appears with your "
            "Federal Taxpayer Number): CONTACT NAME/TITLE (PRINT) | "
            "FEDERAL TAXPAYER NUMBER (ID#): SIGNATURE (IN INK) | col_4: DATE"
        )

    def test_real_qa_not_form_field(self) -> None:
        assert not _looks_like_form_field(
            "A18. The RFP requests that to the extent possible, the "
            "proposed solution protect existing investments in legacy systems."
        )
