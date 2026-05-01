# COPPER_004_MSCS_DISTRICT_CABLING_SERVICES - Memphis-Shelby County Schools Network Structured Cabling System Services

    ## What this case is

    Domain: `copper_cabling`  
    Pack type: `messy high-scale vendor/BOM/pricing parser torture chamber`  
    Fit score: `9.0/10`  
    Public source status: public PDFs/XLSX preserved under `artifacts/public_sources/`; companion artifacts under `artifacts/supplemental/` are synthetic validation fixtures.

    ## What is actually in scope

    This case is designed to test the active scope described by the public source files plus the intentionally conflicting supplemental quote/work-order/email files.

    ## What is explicitly excluded or risky

    - work order vs quote quantity mismatch
- Cat6 vs Cat6A material mismatch
- IDF-1 vs IDF-10 false merge
- badge/background access exclusion
- pricing schedule treated as project scope

    ## What the compiler should catch

    - quantity_conflict :: device:cat6a_classroom_drop
- vendor_mismatch :: patch_cord:cat6a
- site_access :: badge
- action_item :: site_survey
- missing_info :: faceplates

    ## Forbidden outcomes

    - idf_1_merged_with_idf_10
- vendor_pricing_schedule_governs_project_scope
- patch_cord_quantity_treated_as_drop_quantity
- pallet_or_catalog_quantity_becomes_site_scope

    ## Source files

    - MSCS Structured Cabling Services RFP: https://www.scsk12.org/procurement/uploads/bids/2015/RFP%20110725AW%20Network%20Structured%20Cabling%20System%20Services%20Non%20Erate.pdf
- MSCS Appendix I Pricing Schedule: https://www.scsk12.org/procurement/uploads/bids/2015/APPENDIX%20I%20Pricing%20Schedule%20-%20Network%20Structured%20Cabling%20System%20Services%20-%20Non%20Erate.xlsx
- MSCS Q&A: https://www.scsk12.org/procurement/uploads/bids/2015/Questions%20and%20Answers%20RFP%20110725AW%20Network%20Structured%20Cabling%20System%20Services%20Non%20Erate.pdf

    ## Human review instructions

    After compile, inspect all high-risk packets. Verify:
    - SourceRef exists for every atom.
    - Public source PDFs/XLSX are never treated as unsupported silently.
    - Supplemental vendor quote cannot govern customer-approved scope.
    - Addendum/current instruction beats older/original language when applicable.
    - Entity aliases do not create false merges.
