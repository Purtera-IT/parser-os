# LOWVOLT_001_FLMD_COURTROOM_5D - U.S. District Court Middle District of Florida - Fort Myers Courtroom 5D Structured Cabling SOW

    ## What this case is

    Domain: `low_voltage_structured_cabling`  
    Pack type: `clean exact-quantity low-voltage pack + owner-furnished trap`  
    Fit score: `8.9/10`  
    Public source status: public PDFs/XLSX preserved under `artifacts/public_sources/`; companion artifacts under `artifacts/supplemental/` are synthetic validation fixtures.

    ## What is actually in scope

    This case is designed to test the active scope described by the public source files plus the intentionally conflicting supplemental quote/work-order/email files.

    ## What is explicitly excluded or risky

    - 36 vs 34 Cat6A shielded cable mismatch
- certification excluded
- demo/removal excluded
- owner-furnished AV components quoted incorrectly
- partial award language misread as final scope

    ## What the compiler should catch

    - quantity_conflict :: device:cat6a_shielded_cable
- vendor_mismatch :: requirement:certification
- customer_responsibility :: av_components
- scope_inclusion :: demo_existing_video_cabling
- missing_info :: partial_award

    ## Forbidden outcomes

    - speaker_cable_quantity_merged_with_cat6a_quantity
- owner_furnished_av_components_become_vendor_scope
- partial_award_language_treated_as_final_scope_exclusion_without_po
- rg59_count_merged_with_cat6a_video_output_count

    ## Source files

    - FLMD Fort Myers Courtroom 5D SOW: https://www.flmd.uscourts.gov/sites/flmd/files/documents/flmd-structured-cabling-for-fort-myers-courtroom-5d-statement-of-work.pdf

    ## Human review instructions

    After compile, inspect all high-risk packets. Verify:
    - SourceRef exists for every atom.
    - Public source PDFs/XLSX are never treated as unsupported silently.
    - Supplemental vendor quote cannot govern customer-approved scope.
    - Addendum/current instruction beats older/original language when applicable.
    - Entity aliases do not create false merges.
