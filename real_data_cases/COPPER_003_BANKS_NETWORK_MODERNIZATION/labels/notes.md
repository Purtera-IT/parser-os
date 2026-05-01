# COPPER_003_BANKS_NETWORK_MODERNIZATION - Banks School District Network Modernization Project 2026

    ## What this case is

    Domain: `copper_cabling`  
    Pack type: `large-scale clean + explicit missing-info trap`  
    Fit score: `9.2/10`  
    Public source status: public PDFs/XLSX preserved under `artifacts/public_sources/`; companion artifacts under `artifacts/supplemental/` are synthetic validation fixtures.

    ## What is actually in scope

    This case is designed to test the active scope described by the public source files plus the intentionally conflicting supplemental quote/work-order/email files.

    ## What is explicitly excluded or risky

    - 378 vs 360 Cat6A drop mismatch
- patch cord quantity mismatch
- facility drawings missing
- as-built deliverable missing
- testing notice misparsed as quantity

    ## What the compiler should catch

    - quantity_conflict :: device:cat6a_drop
- vendor_mismatch :: patch_cord:3ft_cat6a
- vendor_mismatch :: patch_cord:5ft_cat6a
- missing_info :: facility_drawings
- missing_info :: as_built_dwg
- action_item :: testing_notice

    ## Forbidden outcomes

    - lead_time_or_testing_notice_treated_as_install_date
- missing_drawings_silently_ignored
- fiber_backbone_count_merged_with_copper_drop_count
- patch_cord_count_becomes_drop_count

    ## Source files

    - Banks Network Modernization RFP: https://www.banks.k12.or.us/wp-content/uploads/2026/03/Banks-SD_Network_Modernization_Project_2026.pdf

    ## Human review instructions

    After compile, inspect all high-risk packets. Verify:
    - SourceRef exists for every atom.
    - Public source PDFs/XLSX are never treated as unsupported silently.
    - Supplemental vendor quote cannot govern customer-approved scope.
    - Addendum/current instruction beats older/original language when applicable.
    - Entity aliases do not create false merges.
