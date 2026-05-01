# COPPER_002_WORCESTER_DURKIN_NETWORK_UPGRADES - City of Worcester / WPS Durkin Administration Building Network Upgrades

    ## What this case is

    Domain: `copper_cabling`  
    Pack type: `revision-heavy + MDF/IDF + owner-furnished trap`  
    Fit score: `9.5/10`  
    Public source status: public PDFs/XLSX preserved under `artifacts/public_sources/`; companion artifacts under `artifacts/supplemental/` are synthetic validation fixtures.

    ## What is actually in scope

    This case is designed to test the active scope described by the public source files plus the intentionally conflicting supplemental quote/work-order/email files.

    ## What is explicitly excluded or risky

    - owner-furnished APs quoted incorrectly
- Phase B addendum ignored
- faulty endpoint count missing
- IDF Room 010 entity merge risk
- existing copper re-termination quantity mismatch

    ## What the compiler should catch

    - scope_exclusion :: device:wireless_access_point
- scope_inclusion :: idf:010
- scope_inclusion :: copper:retermination
- quantity_conflict :: copper:retermination
- missing_info :: faulty_endpoints

    ## Forbidden outcomes

    - owner_furnished_ap_becomes_vendor_scope
- addendum_phase_b_ignored
- idf_room_010_merged_with_room_101
- vendor_quote_governs_owner_furnished_scope

    ## Source files

    - Worcester bid page: https://www.worcesterma.gov/finance/purchasing-bids/bids/8689-m6
- Specifications and Drawings PDF: https://www.worcesterma.gov/sites/default/files/bids/8689-M6%20Specificatins%20and%20Drawings.pdf
- Addendum 1 PDF: https://www.worcesterma.gov/sites/default/files/bids/8689-M6-ADDENDUM-1.pdf

    ## Human review instructions

    After compile, inspect all high-risk packets. Verify:
    - SourceRef exists for every atom.
    - Public source PDFs/XLSX are never treated as unsupported silently.
    - Supplemental vendor quote cannot govern customer-approved scope.
    - Addendum/current instruction beats older/original language when applicable.
    - Entity aliases do not create false merges.
