# COPPER_001_SPRING_LAKE_AUDITORIUM - Spring Lake Public Schools FY2025 Structured Cabling RFP + Addendum

    ## What this case is

    Domain: `copper_cabling`  
    Pack type: `messy + revision-heavy + material-spec trap`  
    Fit score: `9.7/10`  
    Public source status: public PDFs/XLSX preserved under `artifacts/public_sources/`; companion artifacts under `artifacts/supplemental/` are synthetic validation fixtures.

    ## What is actually in scope

    This case is designed to test the active scope described by the public source files plus the intentionally conflicting supplemental quote/work-order/email files.

    ## What is explicitly excluded or risky

    - quantity mismatch
- Cat6 UTP/STP material mismatch
- power scope exclusion
- unknown raceway/conduit condition
- certification/testing report missing

    ## What the compiler should catch

    - quantity_conflict :: cabling:rj45
- vendor_mismatch :: material:cat6_utp
- vendor_mismatch :: material:cat6_stp
- scope_exclusion :: scope:power
- missing_info :: raceway_conduit
- missing_info :: requirement:certification
- site_access :: catwalk

    ## Forbidden outcomes

    - original_rfp_governs_over_current_addendum
- vendor_quote_governs_scope
- total_row_extracted_as_real_location
- power_scope_included_as_active_structured_cabling

    ## Source files

    - Spring Lake original RFP: https://resources.finalsite.net/images/v1733493702/springlakeschoolsorg/iiha1hemefukgmincfjw/RFPSCSpringLakeHighSchoolAuditoriumFY25.pdf
- Spring Lake Addendum: https://resources.finalsite.net/images/v1734628088/springlakeschoolsorg/dwra7sh4bp1vfkfasg8c/RFPSCSpringLakeAddendum.pdf

    ## Human review instructions

    After compile, inspect all high-risk packets. Verify:
    - SourceRef exists for every atom.
    - Public source PDFs/XLSX are never treated as unsupported silently.
    - Supplemental vendor quote cannot govern customer-approved scope.
    - Addendum/current instruction beats older/original language when applicable.
    - Entity aliases do not create false merges.
