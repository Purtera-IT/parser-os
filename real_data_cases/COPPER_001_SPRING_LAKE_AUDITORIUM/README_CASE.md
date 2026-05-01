# COPPER_001_SPRING_LAKE_AUDITORIUM

    This folder is a compiler-ready validation pack for `copper_cabling`.

    ## Folder map

    - `artifacts/public_sources/` - downloaded public source PDFs/XLSX and `compiled_public_sources.pdf`.
    - `artifacts/extracted/` - text extraction and source-derived schedules.
    - `artifacts/supplemental/` - synthetic but realistic quote/email/work-order notes created to make the case hard enough for validation.
    - `labels/gold_packets.json` - expected packet-level truth.
    - `labels/notes.md` - human-readable expected truth and review hints.
    - `outputs/` - intentionally empty; compiler output should be written here.

    ## Suggested command

    ```bash
    python scripts/compile_real_data_case.py real_data_cases/COPPER_001_SPRING_LAKE_AUDITORIUM --domain-pack copper_cabling --out real_data_cases/COPPER_001_SPRING_LAKE_AUDITORIUM/outputs/compile_result.json
    ```

    ## Why this case exists

    - Addendum overrides original RFP language after vendor feedback.
- Table includes exact RJ45, Cat6 UTP, and Cat6 STP counts plus a TOTALS row.
- Power is explicitly excluded in Q&A while structured cabling stays in scope.
- Existing raceway/conduit usability is unknown, creating change-order risk.
- Certification/performance verification requirements must be preserved.
