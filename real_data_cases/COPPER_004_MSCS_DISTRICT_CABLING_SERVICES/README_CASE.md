# COPPER_004_MSCS_DISTRICT_CABLING_SERVICES

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
    python scripts/compile_real_data_case.py real_data_cases/COPPER_004_MSCS_DISTRICT_CABLING_SERVICES --domain-pack copper_cabling --out real_data_cases/COPPER_004_MSCS_DISTRICT_CABLING_SERVICES/outputs/compile_result.json
    ```

    ## Why this case exists

    - Includes a real XLSX pricing schedule with many cabling line items.
- District-scale RFP gives MDF/IDF, badge/access, and recurring service context.
- Supplemental work order creates a project-specific scope against the pricing catalog.
- IDF-1 and IDF-10 intentionally stress entity resolution.
- Patch cord vs drop quantities test item-family separation.
