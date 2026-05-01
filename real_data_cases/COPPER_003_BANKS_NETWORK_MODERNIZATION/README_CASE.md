# COPPER_003_BANKS_NETWORK_MODERNIZATION

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
    python scripts/compile_real_data_case.py real_data_cases/COPPER_003_BANKS_NETWORK_MODERNIZATION --domain-pack copper_cabling --out real_data_cases/COPPER_003_BANKS_NETWORK_MODERNIZATION/outputs/compile_result.json
    ```

    ## Why this case exists

    - RFP states 378 Cat6A drops and equal patch cord quantities.
- Public RFP states detailed site maps/facility drawings are available upon request but not attached.
- Testing/certification and as-built requirements are rich closeout evidence.
- Fiber backbone quantities should not merge with copper quantities.
- Numeric testing notice stresses quantity normalizer.
