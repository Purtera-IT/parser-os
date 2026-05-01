# COPPER_002_WORCESTER_DURKIN_NETWORK_UPGRADES

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
    python scripts/compile_real_data_case.py real_data_cases/COPPER_002_WORCESTER_DURKIN_NETWORK_UPGRADES --domain-pack copper_cabling --out real_data_cases/COPPER_002_WORCESTER_DURKIN_NETWORK_UPGRADES/outputs/compile_result.json
    ```

    ## Why this case exists

    - Bid page flags Addendum 1 adding Phase B scope.
- Specifications include owner-provided wireless access points that should not be quoted.
- Addendum introduces existing copper re-termination and new basement IDF cabinet work.
- Faulty endpoint replacement requires missing quantity review.
- Room/entity strings stress MDF/IDF resolution.
