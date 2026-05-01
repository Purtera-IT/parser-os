# LOWVOLT_001_FLMD_COURTROOM_5D

    This folder is a compiler-ready validation pack for `low_voltage_structured_cabling`.

    ## Folder map

    - `artifacts/public_sources/` - downloaded public source PDFs/XLSX and `compiled_public_sources.pdf`.
    - `artifacts/extracted/` - text extraction and source-derived schedules.
    - `artifacts/supplemental/` - synthetic but realistic quote/email/work-order notes created to make the case hard enough for validation.
    - `labels/gold_packets.json` - expected packet-level truth.
    - `labels/notes.md` - human-readable expected truth and review hints.
    - `outputs/` - intentionally empty; compiler output should be written here.

    ## Suggested command

    ```bash
    python scripts/compile_real_data_case.py real_data_cases/LOWVOLT_001_FLMD_COURTROOM_5D --domain-pack low_voltage_structured_cabling --out real_data_cases/LOWVOLT_001_FLMD_COURTROOM_5D/outputs/compile_result.json
    ```

    ## Why this case exists

    - SOW has exact quantities by cable family.
- Appendix A gives location-level terminations.
- Owner-furnished AV equipment must be treated as customer responsibility.
- Testing/certification and demo/removal are explicit.
- Partial award language stresses authority/status handling.
