COPPER_002 — Parser / intake gap handoff (Worcester Durkin validation case)
================================================================================
Generated from: real_data_cases/COPPER_002_WORCESTER_DURKIN_NETWORK_UPGRADES/artifacts
Purpose: These are the file TYPES and concrete samples that fail or under-produce
          atoms in the current Parser OS compiler. Use to improve routing + parsers so
          gold labels (scope_exclusion AP, Phase B, IDF 010, re-termination qty,
          faulty endpoints, etc.) can be reached.

FOLDER LAYOUT
-------------
skipped_no_parser/
  Files where compile logs: "No parser matched artifact ...; skipping file"
  OR manifest shows parser_name: none.

  - extracted/*.txt
      Long plain-text extracts (addendum narrative, spec/drawings text).
      Today: no registered parser reaches confidence >= 0.5 in parser registry.

  - public_sources/*.pdf
      Public PDFs (addendum, compiled sources, spec book).
      Today: PDFs are not in the sample_text read list in choose_parser();
            no dedicated PDF parser in default registry — effectively always skip.

  - supplemental/synthetic_pm_note_faulty_endpoints.txt
      PM note: faulty endpoints / TBD counts — gold expects missing_info.
      Completely skipped (no parser match).

routed_email_zero_atoms/
  Email parser WINS routing (manifest: email_parser_v1) but parse_artifact()
  returns ZERO EvidenceAtoms. Body lines do not match EmailParser line-level
  regexes (instruction / exclusion / open-question) — e.g. "Please do not quote AP"
  is not in the current pattern set.

routed_transcript_zero_atoms/
  source_urls.json is routed (transcript parser in manifest) but produces no
  useful atoms for this tiny JSON — included so JSON/transcript behavior is known.

parses_ok_reference/
  CONTRAST: these two files currently drive almost all COPPER_002 atoms.
  - hardware_scope_schedule.xlsx  -> xlsx_parser (row entities, locations)
  - synthetic_vendor_quote_includes_aps.xlsx -> quote_parser (vendor lines inc. APs)

FILE TYPE SUMMARY (what we are testing the pipeline on)
-------------------------------------------------------
1) .xlsx  — site / hardware scope schedules; vendor quotes (BOM-style).
2) .txt   — (a) extracted legal/spec/addendum prose, (b) synthetic email in
            thread form, (c) PM field notes.
3) .pdf   — public addenda, compiled disclosure pack, large spec set.
4) .json  — small source URL list (edge case for transcript/json routing).

RELEVANT CODE (repository root)
----------------------------------
- app/parsers/registry.py        — choose_parser(), sample_text only for some ext
- app/parsers/email_parser.py    — match() vs parse() gap (zero atoms)
- app/core/compiler.py           — discovers artifacts under case artifacts/

SUCCESS CRITERIA (high level)
------------------------------
- PDFs: either add pdf text extraction + routing, or pre-extract to .txt with a
  parser that reads long prose.
- Generic .txt extracts: route addendum/spec extracts into a prose parser or relax
  transcript/email thresholds for structured headers.
- PM note (faulty endpoints): must not be skipped — needs parser match.
- Owner-furnished AP email: email routed must emit at least customer_instruction /
  exclusion atoms for DO-NOT-QUOTE / WPS-provided AP lines.

Zip path (sibling to this README): ../COPPER_002_parser_gap_handoff.zip
