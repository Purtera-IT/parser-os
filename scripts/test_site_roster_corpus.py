"""Run every mock site-roster PDF through parser-os and verify the
extracted physical_site atoms match the expected site IDs.

Run:
  python scripts/test_site_roster_corpus.py <corpus_dir>

For each PDF in <corpus_dir> and <corpus_dir>/mock_site_rosters/, the
script:
  1. Parses the PDF via OrbitBriefPdfParser.
  2. Pulls the atoms with value.kind == "physical_site".
  3. Compares the parsed site_id set against the expected set.
  4. Prints PASS / FAIL per file and the diff.

Expected sets are hardcoded below — they MUST match the rosters
produced by ``regenerate_doc08_and_mocks.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path


EXPECTED: dict[str, set[str]] = {
    # Doc 08 regen
    "08_site_roster_and_facilities_authoritative.pdf": {
        "ATL-HQ-01", "ATL-WEST-02", "ATL-AIR-03", "ATL-047-04", "ATL-CP-05",
    },
    # Mocks
    "site_roster_01_standard.pdf": {"NYC-HQ-01", "NYC-WEST-02", "BOS-MAIN-03"},
    "site_roster_02_id_last_column.pdf": {"SFO-WEST-01", "NYC-EAST-02", "CHI-MID-03"},
    "site_roster_03_2_column.pdf": {"LON-HQ-01", "LON-WEST-02", "LON-EAST-03"},
    "site_roster_04_8_column_full.pdf": {"TOR-HQ-01", "TOR-NW-02"},
    "site_roster_05_underscore_ids.pdf": {"ATL_HQ_01", "ATL_WEST_02", "ATL_AIR_03"},
    "site_roster_06_numeric_ids.pdf": {"S001", "S002", "S003"},
    "site_roster_07_store_ids.pdf": {"STORE-142", "STORE-143", "STORE-144"},
    "site_roster_08_bldg_ids.pdf": {"BLDG-1", "BLDG-12", "BLDG-A2"},
    "site_roster_09_international.pdf": {
        "TOR-HQ-01", "LON-HQ-01", "FRA-DC-01", "SGP-OFFICE-01",
    },
    "site_roster_10_no_headers.pdf": {"DAL-HQ-01", "DAL-N-02", "DAL-S-03"},
    "site_roster_11_explicit_declaration.pdf": {"HQ", "NW", "SE"},
    "site_roster_12_split_address.pdf": {"SEA-HQ-01", "SEA-BEL-02"},
    "site_roster_13_single_site.pdf": {"PHX-HQ-01"},
    "site_roster_14_many_sites.pdf": {
        "DEN-HQ-01", "DEN-DTC-02", "DEN-AIR-03", "SLC-HQ-04", "PHX-HQ-05",
        "LAS-HQ-06", "ABQ-HQ-07", "BIL-HQ-08", "BOI-HQ-09", "PDX-HQ-10",
        "SEA-HQ-11", "SFO-HQ-12", "LAX-HQ-13", "SAN-HQ-14", "MSP-HQ-15",
        "MKE-HQ-16", "DTW-HQ-17", "IND-HQ-18",
    },
    "site_roster_15_extra_columns.pdf": {"MIA-HQ-01", "MIA-BCH-02"},
    "site_roster_16_phone_email.pdf": {"AUS-HQ-01", "AUS-S-02"},
    "site_roster_17_mixed_id_shapes.pdf": {
        "ATL-HQ-01", "S100", "BLDG-7", "STORE-204", "LON-OFFICE-A",
    },
    "site_roster_18_with_continuation.pdf": {
        "NYC-HQ-01", "NYC-W-02", "NYC-E-05",   # canonical
        "TBD-1", "TBD-2",                       # placeholder rows still emitted
    },
}


def parse_pdf_sites(pdf_path: Path) -> list[dict]:
    """Parse a single PDF and return its physical_site rows."""
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
    parser = OrbitBriefPdfParser()
    output = parser.parse_artifact(
        project_id="corpus_test",
        artifact_id="art_" + pdf_path.stem,
        path=pdf_path,
        domain_pack=None,
    )
    out: list[dict] = []
    for atom in output.atoms:
        val = getattr(atom, "value", None) or {}
        if isinstance(val, dict) and val.get("kind") == "physical_site":
            out.append(val)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_site_roster_corpus.py <corpus_dir>", file=sys.stderr)
        return 2
    corpus = Path(sys.argv[1]).resolve()
    if not corpus.exists():
        print(f"corpus dir not found: {corpus}", file=sys.stderr)
        return 2

    cases: list[tuple[str, Path, set[str]]] = []
    # Doc 08 at root
    doc08 = corpus / "08_site_roster_and_facilities_authoritative.pdf"
    if doc08.exists():
        cases.append((doc08.name, doc08, EXPECTED[doc08.name]))
    # Mocks under mock_site_rosters/
    mock_dir = corpus / "mock_site_rosters"
    if mock_dir.exists():
        for pdf in sorted(mock_dir.glob("site_roster_*.pdf")):
            expected = EXPECTED.get(pdf.name)
            if expected is None:
                print(f"[warn] no expected for {pdf.name}")
                continue
            cases.append((pdf.name, pdf, expected))

    print(f"[corpus] {len(cases)} cases")
    print()

    passed = 0
    failed = 0
    for name, pdf, expected in cases:
        try:
            sites = parse_pdf_sites(pdf)
        except Exception as e:
            print(f"FAIL  {name}  (parser error: {type(e).__name__}: {e})")
            failed += 1
            continue
        site_ids = {s.get("site_id") for s in sites if s.get("site_id")}
        # For lowercase-ID cases, normalize for comparison
        site_ids_upper = {s.upper() for s in site_ids if s}
        expected_upper = {e.upper() for e in expected}
        if site_ids_upper == expected_upper:
            print(f"PASS  {name}  ({len(site_ids)} sites)")
            passed += 1
        else:
            missing = expected_upper - site_ids_upper
            extra = site_ids_upper - expected_upper
            print(f"FAIL  {name}")
            print(f"  expected: {sorted(expected_upper)}")
            print(f"  got:      {sorted(site_ids_upper)}")
            if missing:
                print(f"  missing:  {sorted(missing)}")
            if extra:
                print(f"  extra:    {sorted(extra)}")
            failed += 1

    print()
    print(f"=== {passed} passed, {failed} failed ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
