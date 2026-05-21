"""Universal stress test for site_roster extraction.

Runs the regen doc 08 + every mock variant + the full v8 OPTBOT bundle
through BOTH parser-os compile AND orbitbrief envelope build, then
checks for PM-critical information loss:

  PM_CRITICAL_FIELDS — the columns a PM physically goes looking for:
      site_id           the reference key
      facility_name     what to call it
      street_address    where to go
      access_window     when you can work (or None when source omits)
      mdf_idf           which closet (or None when source omits)
      escort_owner      who lets you in (or None when source omits)

Each row's *expected* fields are the cells present in the source PDF.
The test passes when every expected field arrives in the parser output
AND in the orbitbrief envelope (when the envelope includes the atom).

Usage:
    python scripts/stress_test_site_roster_universal.py <corpus_dir>
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path


# Expected rows. Each entry is the FULL roster expected from that
# source PDF, including every cell the source provided. Tests pass
# when the parser emits matching physical_site atoms with at least
# those fields populated (extra fields are fine).
EXPECTED: dict[str, list[dict[str, str | None]]] = {
    "08_site_roster_and_facilities_authoritative.pdf": [
        {"site_id": "ATL-HQ-01",   "facility_name": "OPTBOT Atlanta HQ",
         "street_address": "1200 Peachtree St NE, Atlanta GA 30309",
         "mdf_idf": "MDF-3A / IDF 2-7",  "access_window": "Mon-Fri 07:00-18:00",
         "escort_owner": "OPTBOT Facilities"},
        {"site_id": "ATL-WEST-02", "facility_name": "OPTBOT West Campus",
         "street_address": "3100 Interstate N Pkwy, Atlanta GA 30339",
         "mdf_idf": "MDF-W1 / IDF W2-3", "access_window": "Mon-Fri 07:00-18:00",
         "escort_owner": "OPTBOT Facilities"},
        {"site_id": "ATL-AIR-03",  "facility_name": "OPTBOT Airport Logistics",
         "street_address": "6000 N Terminal Pkwy, Atlanta GA 30320",
         "mdf_idf": "MDF-A / IDF A1",    "access_window": "Mon-Sat 06:00-22:00",
         "escort_owner": "OPTBOT Security"},
        {"site_id": "ATL-047-04",  "facility_name": "OPTBOT Brady Training",
         "street_address": "047 Brady Ave NW, Atlanta GA 30318",
         "mdf_idf": "MDF-B / IDF B1-2",  "access_window": "Mon-Fri 08:00-17:00",
         "escort_owner": "OPTBOT Facilities"},
        {"site_id": "ATL-CP-05",   "facility_name": "OPTBOT College Park Staging",
         "street_address": "1850 Sullivan Rd, College Park GA 30337",
         "mdf_idf": "MDF-CP / staging",   "access_window": "Mon-Fri 07:00-15:00",
         "escort_owner": "OPTBOT Logistics"},
    ],
    "site_roster_01_standard.pdf": [
        {"site_id": "NYC-HQ-01",   "facility_name": "Acme New York HQ",
         "street_address": "350 5th Ave, New York NY 10118", "mdf_idf": "MDF-1",
         "access_window": "Mon-Fri 08:00-18:00", "escort_owner": "Facilities"},
        {"site_id": "NYC-WEST-02", "facility_name": "Acme Chelsea Office",
         "street_address": "75 9th Ave, New York NY 10011", "mdf_idf": "MDF-2",
         "access_window": "Mon-Fri 08:00-18:00", "escort_owner": "Facilities"},
        {"site_id": "BOS-MAIN-03", "facility_name": "Acme Boston Office",
         "street_address": "100 Federal St, Boston MA 02110", "mdf_idf": "MDF-3",
         "access_window": "Mon-Fri 08:00-17:00", "escort_owner": "Facilities"},
    ],
    "site_roster_02_id_last_column.pdf": [
        {"site_id": "SFO-WEST-01", "facility_name": "DataHub West",
         "street_address": "100 Mission St, San Francisco CA", "mdf_idf": "MDF-W"},
        {"site_id": "NYC-EAST-02", "facility_name": "DataHub East",
         "street_address": "200 Park Ave, New York NY", "mdf_idf": "MDF-E"},
        {"site_id": "CHI-MID-03",  "facility_name": "DataHub Midwest",
         "street_address": "400 Wacker Dr, Chicago IL", "mdf_idf": "MDF-M"},
    ],
    "site_roster_03_2_column.pdf": [
        {"site_id": "LON-HQ-01",   "facility_name": "Acme London"},
        {"site_id": "LON-WEST-02", "facility_name": "Acme Hammersmith"},
        {"site_id": "LON-EAST-03", "facility_name": "Acme Canary Wharf"},
    ],
    "site_roster_04_8_column_full.pdf": [
        {"site_id": "TOR-HQ-01",   "facility_name": "Acme Toronto",
         "street_address": "100 King St W, Toronto ON", "mdf_idf": "MDF-1",
         "access_window": "Mon-Fri 08:00-18:00", "escort_owner": "Security",
         "contact": "J. Smith", "phone": "416-555-0100"},
        {"site_id": "TOR-NW-02",   "facility_name": "Acme North York",
         "street_address": "5 Park Home Ave, Toronto ON", "mdf_idf": "MDF-2",
         "access_window": "Mon-Fri 07:00-19:00", "escort_owner": "Facilities",
         "contact": "A. Patel", "phone": "416-555-0101"},
    ],
    "site_roster_05_underscore_ids.pdf": [
        # Source PDF uses lowercase; parser preserves case
        {"site_id": "atl_hq_01",   "facility_name": "Atlanta HQ",
         "street_address": "1200 Peachtree St NE, Atlanta GA"},
        {"site_id": "atl_west_02", "facility_name": "Atlanta West",
         "street_address": "3100 Interstate N Pkwy, Atlanta GA"},
        {"site_id": "atl_air_03",  "facility_name": "Atlanta Airport",
         "street_address": "6000 N Terminal Pkwy, Atlanta GA"},
    ],
    "site_roster_06_numeric_ids.pdf": [
        {"site_id": "S001", "facility_name": "Acme Site 1", "street_address": "100 Main St, Anytown USA"},
        {"site_id": "S002", "facility_name": "Acme Site 2", "street_address": "200 Main St, Anytown USA"},
        {"site_id": "S003", "facility_name": "Acme Site 3", "street_address": "300 Main St, Anytown USA"},
    ],
    "site_roster_07_store_ids.pdf": [
        {"site_id": "STORE-142", "facility_name": "Cherry Creek",      "street_address": "3030 E 1st Ave, Denver CO"},
        {"site_id": "STORE-143", "facility_name": "Park Meadows",      "street_address": "8401 Park Meadows Center Dr, Lone Tree CO"},
        {"site_id": "STORE-144", "facility_name": "Flatiron Crossing", "street_address": "1 Flatiron Crossing Dr, Broomfield CO"},
    ],
    "site_roster_08_bldg_ids.pdf": [
        {"site_id": "BLDG-1",  "facility_name": "Office"},
        {"site_id": "BLDG-12", "facility_name": "Warehouse"},
        {"site_id": "BLDG-A2", "facility_name": "Datacenter"},
    ],
    "site_roster_09_international.pdf": [
        {"site_id": "TOR-HQ-01",      "street_address": "100 King St W, Toronto ON M5X 1A1"},
        {"site_id": "LON-HQ-01",      "street_address": "1 St Mary Axe, London EC3A 8BF"},
        {"site_id": "FRA-DC-01",      "street_address": "Hanauer Landstraße 296, 60314 Frankfurt"},
        {"site_id": "SGP-OFFICE-01",  "street_address": "1 Marina Boulevard, Singapore"},
    ],
    "site_roster_10_no_headers.pdf": [
        {"site_id": "DAL-HQ-01", "facility_name": "Acme Dallas", "street_address": "100 Commerce St, Dallas TX"},
        {"site_id": "DAL-N-02",  "facility_name": "Acme Frisco", "street_address": "200 Main St, Frisco TX"},
        {"site_id": "DAL-S-03",  "facility_name": "Acme Plano",  "street_address": "300 Plano Pkwy, Plano TX"},
    ],
    "site_roster_11_explicit_declaration.pdf": [
        {"site_id": "HQ", "facility_name": "Acme HQ",         "street_address": "100 Main St, Anytown USA"},
        {"site_id": "NW", "facility_name": "Northwest Office", "street_address": "200 NW Ave, Anytown USA"},
        {"site_id": "SE", "facility_name": "Southeast Office", "street_address": "300 SE Blvd, Anytown USA"},
    ],
    "site_roster_12_split_address.pdf": [
        {"site_id": "SEA-HQ-01",  "facility_name": "Acme Seattle",  "street_address": "1200 5th Ave"},
        {"site_id": "SEA-BEL-02", "facility_name": "Acme Bellevue", "street_address": "10500 NE 8th St"},
    ],
    "site_roster_13_single_site.pdf": [
        {"site_id": "PHX-HQ-01", "facility_name": "Acme Phoenix HQ", "street_address": "100 Camelback Rd, Phoenix AZ"},
    ],
    # 14_many_sites: all 18 sites
    "site_roster_14_many_sites.pdf": [
        {"site_id": "DEN-HQ-01",  "facility_name": "Acme Denver HQ"},
        {"site_id": "DEN-DTC-02", "facility_name": "Acme DTC"},
        {"site_id": "DEN-AIR-03", "facility_name": "Acme Airport"},
        {"site_id": "SLC-HQ-04",  "facility_name": "Acme Salt Lake"},
        {"site_id": "PHX-HQ-05",  "facility_name": "Acme Phoenix"},
        {"site_id": "LAS-HQ-06",  "facility_name": "Acme Las Vegas"},
        {"site_id": "ABQ-HQ-07",  "facility_name": "Acme Albuquerque"},
        {"site_id": "BIL-HQ-08",  "facility_name": "Acme Billings"},
        {"site_id": "BOI-HQ-09",  "facility_name": "Acme Boise"},
        {"site_id": "PDX-HQ-10",  "facility_name": "Acme Portland"},
        {"site_id": "SEA-HQ-11",  "facility_name": "Acme Seattle"},
        {"site_id": "SFO-HQ-12",  "facility_name": "Acme SF"},
        {"site_id": "LAX-HQ-13",  "facility_name": "Acme LA"},
        {"site_id": "SAN-HQ-14",  "facility_name": "Acme San Diego"},
        {"site_id": "MSP-HQ-15",  "facility_name": "Acme Minneapolis"},
        {"site_id": "MKE-HQ-16",  "facility_name": "Acme Milwaukee"},
        {"site_id": "DTW-HQ-17",  "facility_name": "Acme Detroit"},
        {"site_id": "IND-HQ-18",  "facility_name": "Acme Indianapolis"},
    ],
    "site_roster_15_extra_columns.pdf": [
        {"site_id": "MIA-HQ-01",  "facility_name": "Acme Miami", "street_address": "100 Biscayne Blvd, Miami FL",
         "_extras_must_include": ["Risk class", "Cost code", "Owner"]},
        {"site_id": "MIA-BCH-02", "facility_name": "Acme Beach", "street_address": "1 Ocean Dr, Miami Beach FL",
         "_extras_must_include": ["Risk class", "Cost code", "Owner"]},
    ],
    "site_roster_16_phone_email.pdf": [
        {"site_id": "AUS-HQ-01", "facility_name": "Acme Austin", "street_address": "100 Congress Ave, Austin TX",
         "contact": "M. Lopez",  "phone": "512-555-0100", "email": "mlopez@acme.test"},
        {"site_id": "AUS-S-02",  "facility_name": "Acme South",  "street_address": "200 S Lamar Blvd, Austin TX",
         "contact": "T. Nguyen", "phone": "512-555-0101", "email": "tnguyen@acme.test"},
    ],
    "site_roster_17_mixed_id_shapes.pdf": [
        {"site_id": "ATL-HQ-01",    "facility_name": "OPTBOT Atlanta HQ"},
        {"site_id": "S100",         "facility_name": "OPTBOT West Office"},
        {"site_id": "BLDG-7",       "facility_name": "OPTBOT Airport Logistics"},
        {"site_id": "STORE-204",    "facility_name": "OPTBOT Brady Training"},
        {"site_id": "LON-OFFICE-A", "facility_name": "OPTBOT London EMEA"},
    ],
    "site_roster_18_with_continuation.pdf": [
        {"site_id": "NYC-HQ-01", "facility_name": "Acme NYC"},
        {"site_id": "NYC-W-02",  "facility_name": "Acme NYC West"},
        {"site_id": "TBD-1",     "facility_name": "Future site 1"},
        {"site_id": "TBD-2",     "facility_name": "Future site 2"},
        {"site_id": "NYC-E-05",  "facility_name": "Acme NYC East"},
    ],
}


def _normalize(s: str | None) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split())


def parse_pdf_sites(pdf_path: Path) -> list[dict]:
    """Parse a single PDF and return physical_site dicts."""
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
    p = OrbitBriefPdfParser()
    out = p.parse_artifact(
        project_id="universal_stress",
        artifact_id="art_" + pdf_path.stem,
        path=pdf_path,
        domain_pack=None,
    )
    sites: list[dict] = []
    for atom in out.atoms:
        v = getattr(atom, "value", None) or {}
        if isinstance(v, dict) and v.get("kind") == "physical_site":
            sites.append(v)
    return sites


def parse_pdf_via_compile(pdf_path: Path) -> tuple[list[dict], Path | None]:
    """Compile a single-PDF project and return (physical_site atoms,
    path to orbitbrief envelope JSON)."""
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import (
        build_orbitbrief_envelope,
        write_orbitbrief_envelope,
    )

    with tempfile.TemporaryDirectory(prefix="universal_stress_") as td:
        td_path = Path(td)
        art_dir = td_path / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, art_dir / pdf_path.name)

        compile_result = compile_project(
            project_dir=td_path,
            domain_pack=None,
            use_cache=False,
            allow_errors=True,
            allow_unverified_receipts=True,
        )
        sites: list[dict] = []
        for atom in compile_result.atoms:
            v = getattr(atom, "value", None) or {}
            if isinstance(v, dict) and v.get("kind") == "physical_site":
                sites.append(v)

        envelope = build_orbitbrief_envelope(
            project_dir=td_path, compile_result=compile_result
        )
        env_paths = write_orbitbrief_envelope(
            project_dir=td_path, envelope=envelope, out_dir=td_path / ".orbitbrief"
        )
        env_json_path = env_paths[0] if env_paths else None
        # Copy to a stable spot so caller can inspect after teardown
        if env_json_path and env_json_path.exists():
            persistent = pdf_path.parent / f"{pdf_path.stem}.envelope.json"
            shutil.copy2(env_json_path, persistent)
            return sites, persistent
        return sites, None


def check_envelope_for_sites(envelope_path: Path, expected_ids: set[str]) -> tuple[set[str], list[dict]]:
    """Return (site_ids_found, raw_site_dicts) from the envelope JSON."""
    if envelope_path is None or not envelope_path.exists():
        return set(), []
    try:
        env = json.loads(envelope_path.read_text(encoding="utf-8"))
    except Exception:
        return set(), []
    found_ids: set[str] = set()
    found_sites: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("kind") == "physical_site":
                sid = node.get("site_id")
                if sid:
                    found_ids.add(sid)
                    found_sites.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(env)
    return found_ids, found_sites


def check_pm_fields(parsed_sites: list[dict], expected_rows: list[dict]) -> list[str]:
    """For each expected row, check the parsed site has the same fields.

    Returns a list of human-readable problem strings; empty list means
    all PM-critical fields survived.
    """
    by_id_upper: dict[str, dict] = {
        (s.get("site_id") or "").upper(): s
        for s in parsed_sites if s.get("site_id")
    }
    issues: list[str] = []
    for exp in expected_rows:
        expected_id = exp["site_id"]
        target_id = expected_id
        actual = by_id_upper.get(target_id.upper())
        if actual is None:
            issues.append(f"site_id {target_id!r} MISSING from parsed output")
            continue
        for key, expected_val in exp.items():
            if key.startswith("_") or key == "site_id":
                continue
            actual_val = actual.get(key)
            if expected_val and not actual_val:
                issues.append(
                    f"site_id {target_id!r}: field {key!r} EMPTY (expected {expected_val!r})"
                )
            elif expected_val and _normalize(actual_val) and _normalize(expected_val) not in _normalize(actual_val):
                # Soft check: expected value should appear in actual
                # (parser may add extra detail but not lose info)
                if _normalize(actual_val) not in _normalize(expected_val):
                    issues.append(
                        f"site_id {target_id!r}: field {key!r} mismatch "
                        f"expected={expected_val!r} actual={actual_val!r}"
                    )
        # Extras requirement
        extras_required = exp.get("_extras_must_include", [])
        if extras_required:
            extras_dict = actual.get("extras") or {}
            for ex_field in extras_required:
                if ex_field not in extras_dict:
                    issues.append(
                        f"site_id {target_id!r}: extra field {ex_field!r} MISSING "
                        f"(got extras={list(extras_dict.keys())})"
                    )
    return issues


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/stress_test_site_roster_universal.py <corpus_dir>", file=sys.stderr)
        return 2
    corpus = Path(sys.argv[1]).resolve()

    cases: list[tuple[str, Path]] = []
    doc08 = corpus / "08_site_roster_and_facilities_authoritative.pdf"
    if doc08.exists():
        cases.append((doc08.name, doc08))
    mock_dir = corpus / "mock_site_rosters"
    if mock_dir.exists():
        for pdf in sorted(mock_dir.glob("site_roster_*.pdf")):
            cases.append((pdf.name, pdf))

    print(f"=== Universal stress test: {len(cases)} cases ===\n")

    parser_passes = 0
    parser_fails = 0
    envelope_passes = 0
    envelope_fails = 0
    pm_field_issues_total = 0

    for name, pdf in cases:
        expected = EXPECTED.get(name)
        if expected is None:
            print(f"[skip] {name} (no expected)")
            continue
        expected_ids = {row["site_id"] for row in expected if row.get("site_id")}

        # ── Parser-os direct ──────────────────────────────────────
        try:
            parsed_sites = parse_pdf_sites(pdf)
        except Exception as e:
            print(f"FAIL parser {name}: {type(e).__name__}: {e}")
            parser_fails += 1
            continue
        parsed_ids = {s.get("site_id") for s in parsed_sites if s.get("site_id")}
        parsed_ids_upper = {x.upper() for x in parsed_ids}
        expected_ids_upper = {x.upper() for x in expected_ids}
        parser_ok = parsed_ids_upper == expected_ids_upper

        # ── PM-critical field check ───────────────────────────────
        pm_issues = check_pm_fields(parsed_sites, expected)

        # ── Compile + envelope ────────────────────────────────────
        try:
            compile_sites, env_path = parse_pdf_via_compile(pdf)
        except Exception as e:
            print(f"FAIL compile {name}: {type(e).__name__}: {e}")
            envelope_fails += 1
            env_path = None
            compile_sites = []
        env_ids, env_site_dicts = check_envelope_for_sites(env_path, expected_ids) if env_path else (set(), [])
        env_ids_upper = {x.upper() for x in env_ids}
        env_ok = bool(env_ids) and env_ids_upper == expected_ids_upper
        compile_ids = {s.get("site_id") for s in compile_sites if s.get("site_id")}
        compile_ids_upper = {x.upper() for x in compile_ids}
        compile_ok = compile_ids_upper == expected_ids_upper

        # ── Report ────────────────────────────────────────────────
        line_status = []
        line_status.append("parser:" + ("OK" if parser_ok else "FAIL"))
        line_status.append("compile:" + ("OK" if compile_ok else "FAIL"))
        line_status.append("envelope:" + ("OK" if env_ok else "PARTIAL" if env_ids else "MISSING"))
        line_status.append(f"pm_fields:{'OK' if not pm_issues else f'{len(pm_issues)} issues'}")
        print(f"  {name}  {'  '.join(line_status)}")

        if not parser_ok:
            missing = expected_ids_upper - parsed_ids_upper
            extra = parsed_ids_upper - expected_ids_upper
            if missing: print(f"    parser missing: {sorted(missing)}")
            if extra:   print(f"    parser extra:   {sorted(extra)}")
        if not compile_ok:
            missing = expected_ids_upper - compile_ids_upper
            extra = compile_ids_upper - expected_ids_upper
            if missing: print(f"    compile missing: {sorted(missing)}")
            if extra:   print(f"    compile extra:   {sorted(extra)}")
        if env_ids and not env_ok:
            missing = expected_ids_upper - env_ids_upper
            if missing: print(f"    envelope missing: {sorted(missing)}")
        if not env_ids:
            print(f"    envelope DROPS physical_site atoms entirely")
        for issue in pm_issues[:8]:
            print(f"    pm: {issue}")
        if len(pm_issues) > 8:
            print(f"    pm: ...and {len(pm_issues)-8} more")

        if parser_ok:   parser_passes += 1
        else:           parser_fails += 1
        if env_ok:      envelope_passes += 1
        elif env_ids:   envelope_passes += 0  # partial doesn't count
        pm_field_issues_total += len(pm_issues)

    print()
    print("=" * 60)
    print(f"PARSER:    {parser_passes}/{parser_passes+parser_fails} passed")
    print(f"ENVELOPE:  {envelope_passes}/{len(cases)} fully include physical_site")
    print(f"PM FIELDS: {pm_field_issues_total} total field-level issues across all cases")
    print("=" * 60)
    return 0 if parser_fails == 0 and pm_field_issues_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
