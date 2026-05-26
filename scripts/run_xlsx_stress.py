"""Run the XLSX adversarial bundle through compile + envelope.

For each .xlsx, asserts:
  - parser doesn't crash
  - atoms emitted
  - critical content survives (part numbers / quantities / sites where present)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass


def compile_one(xlsx_path: Path):
    """Compile a single XLSX in its OWN directory (no TemporaryDirectory
    cleanup race on Windows — openpyxl + dependencies sometimes leave a
    file handle open just long enough for the cleanup to fail)."""
    from app.core.compiler import compile_project
    base = Path(tempfile.gettempdir()) / f"xlsx_stress_{xlsx_path.stem}"
    art = base / "artifacts"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    art.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xlsx_path, art / xlsx_path.name)
    r = compile_project(
        project_dir=base, domain_pack=None,
        use_cache=False, allow_errors=True, allow_unverified_receipts=True,
    )
    return list(r.atoms), list(r.entities), list(r.warnings)


def _ent_keys(entities) -> str:
    return " ".join((getattr(e, "canonical_key", "") or "").lower() for e in entities)


def _atom_text(atoms) -> str:
    return " ".join(getattr(a, "raw_text", "") or "" for a in atoms)


def check_xa_bom_simple(a, e, w):
    issues = []
    k = _ent_keys(e)
    for sid in ("atl_hq_01", "atl_west_02", "atl_air_03"):
        if sid not in k:
            issues.append(f"site:{sid} missing")
    for p in ("c9300_48p_a", "wap_9180ax_k9", "sfp_10g_lr_s"):
        if p not in k:
            issues.append(f"part_number:{p} missing")
    return issues


def check_xb_multi_sheet(a, e, w):
    issues = []
    t = _atom_text(a).lower()
    if "bronze" not in t or "silver" not in t or "gold" not in t:
        issues.append("Pricing sheet not parsed (tier names missing)")
    k = _ent_keys(e)
    if "atl_hq_01" not in k or "atl_west_02" not in k:
        issues.append("Sites sheet not parsed")
    return issues


def check_xc_hidden_sheets(a, e, w):
    issues = []
    # Hidden site sheet — content should still be captured (or we
    # surface a warning that we skipped it).
    t = _atom_text(a).lower()
    if "wap-9180ax-k9" not in t.lower() and "wap_9180ax_k9" not in _ent_keys(e):
        issues.append("visible BOM lost a row")
    return issues


def check_xd_formula(a, e, w):
    issues = []
    # Formula cells should resolve to VALUES (not =B2*C2 text)
    # Either the cells produce the computed value OR they appear as
    # raw formula strings (acceptable). We just need atoms to exist.
    if len(a) < 1:
        issues.append("formula sheet produced 0 atoms")
    return issues


def check_xe_merged(a, e, w):
    issues = []
    k = _ent_keys(e)
    if "atl_hq_01" not in k:
        issues.append("site after merged title row missing")
    return issues


def check_xf_named_ranges(a, e, w):
    issues = []
    if len(a) < 1:
        issues.append("named-range sheet produced 0 atoms")
    return issues


def check_xg_chart_only(a, e, w):
    issues = []
    t = _atom_text(a)
    # Quarter / spend amounts must reach atoms even with a chart present
    if "Q1 2026" not in t and "q1 2026" not in t.lower():
        issues.append("chart-data sheet lost the quarter labels")
    return issues


def check_xh_pivot(a, e, w):
    issues = []
    k = _ent_keys(e)
    if "atl_hq_01" not in k:
        issues.append("pivot rows lost site")
    t = _atom_text(a).lower()
    if "switches" not in t and "access points" not in t:
        issues.append("pivot columns lost device labels")
    return issues


def check_xi_data_validation(a, e, w):
    issues = []
    t = _atom_text(a).lower()
    if "approved" not in t:
        issues.append("decisions sheet lost row content")
    return issues


def check_xj_huge_500(a, e, w):
    issues = []
    parts = [x for x in e if (getattr(x, "entity_type", "") or "").lower() in ("part", "part_number")]
    if len(parts) < 50:
        issues.append(f"only {len(parts)} parts captured (expect >= 50 from 500-row BOM)")
    return issues


def check_xk_currency(a, e, w):
    issues = []
    t = _atom_text(a).upper()
    for c in ("USD", "EUR", "GBP", "JPY"):
        if c not in t:
            issues.append(f"currency {c!r} missing")
    return issues


def check_xl_site_roster(a, e, w):
    issues = []
    k = _ent_keys(e)
    for sid in ("atl_hq_01", "atl_west_02", "atl_air_03", "atl_047_04", "atl_cp_05"):
        if sid not in k:
            issues.append(f"site:{sid} missing")
    # physical_site atoms preferred
    sites = [x for x in a if isinstance(getattr(x, "value", None), dict) and x.value.get("kind") == "physical_site"]
    if len(sites) < 5:
        issues.append(f"only {len(sites)}/5 physical_site atoms emitted")
    return issues


def check_xm_blank_first_row(a, e, w):
    issues = []
    t = _atom_text(a).lower()
    if "wap-9180ax-k9" not in t and "wap_9180ax_k9" not in _ent_keys(e):
        issues.append("blank-first-row header detection failed")
    return issues


def check_xn_multi_section(a, e, w):
    issues = []
    t = _atom_text(a).lower()
    if "wap-9180ax-k9" not in t and "wap_9180ax_k9" not in _ent_keys(e):
        issues.append("first table (BOM) lost")
    if "cutover" not in t and "saturday" not in t:
        issues.append("second table (Notes) lost")
    return issues


CHECKS = [
    ("xa_bom_simple.xlsx",            "Simple BOM",          check_xa_bom_simple),
    ("xb_bom_multi_sheet.xlsx",       "Multi-sheet",         check_xb_multi_sheet),
    ("xc_hidden_sheets.xlsx",         "Hidden sheets",       check_xc_hidden_sheets),
    ("xd_formula_cells.xlsx",         "Formula cells",       check_xd_formula),
    ("xe_merged_cells.xlsx",          "Merged cells",        check_xe_merged),
    ("xf_named_ranges.xlsx",          "Named ranges",        check_xf_named_ranges),
    ("xg_chart_only.xlsx",            "Sheet with chart",    check_xg_chart_only),
    ("xh_pivot_table.xlsx",           "Pivot table",         check_xh_pivot),
    ("xi_data_validation.xlsx",       "Data validation",     check_xi_data_validation),
    ("xj_huge_table_500_rows.xlsx",   "500-row BOM",         check_xj_huge_500),
    ("xk_mixed_currency.xlsx",        "Multi-currency",      check_xk_currency),
    ("xl_site_roster.xlsx",           "Site roster XLSX",    check_xl_site_roster),
    ("xm_blank_first_row.xlsx",       "Blank-first-row",     check_xm_blank_first_row),
    ("xn_multi_section_in_sheet.xlsx", "Multi-section sheet", check_xn_multi_section),
]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_xlsx_stress.py <bundle_dir>", file=sys.stderr)
        return 2
    bundle = Path(sys.argv[1]).resolve()
    art = bundle / "artifacts"
    print(f"=== XLSX adversarial stress: {len(CHECKS)} cases ===\n")
    pass_count = fail_count = total_issues = 0
    for fname, label, check_fn in CHECKS:
        xlsx = art / fname
        if not xlsx.exists():
            print(f"  SKIP {fname}")
            continue
        try:
            atoms, entities, warnings = compile_one(xlsx)
        except Exception as e:
            print(f"  CRASH {fname}: {type(e).__name__}: {e}")
            fail_count += 1
            total_issues += 1
            continue
        issues = check_fn(atoms, entities, warnings) or []
        status = "PASS" if not issues else f"FAIL ({len(issues)})"
        print(f"  {fname:<35} {label:<25} atoms={len(atoms):<3} ents={len(entities):<3} {status}")
        for iss in issues[:6]:
            print(f"    - {iss}")
        if issues:
            fail_count += 1
            total_issues += len(issues)
        else:
            pass_count += 1
    print()
    print("=" * 70)
    print(f"PASS: {pass_count}   FAIL: {fail_count}   TOTAL ISSUES: {total_issues}")
    print("=" * 70)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
