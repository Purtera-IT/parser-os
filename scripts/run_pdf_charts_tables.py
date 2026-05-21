"""Run the chart/table PDF bundle through compile + envelope."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def compile_one(pdf: Path):
    from app.core.compiler import compile_project
    base = Path(tempfile.gettempdir()) / f"pdf_ct_{pdf.stem}"
    art = base / "artifacts"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    art.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, art / pdf.name)
    r = compile_project(
        project_dir=base, domain_pack=None,
        use_cache=False, allow_errors=True, allow_unverified_receipts=True,
    )
    return list(r.atoms), list(r.entities), list(r.warnings)


def _ent_keys(entities) -> str:
    return " ".join((getattr(e, "canonical_key", "") or "").lower() for e in entities)


def _atom_text(atoms) -> str:
    return " ".join(getattr(a, "raw_text", "") or "" for a in atoms)


def check_a_bar_chart_table(a, e, w):
    iss = []
    t = _atom_text(a).lower()
    for q in ("q1 2026", "q2 2026", "q3 2026", "q4 2026"):
        if q not in t:
            iss.append(f"quarter '{q}' missing")
    k = _ent_keys(e)
    for sid in ("atl_hq_01", "atl_west_02", "atl_air_03"):
        if sid not in k:
            iss.append(f"site:{sid} missing")
    return iss


def check_b_pie(a, e, w):
    iss = []
    t = _atom_text(a).lower()
    for cat in ("switches", "access points", "cameras"):
        if cat not in t:
            iss.append(f"category '{cat}' missing")
    # money entities
    money = [x for x in e if (getattr(x, "entity_type", "") or "") == "money"]
    if len(money) < 3:
        iss.append(f"only {len(money)} money entities (expected >= 3)")
    return iss


def check_c_rotated(a, e, w):
    iss = []
    k = _ent_keys(e)
    for sid in ("atl_hq_01", "atl_west_02", "atl_air_03"):
        if sid not in k:
            iss.append(f"site:{sid} missing")
    for part in ("c9300", "wap_9180ax", "sfp_10g_lr"):
        if part not in k.lower():
            iss.append(f"part containing '{part}' missing")
    return iss


def check_d_nested(a, e, w):
    iss = []
    k = _ent_keys(e)
    if not any(sid in k for sid in ("atl_hq_01", "atl_west_02", "atl_air_03")):
        iss.append("no sites captured from nested-header table")
    return iss


def check_e_image_only(a, e, w):
    iss = []
    # Image-only PDF — no text layer expected; should not crash
    return iss


def check_f_chart_sla(a, e, w):
    iss = []
    t = _atom_text(a).lower()
    for tier in ("bronze", "silver", "gold"):
        if tier not in t:
            iss.append(f"tier '{tier}' missing")
    # SLA constraint atoms expected
    constraints = [x for x in a if str(x.atom_type) == "AtomType.constraint"]
    # Either constraints OR scope items that contain "hours"/"uptime"
    if not constraints and "hours" not in t and "99." not in t:
        iss.append("no SLA constraint signal in atoms")
    return iss


def check_g_color_coded(a, e, w):
    iss = []
    k = _ent_keys(e)
    sites_found = sum(1 for sid in ("atl_hq_01", "atl_west_02", "atl_air_03", "atl_cp_05") if sid in k)
    if sites_found < 3:
        iss.append(f"only {sites_found}/4 sites from RAG status table")
    return iss


def check_h_milestone(a, e, w):
    iss = []
    t = _atom_text(a).lower()
    for mil in ("kickoff", "cutover", "atp", "hypercare"):
        if mil not in t:
            iss.append(f"milestone keyword '{mil}' missing")
    return iss


def check_i_split_table(a, e, w):
    iss = []
    parts = [x for x in e if (getattr(x, "entity_type", "") or "").lower() in ("part", "part_number")]
    if len(parts) < 15:
        iss.append(f"only {len(parts)}/40 parts captured from cross-page table")
    return iss


def check_j_summary_detail(a, e, w):
    iss = []
    k = _ent_keys(e)
    parts = ("wap_9180ax", "c9300", "sfp_10g_lr")
    found = sum(1 for p in parts if p in k.lower())
    if found < 2:
        iss.append(f"only {found}/3 detail-page parts captured")
    # Sites and a money figure expected
    if "atl_hq_01" not in k:
        iss.append("ATL-HQ-01 missing across pages")
    return iss


def check_k_callouts(a, e, w):
    iss = []
    k = _ent_keys(e)
    if not any(sid in k for sid in ("atl_hq_01", "atl_west_02", "atl_air_03", "atl_cp_05")):
        iss.append("no site captured from callout text")
    return iss


def check_l_subtotals(a, e, w):
    iss = []
    t = _atom_text(a)
    if "$293,540" not in t and "293540" not in _ent_keys(e):
        iss.append("grand total $293,540 missing")
    # At least one subtotal should be present
    if "subtotal" not in t.lower():
        iss.append("subtotal labels lost")
    return iss


CHECKS = [
    ("ct_a_bar_chart_plus_table.pdf",     "Bar chart + quarterly table", check_a_bar_chart_table),
    ("ct_b_pie_chart_breakdown.pdf",      "Pie chart + breakdown",       check_b_pie),
    ("ct_c_rotated_table.pdf",            "Landscape 7-col BOM",         check_c_rotated),
    ("ct_d_nested_subheaders.pdf",        "Nested subheaders",           check_d_nested),
    ("ct_e_image_only_table.pdf",         "Image-only table (no crash)", check_e_image_only),
    ("ct_f_chart_and_sla_matrix.pdf",     "Chart + SLA matrix",          check_f_chart_sla),
    ("ct_g_color_coded_table.pdf",        "RAG status table",            check_g_color_coded),
    ("ct_h_milestone_timeline_table.pdf", "Milestone timeline",          check_h_milestone),
    ("ct_i_split_table_across_pages.pdf", "Cross-page 40-row table",     check_i_split_table),
    ("ct_j_summary_then_detail.pdf",      "Summary + detail BOM",        check_j_summary_detail),
    ("ct_k_charts_with_callouts.pdf",     "Chart with callouts",         check_k_callouts),
    ("ct_l_pricing_with_subtotals.pdf",   "Subtotals + grand total",     check_l_subtotals),
]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_pdf_charts_tables.py <bundle_dir>", file=sys.stderr)
        return 2
    bundle = Path(sys.argv[1]).resolve()
    art = bundle / "artifacts"
    print(f"=== Chart/table PDF stress: {len(CHECKS)} cases ===\n")
    pass_count = fail_count = total_issues = 0
    for fname, label, check_fn in CHECKS:
        pdf = art / fname
        if not pdf.exists():
            print(f"  SKIP {fname}")
            continue
        try:
            atoms, entities, warnings = compile_one(pdf)
        except Exception as e:
            print(f"  CRASH {fname}: {type(e).__name__}: {e}")
            fail_count += 1
            total_issues += 1
            continue
        issues = check_fn(atoms, entities, warnings) or []
        status = "PASS" if not issues else f"FAIL ({len(issues)})"
        print(f"  {fname:<40} {label:<28} atoms={len(atoms):<3} ents={len(entities):<3} {status}")
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
