"""Round 2 adversarial check runner.

Each case asserts a specific PM-critical extraction the parser MUST
deliver universally, regardless of the source PDF's quirks.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure stdout can handle unicode (test cases include German / Japanese /
# French / Spanish characters).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass


def compile_one(pdf_path: Path):
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope
    with tempfile.TemporaryDirectory(prefix="r2_stress_") as td:
        td_path = Path(td)
        art = td_path / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, art / pdf_path.name)
        r = compile_project(
            project_dir=td_path, domain_pack=None,
            use_cache=False, allow_errors=True, allow_unverified_receipts=True,
        )
        return list(r.atoms), list(r.entities), list(r.warnings)


def _entity_keys(entities):
    return " ".join((getattr(e, "canonical_key", "") or "").lower() for e in entities)


def _atom_text(atoms):
    return " ".join(getattr(a, "raw_text", "") or "" for a in atoms)


def _kinds(entities):
    out: dict[str, int] = {}
    for e in entities:
        k = getattr(e, "kind", None) or getattr(e, "entity_type", "?")
        out[str(k)] = out.get(str(k), 0) + 1
    return out


def check_a_multi_site(atoms, entities, warnings):
    issues = []
    keys = _entity_keys(entities)
    for sid in ("atl_hq_01", "atl_west_02", "atl_air_03", "atl_cp_05"):
        if sid not in keys:
            issues.append(f"site:{sid} missing")
    # Quantity 50 + 20 both expected
    text = _atom_text(atoms)
    if "50" not in text or "20" not in text:
        issues.append(f"quantity 50 or 20 missing from atom text")
    return issues


def check_b_range_qty(atoms, entities, warnings):
    issues = []
    keys = _entity_keys(entities)
    # Expect at least one quantity entity per sentence: 40, 60, 50, 75, 100
    # (any 2+ of these is acceptable)
    found = sum(1 for n in ("40", "60", "50", "75", "100")
                if f"quantity:{n}" in keys)
    if found < 3:
        issues.append(f"only {found}/5 range quantities captured")
    return issues


def check_c_negative_credit(atoms, entities, warnings):
    issues = []
    keys = _entity_keys(entities)
    # $5,000, $10,000, $2,500, $7,500 — all should be money entities
    found = sum(1 for n in ("5000", "10000", "2500", "7500")
                if f"money:{n}" in keys)
    if found < 2:
        issues.append(f"only {found}/4 credit/discount amounts captured")
    return issues


def check_d_cross_reference(atoms, entities, warnings):
    issues = []
    # P1 response SLA should land as a constraint
    constraints = [a for a in atoms if str(a.atom_type) == "AtomType.constraint"]
    if len(constraints) < 1:
        issues.append("no constraint atoms (SLA defined via cross-reference not picked up)")
    return issues


def check_e_multi_column(atoms, entities, warnings):
    issues = []
    text = _atom_text(atoms).lower()
    # Both columns must contribute: "50 wireless" (left) + "245,000" (right)
    if "wireless access points" not in text:
        issues.append("left column scope text missing")
    if "245,000" not in text and "245000" not in text:
        issues.append("right column pricing text missing")
    return issues


def check_f_footnotes(atoms, entities, warnings):
    issues = []
    text = _atom_text(atoms).lower()
    if "50 access points" not in text:
        issues.append("main body scope lost")
    if "wap-9180ax-k9" not in text and "wap_9180ax_k9" not in text:
        # Footnote part number should still survive somewhere
        issues.append("footnote part number lost (acceptable but worth noting)")
    return issues


def check_g_unicode(atoms, entities, warnings):
    issues = []
    keys = _entity_keys(entities)
    text = _atom_text(atoms)
    # All four sites should appear in entity keys
    for sid in ("muc_hq_01", "tyo_hq_01", "par_hq_01", "mex_hq_01"):
        if sid not in keys:
            issues.append(f"site:{sid} (unicode address) missing")
    # Latin-1 / Latin-Extended characters should survive (reportlab's
    # default font can render these). CJK fonts are not embedded by
    # default — those test as a separate raster path.
    for char in ("ü", "é"):
        if char not in text:
            issues.append(f"unicode char {char!r} missing from atom text")
    return issues


def check_h_rotated_landscape(atoms, entities, warnings):
    issues = []
    text = _atom_text(atoms).lower()
    keys = _entity_keys(entities)
    # All three sites + a couple part numbers should land
    for sid in ("atl_hq_01", "atl_west_02", "atl_air_03"):
        if sid not in keys:
            issues.append(f"site:{sid} from landscape PDF missing")
    if "c9300" not in text.lower():
        issues.append("part text 'C9300' from landscape PDF missing")
    return issues


def check_i_strikethrough(atoms, entities, warnings):
    issues = []
    keys = _entity_keys(entities)
    # Both 50 and 60 should be captured (old + new values)
    if "quantity:50" not in keys or "quantity:60" not in keys:
        issues.append("strikethrough quantities 50/60 not both captured")
    # Old vs new pricing
    money_keys = [k for k in keys.split() if k.startswith("money:")]
    if "money:245000" not in money_keys or "money:255000" not in money_keys:
        issues.append("strikethrough pricing 245000/255000 not both captured")
    return issues


def check_j_multi_currency_conversion(atoms, entities, warnings):
    issues = []
    keys = _entity_keys(entities)
    # USD 100K, EUR 95K, GBP 25K, USD 32K — money entities should cover at least 3
    found = sum(1 for n in ("100000", "95000", "25000", "32000")
                if f"money:{n}" in keys)
    if found < 3:
        issues.append(f"only {found}/4 cross-currency amounts captured")
    return issues


def check_k_discount(atoms, entities, warnings):
    issues = []
    keys = _entity_keys(entities)
    text = _atom_text(atoms).lower()
    # Money: $100,000 (list), $90,000 (net), $200,000 (volume), 10%/5% discounts
    if "money:100000" not in keys or "money:90000" not in keys:
        issues.append("list price or net price after discount missing")
    if "%" not in text:
        issues.append("discount percentage symbol missing from atoms")
    return issues


def check_l_numbered_subsections(atoms, entities, warnings):
    issues = []
    # Each lowest-level subsection should produce an atom
    text = _atom_text(atoms).lower()
    for marker in ("switches", "access points", "cat6a", "dna-e", "managed service"):
        if marker not in text:
            issues.append(f"deep subsection content '{marker}' missing")
    return issues


def check_m_implicit_references(atoms, entities, warnings):
    issues = []
    text = _atom_text(atoms).lower()
    for marker in ("customer", "contractor", "cutover"):
        if marker not in text:
            issues.append(f"implicit-ref subject '{marker}' missing")
    return issues


def check_n_conditional_clauses(atoms, entities, warnings):
    issues = []
    constraints = [a for a in atoms if str(a.atom_type) == "AtomType.constraint"]
    if len(constraints) < 1:
        issues.append("no constraint atoms for conditional SLA clauses")
    # The text uses percentages ("5% credit", "10% credit") not dollar
    # amounts. Both percent thresholds AND the response/uptime numbers
    # should reach the atom text.
    text = _atom_text(atoms).lower()
    for marker in ("99.9", "4 business hours", "5%", "10%"):
        if marker not in text:
            issues.append(f"conditional threshold '{marker}' missing from atom text")
    return issues


def check_o_time_zones(atoms, entities, warnings):
    issues = []
    text = _atom_text(atoms).upper()
    for tz in ("PT", "ET", "PST"):
        if tz not in text:
            issues.append(f"timezone abbrev {tz!r} missing")
    return issues


def check_p_page_break_split(atoms, entities, warnings):
    issues = []
    keys = _entity_keys(entities)
    # Both halves of the split should contribute: quantity 50 AND all 3 sites
    if "quantity:50" not in keys:
        issues.append("quantity 50 (first page) missing")
    for sid in ("atl_hq_01", "atl_west_02", "atl_air_03"):
        if sid not in keys:
            issues.append(f"site:{sid} (second page) missing")
    return issues


def check_q_hybrid_layout(atoms, entities, warnings):
    issues = []
    text = _atom_text(atoms).lower()
    keys = _entity_keys(entities)
    # Bullets, table, and prose should all contribute
    if "access point" not in text:
        issues.append("bullet content (access points) missing")
    if "$60,000" not in text and "money:60000" not in keys:
        issues.append("table cell value missing")
    if "continental us" not in text:
        issues.append("prose footer note missing")
    return issues


def check_r_empty_pdf(atoms, entities, warnings):
    issues = []
    # Should NOT crash — 0 atoms is acceptable
    return issues


def check_s_only_headers(atoms, entities, warnings):
    issues = []
    # Should not crash — headers can produce 0 body atoms
    return issues


def check_t_huge_part_numbers(atoms, entities, warnings):
    issues = []
    parts = [e for e in entities if (getattr(e, "kind", None) or getattr(e, "entity_type", "")) in ("part", "part_number")]
    if len(parts) < 20:
        issues.append(f"only {len(parts)}/30 part numbers captured in large BOM")
    # Each part should have a Qty paired
    qtys = [e for e in entities if (getattr(e, "kind", None) or getattr(e, "entity_type", "")) == "quantity"]
    if len(qtys) < 10:
        issues.append(f"only {len(qtys)} quantity entities (expect >= 10 across 30 rows)")
    return issues


CHECKS = [
    ("r2_a_multi_site_one_row.pdf",          "Multi-site one-row scope", check_a_multi_site),
    ("r2_b_range_qty.pdf",                   "Range / approximate qtys", check_b_range_qty),
    ("r2_c_negative_credit.pdf",             "Negative / credit amounts", check_c_negative_credit),
    ("r2_d_cross_reference.pdf",             "Cross-reference SLA",       check_d_cross_reference),
    ("r2_e_multi_column_layout.pdf",         "Multi-column layout",       check_e_multi_column),
    ("r2_f_footnotes.pdf",                   "Footnote separation",       check_f_footnotes),
    ("r2_g_unicode_addresses.pdf",           "Unicode addresses",         check_g_unicode),
    ("r2_h_rotated_landscape.pdf",           "Landscape page BOM",        check_h_rotated_landscape),
    ("r2_i_strikethrough_revisions.pdf",     "Revisions (strikethrough)", check_i_strikethrough),
    ("r2_j_multi_currency_conversion.pdf",   "Cross-currency conversion", check_j_multi_currency_conversion),
    ("r2_k_discount_percentage.pdf",         "Discount + percentage",     check_k_discount),
    ("r2_l_numbered_subsections.pdf",        "Deep numbered subsections", check_l_numbered_subsections),
    ("r2_m_implicit_references.pdf",         "Implicit references",       check_m_implicit_references),
    ("r2_n_conditional_clauses.pdf",         "Conditional SLA clauses",   check_n_conditional_clauses),
    ("r2_o_time_zones.pdf",                  "Time-zone notation",        check_o_time_zones),
    ("r2_p_page_break_split.pdf",            "Page-break sentence split", check_p_page_break_split),
    ("r2_q_hybrid_bullets_tables.pdf",       "Hybrid bullets+table+prose", check_q_hybrid_layout),
    ("r2_r_empty_pdf.pdf",                   "Empty PDF (no crash)",       check_r_empty_pdf),
    ("r2_s_only_headers.pdf",                "Only-headers PDF (no crash)", check_s_only_headers),
    ("r2_t_huge_part_numbers_table.pdf",     "30-row BOM",                check_t_huge_part_numbers),
]


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_ms_stress_round2.py <bundle_dir>", file=sys.stderr)
        return 2
    bundle = Path(sys.argv[1]).resolve()
    art = bundle / "artifacts"
    print(f"=== Round 2 adversarial stress: {len(CHECKS)} cases ===\n")
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
        kinds = _kinds(entities)
        status = "PASS" if not issues else f"FAIL ({len(issues)})"
        print(f"  {fname:<45} {label:<32} atoms={len(atoms):<2} ents={len(entities):<2} {status}")
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
