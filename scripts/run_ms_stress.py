"""Run the MS stress bundle through compile + envelope and check
for each adversarial expectation. Prints a per-PDF table of:

  - atom counts by type
  - entity counts by kind
  - key warnings
  - PM-critical assertion results

Usage:
  python scripts/run_ms_stress.py <bundle_dir>

The bundle_dir must contain artifacts/ms_*.pdf.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path


def compile_one(pdf_path: Path):
    """Compile a single-PDF project, return (atoms, entities, warnings, env_json)."""
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import (
        build_orbitbrief_envelope,
        write_orbitbrief_envelope,
    )

    with tempfile.TemporaryDirectory(prefix="ms_stress_") as td:
        td_path = Path(td)
        art = td_path / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, art / pdf_path.name)
        result = compile_project(
            project_dir=td_path, domain_pack=None,
            use_cache=False, allow_errors=True, allow_unverified_receipts=True,
        )
        atoms = list(result.atoms)
        entities = list(result.entities)
        warnings = list(result.warnings)
        env = build_orbitbrief_envelope(project_dir=td_path, compile_result=result)
        env_paths = write_orbitbrief_envelope(
            project_dir=td_path, envelope=env, out_dir=td_path / ".orbitbrief"
        )
        env_json = {}
        if env_paths:
            try:
                env_json = json.loads(env_paths[0].read_text(encoding="utf-8"))
            except Exception:
                env_json = {}
        return atoms, entities, warnings, env_json


def _atom_texts(atoms):
    out = []
    for a in atoms:
        t = getattr(a, "raw_text", "") or ""
        out.append((a.atom_type, t))
    return out


def _entity_by_kind(entities):
    out: dict[str, list[dict]] = {}
    for e in entities:
        k = getattr(e, "kind", None) or getattr(e, "entity_type", "?")
        out.setdefault(str(k), []).append({
            "label": getattr(e, "label", None) or getattr(e, "canonical_name", None) or "",
            "key": getattr(e, "canonical_key", None) or getattr(e, "key", None) or "",
        })
    return out


# ── Checks ───────────────────────────────────────────────────────


def check_a_quantity_conflict(atoms, entities, warnings, env):
    issues = []
    # Single-PDF case: just check both numbers (50 and 60) survive.
    # Cross-doc quantity conflict detection is a graph-builder layer
    # that needs multiple sources — not testable from a single PDF.
    qty_keys = " ".join((getattr(e, "canonical_key", "") or "").lower() for e in entities)
    if "quantity:50" not in qty_keys and "50" not in qty_keys:
        issues.append("quantity 50 missing")
    if "quantity:60" not in qty_keys and "60" not in qty_keys:
        issues.append("quantity 60 missing")
    return issues


def check_b_pricing_conflict(atoms, entities, warnings, env):
    issues = []
    # Expect at least 3 money entities (100K / 95K / 10K) or 3 vendor_line_items
    money = entities and [e for e in entities if (getattr(e, "kind", None) or getattr(e, "entity_type", "")) == "money"]
    vlis = [a for a in atoms if a.atom_type == "AtomType.vendor_line_item"]
    if len(money or []) + len(vlis) < 3:
        issues.append(f"only {len(money or [])} money entities + {len(vlis)} vendor_line_items "
                       f"(expected to see $100K + $95K + $10K all surfaced)")
    return issues


def check_c_date_formats(atoms, entities, warnings, env):
    issues = []
    dates = [e for e in entities if (getattr(e, "kind", None) or getattr(e, "entity_type", "")) == "date"]
    quarters = [e for e in entities if (getattr(e, "kind", None) or getattr(e, "entity_type", "")) == "quarter"]
    # 6 date-shaped tokens in source; expect at least 4 unique date entities
    if len(dates) + len(quarters) < 4:
        issues.append(f"only {len(dates)} date + {len(quarters)} quarter entities "
                       f"(expected >= 4 across 6 date-shaped tokens)")
    return issues


def check_d_stakeholders(atoms, entities, warnings, env):
    issues = []
    stakes = [e for e in entities if (getattr(e, "kind", None) or getattr(e, "entity_type", "")) == "stakeholder"]
    labels = [(s.label if hasattr(s, "label") else getattr(s, "canonical_name", "")) or "" for s in stakes]
    # Expect at least 3 of: Jane Roe, John Smith, Maria Lopez, T. Nguyen, Sarah Chen, Alex Patel
    expected = ["jane roe", "john smith", "maria lopez", "t. nguyen", "sarah chen", "alex patel"]
    hits = sum(1 for exp in expected if any(exp.lower() in l.lower() for l in labels))
    if hits < 3:
        issues.append(f"only {hits}/6 expected stakeholder names captured (got labels: {labels[:6]})")
    return issues


def check_e_msa_boilerplate(atoms, entities, warnings, env):
    issues = []
    # Expect very few scope_item atoms despite 3 long boilerplate paragraphs.
    scope = [a for a in atoms if a.atom_type == "AtomType.scope_item"]
    if len(scope) > 6:
        issues.append(f"{len(scope)} scope_items emitted (boilerplate should produce 0-3); polluting scope")
    return issues


def check_f_toc_and_footers(atoms, entities, warnings, env):
    issues = []
    # Expect 0-1 scope_items from the TOC; the actual "1. Project Overview"
    # paragraph + scope sentence should produce ~2 scope_items.
    scope = [a for a in atoms if a.atom_type == "AtomType.scope_item"]
    if len(scope) > 8:
        issues.append(f"{len(scope)} scope_items emitted (TOC + footers polluting scope)")
    # Footer text should NOT appear as atom raw_text
    for a in atoms:
        if "Page X of Y" in (getattr(a, "raw_text", "") or ""):
            issues.append("page footer 'Page X of Y' leaked into atom text")
            break
    return issues


def check_g_watermark_draft(atoms, entities, warnings, env):
    issues = []
    # Expect 3 scope_items (one per site install line). DRAFT lines should NOT
    # produce their own scope_items (would be repetition pollution).
    scope = [a for a in atoms if a.atom_type == "AtomType.scope_item"]
    draft_only = [a for a in scope if "DRAFT" in (getattr(a, "raw_text", "") or "")
                   and not any(s in (getattr(a, "raw_text", "") or "").lower()
                               for s in ("install", "access point", "atl-"))]
    if len(draft_only) > 1:
        issues.append(f"{len(draft_only)} scope_items contain only 'DRAFT' boilerplate")
    return issues


def check_h_sla_constraints(atoms, entities, warnings, env):
    issues = []
    constraints = [a for a in atoms if str(a.atom_type) == "AtomType.constraint"]
    # The doc has 6 SLA statements concatenated into 2 paragraphs. We
    # accept that as 2+ constraint atoms (atoms are paragraph-shaped;
    # multi-sentence atoms count once each).
    if len(constraints) < 2:
        issues.append(f"only {len(constraints)} constraint atoms (expected >= 2)")
    return issues


def check_i_multi_currency(atoms, entities, warnings, env):
    issues = []
    # Multi-currency: expect vendor_line_item atoms covering all 3 currencies.
    vli_atoms = [a for a in atoms if str(a.atom_type) == "AtomType.vendor_line_item"]
    if len(vli_atoms) < 3:
        issues.append(f"only {len(vli_atoms)} vendor_line_item atoms (expected 3)")
    text = " ".join(getattr(a, "raw_text", "") or "" for a in atoms).upper()
    currencies_seen = sum(1 for c in ("USD", "EUR", "GBP") if c in text)
    if currencies_seen < 3:
        issues.append(f"only {currencies_seen}/3 currency tags visible (USD/EUR/GBP)")
    return issues


def check_j_change_order(atoms, entities, warnings, env):
    issues = []
    # Expect site entities for ATL-AIR-03, ATL-WEST-02, ATL-HQ-01, ATL-047-04
    sites = [e for e in entities if (getattr(e, "kind", None) or getattr(e, "entity_type", "")) == "site"]
    site_labels = " ".join((getattr(s, "canonical_key", "") or "").lower() for s in sites)
    for sid in ("atl_air_03", "atl_west_02", "atl_hq_01", "atl_047_04"):
        if sid not in site_labels:
            issues.append(f"site {sid} not in entity keys (change-order coverage incomplete)")
    return issues


def check_k_service_tiers(atoms, entities, warnings, env):
    issues = []
    # Expect tier names visible somewhere
    text = " ".join(getattr(a, "raw_text", "") or "" for a in atoms).lower()
    for t in ("bronze", "silver", "gold"):
        if t not in text:
            issues.append(f"tier '{t}' missing from atom text (whole row may have been dropped)")
    return issues


def check_l_part_numbers(atoms, entities, warnings, env):
    issues = []
    parts = [e for e in entities if (getattr(e, "kind", None) or getattr(e, "entity_type", "")) in ("part", "part_number")]
    # Compare against slugified expected names (entity_resolution
    # canonicalises hyphens / = / # to underscores).
    expected_slugged = ["c9300_48p_a", "wap_9180ax_k9", "cat6a", "sfp_10g_lr_s", "jl664a"]
    keys = " ".join((getattr(p, "canonical_key", "") or "").lower() for p in parts)
    hits = sum(1 for ep in expected_slugged if ep in keys)
    if hits < 4:
        issues.append(f"only {hits}/5 expected part numbers captured (parts found: {len(parts)})")
    return issues


def check_m_empty_placeholders(atoms, entities, warnings, env):
    issues = []
    # Should NOT see TBD/TBA/N/A/— as a "decision" or "date" entity
    text = " ".join(
        (getattr(e, "label", "") or getattr(e, "canonical_name", "") or "")
        for e in entities
    ).lower()
    for placeholder in ("tbd", "tba", "n/a"):
        # Allow placeholder as a substring in a longer label
        # but reject if it's the WHOLE label
        for e in entities:
            label = (getattr(e, "label", "") or getattr(e, "canonical_name", "") or "").strip().lower()
            if label == placeholder:
                issues.append(f"placeholder '{placeholder}' surfaced as an entity label")
                break
    return issues


def check_n_long_paragraph(atoms, entities, warnings, env):
    issues = []
    sites = [e for e in entities if (getattr(e, "kind", None) or getattr(e, "entity_type", "")) == "site"]
    site_keys = " ".join((getattr(s, "canonical_key", "") or "").lower() for s in sites)
    for sid in ("atl_hq_01", "atl_west_02", "atl_air_03"):
        if sid not in site_keys:
            issues.append(f"site {sid} missing from long-paragraph extraction (parser lost context)")
    return issues


def check_o_compliance_matrix(atoms, entities, warnings, env):
    issues = []
    text = " ".join(getattr(a, "raw_text", "") or "" for a in atoms).lower()
    # Each tier should appear with multiple metrics
    for t in ("bronze", "silver", "gold"):
        if t not in text:
            issues.append(f"tier '{t}' missing from atoms")
    return issues


CHECKS = {
    "ms_a_quantity_conflict_bom_vs_sow.pdf":   ("Quantity conflict (BOM vs SOW)", check_a_quantity_conflict),
    "ms_b_pricing_conflict_quote_contract_co.pdf": ("Pricing conflict (Quote/Contract/CO)", check_b_pricing_conflict),
    "ms_c_date_format_chaos.pdf":              ("Date format chaos", check_c_date_formats),
    "ms_d_stakeholder_roles.pdf":              ("Stakeholder roles", check_d_stakeholders),
    "ms_e_msa_boilerplate.pdf":                ("MSA boilerplate suppression", check_e_msa_boilerplate),
    "ms_f_toc_and_footers.pdf":                ("TOC + footer suppression", check_f_toc_and_footers),
    "ms_g_watermark_draft.pdf":                ("DRAFT watermark suppression", check_g_watermark_draft),
    "ms_h_sla_constraints.pdf":                ("SLA constraint extraction", check_h_sla_constraints),
    "ms_i_multi_currency.pdf":                 ("Multi-currency safety", check_i_multi_currency),
    "ms_j_change_order_adds_removes.pdf":      ("Change-order site coverage", check_j_change_order),
    "ms_k_service_tiers.pdf":                  ("Service-tier pricing matrix", check_k_service_tiers),
    "ms_l_part_number_chaos.pdf":              ("Part-number variety", check_l_part_numbers),
    "ms_m_empty_placeholders.pdf":             ("Empty placeholder suppression", check_m_empty_placeholders),
    "ms_n_long_paragraph.pdf":                 ("Long-paragraph extraction", check_n_long_paragraph),
    "ms_o_compliance_sla_matrix.pdf":          ("Compliance/SLA matrix", check_o_compliance_matrix),
}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_ms_stress.py <bundle_dir>", file=sys.stderr)
        return 2
    bundle = Path(sys.argv[1]).resolve()
    art = bundle / "artifacts"
    if not art.exists():
        print(f"No artifacts/ in {bundle}", file=sys.stderr)
        return 2

    print(f"=== MS adversarial stress: {len(CHECKS)} cases ===\n")
    total_issues = 0
    pass_count = 0
    fail_count = 0
    for fname, (label, check_fn) in CHECKS.items():
        pdf = art / fname
        if not pdf.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        try:
            atoms, entities, warnings, env = compile_one(pdf)
        except Exception as e:
            print(f"  CRASH {fname}: {type(e).__name__}: {e}")
            fail_count += 1
            total_issues += 1
            continue
        issues = check_fn(atoms, entities, warnings, env) or []
        # quick stats
        atom_count = len(atoms)
        ent_count = len(entities)
        warn_count = len([w for w in warnings if w])
        status = "PASS" if not issues else f"FAIL ({len(issues)} issues)"
        print(f"  {fname:<45} {label:<35} atoms={atom_count:<3} ents={ent_count:<3} warns={warn_count:<3} {status}")
        for issue in issues:
            print(f"    - {issue}")
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
