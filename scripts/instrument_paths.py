"""Instrumentation: measure what each parser-os emission path actually
contributes on OPTBOT + APS. ZERO code changes — pure observation.

Goal: before deleting/changing any "mess" path, prove how many atoms it
emits and whether they're real or garbage. If a path emits 0 atoms → safe
to delete. If it emits real atoms → understand WHY before changing.

Outputs:
- per-path atom_count
- sample raw_text per path
- sites found per path (site_id values from value.id/site_id)

Coverage today (5 paths to instrument):
  P1: _fitz_site_roster_fallback PRIMARY path (looks_like_site_roster passes,
      extract_site_roster emits)
  P2: _fitz_site_roster_fallback FALLBACK path (looks_like_site_roster fails,
      table_schema_registry.emit_atoms_for_schema fires)
  P3: _text_based_site_roster_extract (text-only when fitz fails)
  P4: LLM site_clusters (chunked or retrieved)
  P5: LLM site_clusters bridge → new atoms (after merge step in v55)

Plus: instrument _enrich_atoms to log site catalog growth (where does
"atl_hq_2026" / "mdf_3a" enter the catalog?).
"""
from __future__ import annotations
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OPTBOT_ROSTER = Path(r"C:\Users\lilli\test_deals\optbot\artifacts\08_site_roster_and_facilities_authoritative.pdf")
APS_B = Path(r"C:\Users\lilli\test_deals\aps_fiber\artifacts\APS_fiber_Attachment_B.pdf")


def _atom_summary(a: Any) -> dict[str, Any]:
    v = getattr(a, "value", None) or {}
    return {
        "id": getattr(a, "id", "")[:24],
        "atom_type": str(getattr(a, "atom_type", "")),
        "site_id": (v.get("site_id") or v.get("id")) if isinstance(v, dict) else None,
        "name": v.get("name") if isinstance(v, dict) else None,
        "address": v.get("address") if isinstance(v, dict) else None,
        "raw_text": (getattr(a, "raw_text", "") or "")[:120],
        "entity_keys": list(getattr(a, "entity_keys", []) or []),
        "confidence": getattr(a, "confidence", None),
    }


# ─────────────────────────────────────────────────────────────────
# P1 + P2: fitz table extractor
# ─────────────────────────────────────────────────────────────────
def measure_fitz_table_paths(pdf: Path, label: str) -> dict[str, Any]:
    """Run _fitz_site_roster_fallback but PATCH the schema_registry import to
    log when the fallback fires. Returns counts from both primary + fallback paths.
    """
    from app.parsers.orbitbrief_pdf import _fitz_site_roster_fallback
    # Monkey-patch identify_schema to log when it's called
    import app.core.table_schema_registry as tsr
    original_identify = tsr.identify_schema
    original_emit = tsr.emit_atoms_for_schema
    call_log = {"identify_called": 0, "emit_called": 0, "emit_returned": 0}

    def _ident_wrap(*a, **kw):
        call_log["identify_called"] += 1
        return original_identify(*a, **kw)

    def _emit_wrap(*a, **kw):
        call_log["emit_called"] += 1
        out = original_emit(*a, **kw)
        call_log["emit_returned"] += len(out) if out else 0
        return out

    tsr.identify_schema = _ident_wrap
    tsr.emit_atoms_for_schema = _emit_wrap
    try:
        atoms = _fitz_site_roster_fallback(
            pdf_path=pdf,
            project_id=label.lower(),
            artifact_id=f"art_{label}",
            parser_version="instrument_v1",
            already_emitted=set(),
        )
    finally:
        tsr.identify_schema = original_identify
        tsr.emit_atoms_for_schema = original_emit

    print(f"  fitz_table: returned {len(atoms)} atoms")
    print(f"    identify_schema called: {call_log['identify_called']} times")
    print(f"    emit_atoms_for_schema called: {call_log['emit_called']} times")
    print(f"    schema-registry FALLBACK emitted: {call_log['emit_returned']} atoms")
    return {
        "path": "fitz_table",
        "total_atoms": len(atoms),
        "schema_fallback_calls": call_log["identify_called"],
        "schema_fallback_emits": call_log["emit_called"],
        "schema_fallback_atoms": call_log["emit_returned"],
        "primary_atoms": len(atoms) - call_log["emit_returned"],
        "atoms": [_atom_summary(a) for a in atoms],
    }


# ─────────────────────────────────────────────────────────────────
# P3: text-based extractor
# ─────────────────────────────────────────────────────────────────
def measure_text_based(pdf: Path, label: str) -> dict[str, Any]:
    from app.parsers.orbitbrief_pdf import _text_based_site_roster_extract
    atoms = _text_based_site_roster_extract(
        pdf_path=pdf,
        project_id=label.lower(),
        artifact_id=f"art_{label}_text",
        parser_version="instrument_v1",
        already_emitted=set(),
    )
    print(f"  text_based: {len(atoms)} atoms")
    return {
        "path": "text_based",
        "atom_count": len(atoms),
        "atoms": [_atom_summary(a) for a in atoms],
    }


# ─────────────────────────────────────────────────────────────────
# P4: LLM site_clusters chunked
# ─────────────────────────────────────────────────────────────────
def measure_llm_chunked(pdf: Path, label: str) -> dict[str, Any]:
    """Build a single-doc by_artifact dict and run _extract_site_clusters_chunked."""
    import fitz
    from app.core.multi_entity_llm import _extract_site_clusters_chunked

    doc = fitz.open(str(pdf))
    text = "\n\n".join(page.get_text() or "" for page in doc)
    doc.close()

    by_artifact = {
        f"art_{label}_chunked": {
            "filename": pdf.name,
            "text": text,
            "atoms": [],
        }
    }
    t0 = time.time()
    out = _extract_site_clusters_chunked(by_artifact)
    rt = time.time() - t0
    canon = [c.get("canonical_name") for c in out if c.get("canonical_name")]
    print(f"  llm_chunked: {len(out)} clusters ({rt:.1f}s)")
    return {
        "path": "llm_chunked",
        "cluster_count": len(out),
        "runtime_sec": round(rt, 1),
        "canonical_names": canon[:20],
        "sample": out[:3],
    }


# ─────────────────────────────────────────────────────────────────
# P5: site catalog (find_authoritative_site_phrases)
# ─────────────────────────────────────────────────────────────────
def measure_site_catalog_growth(pdf: Path, label: str) -> dict[str, Any]:
    """Parse the doc with full pipeline up to enrich_entities, log every
    phrase added to the authoritative_sites catalog with its likely source.
    """
    from app.parsers.orbitbrief_pdf import parse_artifact_full
    from app.core.site_detection import find_authoritative_site_phrases
    # Run parse to get atoms
    atoms = parse_artifact_full(
        pdf_path=pdf,
        project_id=label.lower(),
        artifact_id=f"art_{label}_full",
        parser_version="instrument_v1",
    )
    print(f"  parsed {len(atoms)} atoms total from this doc")

    catalog = find_authoritative_site_phrases(atoms)
    print(f"  catalog size: {len(catalog)} phrases")
    print(f"  catalog: {sorted(catalog)[:30]}")

    # Classify each catalog entry
    suspicious = []
    real_looking = []
    import re
    for phrase in catalog:
        lower = phrase.lower()
        is_mdf = bool(re.search(r"\bmdf[-_ ]?\w", lower)) or bool(re.search(r"\bidf[-_ ]?\w", lower))
        has_year = bool(re.search(r"20\d{2}", lower))
        has_column_keyword = any(k in lower for k in ("asset_type", "asset type", "header", "col_"))
        looks_like_address = bool(re.match(r"^\d+\s+\w", phrase))
        too_generic = lower in ("site", "facility", "location", "building", "office")
        if is_mdf or has_year or has_column_keyword or too_generic:
            suspicious.append({"phrase": phrase, "reason": (
                "MDF/IDF" if is_mdf else
                "year" if has_year else
                "column-keyword" if has_column_keyword else
                "generic"
            )})
        elif looks_like_address:
            real_looking.append({"phrase": phrase, "note": "address-shaped"})
        else:
            real_looking.append({"phrase": phrase})

    return {
        "path": "site_catalog",
        "catalog_total": len(catalog),
        "suspicious_count": len(suspicious),
        "real_looking_count": len(real_looking),
        "suspicious": suspicious[:30],
        "real_looking": real_looking[:20],
    }


# ─────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────
def run_deal(pdf: Path, label: str, skip_llm: bool) -> dict[str, Any]:
    print(f"\n{'='*72}\nINSTRUMENTING: {label}  ({pdf.name})\n{'='*72}")
    results: dict[str, Any] = {"deal": label, "pdf": str(pdf)}

    print("\n[P1+P2] fitz_table emission paths")
    results["fitz"] = measure_fitz_table_paths(pdf, label)

    print("\n[P3] text_based extractor")
    results["text"] = measure_text_based(pdf, label)

    print("\n[P5] site catalog growth")
    try:
        results["catalog"] = measure_site_catalog_growth(pdf, label)
    except Exception as e:
        results["catalog"] = {"error": f"{type(e).__name__}: {e}"}

    if not skip_llm:
        print("\n[P4] LLM chunked extractor")
        try:
            results["llm"] = measure_llm_chunked(pdf, label)
        except Exception as e:
            results["llm"] = {"error": f"{type(e).__name__}: {e}"}
    return results


if __name__ == "__main__":
    skip = "--skip-llm" in sys.argv
    out: dict[str, Any] = {}
    out["optbot"] = run_deal(OPTBOT_ROSTER, "optbot", skip)
    if "--optbot-only" not in sys.argv:
        out["aps_b"] = run_deal(APS_B, "aps_b", skip)

    out_path = REPO / "instrument_paths_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}\n")
    print("=== KEY DECISIONS BASED ON THESE NUMBERS ===")
    for deal in ("optbot", "aps_b"):
        if deal not in out:
            continue
        d = out[deal]
        print(f"\n{deal.upper()}:")
        fitz = d.get("fitz", {})
        print(f"  fitz primary atoms: {fitz.get('primary_atoms', 0)}")
        print(f"  fitz schema-registry FALLBACK atoms: {fitz.get('schema_fallback_atoms', 0)}")
        print(f"    → fix #1 safe to delete? {fitz.get('schema_fallback_atoms') == 0}")
        cat = d.get("catalog", {})
        if "suspicious_count" in cat:
            print(f"  site_catalog: {cat['catalog_total']} phrases ({cat['suspicious_count']} suspicious)")
            print(f"    → fix #2 needed? {cat['suspicious_count'] > 0}")
