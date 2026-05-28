"""Bake-off: test each site-extraction path in isolation on OPTBOT + APS.

Goal: figure out which paths produce clean atoms vs garbage, so we can
make an evidence-based decision on the v55 refactor.

Tests:
  1. PyMuPDF table extractor       — _fitz_site_roster_fallback
  2. Text-based regex extractor    — _text_based_site_roster_extract
  3. LLM site_clusters (current)   — _extract_site_clusters with whole doc
  4. LLM with structural context   — same prompt + known-sites injection

Deals:
  - OPTBOT: doc 08_site_roster_and_facilities_authoritative.pdf
    Ground truth: 5 sites
      ATL-HQ-01, ATL-WEST-02, ATL-AIR-03, ATL-047-04, ATL-CP-05
  - APS: APS_fiber_Attachment_A.pdf
    Ground truth: ~132 sites (APS-001 through APS-132 approximately)

Outputs per test:
  - count of physical_site atoms emitted
  - list of site IDs (or canonical_names for LLM)
  - matches to ground truth (precision/recall)
  - garbage examples (atoms that aren't real sites)
  - runtime in seconds
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Ground truth
OPTBOT_PDF = Path(r"C:\Users\lilli\test_deals\optbot\artifacts\08_site_roster_and_facilities_authoritative.pdf")
APS_PDF = Path(r"C:\Users\lilli\test_deals\aps_fiber\artifacts\APS_fiber_Attachment_A.pdf")

OPTBOT_GT = ["ATL-HQ-01", "ATL-WEST-02", "ATL-AIR-03", "ATL-047-04", "ATL-CP-05"]
# APS ground truth is the numbered roster — we'll detect by shape APS-NNN
APS_GT_PREFIX = "APS-"
APS_GT_COUNT = 132  # approximate, will verify by counting unique APS-### in the doc text


def _site_ids_from_atoms(atoms: list[Any]) -> list[str]:
    out = []
    for a in atoms:
        # parser-os emits atoms using `value` dict (not `structured`)
        v = getattr(a, "value", None) or {}
        if isinstance(v, dict):
            sid = v.get("site_id") or v.get("id") or v.get("name")
            if sid:
                out.append(str(sid))
    return out


def _full_atom_dump(atoms: list[Any]) -> list[dict[str, Any]]:
    """Capture full atom payload for downstream inspection."""
    out = []
    for a in atoms:
        v = getattr(a, "value", None) or {}
        out.append({
            "site_id": v.get("site_id") if isinstance(v, dict) else None,
            "name": v.get("name") if isinstance(v, dict) else None,
            "facility_name": v.get("facility_name") if isinstance(v, dict) else None,
            "address": v.get("address") if isinstance(v, dict) else None,
            "raw_text_snippet": (getattr(a, "raw_text", "") or "")[:200],
        })
    return out


def _classify(found: list[str], gt: list[str], gt_count: int | None = None) -> dict[str, Any]:
    """Compare found vs ground truth. For OPTBOT we have exact GT list,
    for APS we just check shape + count."""
    found_set = {f.upper().strip() for f in found if f}
    if gt:  # exact GT (OPTBOT)
        gt_set = {g.upper().strip() for g in gt}
        tp = found_set & gt_set
        fn = gt_set - found_set
        fp = found_set - gt_set
        return {
            "found": sorted(found_set),
            "true_positive": sorted(tp),
            "false_negative_missed": sorted(fn),
            "false_positive_garbage": sorted(fp),
            "precision": round(len(tp) / max(1, len(found_set)), 3),
            "recall": round(len(tp) / max(1, len(gt_set)), 3),
        }
    else:  # shape GT (APS)
        prefix = APS_GT_PREFIX.upper()
        shape_ok = {f for f in found_set if f.startswith(prefix) and len(f) <= 12}
        garbage = found_set - shape_ok
        return {
            "found_count": len(found_set),
            "found": sorted(found_set)[:20] + (["..."] if len(found_set) > 20 else []),
            "shape_matching_count": len(shape_ok),
            "garbage_examples": sorted(garbage)[:10],
            "gt_count_approx": gt_count,
        }


# ─────────────────────────────────────────────────────────────────────
# Test 1: PyMuPDF table extractor
# ─────────────────────────────────────────────────────────────────────
def test_fitz_table(pdf: Path, project_id: str, label: str) -> dict[str, Any]:
    print(f"\n[1/4] fitz_table  →  {label}")
    from app.parsers.orbitbrief_pdf import _fitz_site_roster_fallback
    t0 = time.time()
    atoms = _fitz_site_roster_fallback(
        pdf_path=pdf,
        project_id=project_id,
        artifact_id=f"art_{label}",
        parser_version="bake_off_v1",
        already_emitted=set(),
    )
    rt = time.time() - t0
    sids = _site_ids_from_atoms(atoms)
    print(f"    found {len(atoms)} atoms, {len(sids)} with site_id ({rt:.2f}s)")
    return {"path": "fitz_table", "runtime_sec": round(rt, 2), "atom_count": len(atoms), "site_ids": sids, "atoms_full": _full_atom_dump(atoms)}


# ─────────────────────────────────────────────────────────────────────
# Test 2: Text-based regex extractor
# ─────────────────────────────────────────────────────────────────────
def test_text_based(pdf: Path, project_id: str, label: str) -> dict[str, Any]:
    print(f"\n[2/4] text_based  →  {label}")
    from app.parsers.orbitbrief_pdf import _text_based_site_roster_extract
    t0 = time.time()
    atoms = _text_based_site_roster_extract(
        pdf_path=pdf,
        project_id=project_id,
        artifact_id=f"art_{label}",
        parser_version="bake_off_v1",
        already_emitted=set(),
    )
    rt = time.time() - t0
    sids = _site_ids_from_atoms(atoms)
    print(f"    found {len(atoms)} atoms, {len(sids)} with site_id ({rt:.2f}s)")
    return {"path": "text_based", "runtime_sec": round(rt, 2), "atom_count": len(atoms), "site_ids": sids, "atoms_full": _full_atom_dump(atoms)}


# ─────────────────────────────────────────────────────────────────────
# Test 3: LLM site_clusters — no structural context
# ─────────────────────────────────────────────────────────────────────
def _pdf_to_excerpt(pdf: Path, max_chars: int = 30000) -> str:
    import fitz
    doc = fitz.open(str(pdf))
    out = []
    n = 0
    for page in doc:
        txt = page.get_text() or ""
        out.append(txt)
        n += len(txt)
        if n > max_chars:
            break
    return "\n\n".join(out)[:max_chars]


def test_llm_no_context(pdf: Path, label: str) -> dict[str, Any]:
    print(f"\n[3/4] llm_no_ctx  →  {label}  (calls Ollama, slow)")
    from app.core.multi_entity_llm import _build_site_clusters_prompt, _call_ollama, _parse_json_object, _normalize_site_clusters
    excerpt = _pdf_to_excerpt(pdf)
    prompt = _build_site_clusters_prompt(excerpt)
    t0 = time.time()
    text = _call_ollama(prompt, max_tokens=2048)
    rt = time.time() - t0
    obj = _parse_json_object(text)
    clusters = []
    if isinstance(obj, dict):
        clusters = _normalize_site_clusters(obj.get("site_clusters"))
    canonical_names = [c.get("canonical_name") for c in clusters if c.get("canonical_name")]
    aliases_all = []
    for c in clusters:
        for a in (c.get("aliases") or []):
            aliases_all.append(a)
    print(f"    found {len(clusters)} clusters, canonical_names={len(canonical_names)}, total_aliases={len(aliases_all)} ({rt:.1f}s)")
    return {
        "path": "llm_no_context",
        "runtime_sec": round(rt, 1),
        "cluster_count": len(clusters),
        "canonical_names": canonical_names,
        "aliases_sample": aliases_all[:30],
    }


# ─────────────────────────────────────────────────────────────────────
# Test 4: LLM site_clusters WITH structural context
# ─────────────────────────────────────────────────────────────────────
def test_llm_with_context(pdf: Path, known_sites: list[str], label: str) -> dict[str, Any]:
    print(f"\n[4/4] llm_w_ctx   →  {label}  (calls Ollama, slow)")
    from app.core.multi_entity_llm import _call_ollama, _parse_json_object, _normalize_site_clusters
    excerpt = _pdf_to_excerpt(pdf)
    known_str = ", ".join(sorted(set(known_sites))[:50]) or "(none)"
    prompt = f"""Identify ADDITIONAL physical sites NOT already in this known list.

KNOWN SITES (already extracted from roster tables — DO NOT re-emit these
or any close variant):

{known_str}

Find any ADDITIONAL physical buildings/sites mentioned in the docs that
are NOT in the known list above. Return empty array if there are none.

EXCLUDE (always):
- Anything that is just a city + zip with no facility name
- Anything that is already a known site (above) or a close alias
- Standards bodies (ANSI, NFPA, ...)
- Vendor / product / SaaS names
- Generic nouns ("the school", "the library")

Output shape:
{{"site_clusters": [
  {{"canonical_name": "<primary name>", "aliases": ["<form 1>", "..."]}},
  ...
]}}

If no additional sites, return: {{"site_clusters": []}}

DOCUMENTS:

{excerpt}

/no_think"""
    t0 = time.time()
    text = _call_ollama(prompt, max_tokens=2048)
    rt = time.time() - t0
    obj = _parse_json_object(text)
    clusters = []
    if isinstance(obj, dict):
        clusters = _normalize_site_clusters(obj.get("site_clusters"))
    canonical_names = [c.get("canonical_name") for c in clusters if c.get("canonical_name")]
    aliases_all = []
    for c in clusters:
        for a in (c.get("aliases") or []):
            aliases_all.append(a)
    print(f"    found {len(clusters)} additional clusters, canonical_names={len(canonical_names)} ({rt:.1f}s)")
    return {
        "path": "llm_with_context",
        "runtime_sec": round(rt, 1),
        "cluster_count": len(clusters),
        "canonical_names": canonical_names,
        "aliases_sample": aliases_all[:30],
        "known_sites_passed": known_sites,
    }


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────
def run_deal(pdf: Path, project_id: str, label: str, gt: list[str], gt_count: int | None) -> dict[str, Any]:
    print(f"\n{'='*72}\nDEAL: {label}  ({pdf.name})\n{'='*72}")
    results: dict[str, Any] = {}

    # Path 1 + 2 first (structural, fast)
    r1 = test_fitz_table(pdf, project_id, label)
    r2 = test_text_based(pdf, project_id, label)

    # Union of structural for context to LLM #4
    structural_sids = list({*(r1["site_ids"]), *(r2["site_ids"])})

    # Path 3 (LLM no context) — only if LLM available
    skip_llm = os.environ.get("BAKE_OFF_SKIP_LLM") == "1"
    if not skip_llm:
        try:
            r3 = test_llm_no_context(pdf, label)
        except Exception as e:
            r3 = {"path": "llm_no_context", "error": f"{type(e).__name__}: {e}"}
    else:
        r3 = {"path": "llm_no_context", "skipped": True}

    # Path 4 (LLM with context)
    if not skip_llm:
        try:
            r4 = test_llm_with_context(pdf, structural_sids, label)
        except Exception as e:
            r4 = {"path": "llm_with_context", "error": f"{type(e).__name__}: {e}"}
    else:
        r4 = {"path": "llm_with_context", "skipped": True}

    # Analysis
    r1["vs_ground_truth"] = _classify(r1.get("site_ids", []), gt, gt_count)
    r2["vs_ground_truth"] = _classify(r2.get("site_ids", []), gt, gt_count)
    if "canonical_names" in r3:
        r3["vs_ground_truth"] = _classify(r3.get("canonical_names", []), gt, gt_count)
    if "canonical_names" in r4:
        r4["vs_ground_truth"] = _classify(r4.get("canonical_names", []), gt, gt_count)

    results["deal"] = label
    results["pdf"] = str(pdf)
    results["ground_truth"] = {"site_ids": gt, "count": gt_count or len(gt)}
    results["structural_union"] = sorted(structural_sids)
    results["paths"] = [r1, r2, r3, r4]
    return results


if __name__ == "__main__":
    out: dict[str, Any] = {}
    out["optbot"] = run_deal(OPTBOT_PDF, "optbot", "optbot", OPTBOT_GT, None)
    out["aps"] = run_deal(APS_PDF, "aps_fiber", "aps_fiber", [], APS_GT_COUNT)

    out_path = REPO / "site_bake_off_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n\nWrote {out_path}")
    print(f"\nSUMMARY:")
    for deal_key in ("optbot", "aps"):
        d = out[deal_key]
        print(f"\n  {deal_key.upper()}:")
        for p in d["paths"]:
            name = p["path"]
            rt = p.get("runtime_sec", "?")
            if "error" in p:
                print(f"    {name:20s}  ERROR: {p['error']}")
            elif p.get("skipped"):
                print(f"    {name:20s}  SKIPPED")
            else:
                count = p.get("atom_count") or p.get("cluster_count") or 0
                gt = p.get("vs_ground_truth") or {}
                if "precision" in gt:
                    print(f"    {name:20s}  {count:4d} found  P={gt['precision']:.2f}  R={gt['recall']:.2f}  ({rt}s)")
                elif "shape_matching_count" in gt:
                    print(f"    {name:20s}  {count:4d} found  shape_ok={gt['shape_matching_count']:3d}  garbage_examples={len(gt['garbage_examples'])}  ({rt}s)")
                else:
                    print(f"    {name:20s}  {count:4d} found  ({rt}s)")
