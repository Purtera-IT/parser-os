"""DEEP INVESTIGATION: why does cloud OPTBOT produce 8 physical_site atoms
when local (with LLM disabled) produces 5?

Hypothesis: the diff is the LLM site_clusters bridge step. With LLM off
it doesn't run. With LLM on, the LLM emits clusters like "HQ", "AIR",
"976 Brady Avenue..." that don't match the structural sites index in
_entities_to_atoms, so the bridge falls through and CREATES NEW
physical_site atoms with site_id derived from the cluster's canonical_name.

This script:
  1. Calls _extract_site_clusters directly with the OPTBOT corpus to
     capture EXACTLY what the LLM produces.
  2. Builds the structural_sites_index that _entities_to_atoms uses.
  3. Walks each cluster through the matcher and reports: MATCH /
     NO-MATCH → what site_id would be derived for new atom.
  4. Shows the precise gap in the matching logic.
"""
from __future__ import annotations
import json, sys, os, re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import fitz
from app.core.multi_entity_llm import (
    _build_site_clusters_prompt,
    _call_ollama,
    _parse_json_object_tolerant,
    _normalize_site_clusters,
)

OPTBOT_DIR = Path(r"C:\Users\lilli\test_deals\optbot\artifacts")


def doc_text(p: Path) -> str:
    d = fitz.open(str(p))
    s = "\n\n".join(page.get_text() or "" for page in d)
    d.close()
    return s


# ============================================================
# STEP 1: get LLM output
# ============================================================
print("="*70)
print("STEP 1: Calling LLM site_clusters extractor on OPTBOT corpus")
print("="*70)
all_text = ""
for pdf in OPTBOT_DIR.glob("*.pdf"):
    txt = doc_text(pdf)
    all_text += f"\n\n=== {pdf.name} ===\n{txt}"
print(f"  Total corpus: {len(all_text)} chars")

prompt = _build_site_clusters_prompt(all_text[:30000])
print("  Calling Ollama (qwen3:32b, max_tokens=16384)...")
import time
t0 = time.time()
raw = _call_ollama(prompt, max_tokens=16384)
rt = time.time() - t0
print(f"  LLM returned {len(raw)} chars in {rt:.1f}s")

obj = _parse_json_object_tolerant(raw, array_key="site_clusters")
clusters = []
if isinstance(obj, dict):
    clusters = _normalize_site_clusters(obj.get("site_clusters"))
print(f"  Parsed {len(clusters)} clusters")
print()

# ============================================================
# STEP 2: simulate structural_sites_index (matching what
# entity_extraction._entities_to_atoms builds from real atoms)
# ============================================================
print("="*70)
print("STEP 2: Simulating structural_sites_index from OPTBOT roster")
print("="*70)

# Real OPTBOT atoms — what's in value dict before LLM bridge runs:
structural_sites = [
    {"site_id": "ATL-HQ-01", "name": "OPTBOT Atlanta HQ",
     "facility_name": "OPTBOT Atlanta HQ",
     "street_address": "1200 Peachtree St NE, Atlanta GA 30309"},
    {"site_id": "ATL-WEST-02", "name": "OPTBOT West Campus",
     "facility_name": "OPTBOT West Campus",
     "street_address": "3100 Interstate N Pkwy, Atlanta GA 30339"},
    {"site_id": "ATL-AIR-03", "name": "OPTBOT Airport Logistics",
     "facility_name": "OPTBOT Airport Logistics",
     "street_address": "6000 N Terminal Pkwy, Atlanta GA 30320"},
    {"site_id": "ATL-047-04", "name": "OPTBOT Brady Training",
     "facility_name": "OPTBOT Brady Training",
     "street_address": "047 Brady Ave NW, Atlanta GA 30318"},
    {"site_id": "ATL-CP-05", "name": "OPTBOT College Park Staging",
     "facility_name": "OPTBOT College Park Staging",
     "street_address": "1850 Sullivan Rd, College Park GA 30337"},
]


def _norm_form(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


structural_index = {}
for atom in structural_sites:
    for field in ("site_id", "name", "facility_name", "street_address"):
        key = _norm_form(atom.get(field))
        if key:
            structural_index[key] = atom

print(f"  Structural index has {len(structural_index)} normalized lookup keys:")
for k in sorted(structural_index.keys())[:8]:
    print(f"    {k!r:50s} -> {structural_index[k]['site_id']}")
print()

# ============================================================
# STEP 3: walk each cluster, report MATCH or NO-MATCH
# ============================================================
print("="*70)
print("STEP 3: For each LLM cluster, simulate the bridge match step")
print("="*70)

stats = {"matched": 0, "no_match_create": 0}
detail = []
for i, cluster in enumerate(clusters, 1):
    canon = cluster.get("canonical_name", "")
    aliases = cluster.get("aliases", []) or []
    forms = [canon] + list(aliases)

    matched = None
    matched_via = None
    for form in forms:
        key = _norm_form(form)
        if key and key in structural_index:
            matched = structural_index[key]
            matched_via = form
            break

    if matched:
        stats["matched"] += 1
        print(f"{i:2d}. MATCH ✓  canon={canon!r}  -> {matched['site_id']} (via form {matched_via!r})")
        print(f"        will MERGE {len(forms)} aliases into structural atom")
    else:
        stats["no_match_create"] += 1
        # _pick_site_id derives an id from canonical_name when no match
        derived_id = re.sub(r"[^A-Za-z0-9]+", "-", canon).strip("-").upper()
        print(f"{i:2d}. NO MATCH ✗  canon={canon!r}")
        print(f"        BRIDGE WOULD CREATE NEW ATOM: site_id={derived_id!r}")
        # Show normalized forms so we can see WHY it didn't match
        nforms = [_norm_form(f) for f in forms[:5]]
        print(f"        normalized forms tried: {nforms}")
        detail.append({"cluster": cluster, "derived_id": derived_id, "normalized_forms": nforms})

print()
print("="*70)
print(f"SUMMARY: {stats['matched']} matched (merge), {stats['no_match_create']} NO-MATCH (ghost create)")
print("="*70)

if detail:
    print()
    print("GHOST ATOMS the bridge will create:")
    for d in detail:
        print(f"  site_id={d['derived_id']!r}  forms={d['normalized_forms']}")

print()
print("CONCLUSION:")
print("  Local test had LLM DISABLED → no clusters → no bridge → 5 clean atoms.")
print("  Cloud has LLM ENABLED → above no-match clusters create ghost atoms.")
print()
print("THE FIX:")
print("  In _entities_to_atoms, when structural physical_site atoms exist,")
print("  NO-MATCH clusters should be DROPPED (not create new atoms).")
print("  Match step is best-effort enrichment; structural roster is canonical.")
