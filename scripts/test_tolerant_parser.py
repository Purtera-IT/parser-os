"""Verify the new _parse_json_object_tolerant recovers sites from the
truncated APS-B raw responses captured during the bake-off."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.multi_entity_llm import (
    _parse_json_object,
    _parse_json_object_tolerant,
    _normalize_site_clusters,
)

REPO = Path(__file__).resolve().parent.parent
raw_a = (REPO / "aps_b_raw_A.txt").read_text(encoding="utf-8")
raw_b = (REPO / "aps_b_raw_B.txt").read_text(encoding="utf-8")

print(f"raw_A.txt: {len(raw_a)} chars")
print(f"raw_B.txt: {len(raw_b)} chars\n")

# Strict path on A
strict_a = _parse_json_object(raw_a)
print(f"strict _parse_json_object(raw_A): {type(strict_a).__name__}  →  cluster_count={(len((strict_a or {}).get('site_clusters', [])) if isinstance(strict_a, dict) else 'N/A')}")

# Tolerant path on A
tol_a = _parse_json_object_tolerant(raw_a, array_key="site_clusters")
n_a = len((tol_a or {}).get("site_clusters", [])) if isinstance(tol_a, dict) else 0
print(f"tolerant _parse_json_object_tolerant(raw_A, 'site_clusters'):  recovered_count={n_a}")
if isinstance(tol_a, dict):
    clusters = _normalize_site_clusters(tol_a.get("site_clusters"))
    print(f"  after _normalize_site_clusters: {len(clusters)} clusters")
    if clusters:
        print(f"  first 3: {[c['canonical_name'] for c in clusters[:3]]}")
        print(f"  last 3:  {[c['canonical_name'] for c in clusters[-3:]]}")

print()

# Strict path on B
strict_b = _parse_json_object(raw_b)
print(f"strict _parse_json_object(raw_B): {type(strict_b).__name__}  →  site_count={(len((strict_b or {}).get('sites', [])) if isinstance(strict_b, dict) else 'N/A')}")

# Tolerant path on B
tol_b = _parse_json_object_tolerant(raw_b, array_key="sites")
n_b = len((tol_b or {}).get("sites", [])) if isinstance(tol_b, dict) else 0
print(f"tolerant _parse_json_object_tolerant(raw_B, 'sites'):  recovered_count={n_b}")
if isinstance(tol_b, dict):
    sites = tol_b.get("sites") or []
    if sites:
        print(f"  first 3 site_nos: {[s.get('site_no') for s in sites[:3]]}")
        print(f"  last 3 site_nos:  {[s.get('site_no') for s in sites[-3:]]}")
