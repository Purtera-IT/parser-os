"""v55 verify: re-test the LLM site extraction on BOTH OPTBOT roster
PDF and APS Attachment B with the new max_tokens=16384 + tolerant parser.

Expected:
  OPTBOT roster doc: ~5 sites (no regression from before)
  APS Att B:         ~132 sites (was 0 before fix)
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz
from app.core.multi_entity_llm import (
    _build_site_clusters_prompt,
    _call_ollama,
    _parse_json_object_tolerant,
    _normalize_site_clusters,
)

OPTBOT_PDF = Path(r"C:\Users\lilli\test_deals\optbot\artifacts\08_site_roster_and_facilities_authoritative.pdf")
APS_PDF = Path(r"C:\Users\lilli\test_deals\aps_fiber\artifacts\APS_fiber_Attachment_B.pdf")


def doc_text(p: Path) -> str:
    d = fitz.open(str(p))
    out = "\n\n".join(page.get_text() or "" for page in d)
    d.close()
    return out


def run(pdf: Path, label: str, expected: int):
    print(f"\n=== {label}  (expected ~{expected} sites) ===")
    text = doc_text(pdf)
    print(f"  pdf_text_chars = {len(text)}")
    prompt = _build_site_clusters_prompt(text[:30000])
    t0 = time.time()
    raw = _call_ollama(prompt, max_tokens=16384)
    rt = time.time() - t0
    print(f"  llm_runtime = {rt:.1f}s  response_chars = {len(raw)}")
    print(f"  response_ends_with_brace = {raw.rstrip().endswith('}')}")
    obj = _parse_json_object_tolerant(raw, array_key="site_clusters")
    clusters = []
    if isinstance(obj, dict):
        clusters = _normalize_site_clusters(obj.get("site_clusters"))
    print(f"  recovered cluster_count = {len(clusters)}")
    if clusters:
        names = [c["canonical_name"] for c in clusters]
        print(f"  first 3 names: {names[:3]}")
        print(f"  last  3 names: {names[-3:]}")
    return {"label": label, "rt": rt, "cluster_count": len(clusters), "names": [c["canonical_name"] for c in clusters]}


if __name__ == "__main__":
    results = []
    results.append(run(OPTBOT_PDF, "OPTBOT roster (08_site_roster_…)", 5))
    results.append(run(APS_PDF,    "APS Attachment B (132-site list)", 132))

    print("\n\nSUMMARY:")
    for r in results:
        print(f"  {r['label']:40s}  {r['cluster_count']:3d} clusters  ({r['rt']:.1f}s)")
    Path("v55_verify_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
