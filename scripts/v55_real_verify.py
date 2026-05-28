"""Test the PRODUCTION _extract_site_clusters function (which now uses
max_tokens=16384 + tolerant parser) on OPTBOT roster and APS Attachment B.
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz
from app.core.multi_entity_llm import _extract_site_clusters

OPTBOT = Path(r"C:\Users\lilli\test_deals\optbot\artifacts\08_site_roster_and_facilities_authoritative.pdf")
APS_B = Path(r"C:\Users\lilli\test_deals\aps_fiber\artifacts\APS_fiber_Attachment_B.pdf")


def text(p):
    d = fitz.open(str(p))
    s = "\n\n".join(page.get_text() or "" for page in d)
    d.close()
    return s


def run(pdf, label, expected):
    print(f"\n=== {label}  expected ~{expected} sites ===")
    excerpt = text(pdf)[:30000]
    print(f"  excerpt: {len(excerpt)} chars")
    t0 = time.time()
    clusters = _extract_site_clusters(excerpt)
    rt = time.time() - t0
    print(f"  clusters: {len(clusters)}  ({rt:.1f}s)")
    if clusters:
        names = [c.get("canonical_name") for c in clusters]
        print(f"  first 3: {names[:3]}")
        print(f"  last  3: {names[-3:]}")
        all_aliases = sum(len(c.get("aliases", [])) for c in clusters)
        print(f"  total aliases across all clusters: {all_aliases}")
    return {"label": label, "clusters": len(clusters), "rt": rt}


if __name__ == "__main__":
    r = []
    r.append(run(OPTBOT, "OPTBOT", 5))
    r.append(run(APS_B,  "APS Attachment B", 132))
    print("\nSUMMARY:")
    for x in r:
        print(f"  {x['label']:30s}  {x['clusters']:3d} clusters  ({x['rt']:.1f}s)")
    Path("v55_real_verify_results.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
