"""Test two LLM prompt designs on APS Attachment B (the actual 132-site roster).

Test A: current 'site_clusters' prompt as-is.
Test B: new 'row-oriented table parse' prompt that gives the LLM column
        headers + page text and asks for one object per row.

Goal: see which strategy recovers more of the 132 sites.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz

from app.core.multi_entity_llm import (
    _build_site_clusters_prompt,
    _call_ollama,
    _parse_json_object,
    _normalize_site_clusters,
)

APS_B = Path(r"C:\Users\lilli\test_deals\aps_fiber\artifacts\APS_fiber_Attachment_B.pdf")


def pdf_full_text(pdf: Path) -> str:
    doc = fitz.open(str(pdf))
    out = []
    for page in doc:
        out.append(page.get_text() or "")
    doc.close()
    return "\n\n".join(out)


def test_A_site_clusters(text: str) -> dict:
    """Current production prompt."""
    print("\n[TEST A] site_clusters prompt (current) ...")
    prompt = _build_site_clusters_prompt(text[:30000])
    t0 = time.time()
    raw = _call_ollama(prompt, max_tokens=4096)
    rt = time.time() - t0
    Path("aps_b_raw_A.txt").write_text(raw, encoding="utf-8")
    obj = _parse_json_object(raw)
    clusters = []
    parse_ok = isinstance(obj, dict)
    if parse_ok:
        clusters = _normalize_site_clusters(obj.get("site_clusters"))
    canon = [c.get("canonical_name") for c in clusters if c.get("canonical_name")]
    print(f"    parse_ok={parse_ok}  raw_len={len(raw)}  raw_ends_with_brace={raw.rstrip().endswith('}')}")
    print(f"    {len(clusters)} clusters, canonical names: {len(canon)} ({rt:.1f}s)")
    return {
        "prompt": "site_clusters",
        "runtime_sec": round(rt, 1),
        "cluster_count": len(clusters),
        "canonical_names": canon[:200],
        "raw_response_length": len(raw),
        "parse_ok": parse_ok,
        "raw_response_snippet": raw[:400],
        "raw_response_tail": raw[-400:],
    }


def test_B_row_oriented(text: str) -> dict:
    """New row-oriented table prompt."""
    print("\n[TEST B] row-oriented table prompt (new) ...")
    prompt = f"""You are parsing one section of a site location table.

The section starts with column headers (Site No, Administrative Site, Street, City, Zip, Lat/Long).
Below the headers are rows. Each row is ONE physical building / school / office.
The site_no is whatever value sits in the "Site No" column (may be a plain integer like 1, 2, 3, ...).

Extract EVERY row you can see. Do NOT cluster or merge rows. Do NOT skip any.
Each row becomes one object.

Output shape:
{{"sites": [
  {{"site_no": "<value from Site No column>",
    "name":    "<value from Administrative Site column>",
    "street":  "<value from Street column>",
    "city":    "<value from City column>",
    "zip":     "<value from Zip column>",
    "lat_long":"<value from Lat,Long column>"}},
  ...
]}}

If a field is missing for a row, use null. Do not invent values.

DOCUMENT TEXT (the table is in here, possibly across multiple pages):

{text}

/no_think"""
    t0 = time.time()
    raw = _call_ollama(prompt, max_tokens=8192)
    rt = time.time() - t0
    Path("aps_b_raw_B.txt").write_text(raw, encoding="utf-8")
    obj = _parse_json_object(raw)
    sites = []
    parse_ok = isinstance(obj, dict)
    if parse_ok:
        s = obj.get("sites")
        if isinstance(s, list):
            sites = s
    print(f"    parse_ok={parse_ok}  raw_len={len(raw)}  raw_ends_with_brace={raw.rstrip().endswith('}')}")
    print(f"    {len(sites)} sites ({rt:.1f}s)")
    return {
        "prompt": "row_oriented",
        "runtime_sec": round(rt, 1),
        "site_count": len(sites),
        "sites_sample": sites[:5] + ([{"...": f"+{len(sites)-5} more"}] if len(sites) > 5 else []),
        "all_site_nos": [s.get("site_no") for s in sites if isinstance(s, dict)],
        "raw_response_length": len(raw),
        "parse_ok": parse_ok,
        "raw_response_snippet": raw[:400],
        "raw_response_tail": raw[-400:],
    }


if __name__ == "__main__":
    text = pdf_full_text(APS_B)
    print(f"APS Attachment B full text: {len(text)} chars across pages")

    out = {
        "pdf": str(APS_B),
        "text_chars": len(text),
        "tests": [
            test_A_site_clusters(text),
            test_B_row_oriented(text),
        ],
    }

    out_path = Path(__file__).resolve().parent.parent / "aps_b_prompt_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("\nSUMMARY:")
    for t in out["tests"]:
        if "cluster_count" in t:
            print(f"  {t['prompt']:20s}  {t['cluster_count']:3d} clusters  ({t['runtime_sec']}s)")
        else:
            print(f"  {t['prompt']:20s}  {t['site_count']:3d} sites    ({t['runtime_sec']}s)")
