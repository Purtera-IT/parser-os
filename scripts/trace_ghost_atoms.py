"""Trace which parser emits each ghost physical_site atom by running the
local pipeline + dumping FULL atom payload (value dict + source_refs).
Tells us exactly which extractor produced each ghost site_id.
"""
from __future__ import annotations
import json, sys, os
os.environ["SOWSMITH_MULTI_ENTITY_DISABLE"] = "1"
os.environ["SOWSMITH_RETRIEVAL_DISABLE"] = "1"

from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.core.compiler import compile_project

OPTBOT_DIR = Path(r"C:\Users\lilli\test_deals\optbot\artifacts")

print("compiling OPTBOT (LLM off) — dumping full physical_site atom payloads")
result = compile_project(
    project_dir=OPTBOT_DIR.parent,
    project_id="optbot",
    domain_pack=None,
    allow_errors=True,
    allow_unverified_receipts=True,
    use_cache=False,
)

phys = []
for a in result.atoms:
    at = getattr(a, "atom_type", None)
    if (at.value if hasattr(at, "value") else str(at)) == "physical_site":
        phys.append(a)

print(f"\n{len(phys)} physical_site atoms total\n")

for i, a in enumerate(phys, 1):
    v = dict(getattr(a, "value", {}) or {})
    srefs = getattr(a, "source_refs", []) or []
    primary = srefs[0] if srefs else None
    print(f"=== atom {i} ===")
    print(f"  id={getattr(a, 'id', '')[:32]}")
    print(f"  site_id = {v.get('site_id')!r}")
    print(f"  id      = {v.get('id')!r}")
    print(f"  name    = {v.get('name')!r}")
    print(f"  facility_name = {v.get('facility_name')!r}")
    print(f"  source_filename = {getattr(primary, 'filename', '') if primary else ''}")
    print(f"  extraction_method = {getattr(primary, 'extraction_method', '') if primary else ''}")
    print(f"  raw_text[:140] = {(getattr(a, 'raw_text', '') or '')[:140]!r}")
    print()
