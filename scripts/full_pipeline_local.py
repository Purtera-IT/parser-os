"""Run the FULL parser-os compile pipeline locally on OPTBOT, just like
the cloud worker would, but skip the LLM stages (set env to disable them).

Goal: see what semantic_dedup produces on the parser output BEFORE any
LLM noise. If the structural pipeline gives 5 clean atoms here, then
the cloud bug is entirely in the LLM extractor + bridge step.
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from collections import Counter

# Disable LLM stages to isolate structural path
os.environ["SOWSMITH_MULTI_ENTITY_DISABLE"] = "1"
os.environ["SOWSMITH_RETRIEVAL_DISABLE"] = "1"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.core.compiler import compile_project

OPTBOT_DIR = Path(r"C:\Users\lilli\test_deals\optbot\artifacts")

print(f"compiling OPTBOT from {OPTBOT_DIR}")
result = compile_project(
    project_dir=OPTBOT_DIR.parent,
    project_id="optbot",
    domain_pack=None,
    allow_errors=True,
    allow_unverified_receipts=True,
    use_cache=False,
)

atoms = list(result.atoms)
print(f"\ntotal atoms: {len(atoms)}")

phys = [a for a in atoms if str(getattr(a, "atom_type", "")).endswith("physical_site") or
                            (hasattr(getattr(a, "atom_type", None), "value") and getattr(a, "atom_type").value == "physical_site")]
print(f"physical_site atoms: {len(phys)}")

print("\n=== physical_site atoms ===")
for i, a in enumerate(phys, 1):
    v = getattr(a, "value", None) or {}
    sid = v.get("site_id") or v.get("id") if isinstance(v, dict) else None
    name = v.get("name") or v.get("facility_name") if isinstance(v, dict) else None
    addr = v.get("address") or v.get("street_address") if isinstance(v, dict) else None
    names_arr = v.get("names", []) if isinstance(v, dict) else []
    ek = list(getattr(a, "entity_keys", []) or [])
    print(f"{i}. site_id={sid!r}")
    print(f"   name: {name!r}")
    print(f"   address: {addr!r}")
    if names_arr:
        print(f"   names[]: {names_arr[:8]}{'...' if len(names_arr)>8 else ''}")
    print(f"   entity_keys: {ek}")
    print()

# Per-type atom breakdown
counter = Counter()
for a in atoms:
    at = getattr(a, "atom_type", None)
    counter[at.value if hasattr(at, "value") else str(at)] += 1
print("\n=== atom_type breakdown ===")
for at, n in counter.most_common(20):
    print(f"  {n:4d}  {at}")
