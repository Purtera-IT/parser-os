"""Run full parser-os compile on APS_fiber locally with v56 fixes
applied (LLM disabled to isolate structural cleanliness).

Target: 159 physical_site atoms (one per APS school), each with
a clean site:* entity_key and structured value fields.
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
from collections import Counter
os.environ["SOWSMITH_MULTI_ENTITY_DISABLE"] = "1"
os.environ["SOWSMITH_RETRIEVAL_DISABLE"] = "1"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.core.compiler import compile_project

APS_DIR = Path(r"C:\Users\lilli\test_deals\aps_fiber\artifacts")

print(f"compiling APS_fiber from {APS_DIR}")
result = compile_project(
    project_dir=APS_DIR.parent,
    project_id="aps_fiber",
    domain_pack=None,
    allow_errors=True,
    allow_unverified_receipts=True,
    use_cache=False,
)

atoms = list(result.atoms)
print(f"\ntotal atoms: {len(atoms)}")

def atype(a):
    at = getattr(a, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at)

phys = [a for a in atoms if atype(a) == "physical_site"]
print(f"physical_site atoms: {len(phys)}")
print()
print("=== first 5 physical_site atoms ===")
for i, a in enumerate(phys[:5], 1):
    v = getattr(a, "value", None) or {}
    print(f"{i}. site_id={v.get('site_id') or v.get('id')!r}")
    print(f"   name={v.get('name') or v.get('facility_name')!r}")
    print(f"   address={v.get('street_address') or v.get('address')!r}")
    print(f"   entity_keys ({len(a.entity_keys or [])}): {list(a.entity_keys or [])[:5]}")
print()
print("=== last 5 physical_site atoms ===")
for i, a in enumerate(phys[-5:], len(phys)-4):
    v = getattr(a, "value", None) or {}
    print(f"{i}. site_id={v.get('site_id') or v.get('id')!r}  name={v.get('name')!r}")

# Per-key counts on physical_site atoms
key_counter = Counter()
for a in phys:
    for k in (a.entity_keys or []):
        if k.startswith("site:"):
            key_counter[k] += 1
print()
print(f"=== site:* key uniqueness across {len(phys)} physical_site atoms ===")
print(f"unique site:* keys: {len(key_counter)}")
# Expect: roughly equal to len(phys), since each atom should have 1 unique site key
print(f"avg site:* keys per atom: {sum(key_counter.values())/max(1,len(phys)):.2f}")

# Atom type breakdown
type_counter = Counter(atype(a) for a in atoms)
print()
print("=== atom_type breakdown ===")
for t, n in type_counter.most_common(20):
    print(f"  {n:5d}  {t}")
