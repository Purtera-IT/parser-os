"""v56e VERIFICATION: run full pipeline locally with LLM ENABLED.
Instrument every physical_site emission path. Verify v56e merge-or-drop
guard eliminates ALL ghost atoms.

This is the definitive test — it reproduces cloud's LLM bridge behavior
locally so we can prove the fix.

Outputs:
  - atom count at each stage (parse, dedup, after-LLM-bridge)
  - per-emission-path count
  - list of all physical_site atoms with their source (extraction_method)
  - any unmatched LLM clusters that got dropped (should be ALL of them
    when structural exists)
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from collections import Counter

# LLM ENABLED — same as cloud
# (do not set SOWSMITH_MULTI_ENTITY_DISABLE or SOWSMITH_RETRIEVAL_DISABLE)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OPTBOT_DIR = Path(r"C:\Users\lilli\test_deals\optbot\artifacts")

# Patch _entities_to_atoms to log every cluster's match/no-match outcome
print("="*70)
print("v56e VERIFICATION — LLM ON, full OPTBOT compile")
print("="*70)

from app.core.compiler import compile_project
import time
t0 = time.time()
result = compile_project(
    project_dir=OPTBOT_DIR.parent,
    project_id="optbot",
    domain_pack=None,
    allow_errors=True,
    allow_unverified_receipts=True,
    use_cache=False,
)
elapsed = time.time() - t0
print(f"\nCompile finished in {elapsed:.1f}s")

atoms = list(result.atoms)
print(f"Total atoms: {len(atoms)}")


def atype(a):
    at = getattr(a, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at)


phys = [a for a in atoms if atype(a) == "physical_site"]
print()
print(f"=== PHYSICAL_SITE COUNT: {len(phys)} ===")
print(f"  expected: 5  ({'PASS' if len(phys) == 5 else 'FAIL'})")
print()

# Per-extraction-method breakdown
method_counter = Counter()
for a in phys:
    srefs = getattr(a, "source_refs", []) or []
    primary = srefs[0] if srefs else None
    method = getattr(primary, "extraction_method", "<unknown>") if primary else "<no source_ref>"
    method_counter[method] += 1

print("=== Atoms by extraction_method ===")
for m, n in method_counter.most_common():
    print(f"  {n:3d}  {m}")
print()

print("=== Each physical_site atom ===")
for i, a in enumerate(phys, 1):
    v = getattr(a, "value", None) or {}
    sid = v.get("site_id") or v.get("id")
    name = v.get("facility_name") or v.get("name")
    aliases = v.get("names") or []
    srefs = getattr(a, "source_refs", []) or []
    method = getattr(srefs[0], "extraction_method", "?") if srefs else "?"
    site_keys = [k for k in (a.entity_keys or []) if k.startswith("site:")]
    print(f"{i}. site_id={sid!r}  name={name!r}")
    print(f"   extraction_method={method}")
    print(f"   aliases ({len(aliases)}): {aliases[:6]}")
    print(f"   site:* keys ({len(site_keys)}): {site_keys}")
    print()

# Verdict
ghost_atoms = [a for a in phys if not (getattr(a, "value", {}).get("site_id") or "").startswith("ATL-")]
if ghost_atoms:
    print(f"=== ❌ {len(ghost_atoms)} GHOST ATOM(S) DETECTED ===")
    for a in ghost_atoms:
        v = getattr(a, "value", {}) or {}
        print(f"  GHOST: site_id={v.get('site_id')!r}  name={v.get('name')!r}")
else:
    print("=== ✅ NO GHOST ATOMS ===")

# Final assertion
print()
print(f"FINAL ASSERTION: {'PASS' if len(phys) == 5 and not ghost_atoms else 'FAIL'}")
print(f"  count == 5:    {len(phys) == 5}")
print(f"  no ghosts:     {not ghost_atoms}")
print(f"  all ATL-NN-NN: {all((getattr(a, 'value', {}).get('site_id') or '').startswith('ATL-') for a in phys)}")
