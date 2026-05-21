"""Run 100+ packages through the full pipeline (parser -> compile -> envelope).

For each artifact OR bundle, captures:
  - crash status
  - atom / entity / packet / edge counts
  - physical_site atom count
  - cross-doc conflict edges

Reports any bundle that fails to compile or produces zero atoms.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def compile_bundle(bundle_path: Path):
    """Compile a bundle directory containing artifacts/ subdir."""
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope, write_orbitbrief_envelope
    base = Path(tempfile.gettempdir()) / f"p100_{bundle_path.name}_{os.getpid()}"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    art = base / "artifacts"
    if (bundle_path / "artifacts").exists():
        shutil.copytree(bundle_path / "artifacts", art)
    elif bundle_path.is_file():
        art.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_path, art / bundle_path.name)
    else:
        return None, "no_artifacts_dir"
    try:
        r = compile_project(
            project_dir=base, domain_pack=None,
            use_cache=False, allow_errors=True, allow_unverified_receipts=True,
        )
    except Exception as e:
        return None, f"compile_crash: {type(e).__name__}: {e}"
    try:
        env = build_orbitbrief_envelope(project_dir=base, compile_result=r)
        envelope_paths = write_orbitbrief_envelope(project_dir=base, envelope=env, out_dir=base / ".ob")
        envelope_json = {}
        if envelope_paths:
            envelope_json = json.loads(envelope_paths[0].read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"envelope_crash: {type(e).__name__}: {e}"
    physical_sites = []
    def walk(n):
        if isinstance(n, dict):
            if n.get("kind") == "physical_site":
                physical_sites.append(n.get("site_id") or "")
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(envelope_json)
    fams = Counter()
    for e in envelope_json.get("edges", []):
        fam = (e.get("metadata") or {}).get("edge_family", "?")
        fams[fam] += 1
    return {
        "atoms": len(envelope_json.get("atoms", [])),
        "entities": len(envelope_json.get("entities", [])),
        "packets": len(envelope_json.get("packets", [])),
        "edges": len(envelope_json.get("edges", [])),
        "physical_sites": len(physical_sites),
        "unique_site_ids": len(set(s for s in physical_sites if s)),
        "edge_families": dict(fams),
    }, None


def list_all_packages():
    """Enumerate every adversarial + domain package on disk."""
    bundles: list[tuple[str, Path]] = []
    individual: list[tuple[str, Path]] = []

    # Multi-artifact bundles
    domain_dir = Path(r"C:\Users\lilli\AppData\Local\Temp\domain_packages")
    if domain_dir.exists():
        for sub in sorted(domain_dir.iterdir()):
            if sub.is_dir() and (sub / "artifacts").exists():
                bundles.append((f"domain/{sub.name}", sub))

    optbot = Path(r"C:\Users\lilli\AppData\Local\Temp\optbot_atl_v8")
    if optbot.exists() and (optbot / "artifacts").exists():
        bundles.append(("optbot_atl_v8", optbot))

    site_corpus = Path(r"C:\Users\lilli\AppData\Local\Temp\site_roster_corpus")
    if site_corpus.exists():
        doc08 = site_corpus / "08_site_roster_and_facilities_authoritative.pdf"
        if doc08.exists():
            individual.append(("site_roster/doc08", doc08))
        mocks = site_corpus / "mock_site_rosters"
        if mocks.exists():
            for f in sorted(mocks.glob("site_roster_*.pdf")):
                individual.append((f"site_roster/{f.stem}", f))

    xdoc = Path(r"C:\Users\lilli\AppData\Local\Temp\xdoc_conflict")
    if xdoc.exists() and (xdoc / "artifacts").exists():
        bundles.append(("xdoc_conflict", xdoc))

    # Single-artifact adversarial PDFs and XLSX
    for source_dir, label in [
        (r"C:\Users\lilli\AppData\Local\Temp\ms_stress\artifacts", "ms_r1"),
        (r"C:\Users\lilli\AppData\Local\Temp\ms_r2\artifacts", "ms_r2"),
        (r"C:\Users\lilli\AppData\Local\Temp\xlsx_stress\artifacts", "xlsx"),
        (r"C:\Users\lilli\AppData\Local\Temp\pdf_ct\artifacts", "pdf_ct"),
        (r"C:\Users\lilli\AppData\Local\Temp\pdf_safety\artifacts", "pdf_safe"),
        (r"C:\Users\lilli\AppData\Local\Temp\docx_stress\artifacts", "docx"),
    ]:
        d = Path(source_dir)
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in (".pdf", ".xlsx", ".docx"):
                individual.append((f"{label}/{f.stem}", f))

    return bundles, individual


def main() -> int:
    bundles, individual = list_all_packages()
    print(f"=== Universal stress: {len(bundles)} bundles + {len(individual)} individual artifacts ===\n")
    pass_count = fail_count = crash_count = 0
    total = 0
    failures: list[tuple[str, str]] = []
    site_totals = 0
    site_unique_totals = 0
    cross_doc_totals = 0
    start = time.time()
    for name, p in bundles + individual:
        total += 1
        out, err = compile_bundle(p)
        if err:
            crash_count += 1
            fail_count += 1
            failures.append((name, err[:200]))
            print(f"  CRASH  {name:<45} {err[:80]}")
            continue
        if out["atoms"] == 0:
            # Empty / safety / metadata-only PDFs are OK
            if any(tag in name for tag in ("pdf_safe", "r2_r_empty", "r2_s_only_headers", "msa_boilerplate", "draft", "dd_multi_section")):
                pass_count += 1
                print(f"  PASS   {name:<45} (intentionally empty)")
                continue
            fail_count += 1
            failures.append((name, "0 atoms"))
            print(f"  FAIL   {name:<45} 0 atoms")
            continue
        pass_count += 1
        site_totals += out["physical_sites"]
        site_unique_totals += out["unique_site_ids"]
        cross_doc_totals += out["edge_families"].get("device_quantity_cross_doc", 0)
        print(f"  PASS   {name:<45} atoms={out['atoms']:<4} ents={out['entities']:<3} sites={out['unique_site_ids']:<2} packets={out['packets']}")
    elapsed = time.time() - start
    print()
    print("=" * 80)
    print(f"TOTAL: {total} packages tested  |  PASS: {pass_count}  |  FAIL: {fail_count}  |  CRASH: {crash_count}")
    print(f"Aggregate physical_site atoms: {site_totals} (unique IDs: {site_unique_totals})")
    print(f"Cross-doc quantity conflicts surfaced: {cross_doc_totals}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 80)
    if failures:
        print("\nFailures:")
        for n, e in failures[:30]:
            print(f"  - {n}: {e}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
