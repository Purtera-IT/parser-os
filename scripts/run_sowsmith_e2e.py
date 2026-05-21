"""Full E2E pipeline verification — parser -> compile -> envelope ->
downstream consumers (SOWSmith / OrbitBrief.scope_truth / RunbookGen /
AtlasDispatch / VisionQC).

For each of the 112 packages, audits:

  1. compile_project succeeds
  2. orbitbrief envelope writes
  3. packets are emitted
  4. each packet has a PacketCertificate
  5. each cert's blast_radius declares the right downstream consumers
  6. SOWSmith.scope_clause appears for scope-relevant packet families
  7. OrbitBrief.scope_truth appears for scope-truth families
  8. RunbookGen.site_steps / AtlasDispatch.site_readiness coverage
  9. Receipt verification (source_replay status)

Prints per-package consumer coverage + aggregate stats.
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


_DOWNSTREAM_CONSUMERS = {
    "OrbitBrief.scope_truth",
    "SOWSmith.scope_clause",
    "SOWSmith.exclusion_clause",
    "RunbookGen.site_steps",
    "AtlasDispatch.site_readiness",
    "VisionQC.photo_requirements",
}


def audit_bundle(bundle_path: Path):
    """Compile + envelope a bundle and audit the downstream-consumer
    coverage. Returns a dict of per-bundle stats or an error string.
    """
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope, write_orbitbrief_envelope

    base = Path(tempfile.gettempdir()) / f"e2e_{bundle_path.name}_{os.getpid()}"
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

    # Receipt status (atoms carry receipts)
    receipt_verified = receipt_unsupported = receipt_failed = 0
    for a in r.atoms:
        for rec in (getattr(a, "receipts", None) or []):
            status = (getattr(rec, "replay_status", "") or "").lower()
            if status == "verified":
                receipt_verified += 1
            elif status == "unsupported":
                receipt_unsupported += 1
            elif status == "failed":
                receipt_failed += 1

    # Packet certificates (each EvidencePacket carries its own cert)
    certs = []
    families = Counter()
    blast_radius_keys = Counter()
    for pkt in (r.packets or []):
        cert = getattr(pkt, "certificate", None)
        if cert is None:
            continue
        certs.append(cert)
        fam = getattr(pkt, "family", None) or getattr(pkt, "packet_family", None) or "?"
        families[str(fam)] += 1
        for br in (getattr(cert, "blast_radius", []) or []):
            blast_radius_keys[br] += 1

    # Build envelope and confirm it contains packets + certificates
    try:
        env = build_orbitbrief_envelope(project_dir=base, compile_result=r)
        paths = write_orbitbrief_envelope(project_dir=base, envelope=env, out_dir=base / ".ob")
        env_json = json.loads(paths[0].read_text(encoding="utf-8")) if paths else {}
    except Exception as e:
        return None, f"envelope_crash: {type(e).__name__}: {e}"

    env_packets = env_json.get("packets", [])
    env_packet_certs = sum(1 for p in env_packets if p.get("certificate"))

    # SOWSmith / OrbitBrief downstream coverage
    sowsmith_scope = blast_radius_keys.get("SOWSmith.scope_clause", 0)
    sowsmith_exclusion = blast_radius_keys.get("SOWSmith.exclusion_clause", 0)
    orbitbrief_scope_truth = blast_radius_keys.get("OrbitBrief.scope_truth", 0)
    runbookgen = blast_radius_keys.get("RunbookGen.site_steps", 0)
    atlas_dispatch = blast_radius_keys.get("AtlasDispatch.site_readiness", 0)
    visionqc = blast_radius_keys.get("VisionQC.photo_requirements", 0)

    return {
        "atoms": len(r.atoms),
        "packets": len(r.packets),
        "certs": len(certs),
        "env_packets": len(env_packets),
        "env_packets_with_cert": env_packet_certs,
        "families": dict(families),
        "sowsmith_scope": sowsmith_scope,
        "sowsmith_exclusion": sowsmith_exclusion,
        "orbitbrief_scope_truth": orbitbrief_scope_truth,
        "runbookgen": runbookgen,
        "atlas_dispatch": atlas_dispatch,
        "visionqc": visionqc,
        "receipt_verified": receipt_verified,
        "receipt_unsupported": receipt_unsupported,
        "receipt_failed": receipt_failed,
    }, None


def list_packages():
    bundles: list[tuple[str, Path]] = []
    individual: list[tuple[str, Path]] = []
    dom = Path(r"C:\Users\lilli\AppData\Local\Temp\domain_packages")
    if dom.exists():
        for sub in sorted(dom.iterdir()):
            if sub.is_dir() and (sub / "artifacts").exists():
                bundles.append((f"domain/{sub.name}", sub))
    optbot = Path(r"C:\Users\lilli\AppData\Local\Temp\optbot_atl_v8")
    if optbot.exists() and (optbot / "artifacts").exists():
        bundles.append(("optbot_atl_v8", optbot))
    xdoc = Path(r"C:\Users\lilli\AppData\Local\Temp\xdoc_conflict")
    if xdoc.exists() and (xdoc / "artifacts").exists():
        bundles.append(("xdoc_conflict", xdoc))
    sc = Path(r"C:\Users\lilli\AppData\Local\Temp\site_roster_corpus")
    if sc.exists():
        d8 = sc / "08_site_roster_and_facilities_authoritative.pdf"
        if d8.exists():
            individual.append(("site_roster/doc08", d8))
        mocks = sc / "mock_site_rosters"
        if mocks.exists():
            for f in sorted(mocks.glob("site_roster_*.pdf")):
                individual.append((f"site_roster/{f.stem}", f))
    for src, lbl in [
        (r"C:\Users\lilli\AppData\Local\Temp\ms_stress\artifacts", "ms_r1"),
        (r"C:\Users\lilli\AppData\Local\Temp\ms_r2\artifacts", "ms_r2"),
        (r"C:\Users\lilli\AppData\Local\Temp\xlsx_stress\artifacts", "xlsx"),
        (r"C:\Users\lilli\AppData\Local\Temp\pdf_ct\artifacts", "pdf_ct"),
        (r"C:\Users\lilli\AppData\Local\Temp\pdf_safety\artifacts", "pdf_safe"),
        (r"C:\Users\lilli\AppData\Local\Temp\docx_stress\artifacts", "docx"),
    ]:
        d = Path(src)
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in (".pdf", ".xlsx", ".docx"):
                    individual.append((f"{lbl}/{f.stem}", f))
    return bundles, individual


def main() -> int:
    bundles, individual = list_packages()
    cases = bundles + individual
    print(f"=== SOWSmith / OrbitBrief E2E audit: {len(cases)} packages ===\n")

    pass_count = 0
    fail_count = 0
    no_packet_count = 0
    failures: list[tuple[str, str]] = []

    total_atoms = total_packets = total_certs = total_env_packets = total_env_certs = 0
    total_sowsmith_scope = total_sowsmith_exclusion = 0
    total_orbitbrief_scope_truth = 0
    total_runbookgen = total_atlas = total_visionqc = 0
    total_receipt_verified = total_receipt_unsupported = total_receipt_failed = 0
    family_totals: Counter[str] = Counter()
    bundles_with_sowsmith = 0
    bundles_with_orbitbrief_scope_truth = 0

    start = time.time()
    for name, p in cases:
        out, err = audit_bundle(p)
        if err:
            fail_count += 1
            failures.append((name, err[:200]))
            print(f"  CRASH    {name:<45} {err[:80]}")
            continue
        total_atoms += out["atoms"]
        total_packets += out["packets"]
        total_certs += out["certs"]
        total_env_packets += out["env_packets"]
        total_env_certs += out["env_packets_with_cert"]
        total_sowsmith_scope += out["sowsmith_scope"]
        total_sowsmith_exclusion += out["sowsmith_exclusion"]
        total_orbitbrief_scope_truth += out["orbitbrief_scope_truth"]
        total_runbookgen += out["runbookgen"]
        total_atlas += out["atlas_dispatch"]
        total_visionqc += out["visionqc"]
        total_receipt_verified += out["receipt_verified"]
        total_receipt_unsupported += out["receipt_unsupported"]
        total_receipt_failed += out["receipt_failed"]
        for f, n in out["families"].items():
            family_totals[f] += n
        if out["sowsmith_scope"] > 0:
            bundles_with_sowsmith += 1
        if out["orbitbrief_scope_truth"] > 0:
            bundles_with_orbitbrief_scope_truth += 1

        # Cert integrity: packets in envelope should each have a certificate
        if out["env_packets"] > 0 and out["env_packets_with_cert"] != out["env_packets"]:
            fail_count += 1
            failures.append((name, f"only {out['env_packets_with_cert']}/{out['env_packets']} envelope packets have a cert"))
            print(f"  FAIL     {name:<45} {out['env_packets_with_cert']}/{out['env_packets']} envelope packets carry cert")
            continue

        if out["packets"] == 0:
            no_packet_count += 1
            print(f"  no-pkt   {name:<45} atoms={out['atoms']:<3} (no scope-bearing pairs)")
            continue

        pass_count += 1
        print(
            f"  PASS     {name:<45} pkts={out['packets']:<2} certs={out['certs']:<2} "
            f"SOWSmith={out['sowsmith_scope']:<2} ScopeTruth={out['orbitbrief_scope_truth']:<2} "
            f"Runbook={out['runbookgen']:<2}"
        )

    elapsed = time.time() - start
    print()
    print("=" * 80)
    print(f"TOTAL: {len(cases)} packages tested  |  PASS: {pass_count}  |  no-packets: {no_packet_count}  |  FAIL: {fail_count}")
    print()
    print("AGGREGATE PIPELINE STATE")
    print(f"  atoms emitted:                        {total_atoms}")
    print(f"  packets emitted:                      {total_packets}")
    print(f"  packet certificates:                  {total_certs}")
    print(f"  envelope packets:                     {total_env_packets}")
    print(f"  envelope packets WITH cert:           {total_env_certs} ({100*total_env_certs/max(1,total_env_packets):.0f}%)")
    print()
    print("DOWNSTREAM CONSUMER COVERAGE (blast_radius declarations)")
    print(f"  OrbitBrief.scope_truth:               {total_orbitbrief_scope_truth} packets across {bundles_with_orbitbrief_scope_truth} bundles")
    print(f"  SOWSmith.scope_clause:                {total_sowsmith_scope} packets across {bundles_with_sowsmith} bundles")
    print(f"  SOWSmith.exclusion_clause:            {total_sowsmith_exclusion} packets")
    print(f"  RunbookGen.site_steps:                {total_runbookgen} packets")
    print(f"  AtlasDispatch.site_readiness:         {total_atlas} packets")
    print(f"  VisionQC.photo_requirements:          {total_visionqc} packets")
    print()
    print("PACKET FAMILY DISTRIBUTION")
    for fam, n in family_totals.most_common():
        print(f"  {fam:<40} {n}")
    print()
    print("RECEIPT VERIFICATION (source_replay)")
    rec_total = total_receipt_verified + total_receipt_unsupported + total_receipt_failed
    if rec_total > 0:
        print(f"  verified:    {total_receipt_verified} ({100*total_receipt_verified/rec_total:.1f}%)")
        print(f"  unsupported: {total_receipt_unsupported} ({100*total_receipt_unsupported/rec_total:.1f}%)")
        print(f"  failed:      {total_receipt_failed} ({100*total_receipt_failed/rec_total:.1f}%)")
    print()
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 80)
    if failures:
        print("\nFailures:")
        for n, e in failures[:30]:
            print(f"  - {n}: {e}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
