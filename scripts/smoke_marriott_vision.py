"""Smoke test: run the parser on Marriott Atlanta DD with vision enabled.

Set:
    OLLAMA_BASE_URL=http://100.114.102.122:11434
    PARSER_OS_VISION_DETECT=1
    PARSER_OS_SCHEMATIC_OVERLAYS=1

Counts how many detections, broken down by modality, plus a per-page
breakdown of overlays produced. Writes a one-page summary to stdout.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    pdf_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if pdf_path_arg is None:
        print("Usage: python -m scripts.smoke_marriott_vision <pdf>", file=sys.stderr)
        return 2
    pdf_path = Path(pdf_path_arg).resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    print(f"[smoke] PDF       = {pdf_path}")
    print(f"[smoke] OLLAMA    = {os.environ.get('OLLAMA_BASE_URL', '(default)')}")
    print(f"[smoke] VISION    = {os.environ.get('PARSER_OS_VISION_DETECT', '0')}")
    print(f"[smoke] OVERLAYS  = {os.environ.get('PARSER_OS_SCHEMATIC_OVERLAYS', '0')}")

    # Reachability ping
    try:
        from orbitbrief_page_os.segmentation.schematic.vision_symbol_detector import (
            is_vision_endpoint_reachable,
        )
        reachable = is_vision_endpoint_reachable()
        print(f"[smoke] reachable = {reachable}")
    except Exception as e:
        print(f"[smoke] vision import failed: {e}")
        reachable = False

    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
    from app.domain.loader import load_domain_pack

    pack_id = os.environ.get("ORBITBRIEF_DOMAIN_PACK", "structured_backbone_fiber_pack")
    pack = load_domain_pack(pack_id)
    print(f"[smoke] domain_pack = {pack.pack_id} (targets={len(pack.detection_targets)})")

    parser = OrbitBriefPdfParser()
    t0 = time.monotonic()
    output = parser.parse_artifact(
        project_id="marriott_smoke",
        artifact_id="marriott_artifact",
        path=pdf_path,
        domain_pack=pack,
    )
    elapsed = time.monotonic() - t0
    print(f"[smoke] parse_artifact: {elapsed:.1f}s, atoms={len(output.atoms)}")

    # Count atoms by type
    by_type: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    for atom in output.atoms:
        atom_type = str(getattr(atom, "atom_type", "unknown"))
        by_type[atom_type] = by_type.get(atom_type, 0) + 1
        value = getattr(atom, "value", {}) or {}
        if isinstance(value, dict):
            mod = value.get("modality")
            if mod:
                by_modality[str(mod)] = by_modality.get(str(mod), 0) + 1

    print("\n[smoke] atoms by type:")
    for k in sorted(by_type, key=lambda x: -by_type[x]):
        print(f"  {by_type[k]:5d}  {k}")

    print("\n[smoke] detection atoms by modality:")
    if not by_modality:
        print("  (no modality-tagged atoms)")
    for k in sorted(by_modality, key=lambda x: -by_modality[x]):
        print(f"  {by_modality[k]:5d}  {k}")

    # Check the derived overlay manifest
    derived_dir = pdf_path.parent / f"{pdf_path.stem}.derived"
    overlays_json = derived_dir / "schematic_overlays.json"
    if overlays_json.exists():
        try:
            manifest = json.loads(overlays_json.read_text())
            overlays = manifest.get("overlays", [])
            print(f"\n[smoke] overlays: {len(overlays)} pages")
            for o in overlays[:25]:
                print(
                    f"  page={o.get('page'):3d}  "
                    f"legends={o.get('legend_count'):3d}  "
                    f"detections={o.get('detection_count'):3d}  "
                    f"file={o.get('relative_path')}"
                )
        except Exception as e:
            print(f"\n[smoke] overlays read failed: {e}")
    else:
        print(f"\n[smoke] no overlays manifest at {overlays_json}")

    # Check vision detect cache
    cache = pdf_path.parent / ".orbitbrief_vision_detect_cache.jsonl"
    if cache.exists():
        try:
            lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
            print(f"\n[smoke] vision cache lines: {len(lines)}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
