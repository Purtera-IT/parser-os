"""Fast vision-detector smoke: one page, real PDF, full pipeline.

Bypasses the full parser. Just runs:
  legend_locator → legend_parser → legend_symbol_crops → region_proposals → vision_symbol_detector

Usage:
  python scripts/smoke_vision_one_page.py <pdf> <legend_page> <target_page> [--max-regions=N]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 4:
        print("Usage: python scripts/smoke_vision_one_page.py <pdf> <legend_page> <target_page> [--max-regions=N]", file=sys.stderr)
        return 2

    pdf_path = Path(sys.argv[1]).resolve()
    legend_page_idx = int(sys.argv[2])
    target_page_idx = int(sys.argv[3])
    max_regions = 20
    for arg in sys.argv[4:]:
        if arg.startswith("--max-regions="):
            max_regions = int(arg.split("=", 1)[1])

    print(f"[mini] pdf            = {pdf_path}")
    print(f"[mini] legend_page    = {legend_page_idx}")
    print(f"[mini] target_page    = {target_page_idx}")
    print(f"[mini] max_regions    = {max_regions}")
    print(f"[mini] OLLAMA         = {os.environ.get('OLLAMA_BASE_URL', '(default)')}")

    import fitz
    from orbitbrief_page_os.segmentation.schematic.legend_locator import (
        locate_legend_candidates,
        page_text_blocks,
    )
    from orbitbrief_page_os.segmentation.schematic.legend_parser import parse_legend
    from orbitbrief_page_os.segmentation.schematic.legend_symbol_crops import (
        extract_legend_symbol_crops,
    )
    from orbitbrief_page_os.segmentation.schematic.region_proposals import (
        propose_regions,
    )
    from orbitbrief_page_os.segmentation.schematic.vision_symbol_detector import (
        detect_symbols_via_vision,
        is_vision_endpoint_reachable,
    )

    print(f"[mini] reachable      = {is_vision_endpoint_reachable()}")

    doc = fitz.open(str(pdf_path))
    try:
        # Step 1: parse legend on the legend page
        legend_page = doc.load_page(legend_page_idx)
        legend_blocks = page_text_blocks(legend_page)
        legend_cands = locate_legend_candidates(
            page_index=legend_page_idx, blocks=legend_blocks
        )
        print(f"[mini] legend_candidates = {len(legend_cands)}")
        legend = None
        for c in sorted(legend_cands, key=lambda x: -x.score):
            legend = parse_legend(
                candidate=c, page_blocks=legend_blocks, sheet_number=None, scope="global"
            )
            if legend is not None:
                break
        if legend is None:
            print("[mini] no legend parsed; abort")
            return 1
        print(f"[mini] legend entries     = {len(legend.entries)}")
        for e in legend.entries[:5]:
            print(f"  - {e.normalized_symbol_text!r:15} {e.label_text}")

        # Step 2: extract symbol crops
        out_dir = pdf_path.parent / f"{pdf_path.stem}.derived"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        crops = extract_legend_symbol_crops(
            legends=[legend],
            pdf_path=pdf_path,
            out_dir=out_dir,
        )
        print(f"[mini] legend crops       = {len(crops)} ({time.monotonic()-t0:.1f}s)")

        # Step 3: propose regions on target page
        target_page = doc.load_page(target_page_idx)
        t0 = time.monotonic()
        proposals = propose_regions(
            page=target_page, page_index=target_page_idx, max_proposals=max_regions
        )
        print(f"[mini] region proposals   = {len(proposals)} ({time.monotonic()-t0:.1f}s)")

        # Step 4: run vision on every proposal
        cache_path = pdf_path.parent / ".orbitbrief_vision_detect_cache.jsonl"
        t0 = time.monotonic()
        vision_dets = detect_symbols_via_vision(
            page=target_page,
            page_index=target_page_idx,
            region_proposals=proposals,
            legend_crops=crops,
            cache_path=cache_path,
            max_regions=max_regions,
        )
        elapsed = time.monotonic() - t0
        print(f"[mini] vision detections  = {len(vision_dets)} in {elapsed:.1f}s")
        for vd in vision_dets:
            print(
                f"  - sym={vd.matched_symbol_text!r:8} "
                f"label={vd.matched_label_text!r:25} "
                f"conf={vd.confidence:.2f} "
                f"bbox=({vd.bbox_pdf[0]:.0f},{vd.bbox_pdf[1]:.0f},{vd.bbox_pdf[2]:.0f},{vd.bbox_pdf[3]:.0f})"
            )
    finally:
        doc.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
