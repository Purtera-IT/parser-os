"""Generate LEGEND.md from the overlay registry."""
from __future__ import annotations

from pathlib import Path

from orbitbrief_page_os.segmentation.overlay.registry import LEGEND_ENTRIES


def _bgr_text(bgr: tuple[int, int, int] | None) -> str:
    return "TBD" if bgr is None else f"BGR {bgr}"


def build_markdown() -> str:
    lines: list[str] = []
    lines.append("# Parser OS Overlay Legend v2")
    lines.append("")
    lines.append("This file is generated from `orbitbrief_page_os.segmentation.overlay.registry`. ")
    lines.append("Production entries document current behavior; reserved entries are intentional extension slots.")
    lines.append("")
    lines.append("## Production layers")
    lines.append("")
    for e in LEGEND_ENTRIES:
        if e.status != "production":
            continue
        lines.append(f"### {e.label}")
        lines.append("")
        lines.append(f"- Layer: `{e.layer.name}`")
        lines.append(f"- Style: {_bgr_text(e.bgr)}; {e.linestyle}" + (f"; hatch: {e.hatch}" if e.hatch else ""))
        lines.append(f"- Meaning: {e.meaning}")
        lines.append(f"- Extraction rule: {e.extraction_rule}")
        lines.append(f"- Golden tests: {', '.join(e.tests)}")
        if e.notes:
            lines.append(f"- Notes: {e.notes}")
        lines.append("")
    lines.append("## Reserved v2 semantic slots")
    lines.append("")
    lines.append("| Color/style key | Layer | Meaning | Extraction rule | Golden tests |")
    lines.append("|---|---:|---|---|---|")
    for e in LEGEND_ENTRIES:
        if e.status != "reserved":
            continue
        style = f"{_bgr_text(e.bgr)}; {e.linestyle}"
        if e.hatch:
            style += f"; {e.hatch} hatch"
        lines.append(
            f"| {e.label}<br>{style} | `{e.layer.name}` | {e.meaning} | {e.extraction_rule} | {', '.join(e.tests)} |"
        )
    lines.append("")
    lines.append("## Add-a-layer checklist")
    lines.append("")
    lines.append("1. Add or enable one pass module under `segmentation/passes/`.")
    lines.append("2. Assign exactly one `OverlayLayer` bit in `overlay_layers.py`.")
    lines.append("3. Add a `LegendEntry` here with color, linestyle/hatch, extraction rule, and test IDs.")
    lines.append("4. Add at least one golden PDF case and structured-box/pixel regression assertion.")
    lines.append("5. Regenerate this file with `python -m orbitbrief_page_os.segmentation.legend.generate_legend --out LEGEND.md`.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="LEGEND.md")
    args = ap.parse_args()
    Path(args.out).write_text(build_markdown())
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
