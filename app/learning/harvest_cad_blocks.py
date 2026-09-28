"""Collect the labelling set for the `block_category` head.

A drawing's most valuable number is a count of what it DRAWS: SP-6 places 106
`5' DESK` blocks, which confirms geometrically the 106 workstations the whole
$110K quote rests on. Deriving that requires knowing a `5' DESK` is a
workstation, and block names are free text per firm. This corpus already holds
`5' DESK` and also `eqklwmew`, `FDKSJ`, `zw$5C48` and `154487`.

So the regex families in `app/parsers/dwg_geometry.py` are a placeholder in
exactly the sense `_LABELING_DOCTRINE.md` means: a guess put in front of a
labeler, whose confirmations and drops become the training set. `\\bTV\\b`
already failed to match `70TV` -- there is no word boundary between two word
characters -- and that class of miss is why this is learned rather than listed.

Two things make a block name labellable without opening the drawing, and both
are collected here:

    the NAME         what the draughtsman typed
    the GEOMETRY     how many were placed, how far apart, how regularly

The geometry is the point. A block inserted 106 times on a 5-foot grid is
workstations whatever it is called, and that signal is what the head should
lean on when the name is `eqklwmew`. It is also the fallback when the head
abstains.

    python -m app.learning.harvest_cad_blocks --blob     # the whole corpus
    python -m app.learning.harvest_cad_blocks a.dxf b.dxf

Writes one JSONL row per (drawing, block name). `category` is left empty for
a human; `guess` is what the current regex would say, so a labeler accepts or
corrects rather than starting from nothing -- and a CORRECTION is worth more
than a confirmation, because a regex can only produce positives.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

#: The categories a block can carry. Deliberately short: a taxonomy nobody can
#: hold in their head produces labels nobody can trust.
CATEGORIES = (
    "workstation", "seating", "display", "door", "window", "plumbing_fixture",
    "millwork", "appliance", "equipment", "electrical_panel", "device",
    "structure", "annotation", "titleblock", "other",
)


def _guess(name: str) -> str:
    """What the parser's regex families would say today."""
    from app.parsers.dwg_geometry import _COUNTS  # noqa: PLC0415
    import re  # noqa: PLC0415

    for pattern, kind, _ in _COUNTS:
        if re.search(pattern, name, re.I):
            return kind
    return ""


def _spacing(points: list[tuple[float, float]]) -> dict[str, Any]:
    """How regularly were these placed?

    A repeating pitch is the name-independent signature of systems furniture,
    and the pitch itself is a fact worth having: SP-6's schedule says 5'-0"
    workstations, and a 60-inch nearest-neighbour median would corroborate it
    from the geometry alone.
    """
    if len(points) < 3:
        return {"count": len(points)}
    near = []
    for i, (x, y) in enumerate(points):
        best = min(((x - a) ** 2 + (y - b) ** 2
                    for j, (a, b) in enumerate(points) if j != i), default=0.0)
        near.append(best ** 0.5)
    med = statistics.median(near)
    # A grid is regular; a scatter is not. Low spread against the median is
    # what separates "106 desks in rows" from "106 unrelated symbols".
    spread = statistics.pstdev(near) / med if med else 0.0
    return {
        "count": len(points),
        "nearest_median": round(med, 2),
        "nearest_spread": round(spread, 3),
        "regular_grid": bool(med > 0 and spread < 0.35),
    }


def read_drawing(path: Path) -> list[dict[str, Any]]:
    """One row per block name in one drawing."""
    import ezdxf  # noqa: PLC0415

    from app.parsers.cad_layers import follows_standard  # noqa: PLC0415
    from app.parsers.dwg_geometry import feet_per_unit  # noqa: PLC0415

    doc = ezdxf.readfile(str(path))
    ft = feet_per_unit(doc)
    layer_names = [str(l.dxf.name) for l in doc.layers]
    standard = follows_standard(layer_names)

    placed: dict[str, list[tuple[float, float]]] = {}
    layers: dict[str, set] = {}
    for e in doc.modelspace():
        if e.dxftype() != "INSERT":
            continue
        try:
            p = e.dxf.insert
            placed.setdefault(e.dxf.name, []).append((float(p.x), float(p.y)))
            layers.setdefault(e.dxf.name, set()).add(str(e.dxf.layer))
        except Exception:  # noqa: BLE001
            continue

    rows = []
    for name, points in sorted(placed.items(), key=lambda kv: -len(kv[1])):
        geom = _spacing(points)
        if ft and geom.get("nearest_median"):
            geom["nearest_median_ft"] = round(geom["nearest_median"] * ft, 2)
        rows.append({
            "drawing": path.name,
            "block_name": name,
            "layers": sorted(layers.get(name, ())),
            "geometry": geom,
            # A drawing whose layers follow the standard gives the head a
            # second, independent hint; one that does not gives it only the
            # name and the spacing, which is the hard case worth training on.
            "layer_standard": standard,
            "units_ft_per_unit": ft,
            "guess": _guess(name),
            "category": "",          # <- the label
            "note": "",
        })
    return rows


def _corpus_paths() -> list[str]:
    """Every CAD file in the deal corpus. Four, today."""
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415
    import os  # noqa: PLC0415

    conn = os.environ.get("SOWSMITH_BLOB_CONN") or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise SystemExit("set SOWSMITH_BLOB_CONN to scan the corpus")
    client = BlobServiceClient.from_connection_string(conn)
    container = client.get_container_client("orbitbrief-artifacts")
    return [b.name for b in container.list_blobs(name_starts_with="deals/")
            if b.name.lower().endswith((".dwg", ".dxf"))]


def main(argv: list[str]) -> int:
    out = Path("cad_blocks.jsonl")
    if "--blob" in argv:
        print("\n".join(_corpus_paths()))
        print("\nDownload these, convert any .dwg with dwg2dxf 0.14+, then "
              "re-run on the .dxf paths.")
        return 0
    paths = [Path(a) for a in argv if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 2
    rows: list[dict[str, Any]] = []
    for p in paths:
        try:
            rows.extend(read_drawing(p))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {p.name}: {type(exc).__name__}: {exc}")
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    labelled = sum(1 for r in rows if r["guess"])
    print(f"{len(rows)} block rows from {len(paths)} drawing(s) -> {out}")
    print(f"   {labelled} carry a regex guess; {len(rows) - labelled} are "
          f"unguessed and are the rows worth a human's time first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
