"""One row, one quantity: read "Hardware to be Installed | 1 UKG DX Clock" by shape.

Every one of the ten Marion County SOWs states its device count in a labelled
property row:

    Hardware to be Installed (By Category) | 1 UKG DX Clock | 1 UKG DX Clock | 1 UKG DX Clock

(the value repeats because the cell is merged across three grid columns). The
BOM table schema keys on HEADER columns -- "qty", "quantity", "count" -- and
this table's header is the merged section title "Requestor Information"
repeated, so no schema fires and the row is left to the LLM, which promoted 3
of the 10 to bom_line and left 7 as scope_item. Ten clocks became three.

Shape decides: a value that is a small integer followed by an item phrase with
no further digits is a quantity. No label vocabulary -- "Hardware", "Equipment",
"Devices", an unlabelled cell all read the same. "1 hr 28 min" (a duration),
"601 Gurley St" (an address), "94575001" (an account code) and "8/19/26" (a
date) all fail the shape and are not taken.
"""

from __future__ import annotations

import re
from typing import Any

from app.parsers.value_shapes import classify_value

# small integer, space, an item phrase that starts with a letter and contains
# no further digits ("1 UKG DX Clock" yes; "1 hr 28 min", "2 x 10ft" no)
_QTY = re.compile(r"^(?P<n>\d{1,4})\s+(?P<item>[A-Za-z][^\d|]{1,80})$")
_LABEL_LIKE = re.compile(r"[:?#]\s*$")


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def quantity_from_cell(cell: Any) -> dict[str, Any] | None:
    s = _norm(cell)
    if not s or classify_value(s) is not None:
        return None
    m = _QTY.match(s)
    if not m:
        return None
    n = int(m.group("n"))
    item = _norm(m.group("item")).rstrip(".,;")
    if n < 1 or len(item) < 2 or not re.search(r"[A-Za-z]{2}", item):
        return None
    return {"kind": "bom_line", "quantity": n, "item": item, "description": s}


def quantities_from_property_row(cells: dict | None) -> list[dict[str, Any]]:
    """Distinct quantity values in one row, each paired with the label before it.

    Requires a non-empty, label-shaped cell immediately before the value, so a
    bare number in a data table is not mistaken for a labelled quantity. Merged
    cells repeat the value; identical repeats collapse to one.
    """
    if not isinstance(cells, dict) or not cells:
        return []
    values = [_norm(v) for v in cells.values()]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i in range(1, len(values)):
        label, val = values[i - 1], values[i]
        if not label or quantity_from_cell(label) is not None or classify_value(label) is not None:
            continue
        q = quantity_from_cell(val)
        if not q or q["description"] in seen:
            continue
        seen.add(q["description"])
        q["label"] = _LABEL_LIKE.sub("", label).strip()
        out.append(q)
    return out
