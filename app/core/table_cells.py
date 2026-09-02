"""Map a table row onto its column names without losing a cell.

`dict(zip(columns, row))` looks obvious and silently discards data whenever two
columns share a name — which happens constantly, because a merged header cell
spanning N columns is expanded into the same text N times by python-docx and by
every other table reader that flattens spans.

Measured on the dev corpus 2026-09-02: 204 of 418 docx (49%) across 99 deals
contained at least one such table, and 3,836 cell values were being dropped. On
deal 010215 that was 141 cells per per-site SOW, including
"Address Line 1 | 601 Gurley St" -- so ten site documents produced three atoms
each and no address, and the deal's sites had to be scraped out of email prose
instead, arriving fused and truncated.

The loss was invisible: nothing warned, and raw_text still carried the full row,
so the document looked parsed.

This lives in core rather than in one parser because the same zip appears in the
docx parser AND in table_schema_registry, which is fed `_columns` by the docx,
xlsx and quote parsers alike. Fixing it in one parser would have left the same
bug standing for every other format.
"""

from __future__ import annotations

from typing import Any, Sequence


def cells_by_column(columns: Sequence[Any] | None, row: Sequence[Any] | None) -> dict[str, Any]:
    """Pair each cell with its column, disambiguating repeats positionally.

    The first occurrence of a name keeps that name, so readers of well-formed
    tables see no change; later repeats get a ``__1``, ``__2`` suffix. A row
    longer than its header keeps its tail under positional keys rather than
    dropping it.
    """
    values = list(row or [])
    names = list(columns or [])
    if not names:
        return {f"col_{i}": v for i, v in enumerate(values)}

    out: dict[str, Any] = {}
    seen: dict[str, int] = {}
    for i, value in enumerate(values):
        raw = names[i] if i < len(names) else ""
        name = str(raw).strip() or f"col_{i}"
        if name in seen:
            seen[name] += 1
            key = f"{name}__{seen[name]}"
        else:
            seen[name] = 0
            key = name
        out[key] = value
    return out
