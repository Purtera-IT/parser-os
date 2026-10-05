"""Checkbox cells in a table: their own facts, never part of a name.

A site list carries a service-type column of Word / Excel checkboxes:

    | Site Location | Service Type                                         |
    | Delphos, OH   | ☐ Assessment ☐ Configuration ☒ Installation ☐ Decom… |

Joined into the row, that read "Delphos, OH | ☐ Assessment ☐ Configuration…"
-- one atom that was neither a site nor a service selection, and a site name
with checkbox text glued onto it. The checkbox cell is a separate fact: which
services were ticked for that row. This module recognises such a cell and
reads it; the parsers emit it as its own atom beside the row's name/address.
"""
from __future__ import annotations

import re
from typing import Any

#: Unchecked / checked glyphs Word, Excel and PDF-to-text produce.
_UNCHECKED = "☐□❑❒○◯"
_CHECKED = "☑☒✓✔✗✘■▣●◉"
_GLYPHS = _UNCHECKED + _CHECKED
_BRACKET = r"\[\s*[xX✓✔]?\s*\]"

_OPTION_RE = re.compile(
    rf"(?P<box>[{_GLYPHS}]|{_BRACKET})\s*(?P<label>[^{_GLYPHS}\[\]|]+?)\s*(?=[{_GLYPHS}]|{_BRACKET}|$)"
)

_SITE_HEADER_RE = re.compile(
    r"\b(?:site|location|facility|store|branch|building|campus|school|office|address|city)\b", re.I
)


def checkbox_options(text: str) -> list[tuple[bool, str]] | None:
    """[(checked, label), ...] when ``text`` is a checkbox cell, else None.

    A checkbox cell is made of box + label pairs and nothing else: at least
    one box, every box followed by a short label, no prose around them.
    """
    s = " ".join(str(text or "").split())
    if not s or not re.search(rf"[{_GLYPHS}]|{_BRACKET}", s):
        return None
    opts: list[tuple[bool, str]] = []
    consumed = 0
    for m in _OPTION_RE.finditer(s):
        label = m.group("label").strip(" ,;:")
        if not label or len(label) > 60:
            return None
        box = m.group("box")
        checked = (box in _CHECKED) or bool(re.search(r"[xX✓✔]", box))
        opts.append((checked, label))
        consumed += len(m.group(0))
    lead = s[: s.find(next(iter(re.findall(rf"[{_GLYPHS}]|{_BRACKET}", s))))].strip()
    # A label before the first box ("Service: ☐ A ☐ B") is allowed; prose is not.
    if lead and (len(lead.split()) > 4 or not lead.endswith(":")):
        return None
    if not opts:
        return None
    # A lone filled square or tick is as often a bullet as a ticked box:
    # without an empty box, a bracket box or a ballot-box glyph (☑ ☒) beside
    # it, one option is not a checkbox cell.
    if len(opts) < 2 and not re.search(rf"[{_UNCHECKED}☑☒]|{_BRACKET}", s):
        return None
    return opts


def is_checkbox_cell(text: str) -> bool:
    return checkbox_options(text) is not None


def checkbox_value(text: str, *, column: str = "", subject: str = "") -> dict[str, Any]:
    """The structured value of a checkbox cell's atom."""
    opts = checkbox_options(text) or []
    col = (column or "").strip()
    return {
        "kind": "checkbox_selection",
        "attribute_kind": re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_") or "checkbox",
        "column": col,
        "subject": subject,
        "site": subject,
        "selected": [l for c, l in opts if c],
        "not_selected": [l for c, l in opts if not c],
        "options": [{"label": l, "checked": c} for c, l in opts],
    }


#: A grid row of checkbox cells is ONE atom ("Delphos, OH: Support"): its
#: label cell, then the ticked options. The unticked options are never atoms
#: and never in the text; with this off they stay on the row atom as
#: structure (``not_selected``, and ``options`` with ``checked: False``), so a
#: requested service the document left unticked can still be flagged. Turn it
#: on to drop them fully: the row atom then names only the ticked options and
#: keeps no raw checkbox cell text.
DROP_UNTICKED_FULLY = False

#: The text after the label when no box in the row is ticked.
NONE_SELECTED = "(none selected)"


def checkbox_row_text(label: str, selected: list[str]) -> str:
    """A checkbox grid row's atom text: ``"<label>: <ticked>, <ticked>"``,
    or ``"<label>: (none selected)"`` when nothing is ticked."""
    label = " ".join(str(label or "").split())
    return f"{label}: {', '.join(selected) if selected else NONE_SELECTED}"


def checkbox_row_value(
    cells: list[tuple[int, str, str]], *, label: str, label_column: str = "",
) -> dict[str, Any]:
    """The checkbox fields of a grid row's ONE atom, from its checkbox cells
    ``[(grid column, column header, cell text), ...]`` in row order: every
    option with the column header (and cell) it came from."""
    options: list[dict[str, Any]] = []
    for ci, col, text in cells:
        for checked, opt in checkbox_options(text) or []:
            options.append({"label": opt, "checked": checked, "column": (col or "").strip(), "cell": ci})
    if DROP_UNTICKED_FULLY:
        options = [o for o in options if o["checked"]]
    out: dict[str, Any] = {
        "checkbox_row": True,
        "subject": label,
        "label_column": (label_column or "").strip(),
        "selected": [o["label"] for o in options if o["checked"]],
        "options": options,
        "option_columns": list(dict.fromkeys(o["column"] for o in options)),
    }
    if not DROP_UNTICKED_FULLY:
        out["not_selected"] = [o["label"] for o in options if not o["checked"]]
    return out


def looks_like_site_column(header: str) -> bool:
    return bool(_SITE_HEADER_RE.search(header or ""))


def checkbox_atom(
    *, project_id: str, artifact_id: str, artifact_type: Any, filename: str,
    text: str, column: str, subject: str, locator: dict[str, Any],
    extraction_method: str, parser_version: str, entity_keys: list[str] | None = None,
    site_row: bool = True, authority_class: Any = None,
) -> Any:
    """One atom for one checkbox cell: its own text (verbatim, highlightable),
    typed ``site_attribute`` (attribute_kind from the column) when the row is
    a site, else ``scope_item``; the row's name/address rides in value."""
    from app.core.ids import stable_id
    from app.core.schemas import (
        AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef,
    )

    raw = " ".join(str(text or "").split())
    aid = stable_id("atm", artifact_id, "checkbox_cell", str(sorted(locator.items(), key=str)), column, raw)
    return EvidenceAtom(
        id=aid,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.site_attribute if site_row else AtomType.scope_item,
        raw_text=raw,
        normalized_text=raw.lower(),
        value=checkbox_value(raw, column=column, subject=subject),
        entity_keys=sorted(set(entity_keys or [])),
        source_refs=[SourceRef(
            id=stable_id("src", aid), artifact_id=artifact_id, artifact_type=artifact_type,
            filename=filename, locator={**locator, "column": column, "extraction": extraction_method},
            extraction_method=extraction_method, parser_version=parser_version,
        )],
        receipts=[],
        authority_class=authority_class or AuthorityClass.contractual_scope,
        confidence=0.8,
        review_status=ReviewStatus.needs_review,
        review_flags=["checkbox_cell"],
        parser_version=parser_version,
    )


def checkbox_row_atom(
    *, project_id: str, artifact_id: str, artifact_type: Any, filename: str,
    label: str, cells: list[tuple[int | None, str, str]], locator: dict[str, Any],
    extraction_method: str, parser_version: str, entity_keys: list[str] | None = None,
    authority_class: Any = None, label_column: str = "", site_row: bool = True,
) -> Any:
    """ONE atom for a table row's checkbox cells, read after the row's own
    label (site) atom: ``"<label>: <ticked>, ..."``, typed ``site_attribute``
    when the row is a site (else ``scope_item``), as the per-cell atoms it
    replaces were; the options under ``value`` (see :func:`checkbox_row_value`)."""
    from app.core.ids import stable_id
    from app.core.schemas import (
        AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef,
    )

    value = {"kind": "checkbox_selection", "site": label,
             **checkbox_row_value(list(cells), label=label, label_column=label_column)}  # type: ignore[arg-type]
    raw = checkbox_row_text(label, value["selected"])
    aid = stable_id("atm", artifact_id, "checkbox_row", str(sorted(locator.items(), key=str)), raw)
    return EvidenceAtom(
        id=aid,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.site_attribute if site_row else AtomType.scope_item,
        raw_text=raw,
        normalized_text=raw.lower(),
        value=value,
        entity_keys=sorted(set(entity_keys or [])),
        source_refs=[SourceRef(
            id=stable_id("src", aid), artifact_id=artifact_id, artifact_type=artifact_type,
            filename=filename, locator={**locator, "extraction": extraction_method},
            extraction_method=extraction_method, parser_version=parser_version,
        )],
        receipts=[],
        authority_class=authority_class or AuthorityClass.contractual_scope,
        confidence=0.8,
        review_status=ReviewStatus.needs_review,
        review_flags=["checkbox_row"],
        parser_version=parser_version,
    )


__all__ = [
    "DROP_UNTICKED_FULLY", "NONE_SELECTED", "checkbox_atom", "checkbox_row_atom", "checkbox_options", "checkbox_row_text",
    "checkbox_row_value", "checkbox_value", "is_checkbox_cell", "looks_like_site_column",
]
