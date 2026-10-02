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


__all__ = ["checkbox_atom", "checkbox_options", "checkbox_value", "is_checkbox_cell", "looks_like_site_column"]
