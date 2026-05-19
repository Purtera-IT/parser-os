"""Sheet metadata parser for drawing title blocks.

Extracts the title-block field set every construction sheet carries:
project name, sheet title, sheet number, scale, issue date,
revision, designer/checker/approver.  The output feeds a
``schematic_sheet_metadata`` atom — one per drawing page — so
reviewers can identify a sheet by name instead of by page index.

Heuristics are text-rule only (no LLM, no classifier). The detector
looks at text blocks inside or near the title-block exclusion region
and pattern-matches each line against a canonical field list.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from orbitbrief_page_os.segmentation.schematic.legend_locator import TextBlock


@dataclass(frozen=True)
class SheetMetadata:
    """One drawing sheet's title-block fields.

    Every field is optional; if the title block doesn't carry a
    field, the corresponding attribute is None. Source bbox is the
    union of the title-block lines we drew the field from, so
    ``source_replay`` can re-render and verify against pixels.
    """

    page_index: int
    bbox: tuple[float, float, float, float] | None
    sheet_number: str | None
    sheet_title: str | None
    project_name: str | None
    scale: str | None
    issue_date: str | None
    revision: str | None
    drafter: str | None
    checker: str | None
    approver: str | None
    client: str | None


_SHEET_TITLE_RE = re.compile(
    r"^(sheet\s+title|drawing\s+title|title)\s*[:\-]?\s*(.+)$",
    re.IGNORECASE,
)
_PROJECT_NAME_RE = re.compile(
    r"^(project(?:\s+name)?)\s*[:\-#]?\s*(.+)$",
    re.IGNORECASE,
)
_SCALE_RE = re.compile(
    r"\bscale\s*[:\-]?\s*([0-9/\"\'.=\-\sNTS]+)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:date|issue\s+date|issued)\s*[:\-]?\s*"
    r"([0-9]{1,4}[\-/.][0-9]{1,2}[\-/.][0-9]{1,4}|"
    r"[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b",
    re.IGNORECASE,
)
_REVISION_RE = re.compile(
    r"\b(?:rev(?:ision)?)\s*[:\-#]?\s*([A-Z]?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_DRAFTER_RE = re.compile(
    r"\b(?:drawn\s+by|drafter|by)\s*[:\-]?\s*([A-Za-z]{1,4}(?:[\s\-][A-Za-z]{1,4})?)\b",
    re.IGNORECASE,
)
_CHECKER_RE = re.compile(
    r"\b(?:checked\s+by|checker|chk)\s*[:\-]?\s*([A-Za-z]{1,4}(?:[\s\-][A-Za-z]{1,4})?)\b",
    re.IGNORECASE,
)
_APPROVER_RE = re.compile(
    r"\b(?:approved\s+by|approver|appr)\s*[:\-]?\s*([A-Za-z]{1,4}(?:[\s\-][A-Za-z]{1,4})?)\b",
    re.IGNORECASE,
)
_CLIENT_RE = re.compile(
    r"^(?:client|owner|for)\s*[:\-]?\s*(.+)$",
    re.IGNORECASE,
)


def _union(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def parse_sheet_metadata(
    *,
    page_index: int,
    blocks: Sequence[TextBlock],
    sheet_number: str | None,
    title_block_bbox: tuple[float, float, float, float] | None,
) -> SheetMetadata | None:
    """Parse title-block fields from a drawing page.

    ``title_block_bbox`` should be the bbox returned by
    ``exclusion_zones._title_block_region`` so we don't accidentally
    pick up body-prose lines that happen to mention "Project" or
    "Scale". When it's ``None`` we walk the whole page but skip blocks
    that obviously aren't title-block-shaped (long sentences > 60 chars
    without any title-block keyword).

    Returns ``None`` when no meaningful fields could be parsed —
    callers should NOT emit an atom for a fieldless page.
    """
    used_bboxes: list[tuple[float, float, float, float]] = []
    sheet_title = None
    project_name = None
    scale = None
    issue_date = None
    revision = None
    drafter = None
    checker = None
    approver = None
    client = None

    def _consider(blk: TextBlock) -> None:
        nonlocal sheet_title, project_name, scale, issue_date, revision
        nonlocal drafter, checker, approver, client
        text = (blk.text or "").strip()
        if not text:
            return
        matched = False
        if sheet_title is None:
            m = _SHEET_TITLE_RE.match(text)
            if m:
                sheet_title = m.group(2).strip()
                matched = True
        if project_name is None:
            m = _PROJECT_NAME_RE.match(text)
            if m:
                project_name = m.group(2).strip()
                matched = True
        if scale is None:
            m = _SCALE_RE.search(text)
            if m:
                scale = m.group(1).strip().rstrip(":").strip()
                matched = True
        if issue_date is None:
            m = _DATE_RE.search(text)
            if m:
                issue_date = m.group(1).strip()
                matched = True
        if revision is None:
            m = _REVISION_RE.search(text)
            if m:
                revision = m.group(1).strip()
                matched = True
        if drafter is None:
            m = _DRAFTER_RE.search(text)
            if m:
                drafter = m.group(1).strip()
                matched = True
        if checker is None:
            m = _CHECKER_RE.search(text)
            if m:
                checker = m.group(1).strip()
                matched = True
        if approver is None:
            m = _APPROVER_RE.search(text)
            if m:
                approver = m.group(1).strip()
                matched = True
        if client is None:
            m = _CLIENT_RE.match(text)
            if m:
                client = m.group(1).strip()
                matched = True
        if matched:
            used_bboxes.append(blk.bbox)

    if title_block_bbox is None:
        for blk in blocks:
            text = (blk.text or "").strip()
            if len(text) > 60 and not any(
                kw in text.lower()
                for kw in ("project", "title", "scale", "date", "rev", "drawn", "checked", "approved", "client", "owner")
            ):
                continue
            _consider(blk)
    else:
        x0, y0, x1, y1 = title_block_bbox
        for blk in blocks:
            bx0, by0, bx1, by1 = blk.bbox
            # Block intersects the title block region?
            if bx1 < x0 or by1 < y0 or bx0 > x1 or by0 > y1:
                continue
            _consider(blk)

    extracted = [
        v
        for v in (sheet_title, project_name, scale, issue_date, revision, drafter, checker, approver, client)
        if v
    ]
    if not extracted and not sheet_number:
        return None
    return SheetMetadata(
        page_index=page_index,
        bbox=_union(used_bboxes) or title_block_bbox,
        sheet_number=sheet_number,
        sheet_title=sheet_title,
        project_name=project_name,
        scale=scale,
        issue_date=issue_date,
        revision=revision,
        drafter=drafter,
        checker=checker,
        approver=approver,
        client=client,
    )
