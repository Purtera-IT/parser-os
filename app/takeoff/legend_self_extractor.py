"""Read symbol → description pairs from the project's own legend page.

The current ``legend_extractor`` loads symbol rules from a hand-tuned YAML
defaults file. That works for Marriott but doesn't generalize — every
project has its own legend with potentially different symbol codes,
class hierarchies, and cable callouts.

This module is the "use the legend as a small classifier at the start
of the parse" piece. Given the page index of the legend sheet, it:

1. Walks the page's native text and groups words into row-phrases
   (via :mod:`nearby_text`).
2. Finds each row that has a short uppercase symbol code in its left
   half and a multi-word description in its right half.
3. Maps the description to a ``normalized_class`` via keyword matching
   (project-agnostic: "WIRELESS NODE" → wireless_node_outlet,
   "POINT OF SALE TERMINAL" → pos_terminal_outlet, etc.).
4. Returns a list of :class:`LegendRule` objects sourced from the PDF.

The legend self-extractor is **additive**: callers fuse its output with
YAML defaults so any symbol present in BOTH gets confidence 0.95+,
symbols only in the PDF are added as new rules, and symbols only in YAML
fall through (covers projects whose legend is hand-drawn / image-only).

Deterministic: same page → same rules, every time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.takeoff.pdf_native import PdfWord, extract_page_words
from app.takeoff.schemas import LegendRule


# ─── Description-keyword → normalized_class table ────────────────────
#
# Order matters: the first match wins. Put more specific keys before
# less specific ones (POINT OF SALE TERMINAL before POINT OF SALE).
_DESCRIPTION_RULES: tuple[tuple[str, dict], ...] = (
    ("POINT OF SALE TERMINAL", {
        "normalized_class": "pos_terminal_outlet",
        "system": "structured_cabling_pos",
        "quote_unit": "cat6_pos_terminal_drop",
        "cable_count": 1, "cable_type": "CAT6 CMP",
    }),
    ("POINT OF SALE PRINTER", {
        "normalized_class": "pos_printer_outlet",
        "system": "structured_cabling_pos",
        "quote_unit": "cat6_pos_printer_drop",
        "cable_count": 1, "cable_type": "CAT6 CMP",
    }),
    ("WIRELESS NODE", {
        "normalized_class": "wireless_node_outlet",
        "system": "structured_cabling_wireless",
        "quote_unit": "cat6_wireless_node_drop",
        "cable_count": 1, "cable_type": "CAT6 CMP",
    }),
    ("CARD READER", {
        "normalized_class": "access_control_card_reader",
        "system": "access_control",
        "quote_unit": "access_control_card_reader_drop",
    }),
    ("PROXIMITY READER", {
        "normalized_class": "access_control_card_reader",
        "system": "access_control",
        "quote_unit": "access_control_card_reader_drop",
    }),
    ("DURESS ALARM", {
        "normalized_class": "duress_alarm_push_button",
        "system": "intrusion_detection",
        "quote_unit": "duress_alarm_location",
    }),
    ("DOOR ALARM", {
        "normalized_class": "door_alarm_contact",
        "system": "intrusion_detection",
        "quote_unit": "door_alarm_contact_drop",
    }),
    ("MOTION DETECTOR", {
        "normalized_class": "motion_detector",
        "system": "intrusion_detection",
        "quote_unit": "motion_detector_drop",
    }),
    ("HOUSE PHONE", {
        "normalized_class": "house_phone_outlet",
        "system": "structured_cabling_voice",
        "quote_unit": "cat6_house_phone_drop",
        "cable_count": 1, "cable_type": "CAT6 CMP",
    }),
    ("HOUSE TELEPHONE", {
        "normalized_class": "house_phone_outlet",
        "system": "structured_cabling_voice",
        "quote_unit": "cat6_house_phone_drop",
    }),
    ("MATV", {
        "normalized_class": "matv_outlet",
        "system": "matv",
        "quote_unit": "matv_tv_drop",
    }),
    ("TELEVISION", {
        "normalized_class": "matv_outlet",
        "system": "matv",
        "quote_unit": "matv_tv_drop",
    }),
    ("CCTV", {
        "normalized_class": "cctv_camera",
        "system": "video_surveillance",
        "quote_unit": "cctv_camera_drop",
    }),
)


# A legend "symbol code" is a short uppercase token in the legend table's
# code column. Accept letters / digits / hyphens, 1-6 chars.
_SYMBOL_CODE_RE = re.compile(r"^[A-Z][A-Z0-9\-]{0,5}$")

# Words that are clearly NOT symbol codes even if they fit the shape.
# Anything that's a cable spec, generic English word, callout marker,
# or installation-detail filler goes here.
_NOT_A_SYMBOL = frozenset({
    # Cable / electrical specs (most common false positives in legend rows).
    "PORT", "PORTS", "CAT6", "CAT5", "RJ45", "POE", "USB",
    "CMP", "CMR", "CL2", "CL3", "PVC", "EMT",
    "AWG", "AFF", "GND", "GROUND",
    # Voltages / current.
    "DC", "AC", "AMP", "AMPS", "VAC", "VDC", "VAR",
    # Common acronyms that look like codes but aren't device legends here.
    "POS",  # POS alone — POS-T / POS-P pass the regex with the hyphen.
    "IT", "BAS", "AV",  # disciplines, not symbols
    "TR", "ER",         # equipment-room markers — captured in nearby_text
    "CC", "NA", "N/A",
    # Generic English words that pass the upper-case shape check.
    "PER", "AND", "OR", "THE", "FOR", "WITH", "SEE", "USE", "ALL",
    "BED", "BOX", "MUD", "RING", "BACK", "JACK", "OUTLET", "PHONE",
    "PLANS", "PLAN", "ABOVE", "BELOW", "NEAR", "FROM", "INTO",
    "SINGLE", "DOUBLE", "TRIPLE", "QUAD", "TYPE", "ANALOG",
    # Likely-typo codes / OCR garbage we saw on Marriott's legend.
    "ZIBGEE", "ZIGBEE", "ZIBEE",
})


@dataclass(frozen=True)
class LegendRow:
    """A single parsed row from a legend table."""
    symbol_code: str
    description: str
    row_y: float  # y center for diagnostics
    confidence: float  # 0..1 — how confident we are this is a legend row


def _group_words_by_row(
    words: list[PdfWord],
    y_tolerance: float = 4.0,
) -> list[list[PdfWord]]:
    """Sort words by y then group consecutive words on the same baseline."""
    if not words:
        return []
    sorted_w = sorted(words, key=lambda w: ((w.y0 + w.y1) / 2.0, w.x0))
    rows: list[list[PdfWord]] = [[sorted_w[0]]]
    cur_y = (sorted_w[0].y0 + sorted_w[0].y1) / 2.0
    for w in sorted_w[1:]:
        wy = (w.y0 + w.y1) / 2.0
        if abs(wy - cur_y) <= y_tolerance:
            rows[-1].append(w)
        else:
            rows.append([w])
            cur_y = wy
    # Sort each row left-to-right.
    return [sorted(row, key=lambda w: w.x0) for row in rows]


# Anchor phrases that strongly indicate a true legend-table row. We
# only accept a row as a legend definition if its description contains
# one of these — that filters out cable specs, callout notes, and other
# uppercase text that happens to be on the same baseline.
_LEGEND_ROW_ANCHORS = (
    "PORT ",       # "4 PORT DATA OUTLET", "1 PORT WIRELESS NODE OUTLET"
    "OUTLET",      # most cabling devices end with "OUTLET"
    " READER",     # "CARD READER", "PROXIMITY READER"
    " ALARM",      # "DURESS ALARM", "DOOR ALARM"
    " DETECTOR",   # "MOTION DETECTOR"
    " STATION",    # "INTERCOM REMOTE STATION"
    " PANEL",      # "SECURITY CONTROL PANEL"
    " TERMINAL",   # "POINT OF SALE TERMINAL"
    " PRINTER",    # "POINT OF SALE PRINTER"
    "JACK",
    "SWITCH",
    "CAMERA",
)


def _classify_row(row: list[PdfWord]) -> LegendRow | None:
    """Try to interpret a row of words as a legend-table row.

    Strict acceptance criteria:

    * The row's text must contain at least one **anchor phrase** (e.g.
      "OUTLET", "READER", "ALARM") — this filters out cable specs,
      callout notes, and other uppercase noise that happens to share
      a baseline with description text.
    * The first uppercase code-shaped token in the row becomes the
      symbol code; we reject tokens that look like cable specs (CMP,
      CAT6) or generic English words (ALL, SEE, USE, …).
    * The description must be ≥ 12 chars and contain at least one
      space (multi-word).

    Returns None when the row fails any of those tests.
    """
    if len(row) < 2:
        return None

    full_row_text = " ".join((w.text or "").strip() for w in row if (w.text or "").strip())
    upper_row = full_row_text.upper()
    if not any(anchor in upper_row for anchor in _LEGEND_ROW_ANCHORS):
        return None

    code_index: int | None = None
    code_text: str | None = None
    for i, w in enumerate(row):
        text = (w.text or "").strip()
        if not text:
            continue
        upper = text.upper()
        if _SYMBOL_CODE_RE.match(upper) and upper not in _NOT_A_SYMBOL:
            code_index = i
            code_text = upper
            break
    if code_text is None or code_index is None:
        return None

    tail = row[code_index + 1 :]
    desc_words = [w.text for w in tail if (w.text or "").strip()]
    if len(desc_words) < 3:
        return None
    description = " ".join(desc_words)
    description = re.sub(r"\s+", " ", description).strip()
    if len(description) < 12 or " " not in description:
        return None
    # Reject paragraph-shaped strings (prose has lots of commas/periods early).
    if description[:30].count(",") >= 2 or description[:20].count(".") >= 1:
        return None

    row_y = (row[code_index].y0 + row[code_index].y1) / 2.0
    return LegendRow(
        symbol_code=code_text,
        description=description,
        row_y=row_y,
        confidence=0.85,
    )


def _normalized_class_for(description: str) -> dict | None:
    """Map a description string to the normalized_class + system + quote unit.

    Returns None when the description doesn't match any known device.
    """
    upper = description.upper()
    for keyword, payload in _DESCRIPTION_RULES:
        if keyword in upper:
            return payload
    return None


def extract_legend_from_page(
    *,
    page,
    confidence: float = 0.95,
) -> list[LegendRule]:
    """Build :class:`LegendRule` list from a PyMuPDF legend page.

    Pure / deterministic. Returns ``[]`` when no rows are recognizable.
    """
    words = extract_page_words(page)
    if not words:
        return []
    rows = _group_words_by_row(words)
    rules: list[LegendRule] = []
    seen_codes: set[tuple[str, str]] = set()
    for row in rows:
        parsed = _classify_row(row)
        if parsed is None:
            continue
        class_info = _normalized_class_for(parsed.description)
        if class_info is None:
            continue
        # Dedupe by (symbol_code, normalized_class) — the legend often
        # has WALL and CEILING variants of the same outlet on adjacent
        # rows; we keep the first definition only.
        key = (parsed.symbol_code, class_info["normalized_class"])
        if key in seen_codes:
            continue
        seen_codes.add(key)

        rules.append(LegendRule(
            raw_symbol=parsed.symbol_code,
            normalized_class=class_info["normalized_class"],
            system=class_info.get("system"),
            description=parsed.description,
            cable_count=class_info.get("cable_count"),
            cable_type=class_info.get("cable_type"),
            quote_unit=class_info.get("quote_unit"),
            source_page=page.number,
            confidence=confidence,
        ))
    return rules


def merge_with_defaults(
    *,
    extracted: list[LegendRule],
    defaults: list[LegendRule],
    accept_new_symbols: bool = False,
) -> tuple[list[LegendRule], list[str]]:
    """Combine PDF-extracted rules with YAML defaults.

    Rules present in BOTH sources are kept (PDF version, but with cable /
    termination / remarks fields back-filled from the default when the
    PDF didn't supply them) and bumped to confidence ≥0.97 (cross-validated).
    Rules only in YAML are kept (some projects don't legend every device).
    Rules only in the PDF are dropped UNLESS ``accept_new_symbols=True``
    — the YAML is the trusted source of valid symbol codes; PDF row
    parsing is too easy to fool without a stricter ML extractor.

    Returns ``(merged_rules, info_messages)``.
    """
    by_symbol_default = {r.raw_symbol.upper(): r for r in defaults}
    info: list[str] = []
    out: list[LegendRule] = []

    extracted_codes: set[str] = set()
    for ex_rule in extracted:
        sym = ex_rule.raw_symbol.upper()
        extracted_codes.add(sym)
        default_rule = by_symbol_default.get(sym)
        if default_rule is None:
            # Symbol on the legend that the YAML doesn't know about.
            # By default we surface this as info but DON'T add the rule
            # to the active set — the symbol_candidate pass would start
            # detecting noise tokens. Operators can promote it to YAML
            # explicitly. Pass accept_new_symbols=True to opt in.
            info.append(
                f"legend_extracted_new_symbol_not_in_defaults: {sym} "
                f"(description: {ex_rule.description or '?'}); "
                "add to rules/low_voltage_symbols.yaml if real"
            )
            if accept_new_symbols:
                out.append(ex_rule)
            continue
        # Back-fill any None fields on the extracted rule from defaults.
        payload = ex_rule.model_dump()
        defaults_payload = default_rule.model_dump()
        for field_name, default_value in defaults_payload.items():
            if payload.get(field_name) in (None, [], "") and default_value not in (None, [], ""):
                payload[field_name] = default_value
        # Bump confidence for cross-validated rules.
        payload["confidence"] = max(payload.get("confidence", 0.9), 0.97)
        out.append(LegendRule(**payload))
        info.append(f"legend_verified_against_pdf: {sym}")

    # Add YAML-only rules at the end — symbols our YAML knows about that
    # the PDF self-extractor didn't catch (legend table parse miss, or
    # the device isn't in this project's legend).
    for sym, default_rule in by_symbol_default.items():
        if sym not in extracted_codes:
            out.append(default_rule)

    return (out, info)


__all__ = [
    "LegendRow",
    "extract_legend_from_page",
    "merge_with_defaults",
]
