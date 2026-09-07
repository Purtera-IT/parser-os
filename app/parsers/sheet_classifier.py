"""Sheet-role classification for spreadsheet inputs.

Real-world estimating workbooks are not scope documents. A single
``Deal_Kit.xlsx`` may carry a deal-financials tab, a master price
catalog, three byte-identical data-validation backing lists, and a
labor-cost build — none of which describe what is being installed for
the customer. The row-level parsers (``xlsx_parser`` / ``quote_parser``)
historically emitted one atom per non-blank row of *every* sheet, so a
workbook with zero real scope produced hundreds of ``scope_item`` atoms
and buried the actual scope (which lived in an attached PDF).

This module classifies each sheet by *role* before any atom is emitted
so the parsers can suppress non-scope sheets. It is deterministic
(name + content heuristics only — no LLM) and intentionally
conservative: a sheet is only suppressed when the signal is strong, so
genuine scope/BOM sheets continue to parse exactly as before.

The single public entry point is :func:`classify_sheet`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SheetRole(str, Enum):
    """What a worksheet actually is, for emission gating."""

    SCOPE = "scope"  # real scope / BOM / asset / site table — parse normally
    EMPTY = "empty"  # no non-blank content
    INSTRUCTIONS = "instructions"  # cover / instructions / terms-only tab
    RATE_CARD = "rate_card"  # rate codes / skill levels / dropdown backing lists
    REFERENCE = "reference"  # named-range / lookup / "do not edit" helper data
    CATALOG = "catalog"  # master price book — no order quantities populated
    FINANCIAL_SUMMARY = "financial_summary"  # internal deal economics, not scope


class SheetDestination(str, Enum):
    """Where a classified sheet's rows are routed.

    The classifier is a *router*, not a binary keep/drop gate. Every
    sheet lands in exactly one bucket so useful pricing is never
    silently discarded and financials can never masquerade as scope.
    """

    SCOPE = "scope"  # mine as scope/BOM/asset/site atoms (the real work)
    COMMERCIAL = "commercial"  # emit as typed commercial atoms (pricing visible)
    DROP = "drop"  # pure backing-data / empty / cover noise — emit nothing


# Role → destination. SCOPE is mined as scope; rate cards, catalogs and
# deal-financials carry pricing the PM needs (routed to COMMERCIAL so
# they surface as money atoms without polluting scope_truth); empty /
# cover / instructions / lookup-helper sheets are genuine noise.
_ROLE_DESTINATION: dict[SheetRole, SheetDestination] = {
    SheetRole.SCOPE: SheetDestination.SCOPE,
    SheetRole.RATE_CARD: SheetDestination.COMMERCIAL,
    SheetRole.CATALOG: SheetDestination.COMMERCIAL,
    SheetRole.FINANCIAL_SUMMARY: SheetDestination.COMMERCIAL,
    SheetRole.EMPTY: SheetDestination.DROP,
    SheetRole.INSTRUCTIONS: SheetDestination.DROP,
    SheetRole.REFERENCE: SheetDestination.DROP,
}


@dataclass
class SheetClassification:
    role: SheetRole
    suppress: bool
    reason: str
    confidence: float
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def destination(self) -> SheetDestination:
        """Routing bucket derived from role (see :data:`_ROLE_DESTINATION`)."""
        return _ROLE_DESTINATION.get(self.role, SheetDestination.SCOPE)


# ── Name-based hints ────────────────────────────────────────────────

# Matched against the sheet name with all non-alphanumerics stripped so
# "SELLL RATES", "Sell_Rates", "Cost Rates" all normalize the same way.
_REFERENCE_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"donotedit"),
    re.compile(r"\bhelper\b|helper$|^helper"),
    re.compile(r"lookup"),
    re.compile(r"dropdown"),
    re.compile(r"validation"),
    re.compile(r"namedrange"),
    re.compile(r"(sell+|cost)rates?$"),
    re.compile(r"^rates?$|ratecard|ratetable"),
    re.compile(r"pricelist|pricebook|pricemaster|masterprice"),
)

_FINANCIAL_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"dealkit|dealsummary"),
    re.compile(r"ganttfinancials|financials?$"),
    re.compile(r"pandl|p&l|profitloss"),
    re.compile(r"marginsummary"),
)


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


# ── Cell-content signal helpers ─────────────────────────────────────

_RATE_CODE_RE = re.compile(r"^ps[-\s][a-z0-9 \-]+$", re.I)
_SKILL_LEVEL_RE = re.compile(r"^l[0-4]\b", re.I)
_BILLING_TYPE_RE = re.compile(
    r"^(t&m|fixed fee.*|per (hour|device|drop|site|asset|day|month|order)|pm fee|materials|travel)$",
    re.I,
)

_FINANCIAL_LABEL_TOKENS: tuple[str, ...] = (
    "oppty #",
    "oppty#",
    "sales rep",
    "total deal revenue",
    "total deal cost",
    "total deal margin",
    "margin %",
    "gross margin",
    "billing type",
    "total labor revenue",
    "total labor cost",
)


def _cells(rows: list[list[Any]]) -> list[str]:
    out: list[str] = []
    for row in rows:
        for c in row:
            s = "" if c is None else str(c).strip()
            if s:
                out.append(s)
    return out


def _nonblank_rows(rows: list[list[Any]]) -> int:
    return sum(
        1 for r in rows if any(("" if c is None else str(c).strip()) for c in r)
    )


def _rate_code_fraction(cells: list[str]) -> float:
    if not cells:
        return 0.0
    hits = sum(
        1
        for c in cells
        if _RATE_CODE_RE.match(c)
        or _SKILL_LEVEL_RE.match(c)
        or _BILLING_TYPE_RE.match(c)
    )
    return hits / len(cells)


def _looks_like_data_header(rows: list[list[Any]]) -> bool:
    """Cheap check for the presence of a real tabular header row.

    Used as a guard so content-based suppression never fires on a sheet
    that actually has a scope/BOM/asset header. Mirrors (a subset of)
    the alias vocabulary the row parsers key on.
    """
    header_tokens = {
        "site", "facility", "building", "room", "floor", "location",
        "device", "asset", "equipment", "camera", "hostname", "serial",
        "qty", "quantity", "count", "order qty", "drops", "# drops",
        "part", "part number", "sku", "model", "description",
        "plate id", "drop id", "outlet id", "ip address", "mac address",
    }
    for row in rows[:25]:
        toks = {
            re.sub(r"\s+", " ", str(c or "").strip().lower()).strip(".:?#")
            for c in row
            if str(c or "").strip()
        }
        if len(toks & header_tokens) >= 2:
            return True
    return False


def _order_qty_col(rows: list[list[Any]]) -> int | None:
    for row in rows[:25]:
        for idx, c in enumerate(row):
            t = re.sub(r"\s+", " ", str(c or "").strip().lower())
            if t in {"order qty", "order quantity", "order qty.", "qty ordered"}:
                return idx
    return None


def _catalog_columns_present(rows: list[list[Any]]) -> bool:
    wanted = {"oem", "manufacturer", "material description", "usa cost",
              "cost $", "sell $", "id #", "id#", "unit cost", "unit price"}
    for row in rows[:25]:
        toks = {str(c or "").strip().lower() for c in row if str(c or "").strip()}
        if any(any(w in t for w in wanted) for t in toks):
            return True
    return False


# A priced services / pricing-template table: a header carrying pricing
# columns. Routes to COMMERCIAL so each priced line becomes a typed commercial
# atom instead of orphan "Unit of Measure: ..." / "Subtotal: ..." fragments on
# the SCOPE path. Universal (column phrasing), not domain-specific.
_PRICED_HEADER_PHRASES: tuple[str, ...] = (
    "pricing element", "unit of measure", "hourly rate", "labor hours",
    "subtotal", "unit price", "extended cost", "ext cost", "price per",
    "materials price", "total cost", "cost per", "rate card",
)
_PRICED_STRONG_WORDS: tuple[str, ...] = ("price", "rate", "subtotal", "cost")


def _priced_table_header(rows: list[list[Any]]) -> bool:
    """True when a leading row is a pricing-table header (>=2 pricing column
    phrases AND a price/rate/cost/subtotal word)."""
    for row in rows[:6]:
        toks = [str(c or "").strip().lower() for c in row if str(c or "").strip()]
        if len(toks) < 2:
            continue
        joined = " | ".join(toks)
        hits = sum(1 for p in _PRICED_HEADER_PHRASES if p in joined)
        if hits >= 2 and any(w in joined for w in _PRICED_STRONG_WORDS):
            return True
    return False


_TECH_RATE_COL = re.compile(
    r"(?:networking\s+)?l[12]\s+(?:euc|technician)|hour\s+minimum|hr\.?\s*min",
    re.I,
)


def _first_header_row(rows: list[list[Any]]) -> int | None:
    for i, row in enumerate(rows[:20]):
        nonblank = [str(c).strip() for c in row if str(c or "").strip()]
        if len(nonblank) >= 2:
            return i
    return None


def _country_rate_card_fraction(rows: list[list[Any]]) -> float:
    """Fraction of data rows on a per-country labor rate table."""
    hdr_i = _first_header_row(rows)
    if hdr_i is None:
        return 0.0
    hdr = [str(c or "").strip().lower() for c in rows[hdr_i]]
    if not any(h == "country" or h.startswith("country") for h in hdr):
        return 0.0
    if not any(_TECH_RATE_COL.search(h) or "technician" in h for h in hdr):
        return 0.0
    data = [
        r for r in rows[hdr_i + 1 :]
        if any(str(c or "").strip() for c in r)
    ]
    if not data:
        return 0.0
    country_rows = 0
    for r in data:
        label = str(r[0] or "").strip()
        if not label:
            continue
        if label.replace(",", "").replace(".", "").replace("$", "").lstrip("-").isdigit():
            continue
        country_rows += 1
    return country_rows / len(data)


def _looks_like_country_rate_card(rows: list[list[Any]]) -> bool:
    """True when the sheet is a global per-country technician rate table."""
    frac = _country_rate_card_fraction(rows)
    nb = _nonblank_rows(rows)
    return frac >= 0.35 and nb >= 5


def sheet_exemplar(sheet_name: str, rows: list[list[Any]]) -> str:
    """What the `sheet` head learns a worksheet's identity from.

    The tab name plus its header row. That pair is what makes a kind of sheet
    recognisable ACROSS deals — "SSRS-SL-CUS001-CustomerOut | Customer Code |
    Customer Description | Order Date" is the same report next month under a
    different deal id, while the rows beneath it are entirely different data.

    Deliberately not the row content: a prototype built from 15,000 customer
    records would match any sheet mentioning those customers, which is a
    coincidence, not a kind.
    """
    header: list[str] = []
    widths = [sum(1 for c in row if c not in (None, "")) for row in rows]
    if widths:
        modal = max(widths) if max(widths) else 0
        # The first row that is at least 60% as wide as the widest — the real
        # header, even when a report preamble sits above it (the live case had
        # its header on row 11 behind a title block).
        for row in rows:
            filled = [str(c).strip() for c in row if c not in (None, "")]
            if modal and len(filled) >= max(3, int(modal * 0.6)):
                header = filled[:14]
                break
    name = " ".join(str(sheet_name or "").split())
    return " | ".join([name, *header])[:600] if header else name


def learned_sheet_role(sheet_name: str, rows: list[list[Any]], *, store: Any = None) -> Any:
    """A PM's standing judgment about this KIND of sheet, if they have made one.

    Consulted only where the classifier would otherwise fall through to
    `default_scope` — the branch that recognises nothing and routes to the most
    consequential bucket anyway. So this can add knowledge and never removes
    any: a sheet the rules DO recognise is not reconsidered here.

    Degrades open. No store, no embedder, no judgment, anything raised at all —
    the caller keeps its existing default. A classifier that fails closed on an
    unreachable store would silently stop mining every spreadsheet.
    """
    try:
        from app.core.decide import get_store
        from app.core.pm_feedback import HEAD_REGISTRY

        st = store if store is not None else get_store()
        if st is None:
            return None
        spec = HEAD_REGISTRY.get("sheet")
        if spec is None:
            return None
        text = sheet_exemplar(sheet_name, rows)
        if not text:
            return None
        from app.core.decide import DecisionScope

        decision = st.resolve(
            relation=spec.relation,
            text=text,
            candidates=list(spec.candidates),
            context="",
            scope=DecisionScope(),
            instruction="",
            relations=None,
        )
        if decision is None or not getattr(decision, "verdict", ""):
            return None
        return decision
    except Exception:
        return None


def classify_sheet(sheet_name: str, rows: list[list[Any]]) -> SheetClassification:
    """Classify a worksheet by role for atom-emission gating.

    ``rows`` is the raw matrix (list of row lists, cells may be ``None``)
    exactly as the parsers already load it.
    """
    nb_rows = _nonblank_rows(rows)
    if nb_rows == 0:
        return SheetClassification(
            role=SheetRole.EMPTY, suppress=True, reason="no_nonblank_rows",
            confidence=1.0, signals={"nonblank_rows": 0},
        )

    norm = _norm_name(sheet_name)
    cells = _cells(rows)
    has_data_header = _looks_like_data_header(rows)

    # 1. Name-based reference/rate-card hints (highest precision). A
    #    sheet literally named "... Do not Edit" / "SELLL RATES" /
    #    "COST RATES" / "Lookup" is backing data regardless of content.
    #
    # "regardless of content" was doing real work here: this returns
    # suppress=True, so EVERY atom on the sheet is dropped, decided on the tab
    # name before anything is read. A tab name is as forgeable as a filename
    # and gets renamed far more often -- a template's "Lookup" or "Helper" tab
    # that now holds the real BOM loses the whole table silently.
    #
    # The guard is the one this file already applies to the FINANCIAL name
    # rule three blocks down, whose comment says it plainly: "the data-header
    # guard already protects any genuine scope table that happens to live
    # under such a name". Rule 1 simply never got it.
    #
    # _looks_like_data_header wants two or more tokens from a scope/BOM
    # vocabulary (site, device, qty, part number, ...), so it does NOT fire on
    # a rate card -- "Code | Skill Level | Rate" contains none of them. Rate
    # cards and price books keep collapsing exactly as before; only a sheet
    # carrying a real scope table escapes.
    #
    # Measured over 17 real workbooks / 126 sheets: 35 are suppressed by this
    # rule and ZERO of them carry a data header, so this changes nothing that
    # currently happens.
    if any(p.search(norm) for p in _REFERENCE_NAME_PATTERNS) and not has_data_header:
        # A price-book / price-list name carries pricing the PM needs, so
        # it routes to COMMERCIAL (CATALOG), not the DROP bucket. A "rate"
        # name is a rate card (also COMMERCIAL). Everything else here
        # (lookup / dropdown / validation / "do not edit" / helper /
        # named range) is pure backing data → REFERENCE → DROP.
        if re.search(r"pricelist|pricebook|pricemaster|masterprice", norm):
            role = SheetRole.CATALOG
        elif "rate" in norm:
            role = SheetRole.RATE_CARD
        else:
            role = SheetRole.REFERENCE
        return SheetClassification(
            role=role, suppress=True, reason=f"reference_name:{sheet_name!r}",
            confidence=0.95, signals={"name_norm": norm},
        )

    # 2b. Content-based per-country labor rate card (Deal Kit "Country" tables).
    # Catches renamed tabs whose content is a global technician rate matrix.
    if _looks_like_country_rate_card(rows) and not has_data_header:
        return SheetClassification(
            role=SheetRole.RATE_CARD, suppress=True,
            reason="country_rate_card_table", confidence=0.9,
            signals={"country_rate_fraction": round(_country_rate_card_fraction(rows), 3)},
        )

    # 3. Content-based rate-card / dropdown backing list. Catches helper
    #    sheets even when renamed: a dense block of rate codes
    #    (PS-L1-ENG-LABOR-...), skill levels (L0-L4), or billing types
    #    with no real data header.
    rate_frac = _rate_code_fraction(cells)
    if rate_frac >= 0.40 and not has_data_header:
        return SheetClassification(
            role=SheetRole.RATE_CARD, suppress=True,
            reason=f"rate_code_density={rate_frac:.2f}", confidence=0.9,
            signals={"rate_code_fraction": round(rate_frac, 3)},
        )

    # 3. Financial summary — internal deal economics, not customer scope.
    lower_cells = " | ".join(c.lower() for c in cells)
    fin_hits = sum(1 for tok in _FINANCIAL_LABEL_TOKENS if tok in lower_cells)
    if fin_hits >= 3 and not has_data_header:
        return SheetClassification(
            role=SheetRole.FINANCIAL_SUMMARY, suppress=True,
            reason=f"financial_labels={fin_hits}", confidence=0.85,
            signals={"financial_label_hits": fin_hits},
        )
    # A financial-named tab (Deal Kit / Gantt Financials / P&L / margin)
    # with no real data header is internal economics — suppress even on
    # weak content signal, since the data-header guard already protects
    # any genuine scope table that happens to live under such a name.
    if (
        any(p.search(norm) for p in _FINANCIAL_NAME_PATTERNS)
        and not has_data_header
        and (fin_hits >= 1 or rate_frac >= 0.10)
    ):
        return SheetClassification(
            role=SheetRole.FINANCIAL_SUMMARY, suppress=True,
            reason=f"financial_name:{sheet_name!r}+labels={fin_hits}",
            confidence=0.8, signals={"financial_label_hits": fin_hits,
                                     "rate_code_fraction": round(rate_frac, 3)},
        )

    # 4. Master price catalog — has cost/sell/OEM columns AND an order-qty
    #    column that is empty on (nearly) every row. Nothing is actually
    #    ordered, so it is reference pricing, not scope.
    oq = _order_qty_col(rows)
    if oq is not None and nb_rows >= 8 and _catalog_columns_present(rows):
        data = [r for r in rows if any(str(c or "").strip() for c in r)]
        filled = sum(1 for r in data if oq < len(r) and str(r[oq] or "").strip())
        empty_frac = 1.0 - (filled / len(data) if data else 0.0)
        if empty_frac >= 0.90:
            return SheetClassification(
                role=SheetRole.CATALOG, suppress=True,
                reason=f"catalog_order_qty_empty={empty_frac:.2f}",
                confidence=0.85,
                signals={"order_qty_empty_fraction": round(empty_frac, 3)},
            )

    # 4b. Priced services / pricing-template table (Pricing Element / Unit of
    #     Measure / Price, or Labor Hours / Hourly Rate / Subtotal). Routes to
    #     COMMERCIAL so priced lines emit as typed commercial atoms, not orphan
    #     label->value fragments on the SCOPE path.
    if _priced_table_header(rows):
        return SheetClassification(
            role=SheetRole.CATALOG, suppress=True,
            reason="priced_table_header", confidence=0.8, signals={},
        )

    # 5. Instructions / cover / terms-only tab with no data table.
    sample = " ".join(c.lower() for c in cells[:60])
    if not has_data_header and (
        "instructions to bidders" in sample
        or "terms and conditions" in sample
        or "instructions to bidder" in sample
    ):
        return SheetClassification(
            role=SheetRole.INSTRUCTIONS, suppress=True,
            reason="instructions_or_terms_only", confidence=0.85,
            signals={},
        )

    # 5b. Cover / title page — a handful of short prose lines with no
    #     tabular structure and no data header. These are single-column
    #     banners (vendor name / project title / notes), not scope, so
    #     each line should not become an atom. Width is capped at 1: a
    #     two-column sheet (e.g. "Item | Notes") is a real table even when
    #     its header words aren't in the data-header vocabulary, so it must
    #     stay SCOPE rather than be mistaken for prose.
    max_row_width = max(
        (sum(1 for c in r if str(c or "").strip()) for r in rows), default=0
    )
    if not has_data_header and max_row_width <= 1 and nb_rows <= 6:
        return SheetClassification(
            role=SheetRole.INSTRUCTIONS, suppress=True,
            reason=f"cover_or_prose_no_table(rows={nb_rows},width={max_row_width})",
            confidence=0.75,
            signals={"nonblank_rows": nb_rows, "max_row_width": max_row_width},
        )

    # Nothing matched. Before defaulting, ask whether a PM has ever judged this
    # KIND of sheet — 45% of real sheets reach this branch, so the default is
    # doing most of the classifying, and it defaults to the bucket that feeds
    # the SOW.
    learned = learned_sheet_role(sheet_name, rows)
    if learned is not None:
        try:
            role = SheetRole(str(learned.verdict))
        except ValueError:
            role = None
        if role is not None:
            return SheetClassification(
                role=role,
                suppress=_ROLE_DESTINATION.get(role) is SheetDestination.DROP,
                reason=f"learned_sheet_role:{getattr(learned, 'correction_id', '') or 'pm'}",
                confidence=float(getattr(learned, "confidence", 0.0) or 0.8),
                signals={"has_data_header": has_data_header, "learned": True},
            )

    # Default: treat as a real scope/BOM/asset table.
    return SheetClassification(
        role=SheetRole.SCOPE, suppress=False, reason="default_scope",
        confidence=0.6, signals={"has_data_header": has_data_header},
    )


__all__ = [
    "sheet_exemplar",
    "learned_sheet_role",
    "SheetRole",
    "SheetDestination",
    "SheetClassification",
    "classify_sheet",
]
