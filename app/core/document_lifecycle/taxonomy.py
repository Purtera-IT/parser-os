"""The document taxonomy. TYPE is chosen by the model from this fixed list;
STAGE is derived here, in code you own, not by the model.

Built from the 244 free-text labels the first corpus run produced -- ~30 real
types described many ways ("photo"/"photos"/"extra photos"/"final photos").
Changing where a type routes is an edit to this table and a re-derive; it does
not require re-running the model.
"""

# type -> (stage, admissibility)
TAXONOMY = {
    # ── discovery: what the customer needs ────────────────────────────────
    "INTAKE_REQUEST":      ("DISCOVERY", "evidence"),
    "RFP":                 ("DISCOVERY", "evidence"),
    "RFQ":                 ("DISCOVERY", "evidence"),
    "QUESTIONNAIRE":       ("DISCOVERY", "evidence"),
    "SITE_SURVEY":         ("DISCOVERY", "evidence"),
    "SITE_LIST":           ("DISCOVERY", "evidence"),
    "FLOOR_PLAN":          ("DISCOVERY", "evidence"),
    "PHOTOS":              ("DISCOVERY", "evidence"),
    "NETWORK_DATA":        ("DISCOVERY", "evidence"),
    # ── quoting: working out the price ────────────────────────────────────
    "ESTIMATING_CALC":     ("QUOTING", "evidence"),
    "BOM":                 ("QUOTING", "evidence"),
    "PRICING_SHEET":       ("QUOTING", "evidence"),
    "RATE_CARD":           ("QUOTING", "reference"),
    "COST_BREAKDOWN":      ("QUOTING", "evidence"),
    # ── our answer: never evidence, always label ──────────────────────────
    "DEAL_KIT":            ("QUOTED_OUTPUT", "label"),
    "QUOTE":               ("QUOTED_OUTPUT", "label"),
    "ROM":                 ("QUOTED_OUTPUT", "label"),
    "PROPOSAL":            ("QUOTED_OUTPUT", "label"),
    "CASE_MANIFEST":       ("QUOTED_OUTPUT", "label"),
    "WIN_WIRE":            ("QUOTED_OUTPUT", "label"),
    # ── contract paper ────────────────────────────────────────────────────
    "SOW_DRAFT":           ("CONTRACTED", "label"),
    "SOW_SIGNED":          ("CONTRACTED", "label"),
    "SOW_TEMPLATE":        ("CONTRACTED", "reference"),
    "CONTRACT":            ("CONTRACTED", "label"),
    "MSA":                 ("CONTRACTED", "reference"),
    "CONTRACT_EXHIBIT":    ("CONTRACTED", "label"),
    # ── delivery: Atlas territory ─────────────────────────────────────────
    "PURCHASE_ORDER":      ("DELIVERY", "atlas"),
    "INVOICE":             ("DELIVERY", "atlas"),
    "INSTALL_INSTRUCTIONS":("DELIVERY", "atlas"),
    "RUNBOOK":             ("DELIVERY", "atlas"),
    "CHANGE_ORDER":        ("DELIVERY", "atlas"),
    "SCHEDULE":            ("DELIVERY", "atlas"),
    "ACCEPTANCE_TEST":     ("DELIVERY", "atlas"),
    "CHECKLIST":           ("DELIVERY", "atlas"),
    # ── closeout ──────────────────────────────────────────────────────────
    "AS_BUILT":            ("CLOSEOUT", "atlas"),
    "CLOSEOUT_PACKET":     ("CLOSEOUT", "atlas"),
    "FINAL_PHOTOS":        ("CLOSEOUT", "atlas"),
    # ── paperwork that is real but says nothing about scope ───────────────
    "NDA":                 ("ADMIN", "neither"),
    "COI":                 ("ADMIN", "neither"),
    "W9":                  ("ADMIN", "neither"),
    "INSURANCE":           ("ADMIN", "neither"),
    "CREDIT_APPLICATION":  ("ADMIN", "neither"),
    # ── escapes ───────────────────────────────────────────────────────────
    "OTHER":               ("UNKNOWN", "quarantine"),   # -> your decision queue
    "UNKNOWN":             ("UNKNOWN", "quarantine"),
}

TYPES = sorted(TAXONOMY)


def normalise(doc_type: str) -> str:
    """`deal kit`, `Deal-Kit`, `DEAL_KIT` all name the same type."""
    import re
    return re.sub(r"[^A-Z0-9]+", "_", (doc_type or "").strip().upper()).strip("_")


def route(doc_type: str):
    """type -> (stage, admissibility). Unrecognised types quarantine, never guess."""
    return TAXONOMY.get(normalise(doc_type), ("UNKNOWN", "quarantine"))
