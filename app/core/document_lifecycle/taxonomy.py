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
    # ── found by hand-reading the 51 documents nothing in the list fitted ──────
    # Each of these was a real recurring shape in the corpus, not a one-off.
    #
    # A CRM note mirrored out of HubSpot as a .txt whose filename is the first
    # words of the note body ("per store- estiamted 5-6 per year.txt"). 22 of the
    # 51. They are correspondence that arrived through the document path, and they
    # carry real pricing ("$5,840/per store", "ballpark of $115/per hour"), so they
    # are evidence -- but they duplicate the note already ingested as
    # correspondence, which is worth deduping later.
    "CRM_NOTE":            ("DISCOVERY", "evidence"),
    # Fireflies transcripts and meeting summaries. Genuine scoping material.
    "CALL_TRANSCRIPT":     ("DISCOVERY", "evidence"),
    # Rack elevations, Ekahau predictive surveys, proposed network architecture.
    # The highest-value evidence in the whole queue: "12 data drops = 36 Cat6,
    # 4 APs = 8 Cat6A, 3 Camera locations = 3 Cat6" is scope with quantities.
    "ENGINEERING_DESIGN":  ("DISCOVERY", "evidence"),
    # Manufacturer product sheets for equipment in scope (Anova, OxBlue, Fortinet).
    "VENDOR_DATASHEET":    ("DISCOVERY", "evidence"),
    # The customer's own build standards -- what we must conform to.
    "TECHNICAL_STANDARDS": ("DISCOVERY", "evidence"),
    # Role specifications used to price labour on staffing engagements.
    "JOB_DESCRIPTION":     ("QUOTING", "evidence"),
    # Our brochures and capability/coverage matrices. Standing, not deal-specific:
    # reference, so they inform without being mistaken for what a customer asked
    # for. Includes partner collateral (CDW one-pagers) for the same reason.
    "SALES_COLLATERAL":    ("QUOTING", "reference"),
    # Customer policy imposed on us (code of conduct, supplier terms). Real, but
    # says nothing about scope.
    "THIRD_PARTY_POLICY":  ("ADMIN", "neither"),
    # Mock and tooling artifacts that leaked into the live corpus -- "Mock
    # Document | Fictional data", "FOR HUMAN REVIEWER ONLY", an INTAKE_REQUEST.md
    # containing "# test". Only 4 corpus-wide, but one was already routed to
    # `label`, which would have made fictional data a TRAINING TARGET. Quarantine
    # is the point of this type.
    "TEST_FIXTURE":        ("UNKNOWN", "quarantine"),

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
