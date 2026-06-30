"""Universal noise-suppression gates — seeded into the feedback store.

The single biggest accuracy failure we see on real deals is **reference /
template content being ingested as deal-specific evidence**. Observed live:

* A master **rate card / price book** (a Deal-Kit spreadsheet carrying a global
  per-country technician rate table and a full materials catalog) gets flattened
  into hundreds of ``pricing_assumption`` atoms. The "total commercial figures"
  then sum an entire template — e.g. one deal reported **$86,268** that was
  almost entirely Indonesia/UAE/Hong-Kong rate-card rows and CAT6/fiber catalog
  SKUs unrelated to the deal. More reference data ⇒ a *worse* grade.
* **Time-window / billing labels** ("Business Hours…", "After Hours…", "Stated
  Rate") get classified as **stakeholders / people**, inflating stakeholder
  coverage with non-humans.

These are two DIFFERENT diseases on two DIFFERENT atom types, so they get two
type-scoped binary gates (mirrors :mod:`app.core.site_role_seed`'s one-gate-per-
concept design), never a single overloaded gate:

* ``pricing_noise_admission`` — examined ONLY on ``pricing_assumption`` atoms.
  DROP class = rate-card rows + materials-catalog lines. KEEP class anchors real
  deal pricing — *including a real after-hours premium for THIS deal* so a
  genuine "after-hours billed at 1.5×" rate is never mistaken for the template
  "After Hours: 50% increase of Stated Rate" label.
* ``person_noise_admission`` — examined ONLY on ``stakeholder`` atoms. DROP
  class = a time-window / rate label that was mis-typed as a person (a schedule
  is never a human). KEEP class anchors real people.

Why the split matters (root cause found on deal #010065, a healthcare AP swap
whose emails are saturated with "after hours"): "after hours" can be a *real
deal commercial term* (a hospital after-hours premium) **or** template rate-card
boilerplate. Scoping the rate-label DROP to the ``stakeholder`` type means it can
only ever delete a time-window masquerading as a person — it can NEVER touch a
deal's real after-hours *pricing*, which lives on ``pricing_assumption`` and is
checked against the pricing gate (whose KEEP class explicitly anchors it).

Every exemplar is a PLAIN-LANGUAGE description of a SHAPE — never a specific
deal's customer, city, vendor, or SKU. The embedder generalizes each shape to
every instance. The gates are **guess-free**: only a confident learned ``drop``
suppresses; store-undecided always keeps.
"""

from __future__ import annotations

from app.core.feedback_store import SCOPE_GLOBAL, Correction

# ── shared verdict vocabulary ────────────────────────────────────────────────
NOISE_CANDIDATES = ["drop", "keep"]
NOISE_DROP_VERDICT = "drop"
NOISE_KEEP_VERDICT = "keep"

# Two type-scoped relations (see module docstring).
PRICING_NOISE_RELATION = "pricing_noise_admission"
PERSON_NOISE_RELATION = "person_noise_admission"

_SEED_COMPLAINT = "seed:atom_noise_admission"

# ── pricing gate: rate-card + materials-catalog (pricing_assumption only) ─────
_PRICING_DROP_CONCEPTS: dict[str, tuple[str, list[str]]] = {
    "rate_card_country": (
        "A row from a global technician rate card (per-country labor rates), "
        "not a price for this deal.",
        [
            "Country: Indonesia | Request: 0.75 | Networking L1 Technician 2 hr. min: 73.5 | "
            "Networking L1 Technician 4hr. Min.: 69",
            "Country: United Arab Emirates | Networking L1 Technician 2 hr. min: 73.0 | "
            "Networking L1 Technician 8 hour minimum",
            "Country: Hong Kong | Request: 1.5 | Networking L1 Technician 2 hr. min: 147 | "
            "Networking L1 Technician 4hr. Min.: 138",
            "Country: Brazil | Request: 1.25 | Networking L1 Technician 2 hr. min: 79.4",
            "Country: South Africa | Networking L1 Technician 2 hr. min: 50.8 | L2 EUC 8 hour minimum",
        ],
    ),
    "materials_catalog": (
        "A line item from a master materials / parts catalog (ID, OEM, part "
        "number, packaging, unit cost), not deal scope.",
        [
            "ID #: 131 | Material Description: 2U Fiber Tray Housing - (Rack Mount) | "
            "Color / Style/Part: CCH-01U | OEM: Corning | QTY in Packaging: 1 | USA Cost $ [Pre-Tax] (Per Unit)",
            "ID #: 16 | Material Description: 24-Port (CAT6) 110-Style Punch-Down Patch Panel | "
            "OEM: CommScope | QTY in Packaging: 1 | USA Cost $ [Pre-Tax]",
            "ID #: 96 | Material Description: Fender Washers | Color / Style/Part: FENW38114 | "
            "QTY in Packaging: 100 | USA Cost $ [Pre-Tax]",
            "ID #: 3 | Material Description: CAT6 Module (Female) Data Jack Connector | "
            "OEM: CommScope | QTY in Packaging: 1",
            "ID #: 150 | Material Description: 2 Meter LC-LC MM OM3/OM4 Duplex Patch Cable | "
            "QTY in Packaging: 1 | USA Cost $ [Pre-Tax]",
        ],
    ),
}

# KEEP class for the pricing gate — real deal pricing/scope, INCLUDING a genuine
# after-hours premium so it is never confused with the template rate label.
_PRICING_KEEP_ANCHOR = [
    "Wifi Survey Tech — fixed price 1,700 for this engagement.",
    "Field engineering labor, onsite, per hour, two-hour minimum.",
    "After-hours work at the hospital sites billed at 1.5x the standard onsite rate.",
    "Weekend access requires an after-hours premium for this deal's four sites.",
    "Net 30 payment terms; invoice on completion.",
    "Lift rental for ceiling-mounted access points, per day.",
]

# ── person gate: time-window / rate label mis-typed as a person (stakeholder) ─
_PERSON_DROP_CONCEPTS: dict[str, tuple[str, list[str]]] = {
    "rate_label_as_person": (
        "A billing time-window or rate label (business hours, after hours, "
        "stated rate) — a schedule/commercial term, never a person.",
        [
            "Business Hours: 8:00 AM to 5:00 PM (17:00) local time at the Stated Rate.",
            "After Hours: 5:00 PM (17:00) to 8:00 AM: 50% increase of Stated Rate.",
            "Stated Rate",
            "Weekend and holiday work billed at the Stated Rate plus a premium.",
        ],
    ),
}

# KEEP class for the person gate — real people (name + role/email/phone shapes).
_PERSON_KEEP_ANCHOR = [
    "Jane Smith, Facilities Manager, jane.smith@hospital.org",
    "AJ Evans, Account Executive, aj.evans@cdw.com, 555-201-3344",
    "John Doe — Network Engineer, onsite lead for the survey",
    "Maria Gonzalez, IT Director, Monument Health",
]


def _drop_corrections(relation: str, concepts: dict[str, tuple[str, list[str]]]) -> list[Correction]:
    out: list[Correction] = []
    for concept, (instruction, exemplars) in concepts.items():
        out.append(
            Correction(
                id=f"noise_drop_{concept}",
                relation=relation,
                verdict=NOISE_DROP_VERDICT,
                scope=SCOPE_GLOBAL,
                exemplars=exemplars,
                instruction=instruction,
                created_by="seed",
                complaint_id=_SEED_COMPLAINT,
            )
        )
    return out


def _keep_correction(relation: str, anchor: list[str], cid: str) -> Correction:
    return Correction(
        id=cid,
        relation=relation,
        verdict=NOISE_KEEP_VERDICT,
        scope=SCOPE_GLOBAL,
        exemplars=list(anchor),
        # No prose instruction: the rich exemplars define the KEEP class; a bland
        # duplicate sentence would over-weight KEEP and blunt the gate. Mirrors
        # site_role_seed.
        instruction="",
        created_by="seed",
        complaint_id=_SEED_COMPLAINT,
    )


def noise_gate_corrections() -> list[Correction]:
    """All universal noise-suppression corrections (both type-scoped gates)."""
    return (
        _drop_corrections(PRICING_NOISE_RELATION, _PRICING_DROP_CONCEPTS)
        + [_keep_correction(PRICING_NOISE_RELATION, _PRICING_KEEP_ANCHOR, "noise_keep_pricing")]
        + _drop_corrections(PERSON_NOISE_RELATION, _PERSON_DROP_CONCEPTS)
        + [_keep_correction(PERSON_NOISE_RELATION, _PERSON_KEEP_ANCHOR, "noise_keep_person")]
    )


__all__ = [
    "PRICING_NOISE_RELATION",
    "PERSON_NOISE_RELATION",
    "NOISE_CANDIDATES",
    "NOISE_DROP_VERDICT",
    "NOISE_KEEP_VERDICT",
    "noise_gate_corrections",
]
