"""Structure-aware semantic dedup.

The fuzzy description-fallback key in ``_value_key`` collapses atoms that
share a truncated description. On OPTBOT this silently dropped distinct
table rows that share a boilerplate description ("Video bar, scheduling
panel, occupancy sensor, cable refresh") but differ in their structured
fields (room name, quantity, site). The fix appends a signature of the
atom's *other* scalar fields to the fuzzy key so distinct rows survive,
while genuine paraphrases (same structured fields) still collapse and the
stable-ID path is untouched.

These are deal-agnostic: no field is special-cased — any differing scalar
field splits a fuzzy group. The change can only ever KEEP more atoms.
"""

from __future__ import annotations

from app.core.semantic_dedup import _scalar_signature, semantic_dedup_atoms


class _Atom:
    _n = 0

    def __init__(self, atom_type, value, confidence=0.8, raw_text=""):
        _Atom._n += 1
        self.id = f"a{_Atom._n}"
        self.atom_type = atom_type
        self.value = value
        self.confidence = confidence
        self.raw_text = raw_text or str(value)
        self.source_refs = []
        self.receipts = []
        self.entity_keys = []
        self.review_flags = []


_BOILERPLATE = "Video bar, scheduling panel, occupancy sensor, cable refresh"


# ── the OPTBOT bug: distinct rooms, shared equipment description ─────


def test_distinct_rows_sharing_boilerplate_are_not_collapsed() -> None:
    a = _Atom("requirement", {"room": "Huddle room", "quantity": 12,
                               "description": _BOILERPLATE})
    b = _Atom("requirement", {"room": "Training or war room", "quantity": 1,
                              "description": _BOILERPLATE})
    out = semantic_dedup_atoms([a, b])
    assert len(out) == 2, "distinct rooms must not collapse on shared description"


def test_same_description_differing_only_in_quantity_survives() -> None:
    a = _Atom("requirement", {"quantity": 12, "description": _BOILERPLATE})
    b = _Atom("requirement", {"quantity": 1, "description": _BOILERPLATE})
    out = semantic_dedup_atoms([a, b])
    assert len(out) == 2


# ── genuine paraphrase: SAME structured fields → still collapses ────


def test_paraphrase_same_fields_still_collapses() -> None:
    # Same room+quantity, description wording diverges only after the 40-char
    # truncation point — a true duplicate of one fact.
    a = _Atom("requirement",
              {"room": "Huddle room", "quantity": 12,
               "description": "Video bar, scheduling panel, occupancy sensor, cable refresh"},
              confidence=0.9)
    b = _Atom("requirement",
              {"room": "Huddle room", "quantity": 12,
               "description": "Video bar, scheduling panel, occupancy sensor, extra cabling"},
              confidence=0.6)
    out = semantic_dedup_atoms([a, b])
    assert len(out) == 1, "same structured fields → one fact, collapse"
    assert out[0].confidence == 0.9  # highest-confidence survivor


# ── stable-ID path is unaffected by field jitter ────────────────────


def test_stable_id_collapses_regardless_of_other_fields() -> None:
    a = _Atom("requirement", {"req_id": "R-07", "priority": "high",
                              "description": "first wording"}, confidence=0.9)
    b = _Atom("requirement", {"req_id": "R-07", "priority": "low",
                              "description": "totally different wording"}, confidence=0.5)
    out = semantic_dedup_atoms([a, b])
    assert len(out) == 1, "same req_id is the authoritative key — collapse"


def test_distinct_ids_never_collapse() -> None:
    a = _Atom("requirement", {"req_id": "R-07", "description": _BOILERPLATE})
    b = _Atom("requirement", {"req_id": "R-08", "description": _BOILERPLATE})
    out = semantic_dedup_atoms([a, b])
    assert len(out) == 2


# ── pure-prose atoms (no distinguishing scalar field) unchanged ─────


def test_pure_prose_paraphrase_still_collapses() -> None:
    # No structured fields at all → signature empty → behaves as before.
    a = _Atom("risk", {"description": "Network outage during business hours could disrupt cutover"})
    b = _Atom("risk", {"description": "Network outage during business hours might disrupt cutover"})
    out = semantic_dedup_atoms([a, b])
    assert len(out) == 1


def test_pure_prose_distinct_text_not_collapsed() -> None:
    a = _Atom("risk", {"description": "Warehouse RF interference may reduce scan reliability"})
    b = _Atom("risk", {"description": "Procurement approval matrix requires CFO signoff"})
    out = semantic_dedup_atoms([a, b])
    assert len(out) == 2


# ── _scalar_signature unit behaviour ────────────────────────────────


def test_signature_excludes_keyed_and_bookkeeping_fields() -> None:
    val = {"description": "long boilerplate text here", "room": "Huddle room",
           "quantity": 12, "kind": "requirement", "_suppression": {"stage": "x"}}
    sig = _scalar_signature(val, exclude=frozenset({"description"}))
    assert "room=" in sig and "quantity=12" in sig
    assert "kind" not in sig and "_suppression" not in sig


def test_signature_skips_long_prose_strings() -> None:
    long_str = "x" * 100
    val = {"description": "boiler", "blob": long_str, "code": "ABC-1"}
    sig = _scalar_signature(val, exclude=frozenset({"description"}))
    assert "code=abc_1" in sig
    assert "blob=" not in sig, "long prose is not an identity token"


def test_signature_empty_when_no_distinguishing_fields() -> None:
    val = {"description": "only a description here", "kind": "risk"}
    assert _scalar_signature(val, exclude=frozenset({"description"})) == ""


# ── MBrany-class physical_site dedup: same place, different site_ids ──


def test_mbrany_three_site_variants_collapse_to_one() -> None:
    """Three physical_site atoms for Highland Park MI 48203 → one site."""
    atom_roster = _Atom(
        "physical_site",
        {
            "kind": "physical_site",
            "id": "HIGHLAND-PARK-MI-48203",
            "site_id": "HIGHLAND-PARK-MI-48203",
            "name": "12575 Oakland Park BLvd, Highland Park, MI 48203",
            "street_address": "12575 Oakland Park BLvd",
            "address": "12575 Oakland Park BLvd",
            "city": "Highland Park",
            "state": "MI",
            "zip": "48203",
            "facility_name": "12575 Oakland Park BLvd, Highland Park, MI 48203",
        },
        confidence=0.9,
    )
    atom_geo_street = _Atom(
        "physical_site",
        {
            "kind": "physical_site",
            "id": "12575_oakland_park_blvd",
            "site_id": "12575_oakland_park_blvd",
            "name": (
                "Location: Mobis North America Work, 12575 Oakland Park BLvd., "
                "Highland Park, MI 48203"
            ),
            "street_address": "Location: Mobis North America Work, 12575 Oakland Park BLvd.",
            "address": "Location: Mobis North America Work, 12575 Oakland Park BLvd.",
            "city": "Highland Park",
            "state": "MI",
            "zip": "48203",
            "inferred": True,
        },
        confidence=0.5,
    )
    atom_geo_street.review_flags = ["geo_fallback_site"]
    atom_entity_backfill = _Atom(
        "physical_site",
        {
            "kind": "physical_site",
            "id": "highland_park_mi_48203",
            "site_id": "highland_park_mi_48203",
            "name": "highland park mi 48203",
            "inferred": True,
            "backfill": "entity_backfill",
        },
        confidence=0.4,
    )
    noise = _Atom("requirement", {"req_id": "R-1", "description": "Install APs"})
    out = semantic_dedup_atoms([atom_roster, atom_geo_street, atom_entity_backfill, noise])
    phys = [a for a in out if a.atom_type == "physical_site"]
    assert len(phys) == 1, f"expected 1 physical_site, got {len(phys)}"
    assert phys[0].value["site_id"] == "HIGHLAND-PARK-MI-48203"
    assert phys[0].value["city"] == "Highland Park"
    assert phys[0].value["state"] == "MI"
    assert phys[0].value["zip"] == "48203"
    assert len(out) == 2  # one site + requirement


def test_multi_city_geo_fallbacks_still_survive() -> None:
    """Distinct cities must not collapse — re-emit only when buckets differ."""
    detroit = _Atom(
        "physical_site",
        {
            "site_id": "DETROIT-MI-48201",
            "city": "Detroit",
            "state": "MI",
            "zip": "48201",
            "street_address": "200 Main St",
        },
        confidence=0.9,
    )
    highland = _Atom(
        "physical_site",
        {
            "site_id": "highland_park_mi_48203",
            "city": "Highland Park",
            "state": "MI",
            "zip": "48203",
            "inferred": True,
        },
        confidence=0.5,
    )
    highland.review_flags = ["geo_fallback_site"]
    santa_fe = _Atom(
        "physical_site",
        {
            "site_id": "santa_fe_nm_87506",
            "city": "Santa Fe",
            "state": "NM",
            "zip": "87506",
            "inferred": True,
        },
        confidence=0.5,
    )
    santa_fe.review_flags = ["geo_fallback_site"]
    out = semantic_dedup_atoms([detroit, highland, santa_fe])
    phys = [a for a in out if a.atom_type == "physical_site"]
    cities = {a.value.get("city") for a in phys}
    assert cities == {"Detroit", "Highland Park", "Santa Fe"}
    assert len(phys) == 3
