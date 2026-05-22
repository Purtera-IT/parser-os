from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id
from app.core.item_identity import canonical_material_key, is_primary_vendor_quantity
from app.core.normalizers import normalize_text
from app.core.schemas import AtomType, AuthorityClass, EdgeType, EntityRecord, EvidenceAtom, EvidenceEdge
from app.domain import get_active_domain_pack
from app.semantic.linker import propose_semantic_link_candidates

# Cap the number of cross-artifact reinforcement targets per source atom.  Without
# this, an N-artifact project with M atoms each yields O(N*M) edges per source;
# OrbitBrief downstream then has to filter noise.  The cap is per source atom
# per target type, picked deterministically by sorted target id.
MAX_CROSS_ARTIFACT_TARGETS_PER_SOURCE = 8

# Stable taxonomy for edge_family — every edge produced by build_edges gets one
# of these so OrbitBrief can filter "show me only the contradictions" or "show
# me only the strong supports" without re-deriving categories from reasons.
EDGE_FAMILY_VALUE_SUPPORT = "value_support"
EDGE_FAMILY_CONSTRAINT_ALIGN = "constraint_alignment"
EDGE_FAMILY_QUANTITY_CONTRADICTION = "quantity_contradiction"
EDGE_FAMILY_PART_NUMBER_QUANTITY_CONFLICT = "part_number_quantity_conflict"
EDGE_FAMILY_EXCLUSION_APPLIES = "exclusion_application"
EDGE_FAMILY_CONSTRAINT_REQUIRES = "constraint_requirement"
EDGE_FAMILY_DEVICE_AGGREGATE_MISMATCH = "device_aggregate_mismatch"
EDGE_FAMILY_MATERIAL_AGGREGATE_MISMATCH = "material_aggregate_mismatch"
# Cross-artifact device-only quantity conflict — BOM/quote/spec atom
# vs SOW/scope atom that share a device key (device:access_point,
# device:switch, ...) and quantity entity keys but have no shared
# part_number to anchor against. False-positive guards are strict:
#   - cross-artifact required (same-file diffs are normal table rows)
#   - same site (or both site-free) required
#   - different authority classes preferred (BOM vs SOW signal)
#   - excludes generic device:unknown
#   - quantity must be >= 5 (single-digit qtys are often template fields)
EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC = "device_quantity_cross_doc"
EDGE_FAMILY_CROSS_ARTIFACT_REINFORCEMENT = "cross_artifact_co_mention"
EDGE_FAMILY_SEMANTIC_LINK = "semantic_link"
# Schematic same-sheet quantity contradiction (PR7) — pairs an aggregated
# detected count with a legend-declared count on the same drawing. Both
# atoms carry bbox provenance so the packetizer's narrow exception can
# certify them despite being from the same artifact / same authority.
EDGE_FAMILY_SCHEMATIC_QUANTITY_CONTRADICTION = "schematic_quantity_contradiction"

# An entity_key that matches more atoms than this threshold is treated as
# too generic to use as a join point (e.g. ``site:campus`` would otherwise
# pair every atom with every other atom).  We still allow these keys to
# participate when *any other* shared key qualifies the pair.  See
# PRODUCTION_GAPS.md P2.2 — without this cap Downey produced 642k edges
# in 38 minutes by joining everything to ``site:campus``.
#
# Threshold: ``max(20, sqrt(N))``.  For VT_CAM (71 atoms) → 20-atom cap.
# For Downey (4,892 atoms) → 70-atom cap.  Sub-linear growth keeps
# candidate_pairs bounded at O(N · sqrt(N) / 2) instead of O(N²).
NOISY_ENTITY_KEY_BUCKET_FLOOR = 20

# Also bound the total number of candidate pairs we will examine.  For
# corpora large enough that even sqrt-bucket keys produce too many
# pairs, this cap prevents blow-up.  Pairs are generated in a stable
# order so deterministic output is preserved up to the cap.
MAX_CANDIDATE_PAIRS = 250_000

# Per-atom edge density cap. With very large corpora (10k+ atoms),
# even the sqrt-bucket cap can produce 25 edges per atom which is
# both noisy for consumers and expensive to construct. This caps
# pairs at ``MAX_PAIRS_PER_ATOM`` * N so per-atom edge density stays
# bounded sub-linearly.
MAX_PAIRS_PER_ATOM = 12

# Part-number entities are the strongest signal for quantity contradictions.
# When two atoms share the same ``part_number:*`` key but report different
# ``Qty: N`` values (or different ``quantity:*`` entity keys), that's a
# real-world cost-proposal mismatch (cf. Natomas CW9166I-B 500 vs 136).
_PART_NUMBER_PREFIX = "part_number:"
_QUANTITY_PREFIX = "quantity:"


# Noun-anchored qty extraction for atoms that carry multiple device
# keys. "Install 50 access points and 5 switches" should bind 50 to
# access_point and 5 to switch so cross-doc binding can compare the
# right quantities per device.
_DEVICE_NOUN_PATTERNS: dict[str, tuple[str, ...]] = {
    "access_point": (r"access\s+points?", r"\baps?\b", r"wireless\s+access\s+points?", r"waps?"),
    "switch": (r"switches?(?:\b|\s)", r"poe\s+switches?", r"core\s+switches?", r"access\s+switches?"),
    "router": (r"routers?", r"edge\s+routers?", r"core\s+routers?"),
    "firewall": (r"firewalls?", r"ngfws?", r"\butms?\b"),
    "ip_camera": (r"ip\s+cameras?", r"cameras?", r"ptz(?:\s+cameras?)?", r"dome\s+cameras?", r"bullet\s+cameras?"),
    "card_reader": (r"card\s+readers?", r"badge\s+readers?", r"\breaders?\b"),
    "controller": (r"controllers?", r"access\s+controllers?", r"door\s+controllers?"),
    "ups": (r"\bupses?\b", r"battery\s+backup"),
    "rack": (r"racks?", r"cabinets?", r"server\s+racks?"),
    "display": (r"displays?", r"monitors?", r"video\s+walls?", r"touchscreens?"),
    "speaker": (r"speakers?", r"ceiling\s+speakers?"),
    "microphone": (r"microphones?", r"\bmics?\b"),
}

# Compile once.
_DEVICE_NOUN_REGEXES: dict[str, list[re.Pattern[str]]] = {
    canonical: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for canonical, patterns in _DEVICE_NOUN_PATTERNS.items()
}


def _noun_anchored_quantity(text: str, device_canonical: str) -> int | None:
    r"""Return the integer quantity that occurs nearest to a device noun.

    ``device_canonical`` is the YAML key (e.g. ``access_point``,
    ``switch``). We look for ``\b\d+\b`` tokens within a 30-char
    window BEFORE the noun (or 15-char window AFTER) in the same
    clause. Sentence boundaries (``.`` ``;`` newline) terminate the
    window so "Install 12 APs. We will install 4 switches" doesn't
    bind 12 to switches. Returns None when no noun match or no
    nearby integer in the same clause.
    """
    if not text:
        return None
    patterns = _DEVICE_NOUN_REGEXES.get(device_canonical)
    if not patterns:
        return None
    best_qty: int | None = None
    best_dist = 10**9
    # Split text into clauses on strong boundaries: ``.``, ``;``, newline.
    # Each clause is processed independently so numbers in one clause
    # can't bind to nouns in another.
    clause_offsets: list[tuple[int, int]] = []  # (start, end)
    cursor = 0
    for sep_match in re.finditer(r"[.;\n]", text):
        clause_offsets.append((cursor, sep_match.end()))
        cursor = sep_match.end()
    if cursor < len(text):
        clause_offsets.append((cursor, len(text)))
    for clause_start, clause_end in clause_offsets:
        clause = text[clause_start:clause_end]
        for pat in patterns:
            for noun_match in pat.finditer(clause):
                n_start = noun_match.start()
                for num_match in re.finditer(r"\b(\d{1,5})\b", clause):
                    raw = num_match.group(1)
                    try:
                        val = int(raw)
                    except ValueError:
                        continue
                    # Noun-anchored has a strong signal (the integer
                    # is right next to the device noun), so we accept
                    # qty:2..4 here even though the broader cross-doc
                    # binding drops them. Real bids do say "4
                    # switches" / "2 firewalls".
                    if val < 2:
                        continue
                    num_pos = num_match.start()
                    before = num_pos < n_start
                    dist = abs(n_start - num_pos)
                    # Tighter windows: 30 chars before the noun
                    # ("Install 50 access points") or 15 chars after
                    # ("access points: 50"). Anything farther is
                    # almost certainly an unrelated number.
                    if before and dist > 30:
                        continue
                    if (not before) and dist > 15:
                        continue
                    # Numbers BEFORE the noun get a strong bonus.
                    effective = dist - (10 if before else 0)
                    if effective < best_dist:
                        best_dist = effective
                        best_qty = val
    return best_qty


def _atoms_are_distinct_positions(a: EvidenceAtom, b: EvidenceAtom) -> bool:
    """True when two same-artifact atoms came from clearly different
    positions in the source doc.

    Compares ``sentence_index`` (set by the markdown sentence splitter
    and other multi-emission parsers), then line numbers in
    ``source_refs[0].locator``, then ``value.row`` / ``value.line``
    fallbacks. Returns False when both atoms came from the same line
    + same sentence (same template row).
    """
    if a.id == b.id:
        return False
    a_val = a.value if isinstance(a.value, dict) else {}
    b_val = b.value if isinstance(b.value, dict) else {}
    a_sent = a_val.get("sentence_index")
    b_sent = b_val.get("sentence_index")
    if a_sent is not None and b_sent is not None and a_sent != b_sent:
        return True
    a_loc = (a.source_refs[0].locator if a.source_refs else {}) or {}
    b_loc = (b.source_refs[0].locator if b.source_refs else {}) or {}
    a_line = a_loc.get("line_start") or a_loc.get("row") or a_loc.get("page") or a_val.get("line")
    b_line = b_loc.get("line_start") or b_loc.get("row") or b_loc.get("page") or b_val.get("line")
    if a_line is not None and b_line is not None and a_line != b_line:
        return True
    return False


def _is_unknown_entity_key(key: str) -> bool:
    """Return True for sentinel '*:unknown' keys we never want to anchor on."""
    return key.endswith(":unknown") or key == "device:unknown" or key == "site:unknown"


def _meaningful_shared_keys(a: EvidenceAtom, b: EvidenceAtom) -> set[str]:
    """Shared entity keys excluding 'unknown' sentinels.

    Cross-artifact reinforcement on '*:unknown' is noise — every artifact that
    failed to classify a device/site would otherwise pair with every other.
    """
    shared = set(a.entity_keys).intersection(set(b.entity_keys))
    return {k for k in shared if not _is_unknown_entity_key(k)}


def _quantity_value(atom: EvidenceAtom) -> float | None:
    value = atom.value.get("quantity") if isinstance(atom.value, dict) else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantity_material_identity(atom: EvidenceAtom) -> str | None:
    """Stable material/line identity from atom value (normalized_item preferred)."""
    if atom.atom_type != AtomType.quantity:
        return None
    v = atom.value if isinstance(atom.value, dict) else {}
    ni = v.get("normalized_item")
    if isinstance(ni, str) and ni.strip():
        return ni.strip().lower()
    item = v.get("item")
    if isinstance(item, str) and item.strip():
        slug = re.sub(r"[^a-z0-9]+", "_", normalize_text(item).lower()).strip("_")
        return slug or None
    return None


def _canonical_material_key(atom: EvidenceAtom) -> str | None:
    """Map roster and vendor quantity identities onto one comparison key."""
    if atom.atom_type != AtomType.quantity:
        return None
    key = canonical_material_key(atom)
    if key:
        return key
    base = _quantity_material_identity(atom)
    if not base:
        return None
    s = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    if not s:
        return None
    if s == "rj45" or s.startswith("rj45_") or "_rj45_" in s or s.endswith("_rj45") or re.search(r"(^|_)rj45(_|$)", s):
        return "rj45"
    if "cat6a" in s:
        return "cat6a"
    if "cat6" in s and "utp" in s:
        return "cat6_utp"
    if "cat6" in s and "stp" in s:
        return "cat6_stp"
    if "cat6" in s and "shield" in s:
        return "cat6_stp"
    if "cat6" in s:
        return "cat6"
    if "fiber" in s or "strand" in s:
        return "fiber"
    return s


_VENDOR_PRIMARY_FILTER_LABEL = (
    "primary_included_unknown_numeric_excluding_optional_alternate_allowance_excluded_tbd_nic_by_others"
)


def _vendor_quote_line_counts_for_primary_total(atom: EvidenceAtom) -> bool:
    """Vendor lines included in primary quote totals (delegates to item_identity)."""
    if atom.authority_class != AuthorityClass.vendor_quote or atom.atom_type != AtomType.quantity:
        return False
    v = atom.value if isinstance(atom.value, dict) else {}
    return is_primary_vendor_quantity(v, raw_text=atom.raw_text)


def _identity_display(identity: str) -> str:
    if identity == "rj45":
        return "RJ45"
    if identity == "cat6_utp":
        return "Cat6 UTP"
    if identity == "cat6_stp":
        return "Cat6 STP"
    return identity


def _roster_vendor_material_totals(
    ordered: list[EvidenceAtom], identity: str
) -> tuple[EvidenceAtom | None, float, list[EvidenceAtom], float, list[EvidenceAtom]]:
    """Roster anchor (aggregate when present), roster total, primary vendor atoms, primary vendor total, excluded vendor atoms."""
    roster = [
        a
        for a in ordered
        if a.authority_class == AuthorityClass.approved_site_roster
        and a.atom_type == AtomType.quantity
        and _canonical_material_key(a) == identity
        and _quantity_value(a) is not None
    ]
    vendor_all = [
        a
        for a in ordered
        if a.authority_class == AuthorityClass.vendor_quote
        and a.atom_type == AtomType.quantity
        and _canonical_material_key(a) == identity
        and _quantity_value(a) is not None
    ]
    primary = [a for a in vendor_all if _vendor_quote_line_counts_for_primary_total(a)]
    excluded = [a for a in vendor_all if a not in primary]
    if not roster or not primary:
        return None, 0.0, [], 0.0, excluded
    vendor_primary_total = sum(float(_quantity_value(a)) for a in primary)
    agg = [a for a in roster if isinstance(a.value, dict) and a.value.get("aggregate") is True]
    if agg:
        anchor = sorted(agg, key=lambda a: a.id)[0]
        roster_total = sum(float(_quantity_value(a)) for a in agg)
    else:
        anchor = sorted(roster, key=lambda a: a.id)[0]
        roster_total = sum(float(_quantity_value(a)) for a in roster)
    return anchor, roster_total, primary, vendor_primary_total, excluded


def _shared_keys(a: EvidenceAtom, b: EvidenceAtom) -> set[str]:
    return set(a.entity_keys).intersection(set(b.entity_keys))


def _site_keys(atom: EvidenceAtom) -> set[str]:
    return {k for k in atom.entity_keys if k.startswith("site:")}


def _device_keys(atom: EvidenceAtom) -> set[str]:
    return {k for k in atom.entity_keys if k.startswith("device:")}


def _floor_room_keys(atom: EvidenceAtom) -> set[str]:
    return {k for k in atom.entity_keys if k.startswith("floor:") or k.startswith("room:") or k.startswith("device:")}


def _plate_keys(atom: EvidenceAtom) -> set[str]:
    return {k for k in atom.entity_keys if k.startswith("plate:")}


def _drop_or_outlet_keys(atom: EvidenceAtom) -> set[str]:
    return {k for k in atom.entity_keys if k.startswith("drop:") or k.startswith("outlet:")}


def _real_device_keys(atom: EvidenceAtom) -> set[str]:
    return {k for k in atom.entity_keys if k.startswith("device:") and k != "device:unknown"}


def _room_floor_location_device_fingerprint(atom: EvidenceAtom) -> frozenset[str]:
    """Room + floor + location + non-unknown device keys (excludes site: and plate:)."""
    parts: set[str] = set()
    for k in atom.entity_keys:
        if k.startswith(("room:", "floor:", "location:")):
            parts.add(k)
        if k.startswith("device:") and k != "device:unknown":
            parts.add(k)
    return frozenset(parts)


def _quantity_pair_comparable_scope(a: EvidenceAtom, b: EvidenceAtom) -> bool:
    """True when two quantity atoms refer to the same comparable scope (not merely same site)."""
    pl_a, pl_b = _plate_keys(a), _plate_keys(b)
    if pl_a and pl_b:
        return bool(pl_a & pl_b)
    if pl_a or pl_b:
        do_a, do_b = _drop_or_outlet_keys(a), _drop_or_outlet_keys(b)
        if do_a and do_b and (do_a & do_b):
            return True
        da, db = _real_device_keys(a), _real_device_keys(b)
        return bool(da & db)
    do_a, do_b = _drop_or_outlet_keys(a), _drop_or_outlet_keys(b)
    if do_a and do_b and (do_a & do_b):
        return True
    da, db = _real_device_keys(a), _real_device_keys(b)
    if da & db:
        return True
    fp_a = _room_floor_location_device_fingerprint(a)
    fp_b = _room_floor_location_device_fingerprint(b)
    return bool(fp_a) and fp_a == fp_b


# PR9 — bid-phase / scope-dimension comparability. A "Base" quote line
# and an "Add Alternate" quote line for the same site can legitimately
# differ in count without being a contradiction. Same for Bid Phase 1
# vs Bid Phase 2, or Owner-Allowance line vs Vendor line. Use this
# alongside _quantity_pair_comparable_scope.

_ALTERNATE_SCOPE_RE = re.compile(
    r"\b(add\s*alt|add\s*alternate|alternate|option|owner\s*allowance)\b",
    re.I,
)
_BASE_SCOPE_RE = re.compile(r"\b(base|base\s*bid)\b", re.I)


def _scope_dimension(atom: EvidenceAtom) -> str:
    """Return ``base`` / ``alternate`` / a normalized phase label /
    ``unspecified`` describing which bid scope this atom belongs to.

    Looks at value-dict hints first (bid_phase, alternate, add_alt,
    section, scope_bucket, quote_section), then falls back to a
    regex over the atom's raw_text + value string.
    """
    value = atom.value if isinstance(atom.value, dict) else {}
    for key in (
        "bid_phase",
        "alternate",
        "add_alt",
        "section",
        "scope_bucket",
        "quote_section",
    ):
        v = value.get(key)
        if v:
            return normalize_text(str(v))

    blob = normalize_text(atom.raw_text or "") + " " + normalize_text(str(value))
    if _ALTERNATE_SCOPE_RE.search(blob):
        return "alternate"
    if _BASE_SCOPE_RE.search(blob):
        return "base"
    return "unspecified"


def _scope_dimensions_compatible(a: EvidenceAtom, b: EvidenceAtom) -> bool:
    """True when two atoms either share a scope dimension or at least
    one is unspecified — used by the part-number quantity conflict
    path (which doesn't otherwise check bid-phase)."""
    a_scope = _scope_dimension(a)
    b_scope = _scope_dimension(b)
    if a_scope == "unspecified" or b_scope == "unspecified":
        return True
    return a_scope == b_scope


def _bom_cells(atom: EvidenceAtom) -> dict[str, Any]:
    """Return the atom's canonical_cells dict (xlsx/csv BOM rows
    populate this), else an empty dict."""
    value = atom.value if isinstance(atom.value, dict) else {}
    cc = value.get("canonical_cells")
    return cc if isinstance(cc, dict) else {}


def _norm_str(v: Any) -> str:
    if v is None:
        return ""
    return normalize_text(str(v))


def _quantity_atoms_are_comparable(a: EvidenceAtom, b: EvidenceAtom) -> bool:
    """True only when two quantity atoms can validly contradict.

    Rejects (post-2-case-review F7 hardening, layered on RF7):

    * non-quantity atoms,
    * atoms with conflicting scope dimensions (Base vs Add Alternate),
    * atoms whose declared ``scope_bucket`` differs (Add Alt 1 vs
      Owner Allowance vs Base vs Contingency),
    * atoms whose declared ``category`` differs (Staging vs
      Cable/Pathway vs Hardware vs Support),
    * atoms whose declared ``uom`` differs (PAIR vs EA vs LF),
    * atoms whose declared ``sku`` differs (gxt5_3000lvrt2uxl_a vs
      gxt5_3000lvrt2uxl_b — different SKU variants of the same base
      part_number are NOT contradictions),
    * atoms from the SAME spreadsheet artifact + same sheet with
      DIFFERENT row numbers (RF7),
    * atoms whose entity_keys are both populated but lack a shared
      ``part_number:`` key.
    """
    if a.atom_type.value != "quantity" or b.atom_type.value != "quantity":
        return False

    a_scope = _scope_dimension(a)
    b_scope = _scope_dimension(b)
    if a_scope != "unspecified" and b_scope != "unspecified" and a_scope != b_scope:
        return False

    # F7 — pull canonical BOM cells from both atoms and reject any
    # mismatch in scope_bucket / category / uom / sku / model /
    # asset_type. Empty values on either side are skipped (we don't
    # penalize cross-source comparisons where one side is a roster
    # row without these fields).
    ac = _bom_cells(a)
    bc = _bom_cells(b)
    for field in (
        "scope_bucket", "category", "uom", "unit_of_measure",
        "sku", "model", "asset_type",
    ):
        av = _norm_str(ac.get(field))
        bv = _norm_str(bc.get(field))
        if av and bv and av != bv:
            return False

    # Boss-review v8 F6 — asset_record rows are different equipment
    # in the same MDF/site. Sharing an entity key like
    # ``part_number:mdf_010`` is NOT enough; they must share an
    # explicit model OR asset_type AND a SKU/model identifier. Drop
    # any cross-asset-record contradiction that doesn't share a
    # model OR asset_type AND has at least one populated model on
    # each side.
    if a.atom_type.value == "asset_record" and b.atom_type.value == "asset_record":
        a_model = _norm_str(ac.get("model"))
        b_model = _norm_str(bc.get("model"))
        a_atype = _norm_str(ac.get("asset_type"))
        b_atype = _norm_str(bc.get("asset_type"))
        if (a_model and b_model and a_model != b_model) or (
            a_atype and b_atype and a_atype != b_atype
        ):
            return False
        # If neither model nor asset_type lines up, refuse.
        if not ((a_model and a_model == b_model) or (a_atype and a_atype == b_atype)):
            return False

    # RF7 — same artifact + same sheet + different rows + same
    # authority class ⇒ different line items unless they explicitly
    # share a ``part_number:`` key.
    a_loc = (a.source_refs[0].locator if a.source_refs else {}) or {}
    b_loc = (b.source_refs[0].locator if b.source_refs else {}) or {}
    a_sheet = a_loc.get("sheet")
    b_sheet = b_loc.get("sheet")
    a_row = a_loc.get("row")
    b_row = b_loc.get("row")
    if (
        a.artifact_id == b.artifact_id
        and a.authority_class == b.authority_class
        and a_sheet
        and a_sheet == b_sheet
        and a_row is not None
        and b_row is not None
        and a_row != b_row
    ):
        a_parts = {k for k in (a.entity_keys or []) if k.startswith("part_number:")}
        b_parts = {k for k in (b.entity_keys or []) if k.startswith("part_number:")}
        if not (a_parts & b_parts):
            return False

    a_entities = set(a.entity_keys or [])
    b_entities = set(b.entity_keys or [])
    if a_entities and b_entities and not (a_entities & b_entities):
        return False

    return True


def _edge_id(project_id: str, edge_type: EdgeType, from_id: str, to_id: str, reason: str) -> str:
    return stable_id("edge", project_id, edge_type.value, from_id, to_id, reason)


def _is_schematic_quantity_atom(atom: EvidenceAtom) -> bool:
    if atom.atom_type != AtomType.quantity:
        return False
    value = atom.value if isinstance(atom.value, dict) else {}
    return bool(value.get("schematic_target_key")) and bool(value.get("schematic_role"))


def _schematic_quantity_signature(atom: EvidenceAtom) -> tuple[Any, ...]:
    """Pairing key for schematic detected/declared quantity atoms.

    Intentionally *does not* include the sheet number: in a normal
    drawing set, the declared count lives on the global legend sheet
    (e.g. T0.01) while the detected count lives on a floor-plan sheet
    (e.g. E1.01). Pairing only by (artifact, target_key) lets the
    same legend govern many drawing pages — which is the whole point
    of the cross-sheet resolver. The edge's reason string still names
    both sheets for the reviewer.
    """
    value = atom.value if isinstance(atom.value, dict) else {}
    return (
        atom.artifact_id,
        value.get("schematic_target_key"),
    )


def _build_schematic_quantity_edges(
    project_id: str, atoms: list[EvidenceAtom]
) -> list[EvidenceEdge]:
    """Pair detected/declared schematic quantity atoms and emit edges.

    Two schematic quantity atoms contradict when they share the same
    artifact, the same sheet, and the same target key, and one is
    marked ``schematic_role="detected"`` while the other is
    ``schematic_role="declared"``. The numeric quantities must
    disagree by at least 1 to qualify (drawings often round
    rough-in callouts so an exact-match pass is too sensitive in
    other directions but a difference of <1 is rarely meaningful).
    """
    by_sig: dict[tuple[Any, ...], dict[str, list[EvidenceAtom]]] = {}
    for atom in atoms:
        if not _is_schematic_quantity_atom(atom):
            continue
        sig = _schematic_quantity_signature(atom)
        role = str(atom.value.get("schematic_role"))
        by_sig.setdefault(sig, {}).setdefault(role, []).append(atom)
    edges: list[EvidenceEdge] = []
    for sig, by_role in by_sig.items():
        detected = sorted(by_role.get("detected", []), key=lambda a: a.id)
        declared = sorted(by_role.get("declared", []), key=lambda a: a.id)
        if not detected or not declared:
            continue
        # One-to-one pairing in deterministic ID order.
        for det, dec in zip(detected, declared):
            det_qty = det.value.get("quantity")
            dec_qty = dec.value.get("quantity")
            try:
                if det_qty is None or dec_qty is None:
                    continue
                if abs(float(det_qty) - float(dec_qty)) < 1.0:
                    continue
            except (TypeError, ValueError):
                continue
            det_sheet = det.value.get("schematic_sheet_number")
            dec_sheet = dec.value.get("schematic_sheet_number")
            edges.append(
                _build_edge(
                    project_id,
                    EdgeType.contradicts,
                    det,
                    dec,
                    (
                        f"Schematic quantity contradiction target {sig[1]}: "
                        f"detected={det_qty} on sheet {det_sheet or '?'} vs "
                        f"declared={dec_qty} on sheet {dec_sheet or '?'}"
                    ),
                    0.9,
                    edge_family=EDGE_FAMILY_SCHEMATIC_QUANTITY_CONTRADICTION,
                )
            )
    return edges


def _build_edge(
    project_id: str,
    edge_type: EdgeType,
    from_atom: EvidenceAtom,
    to_atom: EvidenceAtom,
    reason: str,
    confidence: float,
    *,
    metadata: dict[str, Any] | None = None,
    edge_family: str | None = None,
) -> EvidenceEdge:
    """Build an :class:`EvidenceEdge` and stamp cross-artifact provenance.

    ``metadata['cross_artifact']`` is set to ``True`` whenever the two
    atoms come from different source artifacts.  ``from_artifact_id`` /
    ``to_artifact_id`` are also added so OrbitBrief can render a
    "this PDF says X but this email says Y" badge directly from the
    edge alone.

    ``edge_family`` (when provided) lands in ``metadata['edge_family']`` so
    downstream consumers can filter the graph by family without re-parsing
    the human-readable reason string.
    """
    meta = dict(metadata or {})
    meta.setdefault("from_artifact_id", from_atom.artifact_id)
    meta.setdefault("to_artifact_id", to_atom.artifact_id)
    if from_atom.artifact_id != to_atom.artifact_id:
        meta.setdefault("cross_artifact", True)
    else:
        meta.setdefault("cross_artifact", False)
    if edge_family and "edge_family" not in meta:
        meta["edge_family"] = edge_family
    return EvidenceEdge(
        id=_edge_id(project_id, edge_type, from_atom.id, to_atom.id, reason),
        project_id=project_id,
        from_atom_id=from_atom.id,
        to_atom_id=to_atom.id,
        edge_type=edge_type,
        reason=reason,
        confidence=confidence,
        metadata=meta,
    )


def build_edges(project_id: str, atoms: list[EvidenceAtom], entities: list[EntityRecord]) -> list[EvidenceEdge]:
    pack = get_active_domain_pack()
    exclusion_patterns = [normalize_text(pattern) for pattern in pack.exclusion_patterns]
    constraint_patterns = [
        normalize_text(pattern)
        for patterns in pack.constraint_patterns.values()
        for pattern in patterns
    ]
    edges: list[EvidenceEdge] = []
    seen: set[str] = set()

    def push(edge: EvidenceEdge) -> None:
        if edge.id in seen:
            return
        seen.add(edge.id)
        edges.append(edge)

    ordered = sorted(atoms, key=lambda a: a.id)
    atom_by_id = {atom.id: atom for atom in ordered}

    # ─── Inverted index: entity_key → list of atom indices ───
    # Replaces the O(n²) double-loop with O(sum of bucket²) pair generation.
    # See PRODUCTION_GAPS.md P2.2.  Without this, Downey's 4,892 atoms produced
    # ~12M atom pairs and a 38-minute graph_build.
    key_to_indices: dict[str, list[int]] = {}
    for idx, atom in enumerate(ordered):
        # Dedupe entity_keys per atom — duplicates (e.g. an atom whose
        # alias matcher registered ``device:ip_camera`` twice) would
        # otherwise insert the same idx into the bucket multiple times,
        # which downstream produces self-pair (i==j) candidates and
        # ultimately a ``supports`` self-loop edge that fails graph
        # invariants. Dedupe here, not in the atom — keys may come from
        # multiple legitimate sources upstream.
        seen_keys: set[str] = set()
        for k in atom.entity_keys:
            if _is_unknown_entity_key(k) or k in seen_keys:
                continue
            seen_keys.add(k)
            key_to_indices.setdefault(k, []).append(idx)

    # Treat keys that match too many atoms as "noisy" — they'd otherwise pair
    # every atom with every other.  Common culprits: ``site:campus``,
    # ``device:storage``, broad typed-alias hits.  Atoms with only noisy keys
    # in common are skipped; pairs that share at least one informative key
    # are still processed even if they also share noisy keys.
    #
    # Threshold scales sub-linearly with corpus size so that candidate_pairs
    # grows as O(N · sqrt(N)), not O(N²).
    import math
    noisy_threshold = max(
        NOISY_ENTITY_KEY_BUCKET_FLOOR,
        int(math.sqrt(max(len(ordered), 1))),
    )
    informative_keys = {
        k for k, indices in key_to_indices.items()
        if 2 <= len(indices) <= noisy_threshold
    }
    # Part-number / quantity / address keys are always informative regardless
    # of bucket size — they're the highest-precision signals for contradictions.
    for k in list(key_to_indices.keys()):
        if k.startswith(_PART_NUMBER_PREFIX) or k.startswith(_QUANTITY_PREFIX) or k.startswith("address:"):
            informative_keys.add(k)

    # Generate candidate pairs from informative keys only.  Each pair is
    # processed once (by sorted (i, j) tuple) regardless of how many keys
    # it shares.  Process keys in ascending bucket size so the pairs we
    # cap-out have the biggest noisy buckets, not the precise ones.
    #
    # Scale governor: cap pairs at ``min(MAX_CANDIDATE_PAIRS,
    # N * MAX_PAIRS_PER_ATOM)`` so a 10k-row BOM doesn't generate
    # 250k pairs (which would otherwise produce 25 edges per atom).
    pair_cap = min(MAX_CANDIDATE_PAIRS, len(ordered) * MAX_PAIRS_PER_ATOM)
    candidate_pairs: set[tuple[int, int]] = set()
    keys_by_bucket_size = sorted(
        informative_keys, key=lambda k: (len(key_to_indices.get(k, [])), k)
    )
    for k in keys_by_bucket_size:
        indices = key_to_indices.get(k) or []
        if len(indices) < 2:
            continue
        for ii in range(len(indices)):
            if len(candidate_pairs) >= pair_cap:
                break
            i = indices[ii]
            for jj in range(ii + 1, len(indices)):
                if len(candidate_pairs) >= pair_cap:
                    break
                j = indices[jj]
                if i == j:
                    # Defensive: self-pairs would build a self-loop edge
                    # of type ``supports`` that the graph invariants
                    # validator rejects. The dedupe above should make
                    # this unreachable, but we belt-and-suspender it
                    # because graph invariants are a hard CI gate.
                    continue
                if i < j:
                    candidate_pairs.add((i, j))
                else:
                    candidate_pairs.add((j, i))
        if len(candidate_pairs) >= pair_cap:
            break

    for i, j in sorted(candidate_pairs):
        a = ordered[i]
        b = ordered[j]
        shared = _shared_keys(a, b)
        if not shared:
            continue

        # supports: same entity keys + same atom_type + same normalized value/quantity.
        quantity_a = _quantity_value(a)
        quantity_b = _quantity_value(b)
        if a.atom_type == b.atom_type:
            same_value = normalize_text(str(a.value)) == normalize_text(str(b.value))
            same_quantity = quantity_a is not None and quantity_b is not None and quantity_a == quantity_b
            if same_value or same_quantity:
                push(
                    _build_edge(
                        project_id,
                        EdgeType.supports,
                        a,
                        b,
                        "Atoms support each other with matching entity, type, and value",
                        0.88,
                        edge_family=EDGE_FAMILY_VALUE_SUPPORT,
                    )
                )
        if (
            a.atom_type.value == "constraint"
            and b.atom_type.value == "constraint"
            and _site_keys(a).intersection(_site_keys(b))
        ):
            push(
                _build_edge(
                    project_id,
                    EdgeType.supports,
                    a,
                    b,
                    "Constraint atoms align on same site context",
                    0.84,
                    edge_family=EDGE_FAMILY_CONSTRAINT_ALIGN,
                )
            )

        # contradicts: same comparable scope + quantity differs (not mere co-site line items).
        if a.atom_type.value == "quantity" and b.atom_type.value == "quantity":
            if quantity_a is None or quantity_b is None or quantity_a == quantity_b:
                pass  # skip block below; fall through to other rules
            else:
                sites_a = _site_keys(a)
                sites_b = _site_keys(b)
                ok = True
                if sites_a and sites_b and sites_a != sites_b:
                    ok = False
                if (
                    ok
                    and a.authority_class == AuthorityClass.approved_site_roster
                    and b.authority_class == AuthorityClass.approved_site_roster
                ):
                    ida, idb = _canonical_material_key(a), _canonical_material_key(b)
                    if ida is None or idb is None or ida != idb:
                        ok = False
                if ok and not _quantity_pair_comparable_scope(a, b):
                    ok = False
                # PR9 gate — bid-phase / scope-dimension comparability.
                # Refuses Base vs Add-Alt or disjoint-entity contradictions.
                if ok and not _quantity_atoms_are_comparable(a, b):
                    ok = False
                if ok:
                    push(
                        _build_edge(
                            project_id,
                            EdgeType.contradicts,
                            a,
                            b,
                            f"Quantity mismatch {quantity_a:g} vs {quantity_b:g} for shared entity context",
                            0.9,
                            edge_family=EDGE_FAMILY_QUANTITY_CONTRADICTION,
                        )
                    )

        # ─── P0.5: part_number-driven quantity conflict ───
        # When two atoms share a ``part_number:*`` entity but their
        # ``quantity:*`` entity keys differ, that's a cost-proposal vs
        # equipment-list mismatch (e.g. Natomas CW9166I-B 500 vs 136).
        # This rule fires regardless of atom_type so it catches scope_item
        # atoms that carry "Qty: N" patterns extracted by entity_extraction.
        shared_parts = {k for k in shared if k.startswith(_PART_NUMBER_PREFIX)}
        if shared_parts:
            qty_keys_a = {k for k in a.entity_keys if k.startswith(_QUANTITY_PREFIX)}
            qty_keys_b = {k for k in b.entity_keys if k.startswith(_QUANTITY_PREFIX)}
            if qty_keys_a and qty_keys_b and qty_keys_a != qty_keys_b:
                # Boss-review F7 + v8 F6 — apply the FULL comparability
                # gate here too. Sharing a normalized part_number key
                # is not enough: two BOM/asset rows whose
                # scope_bucket / category / uom / sku / model /
                # asset_type differ are different items and must not
                # contradict.
                ac = _bom_cells(a)
                bc = _bom_cells(b)
                cell_mismatch = False
                for field in (
                    "scope_bucket", "category", "uom", "unit_of_measure",
                    "sku", "model", "asset_type",
                ):
                    av = _norm_str(ac.get(field))
                    bv = _norm_str(bc.get(field))
                    if av and bv and av != bv:
                        cell_mismatch = True
                        break
                if not cell_mismatch and (
                    a.atom_type.value == "asset_record"
                    and b.atom_type.value == "asset_record"
                ):
                    a_model = _norm_str(ac.get("model"))
                    b_model = _norm_str(bc.get("model"))
                    a_atype = _norm_str(ac.get("asset_type"))
                    b_atype = _norm_str(bc.get("asset_type"))
                    if not (
                        (a_model and a_model == b_model)
                        or (a_atype and a_atype == b_atype)
                    ):
                        cell_mismatch = True
                if cell_mismatch:
                    pass
                # PR9 gate — refuse Base vs Add-Alt cost-proposal lines
                # for the same part number; those legitimately differ.
                elif not _scope_dimensions_compatible(a, b):
                    pass
                else:
                    qa_display = sorted(k.split(":", 1)[1] for k in qty_keys_a)
                    qb_display = sorted(k.split(":", 1)[1] for k in qty_keys_b)
                    part_display = sorted(k.split(":", 1)[1] for k in shared_parts)
                    push(
                        _build_edge(
                            project_id,
                            EdgeType.contradicts,
                            a,
                            b,
                            (
                                f"Quantity contradiction for part {','.join(part_display)}: "
                                f"{','.join(qa_display)} vs {','.join(qb_display)}"
                            ),
                            0.92,
                            edge_family=EDGE_FAMILY_PART_NUMBER_QUANTITY_CONFLICT,
                        )
                    )

    # Schematic same-sheet quantity contradiction (PR7) — pairs an
    # aggregated detected count with a legend-declared count on the
    # same drawing. Both atoms carry bbox provenance so the
    # packetizer's narrow same-artifact exception can certify them.
    for edge in _build_schematic_quantity_edges(project_id, ordered):
        push(edge)

    # excludes: exclusion atom mentions entity key in another atom.
    # Uses the entity-key index so we only iterate atoms that actually
    # share an entity_key with the exclusion (was O(exclusions × atoms)).
    exclusions = [a for a in ordered if a.atom_type.value == "exclusion"]
    exclusions.extend(
        [
            atom
            for atom in ordered
            if atom.atom_type.value == "customer_instruction"
            and any(pattern in normalize_text(atom.raw_text) for pattern in exclusion_patterns)
        ]
    )
    exclusions = sorted({atom.id: atom for atom in exclusions}.values(), key=lambda atom: atom.id)
    for ex in exclusions:
        ex_keys = {k for k in ex.entity_keys if not _is_unknown_entity_key(k)}
        if not ex_keys:
            continue
        target_idx_set: set[int] = set()
        for k in ex_keys:
            for idx in key_to_indices.get(k, []):
                target_idx_set.add(idx)
        for idx in sorted(target_idx_set):
            target = ordered[idx]
            if target.id == ex.id:
                continue
            shared_keys = ex_keys.intersection(set(target.entity_keys))
            meaningful = {k for k in shared_keys if not _is_unknown_entity_key(k)}
            if meaningful:
                push(
                    _build_edge(
                        project_id,
                        EdgeType.excludes,
                        ex,
                        target,
                        "Exclusion atom applies to target entity context",
                        0.9,
                        edge_family=EDGE_FAMILY_EXCLUSION_APPLIES,
                    )
                )

    # requires: constraint shares site with scope/quantity atoms.
    # Same index-driven optimization.
    constraints = [a for a in ordered if a.atom_type.value == "constraint"]
    constraints.extend(
        [
            atom
            for atom in ordered
            if atom.atom_type.value == "customer_instruction"
            and any(pattern in normalize_text(atom.raw_text) for pattern in constraint_patterns)
        ]
    )
    constraints = sorted({atom.id: atom for atom in constraints}.values(), key=lambda atom: atom.id)
    for constraint in constraints:
        sites = _site_keys(constraint)
        if not sites:
            continue
        target_idx_set = set()
        for site_key in sites:
            for idx in key_to_indices.get(site_key, []):
                target_idx_set.add(idx)
        for idx in sorted(target_idx_set):
            target = ordered[idx]
            if target.id == constraint.id or target.atom_type.value not in {"scope_item", "quantity"}:
                continue
            shared_sites = sites.intersection(_site_keys(target))
            meaningful_sites = {k for k in shared_sites if not _is_unknown_entity_key(k)}
            if meaningful_sites:
                push(
                    _build_edge(
                        project_id,
                        EdgeType.requires,
                        constraint,
                        target,
                        "Constraint requires adherence for same site context",
                        0.86,
                        edge_family=EDGE_FAMILY_CONSTRAINT_REQUIRES,
                    )
                )

    # Aggregate device quantity contradiction: approved_site_roster vs vendor_quote.
    by_device_and_authority: dict[tuple[str, AuthorityClass], dict[str, object]] = {}
    for atom in ordered:
        if atom.atom_type.value != "quantity":
            continue
        qty = _quantity_value(atom)
        if qty is None:
            continue
        for device_key in _device_keys(atom):
            key = (device_key, atom.authority_class)
            bucket = by_device_and_authority.setdefault(key, {"total": 0.0, "atoms": []})
            bucket["total"] = float(bucket["total"]) + qty
            bucket["atoms"].append(atom)

    device_keys = sorted({k[0] for k in by_device_and_authority})
    for device_key in device_keys:
        if device_key == "device:unknown":
            continue
        approved = by_device_and_authority.get((device_key, AuthorityClass.approved_site_roster))
        vendor = by_device_and_authority.get((device_key, AuthorityClass.vendor_quote))
        if not approved or not vendor:
            continue
        approved_total = float(approved["total"])
        vendor_total = float(vendor["total"])
        if approved_total == vendor_total:
            continue
        from_atom = sorted(approved["atoms"], key=lambda a: a.id)[0]
        to_atom = sorted(vendor["atoms"], key=lambda a: a.id)[0]
        # RF7 — only emit when the chosen pair are actually
        # comparable. The aggregate path defaults to the
        # lexicographically-first atom; if they share no
        # part_number and live on different rows of the same
        # workbook, they're different SKUs that happen to share a
        # generic device:* key.
        if not _quantity_atoms_are_comparable(from_atom, to_atom):
            continue
        reason = (
            f"Aggregate scoped quantity {int(approved_total) if approved_total.is_integer() else approved_total:g} "
            f"does not match vendor quantity {int(vendor_total) if vendor_total.is_integer() else vendor_total:g} "
            f"for {device_key}"
        )
        push(
            _build_edge(
                project_id,
                EdgeType.contradicts,
                from_atom,
                to_atom,
                reason,
                0.95,
                edge_family=EDGE_FAMILY_DEVICE_AGGREGATE_MISMATCH,
            )
        )

    # ─── Cross-artifact device-only quantity conflict ────────────────
    # Catches the universal managed-services failure mode where:
    #
    #   BOM PDF says:  "50 wireless access points"  (vendor_quote)
    #   SOW PDF says:  "60 access points across three sites" (contractual_scope)
    #
    # No shared part_number, so PART_NUMBER_QUANTITY_CONFLICT can't fire;
    # the atoms are scope_item / vendor_line_item, so the strict
    # DEVICE_AGGREGATE_MISMATCH (which only pairs approved_site_roster
    # vs vendor_quote on atom_type=quantity) can't fire either.
    #
    # Strict false-positive guards:
    #   - cross-artifact (same-file diff is normal table rows)
    #   - no shared part_number (PART_NUMBER_QUANTITY_CONFLICT handles those)
    #   - excludes device:unknown
    #   - excludes single-digit quantities (qty 1-4 is often template
    #     "1 each" or generic count, not a deal-level scope value)
    #   - excludes when atoms reference different sites (site-scoped
    #     counts are independent)
    #   - excludes when both atoms are exclusion / boilerplate / risk
    for_device_pairs_seen: set[tuple[str, str, str]] = set()
    by_device: dict[str, list] = {}
    for atom in ordered:
        # Walk every device key on every atom (any atom that carries
        # a quantity:* + device:* combo is a candidate, regardless of
        # whether atom_type is quantity / scope_item / vendor_line_item).
        qty_keys = {k for k in atom.entity_keys if k.startswith(_QUANTITY_PREFIX)}
        if not qty_keys:
            continue
        if any(k.startswith(_PART_NUMBER_PREFIX) for k in atom.entity_keys):
            # If THIS atom has a part_number, leave it to the part-number
            # path — only emit device-only conflicts when at least one
            # side has no part_number anchor.
            pass
        for dk in _device_keys(atom):
            if not dk or dk == "device:unknown":
                continue
            by_device.setdefault(dk, []).append(atom)

    # Scale cap: when a single device shows up in too many atoms (e.g.
    # a 10k-row BOM where every row mentions "switch"), the N^2 pair
    # check explodes. Limit per-device candidates so the inner loop
    # stays linear-ish. Aggregate rows / governing atoms keep priority.
    DEVICE_QTY_BUCKET_CAP = 200
    for device_key, atoms_for_device in by_device.items():
        if len(atoms_for_device) < 2:
            continue
        # Filter atoms whose authority/atom_type suggests they're NOT
        # a real scope statement (exclusion text, boilerplate prose).
        candidates = [
            a for a in atoms_for_device
            if a.atom_type.value not in {"exclusion", "compliance", "risk", "open_question"}
        ]
        if len(candidates) < 2:
            continue
        # Bucket cap: when a device appears in too many atoms (e.g. a
        # 10k-row BOM where every row mentions "switch"), prefer atoms
        # that look like roster aggregates / governing claims and cap
        # the bucket to ``DEVICE_QTY_BUCKET_CAP``. This keeps the
        # contradiction surface honest without doing 50M pair checks.
        if len(candidates) > DEVICE_QTY_BUCKET_CAP:
            def _priority(a: EvidenceAtom) -> tuple[int, str]:
                ac = a.authority_class.value if hasattr(a.authority_class, "value") else str(a.authority_class)
                # approved_site_roster / contractual_scope first;
                # vendor_quote / customer_email next; everything else last.
                tier = 0 if ac in {"approved_site_roster", "contractual_scope", "customer_current_authored"} else (
                    1 if ac in {"vendor_quote", "formal_sow", "current_addendum"} else 2
                )
                return (tier, a.id)
            candidates = sorted(candidates, key=_priority)[:DEVICE_QTY_BUCKET_CAP]
        # Build pairs that satisfy the cross-doc / different-qty /
        # no-shared-part-number / same-site-context guards.
        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                if a.artifact_id == b.artifact_id:
                    # Same artifact — allow ONLY when the two atoms
                    # come from clearly different positions in the
                    # source doc (different line OR different
                    # sentence_index). Same-row table cells with the
                    # same qty get filtered out by the value-set
                    # intersection check below, so we don't need extra
                    # guards here. This unlocks intra-doc self-
                    # contradictions ("24 cameras... 30 cameras...")
                    # while keeping the original cross-doc behavior.
                    if not _atoms_are_distinct_positions(a, b):
                        continue
                # Multi-device atoms now bind quantities to specific
                # device nouns via ``_noun_anchored_quantity``. If both
                # atoms can pin a specific qty to ``device_key`` via
                # noun proximity, treat those pinned values as the
                # single qty for the comparison below. Otherwise fall
                # back to the original "single device + single qty"
                # ambiguity guard.
                a_devices = {k for k in a.entity_keys if k.startswith("device:") and k != "device:unknown"}
                b_devices = {k for k in b.entity_keys if k.startswith("device:") and k != "device:unknown"}
                a_qty_keys = {k for k in a.entity_keys if k.startswith(_QUANTITY_PREFIX)}
                b_qty_keys = {k for k in b.entity_keys if k.startswith(_QUANTITY_PREFIX)}
                if not a_qty_keys or not b_qty_keys:
                    continue
                device_canonical = device_key.split(":", 1)[1]
                pinned_a: int | None = None
                pinned_b: int | None = None
                if len(a_devices) > 1 or len(b_devices) > 1 or len(a_qty_keys) > 1 or len(b_qty_keys) > 1:
                    pinned_a = _noun_anchored_quantity(a.raw_text or "", device_canonical)
                    pinned_b = _noun_anchored_quantity(b.raw_text or "", device_canonical)
                    if pinned_a is None or pinned_b is None:
                        # Couldn't disambiguate — preserve the old
                        # safety guards. Multi-device or multi-qty
                        # without a noun-anchor stays skipped.
                        continue
                    # Substitute the pinned values into the
                    # comparison sets so the downstream logic uses
                    # the device-specific qty for each side.
                    a_qty_keys = {f"quantity:{pinned_a}"}
                    b_qty_keys = {f"quantity:{pinned_b}"}
                a_parts = {k for k in a.entity_keys if k.startswith(_PART_NUMBER_PREFIX)}
                b_parts = {k for k in b.entity_keys if k.startswith(_PART_NUMBER_PREFIX)}
                # Skip when a shared part_number exists — that's the
                # part-number conflict path.
                if a_parts and b_parts and a_parts.intersection(b_parts):
                    continue
                # Site-scope guard: if BOTH atoms reference sites and
                # the site sets are disjoint, they're talking about
                # different deployments.
                sites_a = _site_keys(a)
                sites_b = _site_keys(b)
                if sites_a and sites_b and not sites_a.intersection(sites_b):
                    continue
                # Compute candidate (a_qty, b_qty) pairs — the smallest
                # informative pair (avoid spamming N*M for one device).
                def _qty_val(k: str) -> int | None:
                    try:
                        return int(k.split(":", 1)[1])
                    except (ValueError, IndexError):
                        return None
                a_vals = sorted({v for v in (_qty_val(k) for k in a_qty_keys) if v is not None})
                b_vals = sorted({v for v in (_qty_val(k) for k in b_qty_keys) if v is not None})
                # Single-digit guard. When binding came from a
                # noun-anchored pin (pinned_a/pinned_b set above), the
                # signal is strong enough to keep qty:2..4. Otherwise
                # the template-row noise risk is real and we keep
                # qty>=5.
                min_qty = 2 if (pinned_a is not None or pinned_b is not None) else 5
                a_vals = [v for v in a_vals if v >= min_qty]
                b_vals = [v for v in b_vals if v >= min_qty]
                if not a_vals or not b_vals:
                    continue
                # The conflict fires when there is NO common value AND
                # the difference between the closest pair is non-trivial.
                if set(a_vals).intersection(b_vals):
                    continue
                # Closest cross-set pair:
                pair = min(
                    ((av, bv) for av in a_vals for bv in b_vals),
                    key=lambda p: abs(p[0] - p[1]),
                )
                av, bv = pair
                dedup_key = (device_key, a.id, b.id)
                if dedup_key in for_device_pairs_seen:
                    continue
                for_device_pairs_seen.add(dedup_key)
                # Pick deterministic from/to (lex by atom id)
                from_atom, to_atom = (a, b) if a.id < b.id else (b, a)
                is_intra_doc = a.artifact_id == b.artifact_id
                scope_label = "intra-doc" if is_intra_doc else "Cross-artifact"
                reason = (
                    f"{scope_label} quantity mismatch for {device_key}: {av} vs {bv}"
                )
                push(
                    _build_edge(
                        project_id,
                        EdgeType.contradicts,
                        from_atom,
                        to_atom,
                        reason,
                        0.78,
                        edge_family=EDGE_FAMILY_DEVICE_QUANTITY_CROSS_DOC,
                    )
                )

    # Material / line-item identity: governing approved_site_roster vs vendor_quote (normalized_item).
    # Roster aggregate row (when present) defines scope quantity; vendor is summed per identity; never reversed.
    identities = sorted(
        {
            ident
            for atom in ordered
            if (ident := _canonical_material_key(atom)) is not None
            and atom.atom_type == AtomType.quantity
            and atom.authority_class
            in {AuthorityClass.approved_site_roster, AuthorityClass.vendor_quote}
        }
    )
    for identity in identities:
        anchor, roster_total, primary_vendors, vendor_primary_total, excluded_vendors = _roster_vendor_material_totals(
            ordered, identity
        )
        if anchor is None or not primary_vendors:
            continue
        if roster_total == vendor_primary_total:
            continue
        vendor_rep = sorted(primary_vendors, key=lambda a: a.id)[0]
        r_display = int(roster_total) if roster_total.is_integer() else roster_total
        v_display = int(vendor_primary_total) if vendor_primary_total.is_integer() else vendor_primary_total
        delta = float(roster_total) - float(vendor_primary_total)
        disp = _identity_display(identity)
        if delta > 0:
            delta_note = f"vendor quote short by {int(delta) if float(delta).is_integer() else round(delta, 4)}"
        elif delta < 0:
            over = -delta
            oi = int(over) if float(over).is_integer() else round(over, 4)
            delta_note = f"vendor quote over by {oi} vs addendum (differs by +{oi})"
        else:
            delta_note = "quantities match"
        reason = (
            f"{disp}: approved_site_roster aggregate {r_display:g} vs vendor_quote primary-line total {v_display:g}; "
            f"{delta_note}."
        )
        meta: dict[str, Any] = {
            "identity": identity,
            "roster_quantity": float(roster_total),
            "vendor_quantity": float(vendor_primary_total),
            "delta": float(delta),
            "roster_atom_id": anchor.id,
            "vendor_atom_ids": sorted(a.id for a in primary_vendors),
            "vendor_excluded_atom_ids": sorted(a.id for a in excluded_vendors),
            "roster_authority_class": AuthorityClass.approved_site_roster.value,
            "vendor_authority_class": AuthorityClass.vendor_quote.value,
            "comparison_basis": "aggregate_roster_vs_summed_vendor_quote",
            "included_vendor_line_filter": _VENDOR_PRIMARY_FILTER_LABEL,
            "preferred_packet_family": "quantity_conflict" if identity == "rj45" else "vendor_mismatch",
        }
        push(
            _build_edge(
                project_id,
                EdgeType.contradicts,
                anchor,
                vendor_rep,
                reason,
                0.96,
                metadata=meta,
                edge_family=EDGE_FAMILY_MATERIAL_AGGREGATE_MISMATCH,
            )
        )

    # Cross-artifact reinforcement: when an instruction or constraint in
    # one artifact (e.g. customer email, transcript decision) lines up
    # with a scope/quantity/exclusion in another (e.g. PDF SOW, XLSX
    # roster), record a "supports" edge with ``cross_artifact=True`` so
    # downstream packetizers can show OrbitBrief readers a single
    # cross-source story instead of two unrelated atoms.
    cross_artifact_pairs: list[tuple[AtomType, set[AtomType]]] = [
        (AtomType.customer_instruction, {AtomType.scope_item, AtomType.quantity, AtomType.exclusion}),
        (AtomType.decision, {AtomType.scope_item, AtomType.quantity, AtomType.exclusion, AtomType.constraint}),
        (AtomType.action_item, {AtomType.scope_item, AtomType.quantity}),
        (AtomType.constraint, {AtomType.scope_item, AtomType.quantity}),
        (AtomType.exclusion, {AtomType.scope_item, AtomType.quantity}),
        (AtomType.open_question, {AtomType.scope_item, AtomType.quantity, AtomType.constraint}),
    ]
    seeds_by_type: dict[AtomType, list[EvidenceAtom]] = {}
    for atom in ordered:
        seeds_by_type.setdefault(atom.atom_type, []).append(atom)
    # Pre-compute target-id sets per atom_type so we can intersect against
    # the entity-key index for O(shared_keys × bucket_size) lookups instead
    # of O(sources × targets).
    target_ids_by_type: dict[AtomType, set[str]] = {
        atype: {atom.id for atom in atoms_of_type}
        for atype, atoms_of_type in seeds_by_type.items()
    }
    for source_type, target_types in cross_artifact_pairs:
        sources = seeds_by_type.get(source_type) or []
        target_id_pool: set[str] = set()
        for tt in target_types:
            target_id_pool |= target_ids_by_type.get(tt, set())
        if not target_id_pool:
            continue
        for source in sources:
            src_keys = {k for k in source.entity_keys if not _is_unknown_entity_key(k)}
            if not src_keys:
                continue
            # Use the entity-key index to find candidate targets — only atoms
            # that actually share at least one informative key with the
            # source make it into ``scored``.
            candidate_idxs: set[int] = set()
            for k in src_keys:
                for idx in key_to_indices.get(k, []):
                    candidate_idxs.add(idx)
            scored: list[tuple[int, str, EvidenceAtom, set[str]]] = []
            for idx in candidate_idxs:
                target = ordered[idx]
                if target.id == source.id:
                    continue
                if target.id not in target_id_pool:
                    continue
                if source.artifact_id == target.artifact_id:
                    continue
                tgt_keys = {k for k in target.entity_keys if not _is_unknown_entity_key(k)}
                shared = src_keys.intersection(tgt_keys)
                if not shared:
                    continue
                scored.append((len(shared), target.id, target, shared))
            scored.sort(key=lambda row: (-row[0], row[1]))
            for _, _, target, shared in scored[:MAX_CROSS_ARTIFACT_TARGETS_PER_SOURCE]:
                shared_label = ", ".join(sorted(shared)[:4])
                push(
                    _build_edge(
                        project_id,
                        EdgeType.supports,
                        source,
                        target,
                        f"Cross-artifact reinforcement on {shared_label}",
                        0.78,
                        metadata={
                            "shared_entity_keys": sorted(shared),
                            "source_atom_type": source.atom_type.value,
                            "target_atom_type": target.atom_type.value,
                            "edge_family": EDGE_FAMILY_CROSS_ARTIFACT_REINFORCEMENT,
                        },
                    )
                )

    semantic_candidates = propose_semantic_link_candidates(ordered, domain_pack=pack)
    for candidate in semantic_candidates:
        if candidate.status != "accepted":
            continue
        from_atom = atom_by_id.get(candidate.from_atom_id)
        to_atom = atom_by_id.get(candidate.to_atom_id)
        if from_atom is None or to_atom is None:
            continue
        if candidate.proposed_edge_type == EdgeType.contradicts:
            continue
        reason = (
            "semantic_candidate_linker "
            f"method={candidate.method} score={candidate.similarity_score:.3f} "
            f"status={candidate.status}; {candidate.reason}"
        )
        push(
            _build_edge(
                project_id=project_id,
                edge_type=candidate.proposed_edge_type,
                from_atom=from_atom,
                to_atom=to_atom,
                reason=reason,
                confidence=min(0.99, max(0.5, candidate.similarity_score)),
                edge_family=EDGE_FAMILY_SEMANTIC_LINK,
            )
        )

    edges.sort(key=lambda e: e.id)
    return edges


def build_entity_edges(atoms: list[EvidenceAtom]):
    """Compatibility wrapper for older call sites."""
    return build_edges(project_id="unknown_project", atoms=atoms, entities=[])
