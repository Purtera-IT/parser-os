from __future__ import annotations

import re
from typing import Any

from app.core.ids import stable_id
from app.core.item_identity import canonical_material_key, is_primary_vendor_quantity
from app.core.normalizers import normalize_text
from app.core.schemas import AtomType, AuthorityClass, EdgeType, EntityRecord, EvidenceAtom, EvidenceEdge
from app.domain import get_active_domain_pack
from app.semantic.linker import propose_semantic_link_candidates


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


def _edge_id(project_id: str, edge_type: EdgeType, from_id: str, to_id: str, reason: str) -> str:
    return stable_id("edge", project_id, edge_type.value, from_id, to_id, reason)


def _build_edge(
    project_id: str,
    edge_type: EdgeType,
    from_atom: EvidenceAtom,
    to_atom: EvidenceAtom,
    reason: str,
    confidence: float,
    *,
    metadata: dict[str, Any] | None = None,
) -> EvidenceEdge:
    return EvidenceEdge(
        id=_edge_id(project_id, edge_type, from_atom.id, to_atom.id, reason),
        project_id=project_id,
        from_atom_id=from_atom.id,
        to_atom_id=to_atom.id,
        edge_type=edge_type,
        reason=reason,
        confidence=confidence,
        metadata=dict(metadata or {}),
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

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
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
                    )
                )

            # contradicts: same comparable scope + quantity differs (not mere co-site line items).
            if a.atom_type.value == "quantity" and b.atom_type.value == "quantity":
                if quantity_a is None or quantity_b is None or quantity_a == quantity_b:
                    continue
                sites_a = _site_keys(a)
                sites_b = _site_keys(b)
                if sites_a and sites_b and sites_a != sites_b:
                    continue
                if (
                    a.authority_class == AuthorityClass.approved_site_roster
                    and b.authority_class == AuthorityClass.approved_site_roster
                ):
                    ida, idb = _canonical_material_key(a), _canonical_material_key(b)
                    if ida is None or idb is None or ida != idb:
                        continue
                if not _quantity_pair_comparable_scope(a, b):
                    continue
                push(
                    _build_edge(
                        project_id,
                        EdgeType.contradicts,
                        a,
                        b,
                        f"Quantity mismatch {quantity_a:g} vs {quantity_b:g} for shared entity context",
                        0.9,
                    )
                )

    # excludes: exclusion atom mentions entity key in another atom.
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
        ex_keys = set(ex.entity_keys)
        for target in ordered:
            if target.id == ex.id:
                continue
            if ex_keys.intersection(set(target.entity_keys)):
                push(
                    _build_edge(
                        project_id,
                        EdgeType.excludes,
                        ex,
                        target,
                        "Exclusion atom applies to target entity context",
                        0.9,
                    )
                )

    # requires: constraint shares site with scope/quantity atoms.
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
        for target in ordered:
            if target.id == constraint.id or target.atom_type.value not in {"scope_item", "quantity"}:
                continue
            if sites.intersection(_site_keys(target)):
                push(
                    _build_edge(
                        project_id,
                        EdgeType.requires,
                        constraint,
                        target,
                        "Constraint requires adherence for same site context",
                        0.86,
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
        reason = (
            f"Aggregate scoped quantity {int(approved_total) if approved_total.is_integer() else approved_total:g} "
            f"does not match vendor quantity {int(vendor_total) if vendor_total.is_integer() else vendor_total:g} "
            f"for {device_key}"
        )
        push(_build_edge(project_id, EdgeType.contradicts, from_atom, to_atom, reason, 0.95))

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
            )
        )

    edges.sort(key=lambda e: e.id)
    return edges


def build_entity_edges(atoms: list[EvidenceAtom]):
    """Compatibility wrapper for older call sites."""
    return build_edges(project_id="unknown_project", atoms=atoms, entities=[])
