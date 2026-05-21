from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.core.entity_extraction import (
    _coalesce_alias_groups,
    _emit_site_aliases_from_text,
)
from app.core.ids import stable_id
from app.core.normalizers import normalize_entity_key, normalize_text
from app.core.schemas import EntityRecord, EvidenceAtom, ReviewStatus
from app.domain import get_active_domain_pack
from app.domain.schemas import DomainPack

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback path
    fuzz = None


def _fuzzy_score(a: str, b: str) -> float:
    if fuzz is not None:
        return float(fuzz.ratio(a, b))
    return SequenceMatcher(a=a, b=b).ratio() * 100.0


def _build_alias_index(pack: DomainPack) -> dict[str, dict[str, str]]:
    """Build ``{entity_type: {normalized_alias: canonical_value}}`` from a
    domain pack so any alias spotted in an atom resolves to the same
    canonical key.

    Order of precedence:
    * ``device_aliases`` -> entity_type ``device``
    * ``entity_types[].aliases`` -> entity_type as named
    """
    out: dict[str, dict[str, str]] = {"device": {}}
    for canonical, aliases in (pack.device_aliases or {}).items():
        slot = out.setdefault("device", {})
        slot[canonical.lower()] = canonical
        for alias in aliases:
            slot[normalize_text(alias)] = canonical
    for entity in pack.entity_types or []:
        slot = out.setdefault(entity.name, {})
        slot[entity.name.lower()] = entity.name
        for alias in entity.aliases:
            slot[normalize_text(alias)] = entity.name
        for example in entity.examples:
            slot[normalize_text(example)] = example
    return out


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _canonical_alias_key(
    entity_type: str,
    raw_value: str,
    alias_index: dict[str, dict[str, str]],
) -> str:
    """Return the canonical entity key for ``raw_value`` using the
    domain-pack alias table; falls back to the regular normalizer.

    A4 fix: the substring-match path used to collapse specific
    instances like ``branch_42`` / ``building_13`` / ``site_7``
    into the generic entity type ``site`` because the pack's
    ``entity_types['site'].aliases`` contains ``branch`` /
    ``building`` / ``site``. We now skip the substring fallback
    when the raw_value contains a digit — that's the signal that
    the raw_value is a specific instance rather than a bare
    generic word the pack wants to route. Exact-match remains
    enabled so a bare ``branch`` still routes to ``site``.
    """
    table = alias_index.get(entity_type) or {}
    needle = normalize_text(raw_value)
    if needle and needle in table:
        return f"{entity_type}:{_slugify(table[needle])}"
    # A4 fix: the substring fallback used to over-route specific
    # instances like ``branch_42`` / ``building_c`` / ``warehouse_rf``
    # into the generic entity type ``site`` because the pack's
    # site-alias table contains ``branch`` / ``building`` /
    # ``warehouse`` as routing-only entries. Guard with two
    # constraints so specificity is preserved:
    #   1. Skip for entity_type == "site" entirely. Sites are
    #      canonicalized by ``_emit_sites`` so the substring
    #      fallback only adds noise.
    #   2. For other entity types, only apply when the needle is
    #      short enough that the alias represents most of it
    #      (length ratio ≥ 0.6). Avoids "rugged_logistics_tablet"
    #      collapsing into a single device alias.
    if needle and entity_type != "site":
        for alias, canonical in table.items():
            if not alias:
                continue
            if alias in needle and len(alias) / max(1, len(needle)) >= 0.6:
                return f"{entity_type}:{_slugify(canonical)}"
    return normalize_entity_key(entity_type, raw_value)


def extract_entity_records(
    project_id: str,
    atoms: list[EvidenceAtom],
    *,
    pack: DomainPack | None = None,
) -> list[EntityRecord]:
    pack = pack or get_active_domain_pack()
    alias_index = _build_alias_index(pack)
    grouped: dict[str, dict] = {}
    for atom in atoms:
        for key in atom.entity_keys:
            if ":" not in key:
                continue
            entity_type, raw_value = key.split(":", 1)
            canonical_key = _canonical_alias_key(entity_type, raw_value, alias_index)
            # ``normalize_entity_key`` returns "" for generic pseudo-values
            # (e.g. site:"ALL" / site:"N/A" / site:"Various"). Skipping
            # here prevents an IndexError below at
            # ``canonical_key.split(":", 1)[1]`` and also prevents the
            # noisy "ALL → unnamed site" record from leaking into outputs.
            if not canonical_key or ":" not in canonical_key:
                continue
            if canonical_key not in grouped:
                grouped[canonical_key] = {
                    "entity_type": entity_type,
                    "aliases": set(),
                    "source_atom_ids": set(),
                }
            grouped[canonical_key]["aliases"].add(key)
            grouped[canonical_key]["source_atom_ids"].add(atom.id)

    records: list[EntityRecord] = []
    for canonical_key in sorted(grouped):
        info = grouped[canonical_key]
        records.append(
            EntityRecord(
                id=stable_id("ent", project_id, canonical_key),
                project_id=project_id,
                entity_type=info["entity_type"],
                canonical_key=canonical_key,
                canonical_name=canonical_key.split(":", 1)[1].replace("_", " "),
                aliases=sorted(info["aliases"]),
                source_atom_ids=sorted(info["source_atom_ids"]),
                confidence=1.0,
                review_status=ReviewStatus.auto_accepted,
            )
        )
    return records


def collect_stakeholder_alias_groups(atoms: list[EvidenceAtom]) -> list[frozenset[str]]:
    """D3: collapse multiple surface forms of the same person.

    Real MSP deals reference one person multiple ways:
      * Full name:        ``stakeholder:renee_watkins``
      * Surname only:     ``stakeholder:watkins`` (parser-os emits
                          these when an honorific like ``Ms. Watkins``
                          appears)
      * Initial + surname ``stakeholder:r_watkins`` (parser-os emits
                          these when ``R. Watkins`` appears — see the
                          companion change in ``_emit_stakeholders``)

    Detection is purely key-shape based (no text scanning needed,
    no false positives across documents):

      For every ``stakeholder:*`` canonical key in the project,
      classify it as either FULL (≥2 tokens, at least one with 2+
      lowercase letters) or THIN (1 token, OR 2 tokens where the
      first is a single letter or single letter + dot — i.e. an
      initial). A THIN key fuses into a FULL key when the surname
      matches AND the project contains *exactly one* FULL key
      with that surname. If two people share the same surname
      (``Watkins`` could be Renee or Bob), the THIN key is
      ambiguous and stays unfused.

    Returns alias groups in the same format as
    ``collect_site_alias_groups`` so ``fuse_alias_groups`` can
    consume both without changes.
    """
    if not atoms:
        return []
    all_stakeholder_keys: set[str] = set()
    for atom in atoms:
        for k in atom.entity_keys:
            if isinstance(k, str) and k.startswith("stakeholder:"):
                all_stakeholder_keys.add(k)
    if len(all_stakeholder_keys) < 2:
        return []

    def classify(key: str) -> tuple[str, str]:
        """Return (shape, surname). shape ∈ {full, thin}. surname is
        the last token (slug) used for matching.
        """
        slug = key.split(":", 1)[1]
        tokens = [t for t in slug.split("_") if t]
        if not tokens:
            return "", ""
        # Initials-prefixed form: ``r_watkins`` or ``r_dot_watkins``
        # (the latter shouldn't appear after slugify but be defensive).
        if len(tokens) == 2 and len(tokens[0]) == 1:
            return "thin", tokens[-1]
        # Single-token form: ``watkins`` only.
        if len(tokens) == 1:
            return "thin", tokens[0]
        # Otherwise it's a full name (≥2 multi-letter tokens).
        return "full", tokens[-1]

    surname_to_full_keys: dict[str, set[str]] = {}
    thin_keys_by_surname: dict[str, set[str]] = {}
    for key in all_stakeholder_keys:
        shape, surname = classify(key)
        if not surname:
            continue
        if shape == "full":
            surname_to_full_keys.setdefault(surname, set()).add(key)
        elif shape == "thin":
            thin_keys_by_surname.setdefault(surname, set()).add(key)

    groups: list[set[str]] = []
    for surname, thin_keys in thin_keys_by_surname.items():
        full_matches = surname_to_full_keys.get(surname, set())
        # Ambiguous: 0 full matches (nothing to fuse into) or ≥2
        # (could be either person). Skip — keep the thin key
        # standalone so a downstream reviewer sees it.
        if len(full_matches) != 1:
            continue
        canonical = next(iter(full_matches))
        group = {canonical}
        group.update(thin_keys)
        groups.append(group)
    return _coalesce_alias_groups(groups)


def collect_site_alias_groups(atoms: list[EvidenceAtom]) -> list[frozenset[str]]:
    """Scan every atom's raw_text for site-alias co-mention patterns
    and return the union of all discovered alias groups.

    Each group is a set of canonical site keys (``site:atl_hq``,
    ``site:atlanta_headquarters``, ...) that the text asserts refer
    to the same physical place. The groups feed into
    :func:`fuse_alias_groups` so the entity_resolution stage collapses
    them into one canonical EntityRecord per physical site.
    """
    if not atoms:
        return []
    all_groups: list[set[str]] = []
    for atom in atoms:
        text = atom.raw_text or ""
        if not text:
            continue
        for group in _emit_site_aliases_from_text(text):
            all_groups.append(set(group))
    return _coalesce_alias_groups(all_groups)


def _pick_canonical(group: frozenset[str]) -> str:
    """Choose a canonical key for an alias group.

    For ``stakeholder:*`` groups we prefer the *most specific* key
    (most tokens, then alphabetically first) so that a fused group
    of ``{r_watkins, renee_watkins, watkins}`` picks
    ``renee_watkins`` as canonical instead of the alphabetically
    earlier ``r_watkins``. For all other entity types (sites,
    devices, etc.) we keep the alphabetical preference so existing
    site fusion behavior is unchanged.
    """
    if not group:
        return ""
    # Stakeholder preference: prefer keys whose first token is a
    # real word (≥2 chars) over initials, then prefer more tokens,
    # then alphabetical. Picks ``renee_watkins`` over both
    # ``r_watkins`` (initial first token) and ``watkins`` (single
    # token).
    members = sorted(group)
    if all(k.startswith("stakeholder:") for k in members):
        def stake_key(key: str) -> tuple[int, int, str]:
            slug = key.split(":", 1)[1]
            tokens = [t for t in slug.split("_") if t]
            first_is_initial = (len(tokens) >= 1 and len(tokens[0]) == 1)
            return (
                1 if first_is_initial else 0,  # 0 wins (full name first)
                -(len(tokens)),                 # more tokens wins
                key,                            # alphabetical tie-break
            )
        return sorted(members, key=stake_key)[0]
    return members[0]


def fuse_alias_groups(
    records: list[EntityRecord],
    alias_groups: list[frozenset[str]],
) -> list[EntityRecord]:
    """Collapse EntityRecords whose canonical_keys appear in the same
    alias group into a single record per group.

    Canonical-key choice: for sites and other entities, the
    alphabetically-first key in the group (deterministic, stable).
    For stakeholders, the most-specific key (most tokens, then
    alphabetical) so ``{r_watkins, renee_watkins, watkins}`` picks
    ``renee_watkins`` as canonical, not ``r_watkins``.

    All other keys in the group are folded into the merged
    record's ``aliases`` list. Source atoms and confidence are
    merged the same way ``resolve_aliases`` does for fuzzy-matched
    records.

    Records whose keys aren't in any group pass through unchanged.
    """
    if not records or not alias_groups:
        return records
    # Map each member key → canonical key for its group.
    canonical_map: dict[str, str] = {}
    for group in alias_groups:
        if not group:
            continue
        canonical = _pick_canonical(group)
        for key in group:
            canonical_map[key] = canonical
    # Group records by their (mapped) canonical key.
    by_canonical: dict[str, list[EntityRecord]] = {}
    for r in records:
        canon = canonical_map.get(r.canonical_key, r.canonical_key)
        by_canonical.setdefault(canon, []).append(r)
    out: list[EntityRecord] = []
    for canon, recs in by_canonical.items():
        if len(recs) == 1 and recs[0].canonical_key == canon:
            out.append(recs[0])
            continue
        # Prefer the record whose canonical_key == canon as the primary
        # (so its EntityRecord.id stays stable); otherwise take the first.
        primary = next((r for r in recs if r.canonical_key == canon), recs[0])
        merged = primary.model_copy(deep=True)
        merged.canonical_key = canon
        merged.canonical_name = canon.split(":", 1)[1].replace("_", " ")
        all_aliases: set[str] = set()
        all_atoms: set[str] = set()
        min_confidence = 1.0
        worst_status = ReviewStatus.auto_accepted
        for r in recs:
            all_aliases.update(r.aliases)
            all_aliases.add(r.canonical_key)
            all_atoms.update(r.source_atom_ids)
            min_confidence = min(min_confidence, r.confidence)
            if r.review_status == ReviewStatus.needs_review:
                worst_status = ReviewStatus.needs_review
        merged.aliases = sorted(all_aliases)
        merged.source_atom_ids = sorted(all_atoms)
        # Co-mention fusion is lower-confidence than fuzzy-string
        # matching: assert 0.85 unless any constituent was already
        # lower, in which case keep the minimum.
        merged.confidence = min(min_confidence, 0.85) if len(recs) > 1 else min_confidence
        merged.review_status = worst_status
        out.append(merged)
    out.sort(key=lambda r: (r.entity_type, r.canonical_key))
    return out


# Entity types whose canonical_name is a number / structured value.
# Fuzzy string-similarity makes no sense here (`100000` vs `10000`
# scores high but means very different things; `2026-04-30` vs
# `2026-04-03` differ by one digit). These types only merge when the
# canonical_key matches EXACTLY.
_EXACT_MATCH_ENTITY_TYPES: frozenset[str] = frozenset({
    "money", "quantity", "date", "milestone", "quarter", "part_number",
    "phone", "email", "zip", "address",
})


def resolve_aliases(records: list[EntityRecord]) -> list[EntityRecord]:
    if not records:
        return []

    ordered = sorted(records, key=lambda r: (r.entity_type, r.canonical_key, r.id))
    consumed: set[str] = set()
    resolved: list[EntityRecord] = []

    for record in ordered:
        if record.id in consumed:
            continue
        merged = record.model_copy(deep=True)
        consumed.add(record.id)
        exact_only = merged.entity_type in _EXACT_MATCH_ENTITY_TYPES

        for other in ordered:
            if other.id in consumed or other.entity_type != merged.entity_type:
                continue
            if other.canonical_key == merged.canonical_key:
                score = 100.0
            elif exact_only:
                # Skip fuzzy merging for numeric / structured entity
                # types — "100000" vs "10000" must NOT collapse.
                continue
            else:
                score = _fuzzy_score(merged.canonical_name, other.canonical_name)

            if score >= 92:
                merged.aliases = sorted(set(merged.aliases) | set(other.aliases) | {other.canonical_key})
                merged.source_atom_ids = sorted(set(merged.source_atom_ids) | set(other.source_atom_ids))
                consumed.add(other.id)
            elif 82 <= score < 92:
                merged.aliases = sorted(set(merged.aliases) | set(other.aliases) | {other.canonical_key})
                merged.source_atom_ids = sorted(set(merged.source_atom_ids) | set(other.source_atom_ids))
                merged.review_status = ReviewStatus.needs_review
                merged.confidence = min(merged.confidence, 0.82)
                consumed.add(other.id)

        resolved.append(merged)

    resolved.sort(key=lambda r: (r.entity_type, r.canonical_key))
    return resolved
