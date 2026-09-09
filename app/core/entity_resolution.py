from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher

from app.core import progress
from app.core.entity_extraction import (
    _coalesce_alias_groups,
    _emit_site_aliases_from_text,
)
from app.core.ids import stable_id
from app.core.entity_extraction import is_site_boilerplate_slug
from app.core.normalizers import normalize_entity_key, normalize_text
from app.core.schemas import EntityRecord, EvidenceAtom, ReviewStatus
from app.domain import get_active_domain_pack
from app.domain.schemas import DomainPack

try:
    from rapidfuzz import fuzz, process as _rf_process
except Exception:  # pragma: no cover - fallback path
    fuzz = None
    _rf_process = None


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
        if info["entity_type"] == "site":
            slug = canonical_key.split(":", 1)[-1]
            if is_site_boilerplate_slug(slug):
                continue
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

    # v44: CROSS-DOC SEMANTIC DEDUP at entity-resolution layer.
    # The tournament canonicalization runs WITHIN a single retrieval
    # pass — it doesn't see entities created by the regex emitters or
    # by different LLM passes that resolved to slightly-different
    # canonical_keys. This final pass embeds the canonical_name of
    # every entity within each entity_type, finds same-type pairs
    # with cosine_sim > 0.88, and asks the LLM "same canonical?
    # if yes, which is better". Union-find merges the entity records
    # that pass the test.
    import os as _os
    if not _os.environ.get("SOWSMITH_FINAL_DEDUP_DISABLE") and len(records) >= 4:
        try:
            from app.core.rag_extras import run_tournament
            from app.core.embedding_retrieval import (
                embed_texts as _embed_texts,
                embedding_endpoint_reachable,
            )
            from app.core.multi_entity_llm import (
                _call_ollama, _parse_json_object,
            )
            if embedding_endpoint_reachable():
                # Group by entity_type for type-scoped dedup
                by_type: dict[str, list[int]] = {}
                for idx, r in enumerate(records):
                    by_type.setdefault(r.entity_type, []).append(idx)
                merged_indices: set[int] = set()
                new_canon_for: dict[int, str] = {}
                for et, indices in by_type.items():
                    if et in {"phone", "email", "money", "date",
                              "milestone", "address", "part_number",
                              "quarter", "qa"}:
                        # Exact-match types skip semantic dedup
                        continue
                    if len(indices) < 3:
                        continue
                    canon_names = [records[i].canonical_name for i in indices]
                    vecs = _embed_texts(canon_names)
                    if vecs.size == 0:
                        continue
                    # Wrap each record into the shape run_tournament expects
                    items = [
                        {"text": records[i].canonical_name, "_idx": i}
                        for i in indices
                    ]
                    deduped = run_tournament(
                        items, vecs,
                        entity_type=et,
                        canonical_key="text",
                        llm_call=lambda p, mt: _call_ollama(p, max_tokens=mt),
                        parse_json=_parse_json_object,
                        sim_threshold=0.88,
                        max_pairs=40,
                    )
                    # Items that were absorbed into a merge get
                    # _merged_from > 1; we drop them and update the
                    # canonical record with the new canon
                    for d in deduped:
                        merged_from = d.get("_merged_from", 0)
                        if merged_from and merged_from > 1:
                            kept_idx = d.get("_idx")
                            if kept_idx is not None:
                                new_canon_for[kept_idx] = d.get("text", "")
                # Strip merged-away records
                if new_canon_for:
                    # In this v44 first cut we just update canonical_name
                    # without restructuring entity ids. Future v45 can do
                    # full merge with id rewrite.
                    for idx, new_name in new_canon_for.items():
                        if new_name and len(new_name) >= 3:
                            records[idx] = records[idx].model_copy(
                                update={"canonical_name": new_name}
                            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Final cross-doc semantic dedup failed: %s", e,
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


def collapse_duplicate_atoms(atoms: list) -> list:
    """v48 — collapse near-duplicate atoms emitted by repeated doc sections.

    Two atoms are duplicates when:
      1. normalized_text / raw_text is IDENTICAL (PLIR repeat), OR
      2. raw_text SequenceMatcher similarity > 0.92 AND same artifact_id.

    Keep the higher-confidence copy. Intra-doc only — cross-doc repetition
    is intentional evidence corroboration.
    """
    if not atoms:
        return atoms
    by_artifact: dict[str, list] = {}
    other: list = []
    for atom in atoms:
        aid = getattr(atom, "artifact_id", None)
        if aid:
            by_artifact.setdefault(aid, []).append(atom)
        else:
            other.append(atom)
    result: list = list(other)

    def _atype(atom) -> str:
        t = getattr(atom, "atom_type", None)
        return t.value if hasattr(t, "value") else str(t or "")

    # v57: the two catch-all/untyped families are treated as ONE
    # equivalence class for exact-text dedup. The parsers' page-level
    # recall passes routinely emit the SAME sentence as both a scope_item
    # and an entity — different type GUESSES of one source span, not two
    # facts. Before the type-aware classifier runs they are pure
    # duplicates, so collapse identical-text copies across these generic
    # types and keep the highest-confidence one. Structured types
    # (raw_table_row, bom_line, physical_site, ...) are NOT folded — they
    # legitimately co-exist as distinct facets of the same row. Nor is
    # customer_instruction folded — it is a meaningful classification
    # that downstream stages (and the transcript gate) assert on.
    _GENERIC_TYPES = {"scope_item", "entity"}

    def _dedup_type(atom) -> str:
        at = _atype(atom)
        return "_generic" if at in _GENERIC_TYPES else at

    for aid, art_atoms in by_artifact.items():
        # v50.1: dedupe key INCLUDES atom_type so a raw_table_row and
        # a bom_line sourced from the SAME table row both survive —
        # they represent different facets of the data (raw vs typed
        # classification). Same goes for vendor_line_item vs bom_line.
        seen_normalized: dict[tuple, object] = {}
        unique: list = []
        for atom in sorted(art_atoms, key=lambda a: getattr(a, "confidence", 0.0), reverse=True):
            norm = getattr(atom, "normalized_text", None) or getattr(atom, "raw_text", "") or ""
            if not norm:
                unique.append(atom)
                continue
            norm_key = (_dedup_type(atom), norm.strip().lower())
            if norm_key not in seen_normalized:
                seen_normalized[norm_key] = atom
                unique.append(atom)
        # Second pass: fuzzy dedup on long prose only (≥50 chars).
        # Structured rows (physical_site, BOM, site_allocation, tasks, etc.)
        # have semantic keys and should not pay the O(n²) SequenceMatcher
        # cost. APS Attachment B has 100+ long but distinct physical_site
        # rows; comparing every row to every row can dominate compile time
        # and is the wrong dedup layer anyway.
        fuzzy_dedup_types = {
            "scope_item", "constraint", "assumption", "exclusion",
            "decision", "action_item", "open_question", "customer_instruction",
        }
        final: list = []
        # Bucket by atom type plus the first few normalized tokens so even
        # prose comparisons stay local; exact duplicates were already removed
        # above. Buckets hold only the truncated rep STRINGS (not atoms) so the
        # near-dup check is a single C-vectorized rapidfuzz call per atom.
        #
        # v58 robustness (#75): the old path ran pure-Python difflib in an
        # O(n²) all-pairs loop. When boilerplate makes thousands of *distinct*
        # prose atoms share the same 8-token bucket key (e.g. a 39k-atom
        # spreadsheet deal where every scope row starts "contractor shall
        # provide and install ..."), that loop hit millions of ~1ms comparisons
        # and burned hours. Two universal fixes, neither alters behaviour on
        # normal/small buckets:
        #   1. rapidfuzz.fuzz.ratio is mathematically identical to difflib's
        #      SequenceMatcher.ratio but C-accelerated (~100x), and
        #      process.extractOne short-circuits the "matches any rep?" scan in
        #      one call via score_cutoff.
        #   2. a per-bucket representative cap bounds the absolute worst case to
        #      O(n · cap): once a bucket holds `cap` distinct reps we stop
        #      growing it (exact-dedup already ran above, so the marginal
        #      near-dup beyond `cap` distinct prose items is negligible).
        max_reps = max(1, int(os.environ.get("SOWSMITH_FUZZY_DEDUP_MAX_REPS", "400")))
        fuzzy_buckets: dict[tuple[str, str], list[str]] = {}
        for atom in progress.track(
            unique, desc=f"dedup {str(aid)[:8]}", total=len(unique), min_total=500
        ):
            atype = _atype(atom)
            rt = getattr(atom, "raw_text", "") or ""
            if len(rt) < 50 or atype not in fuzzy_dedup_types:
                final.append(atom)
                continue
            norm = (getattr(atom, "normalized_text", None) or rt).strip().lower()
            bucket_key = (atype, " ".join(norm.split()[:8]))
            reps = fuzzy_buckets.setdefault(bucket_key, [])
            rt500 = rt[:500]
            is_dup = False
            if reps:
                if _rf_process is not None:
                    # One C call over all reps; returns None if none clear 92.
                    is_dup = (
                        _rf_process.extractOne(
                            rt500, reps, scorer=fuzz.ratio, score_cutoff=92.0
                        )
                        is not None
                    )
                else:  # pragma: no cover - difflib fallback when rapidfuzz absent
                    for ext in reps:
                        if SequenceMatcher(None, rt500, ext).ratio() > 0.92:
                            is_dup = True
                            break
            if not is_dup:
                final.append(atom)
                if len(reps) < max_reps:
                    reps.append(rt500)
        result.extend(final)
    return result


def complete_truncated_site_values(site_objects: list[dict]) -> list[dict]:
    """v48 supplemental — cross-doc prefix completion for truncated PDF
    site attribute values.

    Doc 08 (PDF) clips at column boundaries: 'ATL-WEST-02' → 'ATL-WEST-0'.
    Doc 02 (DOCX) has the complete value. When one is a tight prefix of
    another (≤4 chars longer), promote the truncated entry to the full value.

    Only applies to id/address/mdf_idf attributes. Prefix must be ≥5 chars.
    """
    if not site_objects:
        return site_objects
    attrs_to_check = ("id", "address", "mdf_idf")
    for attr in attrs_to_check:
        all_values: list[tuple[int, str]] = [
            (i, s[attr]) for i, s in enumerate(site_objects)
            if s.get(attr) and len(s[attr]) >= 5
        ]
        all_values.sort(key=lambda x: len(x[1]), reverse=True)
        for long_idx, long_val in all_values:
            for short_idx, short_val in all_values:
                if short_idx == long_idx:
                    continue
                if len(short_val) >= len(long_val):
                    continue
                if len(short_val) < 5:
                    continue
                if long_val.startswith(short_val) and len(long_val) - len(short_val) <= 4:
                    site_objects[short_idx][attr] = long_val
                    if attr == "id":
                        names = site_objects[short_idx].get("names", [])
                        if short_val in names:
                            names[names.index(short_val)] = long_val
                        site_objects[short_idx]["names"] = names
    return site_objects


from app.core.site_duplicate_candidates import (
    SAME_SITE,
    SITE_PAIR_CANDIDATES,
    SITE_PAIR_RELATION,
    pair_exemplar,
)


def _atom_type_of(atom: Any) -> str:
    """An atom's type, whether it is an object or a serialised dict."""
    value = getattr(atom, "atom_type", None)
    if value is None and isinstance(atom, dict):
        value = atom.get("atom_type")
    return value.value if hasattr(value, "value") else str(value or "")


def _normalize_for_evidence(text: Any) -> str:
    """Casing- and punctuation-blind form, for asking "is this string in the
    documents at all". Mirrors the normalisation base-health uses to judge a
    display name against its own source text."""
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _deal_evidence_text(atoms: Any) -> str:
    """Every atom's text, normalised and joined. Built once per compile."""
    parts: list[str] = []
    for atom in atoms or []:
        text = getattr(atom, "text", None)
        if text is None and isinstance(atom, dict):
            text = atom.get("text")
        if text:
            parts.append(_normalize_for_evidence(text))
    return " || ".join(parts)


def _name_appears_in_evidence(name: str, evidence: str) -> bool:
    """Is this name anywhere in the deal's own evidence?

    Degrades OPEN: with no evidence text collected we cannot prove a name was
    invented, and refusing every alias on that basis would be worse than the
    problem. Absence of proof is not proof of absence.
    """
    if not evidence:
        return True
    probe = _normalize_for_evidence(name)
    if not probe:
        return False
    return probe in evidence


def collect_site_alias_groups(atoms: list[EvidenceAtom]) -> list[frozenset[str]]:
    """Scan every atom's raw_text for site-alias co-mention patterns
    and return the union of all discovered alias groups.

    Each group is a set of canonical site keys (``site:atl_hq``,
    ``site:atlanta_headquarters``, ...) that the text asserts refer
    to the same physical place. The groups feed into
    :func:`fuse_alias_groups` so the entity_resolution stage collapses
    them into one canonical EntityRecord per physical site.

    Two passes:
      1. Co-mention patterns from raw text ("X (also known as Y)").
      2. Key-shape inference: ``site:atl_hq`` and ``site:atl_hq_01``
         are obvious aliases of the same physical site — the longer
         form just adds a row-number suffix. Same for ``atl_west``
         and ``atl_west_02``.
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

    # Pass 2: key-shape inference.
    # Collect every site key actually emitted across atoms, then
    # group keys whose slugs share a prefix and differ only by a
    # numeric suffix.
    all_site_keys: set[str] = set()
    for atom in atoms:
        for k in (atom.entity_keys or []):
            if k.startswith("site:"):
                all_site_keys.add(k)
    # Group by the prefix-without-trailing-digits
    import re as _re
    prefix_map: dict[str, set[str]] = {}
    for k in all_site_keys:
        slug = k[len("site:"):]
        # Strip trailing `_NN` (1-3 digits) suffix if present
        m = _re.match(r"^(.+?)_(\d{1,3})$", slug)
        prefix = m.group(1) if m else slug
        prefix_map.setdefault(prefix, set()).add(k)
    for prefix, group in prefix_map.items():
        if len(group) >= 2:
            # Also include the bare-prefix form (no suffix) when it
            # exists as a separate site key.
            bare = f"site:{prefix}"
            if bare in all_site_keys:
                group.add(bare)
            all_groups.append(group)

    # Separators are not identity.
    #
    # A customer spells their own name both ways inside one deal — live 010302
    # writes "SymphonyAI" three times and "Symphony AI" twice — and the
    # extractor slugs whichever spelling it saw. One office becomes:
    #
    #     site:symphonyai_hillview_office
    #     site:symphony_ai_hillview_office
    #
    # Nothing joined them, so the deal carried two sites, whichever key won a
    # given compile became the row (which read as a site "renaming itself"
    # between runs), and no lesson taught about one could ever match the other.
    #
    # Two keys whose letters and digits are identical in order are the same
    # words with the separators moved. That is a spelling difference, not two
    # places — "atl_hq_01" and "atlhq01" cannot be different sites.
    letters_map: dict[str, set[str]] = {}
    for k in all_site_keys:
        bare_letters = _re.sub(r"[^a-z0-9]+", "", k[len("site:"):].lower())
        if bare_letters:
            letters_map.setdefault(bare_letters, set()).add(k)
    for group in letters_map.values():
        if len(group) >= 2:
            all_groups.append(set(group))

    # ─── LLM SITE-CLUSTER FUSION (v35) ───
    # Pick up the LLM's site_clusters output from the session cache
    # (stashed by extract_all_entities_with_llm during enrich_atoms).
    # Each cluster is {canonical_name, aliases[]} — convert to a
    # frozenset of site:<slug> keys so fuse_alias_groups collapses
    # the surface forms into one canonical EntityRecord.
    #
    # Real-world impact: OPTBOT goes from 13 site entities (one per
    # surface form: code + friendly name + 3 addresses × 4 sites)
    # to 5 canonical entities (one per physical place). The LLM
    # already knows which surface forms are the same place; this
    # just plumbs that knowledge into the fusion stage.
    try:
        from app.core.multi_entity_llm import get_session_site_clusters
        llm_clusters = get_session_site_clusters(atoms)
    except Exception:
        llm_clusters = []
    if llm_clusters:
        # An alias has to be IN the documents.
        #
        # The boilerplate check below is a word list, and a word list only
        # knows the words somebody already wrote down. The universal test is
        # provenance: a name the deal's own evidence never contains was
        # invented by the model, whatever it looks like. Measured on a
        # 140-envelope sample (2026-09-07), 84 of 242 alias names attached to
        # site atoms appear NOWHERE in their deal — "rpdu 230 sites",
        # "backup modem 284 sites" — 35% of the alias fabrications.
        #
        # Built once per deal; a name absent here cannot be a surface form of
        # anything, so it can neither be an alias nor merge two sites.
        evidence = _deal_evidence_text(atoms)
        for cluster in llm_clusters:
            aliases = cluster.get("aliases") or []
            site_keys = set()
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    if not _name_appears_in_evidence(alias, evidence):
                        continue
                    slug = _re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")
                    # The LLM's cluster aliases went in unchecked, which is how
                    # a pricing unit became an alias of a real office.
                    if slug and not is_site_boilerplate_slug(slug):
                        site_keys.add(f"site:{slug}")
            canon = cluster.get("canonical_name")
            if isinstance(canon, str) and canon.strip():
                slug = _re.sub(r"[^a-z0-9]+", "_", canon.lower()).strip("_")
                if slug and not is_site_boilerplate_slug(slug):
                    site_keys.add(f"site:{slug}")
            if len(site_keys) >= 2:
                all_groups.append(site_keys)

    # ─── SEMANTIC SITE FUSION (decide() chokepoint) ───
    # Everything above merges site keys by *string* shape only: shared slug,
    # numeric-suffix, or an explicit co-mention/LLM-cluster alias. None of it
    # can see that a site CODE ("atl_air_03") and its FRIENDLY NAME
    # ("atlanta_air_office") are the same physical place — different slugs,
    # different tokens, no shared substring. semantic_site_fusion_groups routes
    # that judgment through decide() (STORE kNN → LLM → UNDECIDED), so a PM's
    # learned correction merges them and every structurally-similar pair after.
    # Off by default (no store wired → decide() returns fallback → zero merges →
    # byte-identical to the deterministic pipeline above).
    universe: set[str] = set(all_site_keys)
    for g in all_groups:
        universe |= {k for k in g if isinstance(k, str) and k.startswith("site:")}
    # Hand the fusion pass the rows, not just the keys. Without them it
    # compares "palo alto ca 94304" against "symphonyai hillview office" and
    # cannot see that 3300 Hillview Ave is the first one's address — the single
    # fact the judgment turns on.
    site_rows: dict[str, dict[str, Any]] = {}
    for atom in atoms or []:
        if _atom_type_of(atom) != "physical_site":
            continue
        value = getattr(atom, "value", None)
        if value is None and isinstance(atom, dict):
            value = atom.get("value") or atom.get("structured")
        if not isinstance(value, dict):
            continue
        for key in (getattr(atom, "entity_keys", None) or (atom.get("entity_keys") if isinstance(atom, dict) else None) or []):
            if isinstance(key, str) and key.startswith("site:") and key not in site_rows:
                site_rows[key] = {
                    "site": key,
                    "facility_name": value.get("facility_name") or value.get("name"),
                    "street_address": value.get("street_address") or value.get("address"),
                    "city": value.get("city"),
                    "state": value.get("state"),
                    # The postcode is what makes a street address identifying.
                    # Without it these rows could not be compared on address at
                    # all, so the same building published two and three times.
                    "zip": value.get("zip") or value.get("postal_code"),
                }
    # Deterministic first, and NOT behind the neural flag: two rows carrying the
    # same street and the same postcode are the same building, which is what an
    # address means rather than something to be judged. Running it here also
    # spares the learned pass from being asked about pairs that are not in
    # question.
    all_groups.extend(address_identity_groups(universe, site_rows))
    all_groups.extend(semantic_site_fusion_groups(universe, site_rows))

    # ─── HYGIENE PASS ON ALIAS GROUPS ───
    # Drop any site:* key that fails hygiene before grouping is
    # finalized. Otherwise the proper-noun regex span scan in
    # _emit_site_aliases_from_text can introduce junk keys like
    # "site:pre_bid_meeting_location" into a group of real sites,
    # and _pick_canonical may then promote the junk key to be the
    # canonical of the merged entity.
    # NEURAL GHOST-REJECTION GATE (decide() store) — runs BEFORE the lexical
    # denylist so a PM-taught, presentation-aware role correction OVERRIDES the
    # hand-curated deal-specific list. Off by default → empty drop set. This is
    # the universal replacement that lets _is_obvious_non_site (the deal-specific
    # "cheating") be retired via verify-gate once coverage is proven live.
    role_drops: set[str] = set()
    try:
        universe_keys: set[str] = set()
        for group in all_groups:
            universe_keys |= {k for k in group if isinstance(k, str)}
        role_drops = semantic_site_role_drops(universe_keys)
    except Exception:  # pragma: no cover - gate must never break resolution
        role_drops = set()

    try:
        from app.core.site_llm_verify import _is_obvious_non_site
    except Exception:
        _is_obvious_non_site = None  # type: ignore
    if _is_obvious_non_site is not None or role_drops:
        cleaned: list[set[str]] = []
        for group in all_groups:
            kept = set()
            for k in group:
                if k.startswith("site:"):
                    # Neural gate first: a confident PM-taught reject drops it.
                    if k in role_drops:
                        continue
                    phrase = k[len("site:"):].replace("_", " ")
                    if _is_obvious_non_site is not None and _is_obvious_non_site(phrase):
                        continue
                kept.add(k)
            if len(kept) >= 2:
                cleaned.append(kept)
        all_groups = cleaned

    return _coalesce_alias_groups(all_groups)


#: Above this many sites, stop enumerating every pair and ask only about the
#: shortlist. 30 sites is 435 pairs, which is a sweep worth doing; 135 sites is
#: 9,045 and a 437-site rollout is 95,266, which is not.
_EXHAUSTIVE_PAIR_CEILING = 30


def _address_identity(row: dict[str, Any]) -> str:
    """``street|postcode`` for a row, or "" when it is not identifying.

    Street ALONE is not identity -- "Main St" exists in every town. A postcode
    alone is not either; it covers many buildings. Together they are, which is
    why a postal address is written that way.

    The city is deliberately not part of this. It is the field most often wrong
    or missing (deal 02557291 published one row whose city read "Gregg St", a
    fragment of its own street), and it carries no information the postcode
    does not already carry.
    """
    from app.core.address_parse import _street_for_dedup

    street = _street_for_dedup(
        str(row.get("street_address") or row.get("address") or "")
    )
    code = re.sub(r"\s+", "", str(row.get("zip") or row.get("postal_code") or "")).upper()
    if not street or not code:
        return ""
    return f"{street}|{code}"


def address_identity_groups(
    site_keys: set[str],
    rows_by_key: dict[str, dict[str, Any]] | None = None,
) -> list[set[str]]:
    """Group site keys that carry the SAME street address and postcode.

    Deal 02557291 published 2205 Gregg St three times -- once from the note's
    roster (``site:2205_gregg_st``), once from the entity backfill
    (``site:loc_2205_gregg_st``, whose city read "Gregg St"), and once as
    ``site:site_1``. Three extractors, three keys, one building. The learned
    fusion pass asked about all six pairs and returned ``fallback:None`` for
    every one, because nothing had been taught about them and the LLM tier is
    budgeted off -- so a duplicate this obvious survived to the roster.

    It should never have been a question. Two rows with the same street and the
    same postcode are the same place; that is what an address IS. Asking a
    model to confirm it is both slower and less reliable than reading it.

    Unlike the learned pass this is not gated on ``SOWSMITH_NEURAL_SITE_FUSION``
    -- there is nothing neural about it -- and it abstains completely when
    either row lacks a street or a postcode, which is the common case for a
    site known only by name.
    """
    keys = sorted(k for k in site_keys if isinstance(k, str) and k.startswith("site:"))
    if len(keys) < 2:
        return []
    by_address: dict[str, set[str]] = {}
    for key in keys:
        row = (rows_by_key or {}).get(key)
        if not isinstance(row, dict):
            continue
        ident = _address_identity(row)
        if ident:
            by_address.setdefault(ident, set()).add(key)
    groups = [g for g in by_address.values() if len(g) >= 2]
    if groups:
        logging.getLogger(__name__).info(
            "address_identity: %d site key(s) share %d address(es) -> merged %s",
            sum(len(g) for g in groups), len(groups), [sorted(g) for g in groups],
        )
    return groups


def semantic_site_fusion_groups(
    site_keys: set[str],
    rows_by_key: dict[str, dict[str, Any]] | None = None,
) -> list[set[str]]:
    """Merge physical-site keys that name the same place but slug-equality misses.

    The deterministic passes in :func:`collect_site_alias_groups` merge two
    site keys only when their slugs are identical (modulo a trailing ``_NN``
    suffix) or an explicit alias/LLM-cluster ties them. They are blind to the
    most common real-world dupe: a site **code** and its **friendly name** for
    the same physical location — ``site:atl_air_03`` vs
    ``site:atlanta_air_office``. Different slugs, different tokens, no shared
    substring, so every string heuristic leaves them split.

    This pass asks the question the *right* way — through the decide()
    chokepoint, once per ambiguous pair::

        STORE (a PM-taught kNN correction)  →  LLM (reads both names)  →  UNDECIDED

    A PM teaches the merge **as text** ("atlanta air office IS atl-air-03");
    the feedback store embeds it and every structurally-similar code/friendly
    pair — in this deal and the next — deflects from the LLM. We never add a
    regex or keyword list: an unrecognized pair the store and LLM can't resolve
    stays **separate** (undecided), so we never invent a merge.

    Returns extra alias groups (each a set of ≥2 ``site:`` keys judged the same
    place) to fold into the fusion stage. Safe no-op when:

      * ``SOWSMITH_NEURAL_SITE_FUSION`` is unset/0 (default), or
      * no feedback store is wired (decide() returns fallback for every pair →
        zero merges → byte-identical to the deterministic pipeline).
    """
    if os.environ.get("SOWSMITH_NEURAL_SITE_FUSION", "") in ("", "0", "false"):
        return []
    keys = sorted(k for k in site_keys if isinstance(k, str) and k.startswith("site:"))
    if len(keys) < 2:
        return []

    try:
        from app.core.decide import DecisionScope, decide
    except Exception:  # pragma: no cover - decide must always import
        return []

    scope = DecisionScope()

    def phrase(k: str) -> str:
        return k[len("site:"):].replace("_", " ").strip()

    def describe(k: str) -> dict[str, Any]:
        """The row this key names, or a stand-in built from the key itself."""
        row = (rows_by_key or {}).get(k)
        return dict(row) if isinstance(row, dict) else {"site": k, "facility_name": phrase(k)}

    # Enumerate the candidate pairs and their decision texts once.
    #
    # Every pair is O(n^2) with no upper bound on n: a 135-site deal is 9,045
    # pairs and a 437-site rollout is 95,266. Past a ceiling, ask only about
    # the pairs a PERSON would be asked about — an unlocated site and the one
    # located site whose address carries a word from its name. A handful per
    # deal instead of tens of thousands, and the same shortlist the panel
    # shows, so the pass and the question agree on what is even a candidate.
    #
    # Below the ceiling the exhaustive sweep stays, because it catches the pair
    # the shortlist cannot: a site CODE and its FRIENDLY NAME ("atl_air_03" /
    # "atlanta_air_office") share no token with any address.
    pairs: list[tuple[str, str, str]] = []  # (key_a, key_b, pair_text)
    if len(keys) > _EXHAUSTIVE_PAIR_CEILING:
        from app.core.site_duplicate_candidates import site_duplicate_candidates

        known = set(keys)
        for cand in site_duplicate_candidates([describe(k) for k in keys]):
            a, b = str(cand.get("unlocated") or ""), str(cand.get("located") or "")
            if a in known and b in known:
                pairs.append((a, b, str(cand.get("exemplar") or "")))
    else:
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if not (phrase(keys[i]) and phrase(keys[j])):
                    continue
                # ONE exemplar builder, shared with the shortlist a PM answers.
                # This was slug-only, which hid the address the judgment turns
                # on AND differed from the string the answer was taught on — so
                # an answer could never be retrieved here.
                pairs.append((
                    keys[i], keys[j],
                    pair_exemplar(describe(keys[i]), describe(keys[j])),
                ))
    if not pairs:
        return []

    # BATCH / "parallel": warm the shared embedding cache with every pair text
    # in ONE round-trip before the loop. The store's per-pair embed inside
    # decide() then resolves from cache instead of N sequential embed calls.
    # We deliberately do NOT fan out concurrent LLM calls — the single remote
    # Ollama host serializes and thrashes under concurrency; the real speedup
    # is batched embeddings + a warm store driving LLM escalations toward zero.
    try:
        from app.core.embedding_retrieval import embed_texts
        embed_texts([t for _, _, t in pairs])
    except Exception:  # pragma: no cover - cache warming is best-effort
        pass

    # Safety valve: cap LLM escalations per compile so a pathological deal with
    # dozens of sites can't fan out into hundreds of cold LLM calls. Past the
    # cap, undecided pairs simply stay separate + flagged (the safe default) —
    # never a guessed merge.
    try:
        llm_budget = int(os.environ.get("SOWSMITH_SITE_FUSION_LLM_BUDGET", "80"))
    except ValueError:
        llm_budget = 80

    verdicts: dict[str, int] = {}
    # Union-find over the candidate site keys; each confident "same" verdict
    # unions two keys, so a code that matches several surface forms collapses
    # them all into one cluster transitively.
    parent: dict[str, str] = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, pair_text in pairs:
        if find(a) == find(b):
            continue  # already merged transitively — skip the round-trip
        pa, pb = phrase(a), phrase(b)
        d = decide(
            relation=SITE_PAIR_RELATION,
            text=pair_text,
            candidates=list(SITE_PAIR_CANDIDATES),
            instruction=(
                "Two site names from one deal, separated by '||'. Decide if "
                "they name the SAME physical location (e.g. a site code and "
                "its friendly office name, or a name and its address) or two "
                "DISTINCT locations."
            ),
            context=f"Deal site names under comparison: {pa!r} and {pb!r}.",
            scope=scope,
            # Let a confident STORE hit decide for free; only spend an LLM call
            # while we still have budget. Out of budget → store-only (llm=False),
            # so unknown pairs stay distinct rather than guessed or expensive.
            llm=llm_budget > 0,
        )
        if d.source == "llm":
            llm_budget -= 1
        verdicts[f"{d.source}:{d.verdict}"] = verdicts.get(f"{d.source}:{d.verdict}", 0) + 1
        if d.verdict == SAME_SITE:
            union(a, b)

    groups: dict[str, set[str]] = {}
    for k in keys:
        groups.setdefault(find(k), set()).add(k)
    merged = [g for g in groups.values() if len(g) >= 2]

    # Say what was asked and what came back.
    #
    # A merge that does not happen is indistinguishable, from outside, from a
    # pass that never ran — which is how six separate fixes were shipped
    # against a symptom nobody could see the cause of. One line closes that:
    # pairs asked, what decide() said, and whether anything actually merged.
    logging.getLogger(__name__).info(
        "site_fusion: %d site keys, %d pairs asked, verdicts=%s, merged %d group(s) %s",
        len(keys), len(pairs), verdicts or "{}", len(merged),
        [sorted(g) for g in merged] or "",
    )
    return merged


def semantic_site_role_drops(site_keys: set[str]) -> set[str]:
    """Universal ghost-rejection gate for ``site:`` keys via the decide() store.

    The deterministic hygiene below uses ``_is_obvious_non_site`` — a ~300-entry
    hand-curated, DEAL-SPECIFIC denylist (literal place names, code prefixes).
    It is the "cheating" the store is meant to retire: it never generalizes to
    the next deal's phrasing and a PM can't correct it.

    This pass asks the same question the *right* way — through decide()
    (STORE kNN → LLM → UNDECIDED) on the **role** of each surface form:

        canonical_site   — a real physical job site (KEEP)
        site_attribute   — an attribute OF a site: an escort/owner contact, an
                           MDF/IDF closet, an access window (DROP — not a site)
        not_a_site       — boilerplate / schedule / non-site noise (DROP)

    A PM teaches the role **presentationally** ("a column of work hours is a
    schedule, not a site"; "an MDF/IDF label is equipment inside a site") from a
    few example VALUES — not from specific site names — so the store generalizes
    across deals with zero shared tokens (proven: 0% false kills, ghosts
    dropped). It is a deliberately **one-sided** gate:

        store-confident verdict in {site_attribute, not_a_site}  →  DROP ghost
        canonical_site  OR  store abstains (verdict None)         →  KEEP (safe)

    STORE-ONLY by design (``llm=False``): the LLM tier is **never** allowed to
    manufacture a drop here. A site key reaching this stage is often a bare CODE
    slug ("atl 047 04") stripped of its facility name; the calibrated store
    correctly ABSTAINS on those (no taught canonical is a bare code), so they
    survive. The LLM, handed a context-free code, would *guess* a reject verdict
    and false-kill a real site — measured 3/5 false kills with the LLM on, 0/5
    store-only. So only the calibrated kNN/head, which abstains under
    uncertainty, may drop. An unknown form the store can't confidently reject
    **survives** — we never invent a drop. Returns ``site:`` keys to drop.

    Safe no-op when:
      * ``SOWSMITH_NEURAL_SITE_ROLE_GATE`` is unset/0 (default), or
      * no feedback store is wired (store abstains for every key → empty drop
        set → byte-identical to the deterministic pipeline).
    """
    # ON by default — the learned gate is the primary site ghost-rejection path
    # (the denylist is trimmed to an irreducible residue behind it). Explicitly
    # set the flag to "0"/"false" to disable. Safe-degrading regardless: with no
    # store wired or the embedder unreachable, every decide() abstains → empty
    # drop set → byte-identical to not running the gate.
    if os.environ.get("SOWSMITH_NEURAL_SITE_ROLE_GATE", "1").strip().lower() in ("0", "false", "no", "off"):
        return set()
    keys = sorted(k for k in site_keys if isinstance(k, str) and k.startswith("site:"))
    if not keys:
        return set()

    try:
        from app.core.decide import DecisionScope, decide
    except Exception:  # pragma: no cover - decide must always import
        return set()
    try:
        from app.core.site_role_seed import (
            CONCEPT_CANDIDATES,
            ROLE_CANDIDATES,
            ROLE_RELATION,
            _CONCEPT_DROP_VERDICT,
            _ROLE_DROP_VERDICTS,
            concept_relations,
            is_address_like,
            looks_like_parse_fragment,
        )
    except Exception:  # pragma: no cover - seed module must always import
        return set()

    scope = DecisionScope()

    def phrase(k: str) -> str:
        return k[len("site:"):].replace("_", " ").strip()

    texts = [(k, phrase(k)) for k in keys]
    texts = [(k, p) for k, p in texts if p]
    if not texts:
        return set()

    # A street address / ZIP IS a real site — the fusion gate DEDUPS it, it is
    # never a non-site. Exclude it from the role/concept reject pass entirely so
    # no learned gate can ever drop a real address.
    texts = [(k, p) for k, p in texts if not is_address_like(p)]
    if not texts:
        return set()

    # BATCH: warm the shared embedding cache with every phrase in ONE round-trip
    # so the per-key resolve() inside decide() hits cache, not N embed calls.
    try:
        from app.core.embedding_retrieval import embed_texts
        embed_texts([p for _, p in texts])
    except Exception:  # pragma: no cover - cache warming is best-effort
        pass

    concept_rels = concept_relations()
    drops: set[str] = set()
    for k, p in texts:
        # (0) Structural parse-garbage: a lone truncated token ("philad",
        # "barcelon"). No name list — keys on SHAPE. Conservative (lone short
        # tokens only) so it can never eat a real multi-word facility name.
        if looks_like_parse_fragment(p):
            drops.add(k)
            continue

        # (1) Aggressive 3-way ROLE head — the calibrated workhorse. One-sided:
        # only a CONFIDENT store reject verdict drops; canonical_site OR an
        # abstention (verdict None) both KEEP.
        d = decide(
            relation=ROLE_RELATION,
            text=p,
            candidates=list(ROLE_CANDIDATES),
            instruction=(
                "A single value pulled from a site-roster row. Decide its ROLE: "
                "a real physical job site (canonical_site); an attribute OF a "
                "site such as an escort/owner contact, an MDF/IDF network closet, "
                "or an access/work window (site_attribute); or non-site "
                "boilerplate (not_a_site)."
            ),
            context="Deciding whether this roster value is itself a site.",
            scope=scope,
            # STORE-ONLY: never let the LLM guess a drop on a context-free site
            # code. Only the calibrated store, which abstains under uncertainty,
            # may reject — keeping false kills at zero.
            llm=False,
        )
        if d.verdict in _ROLE_DROP_VERDICTS:
            drops.add(k)
            continue

        # (2) UNION the tight per-concept binary gates. Each is its own kNN
        # (reject vs real_site) with a calibrated head; a drop fires if ANY one
        # confidently rejects. max-sim is additive across concepts, so coverage
        # broadens without any single head collapsing.
        for rel in concept_rels:
            cd = decide(
                relation=rel,
                text=p,
                candidates=list(CONCEPT_CANDIDATES),
                instruction="Decide whether this roster value is a physical site.",
                context="Deciding whether this roster value is itself a site.",
                scope=scope,
                llm=False,
            )
            if cd.verdict == _CONCEPT_DROP_VERDICT:
                drops.add(k)
                break
    return drops


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
    # Site preference: prefer the most-specific form (longest slug
    # ending in a numeric suffix wins, then more tokens, then
    # alphabetical). Picks ``atl_hq_01`` as canonical over
    # ``atl_hq``, and ``atl_west_02`` over ``atl_west`` /
    # ``atl_west_0`` (the truncated PDF-wrap form).
    if all(k.startswith("site:") for k in members):
        import re as _re
        def site_key(key: str) -> tuple[int, int, int, str]:
            slug = key.split(":", 1)[1]
            m = _re.match(r"^.+_(\d{1,3})$", slug)
            # Has numeric suffix? Length of that suffix as proxy for
            # specificity (atl_hq_01 has 2-digit, atl_west_0 has
            # 1-digit). Prefer 2+ digit complete suffixes over
            # 1-digit truncated ones.
            has_suffix = 1 if m else 0
            suffix_len = len(m.group(1)) if m else 0
            tokens = [t for t in slug.split("_") if t]
            return (
                -has_suffix,        # has-suffix wins
                -suffix_len,        # 2-digit wins over 1-digit truncated
                -len(tokens),       # more tokens wins
                key,                # alphabetical
            )
        return sorted(members, key=site_key)[0]
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
    "part",  # xlsx_parser / quote_parser emit under "part:", not "part_number:"
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
