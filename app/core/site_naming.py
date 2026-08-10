"""Give every site row a name a PM can read.

A site row earns a ``facility_name`` only when a ``physical_site`` atom
anchors it — the roster extractor read it out of a table, so the name came
along with the row. Sites the deal only ever mentions in prose never get
that atom. They still become rows, because the entity emitters correctly
recognise them as sites, but the emitter slugifies the phrase to build the
key and throws the phrase away. What reaches the PM is the identifier::

    site:prudential_center_office        ← what the dossier shows
    "Prudential Center office"           ← what the document said

Both name the same place; only one of them is readable. This module
recovers the second from evidence the compile already holds, in two tiers:

  Tier 1 — the document's own words.
    Scan the atom text for a contiguous run of tokens whose slug is
    exactly the site's slug. ``site:wind_creek_atmore`` is named by the
    sentence that produced it: "Wind Creek Atmore is soliciting
    proposals...". This is a projection of the source, not a guess — the
    phrase is the one the emitter slugified.

  Tier 2 — an identity string the compile already established.
    Sites that arrive through alias collapse have no surface phrase of
    their own: ``site:azs_1`` is what survived merging
    ``site:maricopa_county_iron_mountain_data_centers_azs_1_scottsdale``.
    Look through the names, aliases and document cells for the shortest
    string whose slug contains the site's slug at a token boundary AND
    which one of the row's own aliases corroborates. For AZS-1 that is
    the workbook cell "Iron Mountain Data Centers AZS-1 - Scottsdale".

Never invents a name. A site the documents describe only by city and ZIP
("Atmore, AL 36502") has no facility name to recover, and stays unnamed —
which is correct, and is why tier 2 rejects candidates that merely echo
the key back with the underscores taken out.

Only ever fills blanks. A row that already carries a ``facility_name`` is
not read, not scored, and not touched; a deal where every site is anchored
does no work at all.

Pure functions, no I/O, no LLM.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

# A token is a word, keeping the internal hyphen/apostrophe that
# ``AZS-1`` and ``Levy's`` depend on — the slugifier collapses those to
# ``_`` either way, so matching has to see them as one token to line up
# with the key.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# A token that carries a word, as opposed to a bare number or ordinal.
_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Longest site slug we will look for as a phrase. Beyond this a "site
# name" is a sentence fragment, and scanning every window that long over
# every atom costs more than the name is worth.
_MAX_PHRASE_TOKENS = 8

# Bounds on a string that can serve as a display name. Shorter than this
# is an abbreviation, longer is a paragraph.
_MIN_NAME_LEN = 3
_MAX_NAME_LEN = 120

# How deep to walk a document's structured projection looking for cells.
_MAX_WALK_DEPTH = 12


def _slug(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").lower()).strip("_")


def _casedness(s: str) -> int:
    """How much original casing a string preserves.

    ``Iron Mountain Data Centers AZS-1 - Scottsdale`` and ``iron mountain
    data centers azs 1 scottsdale`` slugify identically; the first is the
    one the document actually printed.
    """
    return sum(1 for ch in s if ch.isupper())


def _atom_text(atom: Any) -> str:
    for field in ("raw_text", "text"):
        v = getattr(atom, field, None)
        if isinstance(v, str) and v:
            return v
    if isinstance(atom, Mapping):
        for field in ("raw_text", "text"):
            v = atom.get(field)
            if isinstance(v, str) and v:
                return v
    return ""


def _atom_value(atom: Any) -> Mapping[str, Any]:
    for field in ("value", "structured"):
        v = getattr(atom, field, None)
        if isinstance(v, Mapping):
            return v
    if isinstance(atom, Mapping):
        for field in ("value", "structured"):
            v = atom.get(field)
            if isinstance(v, Mapping):
                return v
    return {}


def _is_name_shaped(s: Any) -> bool:
    return (
        isinstance(s, str)
        and _MIN_NAME_LEN <= len(s.strip()) <= _MAX_NAME_LEN
        and any(c.isalpha() for c in s)
    )


def _walk_strings(obj: Any, depth: int = 0) -> Iterator[str]:
    """Yield every name-shaped string in a nested projection."""
    if depth > _MAX_WALK_DEPTH:
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            yield from _walk_strings(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v, depth + 1)
    elif _is_name_shaped(obj):
        yield obj.strip()


def _scan_phrases(text: str, wanted: set[str], max_tokens: int) -> Iterator[tuple[str, str]]:
    """Yield ``(slug, phrase)`` for every token run in ``text`` whose slug
    is one of ``wanted``.

    A run must carry at least two worded tokens. Anything less is the key
    spelled differently — "AZS-1" or "HC 238" for ``site:azs_1`` and
    ``site:hc_238`` — and handing that back as the display name would look
    like a fix while telling the PM nothing they could not already read
    off the identifier.
    """
    tokens = list(_TOKEN_RE.finditer(text))
    worded = [bool(_WORD_RE.search(t.group(0))) for t in tokens]
    for i in range(len(tokens)):
        # j is exclusive, and starts at i+2 so the shortest window is two
        # tokens wide.
        for j in range(i + 2, min(len(tokens), i + max_tokens) + 1):
            if sum(worded[i:j]) < 2:
                continue
            phrase = text[tokens[i].start():tokens[j - 1].end()]
            slug = _slug(phrase)
            if slug in wanted:
                yield slug, phrase


def _better(candidate: str, current: str | None) -> bool:
    if current is None:
        return True
    return _casedness(candidate) > _casedness(current)


def _identity_pool(
    atoms: Iterable[Any], documents: Iterable[Mapping[str, Any]] | None
) -> dict[str, str]:
    """slug → the best-cased string the compile has for that slug.

    Draws on what the compile already established about identity: the
    name/address fields and alias lists of every atom, plus the cells of
    each document's structured projection — which is where a workbook's
    own spelling of a site survives when no atom captured it.
    """
    pool: dict[str, str] = {}

    def offer(s: Any) -> None:
        if not _is_name_shaped(s):
            return
        s = s.strip()
        slug = _slug(s)
        if slug and _better(s, pool.get(slug)):
            pool[slug] = s

    for atom in atoms:
        value = _atom_value(atom)
        for field in ("name", "facility_name", "street_address", "address"):
            offer(value.get(field))
        for field in ("names", "aliases", "alternative_names"):
            listed = value.get(field)
            if isinstance(listed, (list, tuple)):
                for item in listed:
                    offer(item)

    for doc in documents or ():
        structured = doc.get("structured") if isinstance(doc, Mapping) else None
        for s in _walk_strings(structured):
            offer(s)

    return pool


def _alias_slugs(row: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for alias in row.get("aliases") or ():
        if not isinstance(alias, str):
            continue
        body = alias[len("site:"):] if alias.startswith("site:") else alias
        slug = _slug(body)
        if slug:
            out.add(slug)
    return out


def _contains_at_boundary(haystack: str, needle: str) -> bool:
    return re.search(rf"(?:^|_){re.escape(needle)}(?:_|$)", haystack) is not None


def recover_site_display_names(
    *,
    sites: Sequence[Mapping[str, Any]],
    atoms: Sequence[Any],
    documents: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Return ``{site_key: display_name}`` for site rows that have no name.

    ``sites`` are ``site_readiness`` rows. Rows that already carry a
    ``facility_name`` are skipped entirely — this only fills blanks, and
    a deal whose sites are all roster-anchored returns ``{}`` without
    reading a single atom.
    """
    targets: dict[str, Mapping[str, Any]] = {}
    for row in sites:
        if not isinstance(row, Mapping) or row.get("facility_name"):
            continue
        key = row.get("site") or row.get("site_key") or ""
        if not isinstance(key, str) or not key.startswith("site:"):
            continue
        slug = key[len("site:"):]
        if slug:
            targets[slug] = row
    if not targets:
        return {}

    resolved: dict[str, str] = {}

    # ── Tier 1: the phrase the emitter slugified, recovered from the
    # text it was slugified from.
    wanted = {s for s in targets if 2 <= len(s.split("_")) <= _MAX_PHRASE_TOKENS}
    if wanted:
        best: dict[str, str] = {}
        for atom in atoms:
            text = _atom_text(atom)
            if not text:
                continue
            for slug, phrase in _scan_phrases(text, wanted, _MAX_PHRASE_TOKENS):
                if _better(phrase, best.get(slug)):
                    best[slug] = phrase
        resolved.update(best)

    # ── Tier 2: for sites that arrived through alias collapse and so have
    # no surface phrase of their own, the established identity string that
    # one of the row's aliases corroborates.
    unresolved = {s: row for s, row in targets.items() if s not in resolved}
    if unresolved:
        pool = _identity_pool(atoms, documents)
        for slug, row in unresolved.items():
            aliases = _alias_slugs(row)
            candidates: list[tuple[int, str]] = []
            for pool_slug, phrase in pool.items():
                if pool_slug == slug:
                    # The key with its underscores taken out. Not a name.
                    continue
                if not _contains_at_boundary(pool_slug, slug):
                    continue
                corroborated = any(
                    pool_slug == a or pool_slug in a or a in pool_slug
                    for a in aliases
                )
                if not corroborated:
                    continue
                candidates.append((len(pool_slug), phrase))
            if candidates:
                # Shortest wins: the longer strings are this site fused
                # with its neighbours ("Maricopa County Iron Mountain Data
                # Centers Azs 1 Scottsdale" names two sites, not one).
                candidates.sort()
                resolved[slug] = candidates[0][1]

    return {f"site:{slug}": name for slug, name in resolved.items()}


__all__ = ["recover_site_display_names"]
