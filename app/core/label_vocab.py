"""Closed vocabularies for entity keys and ``scope_category`` -- ``app/core/label_vocab.json``.

Both fields were open: any ``prefix:value`` was a valid entity key, and any
string a valid SOW group. On 000132 the entity prefixes grew to 40-odd kinds
with near-duplicates (``qty``/``quantity``, ``money``/``amount``,
``party``/``org``), so the claims head would learn spellings rather than keys.
Labeling rule 50 settles one spelling per thing; the JSON records it.

``entity_tag(key)`` and ``scope_category(value)`` say, for one stored value:

* ``ok``       -- on the list as written;
* ``alias``    -- an old spelling with exactly one canonical spelling, which
                  ``canonical`` holds (``billing:time_and_materials`` ->
                  ``billing:t_and_m``, ``org:*`` -> ``party:*``);
* ``off_list`` -- not on the list and no one-to-one rename. The value is kept
                  as written and counted; ``proposal`` names the likely
                  canonical form when one is known, for a person to decide.

``label`` is a name for the count that never carries an open value (a person's
or a customer's name): an open prefix is counted as ``prefix:*``, and only a
closed list's slug is spelled out.

ml/c3/vocab.py reads the same JSON for the C3 model (that package never
imports ``app``); tests/test_label_vocab.py holds the two to the same answers.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

VOCAB_PATH = Path(__file__).with_name("label_vocab.json")

OK, ALIAS, OFF_LIST = "ok", "alias", "off_list"

#: A value safe to print in a count: a short slug. Anything else (free text a
#: person typed) is counted under a placeholder, so no customer text leaks.
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.\-]{0,47}$")


@dataclass(frozen=True)
class Check:
    value: str        # as stored
    canonical: str    # what training reads (== value unless status is alias)
    status: str       # ok | alias | off_list
    label: str        # the count's name (no open values)
    proposal: str = ""  # off_list only: a likely canonical form, for review


@lru_cache(maxsize=1)
def load_vocab() -> dict[str, Any]:
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _patterns() -> tuple[tuple[re.Pattern[str], str, str], ...]:
    return tuple((re.compile(p["match"]), p["to"], p["name"])
                 for p in load_vocab()["entity_tags"].get("pattern_aliases") or [])


def safe(value: str, placeholder: str = "<free text>") -> str:
    return value if _SLUG.match(value) else placeholder


def entity_prefixes() -> tuple[str, ...]:
    return tuple(load_vocab()["entity_tags"]["prefixes"])


def closed_values(prefix: str) -> tuple[str, ...] | None:
    spec = load_vocab()["entity_tags"]["prefixes"].get(prefix) or {}
    return tuple(spec["values"]) if spec.get("values") else None


def scope_categories() -> tuple[str, ...]:
    return tuple(load_vocab()["scope_category"]["values"])


def canonical_key(key: str) -> str:
    """The one-to-one canonical spelling of an entity key, else the key as is."""
    tags = load_vocab()["entity_tags"]
    k = str(key).strip()
    if k in tags["key_aliases"]:
        return tags["key_aliases"][k]
    for rx, to, _ in _patterns():
        if rx.match(k):
            return rx.sub(to, k)
    prefix, sep, rest = k.partition(":")
    if sep and prefix in tags["prefix_aliases"]:
        return canonical_key(f"{tags['prefix_aliases'][prefix]}:{rest}")
    return k


def entity_tag(key: Any) -> Check:
    tags = load_vocab()["entity_tags"]
    k = str(key).strip()
    prefix, sep, rest = k.partition(":")
    if not sep or not prefix:
        return Check(k, k, OFF_LIST, "(no prefix)")
    canon = canonical_key(k)
    if canon != k:
        if k in tags["key_aliases"]:
            label = f"{k} -> {canon}"
        else:
            label = next((name for rx, _, name in _patterns() if rx.match(k)), "")
            if not label:
                label = f"{prefix}:* -> {canon.partition(':')[0]}:*"
        return Check(k, canon, ALIAS, label)
    if prefix not in tags["prefixes"]:
        review = tags.get("review_aliases", {}).get(prefix)
        return Check(k, k, OFF_LIST, f"{safe(prefix, '<odd prefix>')}:*",
                     f"{review}:* (review)" if review else "")
    values = closed_values(prefix)
    if values is not None and rest not in values:
        return Check(k, k, OFF_LIST, f"{prefix}:{safe(rest)}")
    return Check(k, k, OK, f"{prefix}:*")


def canonical_keys(keys: Iterable[Any]) -> list[str]:
    """Entity keys with each one-to-one alias renamed; order kept, repeats folded."""
    out: list[str] = []
    for k in keys:
        c = canonical_key(str(k))
        if c and c not in out:
            out.append(c)
    return out


def _slug(value: str) -> str:
    return re.sub(r"[\s\-/]+", "_", value.strip().lower())


def scope_category(value: Any) -> Check:
    sc = load_vocab()["scope_category"]
    v = str(value if value is not None else "").strip()
    if v in sc["values"]:
        return Check(v, v, OK, v)
    canon = sc.get("aliases", {}).get(v) or sc.get("aliases", {}).get(_slug(v))
    if canon is None and _slug(v) in sc["values"]:
        canon = _slug(v)            # case and spacing only: "Service Model"
    if canon is not None:
        return Check(v, canon, ALIAS, f"{safe(v)} -> {canon}")
    return Check(v, v, OFF_LIST, safe(v))
