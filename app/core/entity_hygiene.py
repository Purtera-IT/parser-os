"""Entity hygiene — stop products / SKUs / frameworks / SaaS tools
from being miscategorized as physical sites.

The corpus review found ``site:*`` keys like ``site:belden_cat6_cmp``,
``site:cisa_vulnerability_playbook``, ``site:servicenow``,
``site:genetec_synergis``, ``site:apc_ups``. These broke
``site_reality`` clustering downstream and surfaced as fake locations
in briefs and review UI.

This module provides a single public function,
:func:`filter_entity_keys_for_atom`, that removes ``site:*`` keys
whose evidence text is dominated by product / vendor / framework /
SaaS / standard vocabulary unless the text *also* contains explicit
physical-place vocabulary (school / building / address / MDF / IDF / …).

Non-``site:*`` keys are passed through unchanged — vendor / device /
part_number / standard keys are useful exactly where this filter is
strict.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


_SITE_POSITIVE_RE = re.compile(
    r"\b("
    r"school|campus|building|bldg|center|centre|hospital|clinic|library|"
    r"auditorium|courthouse|warehouse|office|plant|facility|mdf|idf|"
    r"district core|main campus|annex|tower|floor|suite|room|closet|"
    r"\d{2,6}\s+[a-z0-9 .'-]+\s+(st|street|rd|road|ave|avenue|dr|drive|blvd|way|lane|ln)"
    r")\b",
    re.I,
)
_SITE_NEGATIVE_RE = re.compile(
    r"\b("
    r"cat\s?6|cat\s?6a|belden|panduit|commscope|leviton|"
    r"cisco|meraki|juniper|aruba|palo alto|fortinet|fortigate|"
    r"genetec|axis|hanwha|milestone|lenel|hid|mercury|"
    r"apc|ups|server|switch|router|firewall|camera|reader|license|lic|sku|"
    r"servicenow|pagerduty|logicmonitor|sentinel|cisa|nist|pci|hipaa|nfpa|"
    r"vulnerability|incident|playbook|runbook|workflow|sla|contract|renewal"
    r")\b",
    re.I,
)


def _atom_text_blob(atom: Any) -> str:
    """Concatenate the atom's text-bearing fields for regex evaluation."""
    parts: list[str] = []
    for attr in ("raw_text", "normalized_text"):
        v = getattr(atom, attr, None) or ""
        if isinstance(v, str):
            parts.append(v)
    val = getattr(atom, "value", None)
    if val is not None:
        parts.append(str(val))
    return " ".join(p for p in parts if p)


def filter_entity_keys_for_atom(atom: Any, keys: Iterable[str]) -> list[str]:
    """Return a sorted, de-duplicated list of entity keys with bogus
    ``site:*`` keys removed.

    A ``site:*`` key is kept only if the atom's evidence blob plus the
    site key's surface form contains at least one positive
    physical-place token, AND is not dominated by negative
    product/framework tokens with no positive tokens.
    """
    kept: list[str] = []
    blob = _atom_text_blob(atom)

    for key in keys:
        if not isinstance(key, str):
            continue
        if not key.startswith("site:"):
            kept.append(key)
            continue

        candidate = key.replace("site:", "").replace("_", " ")

        # Drop when the candidate name itself is clearly built from
        # product / SaaS / framework words and lacks any positive
        # site-vocabulary anchor. This is the dominant rule —
        # ``site:belden_cat6_cmp`` is bogus regardless of whether
        # the surrounding atom text happens to mention an MDF/IDF.
        cand_neg = bool(_SITE_NEGATIVE_RE.search(candidate))
        cand_pos = bool(_SITE_POSITIVE_RE.search(candidate))
        if cand_neg and not cand_pos:
            continue

        # Drop when the full evidence blob is negative-dominated and
        # carries no positive anchor anywhere — the parser-side
        # extractor probably pulled the wrong span as a "site".
        test_blob = f"{candidate} {blob}"
        if _SITE_NEGATIVE_RE.search(test_blob) and not _SITE_POSITIVE_RE.search(
            test_blob
        ):
            continue

        # Otherwise keep. We deliberately accept site keys with no
        # explicit positive vocabulary match (e.g. ``site:west_wing``,
        # ``site:annex_b``) because real proper-noun site names rarely
        # contain the literal words "school" / "building" — they ARE
        # the site name.
        kept.append(key)

    return sorted(set(kept))


__all__ = ["filter_entity_keys_for_atom"]
