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
    r"vulnerability|incident|playbook|runbook|workflow|sla|contract|renewal|"
    # Vendor product names that contain a positive site word
    # ("center", "campus", …) and would otherwise survive on the
    # positive token alone. Keeps "security center", "command center",
    # etc. from being miscategorized as physical sites.
    r"security center|command center|operations center|"
    r"synergis|streamvault|omnicast|palo alto networks|"
    r"axis communications|cisco systems"
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
    kept, _dropped = _filter_with_audit(atom, keys, blob=_atom_text_blob(atom))
    return kept


def filter_entity_keys_with_audit(
    atom: Any, keys: Iterable[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Same as :func:`filter_entity_keys_for_atom` but also returns
    one audit record per dropped candidate so the compiler can write
    them to ``hygiene_audit.jsonl``. Each record carries:

      {
        "atom_id": str,
        "dropped_site_candidate": str,
        "reason": "negative_term_match" | "no_positive_evidence",
        "negative_terms": [str, ...],
        "positive_terms": [str, ...],
        "source_atom_id": str,
      }
    """
    return _filter_with_audit(atom, keys, blob=_atom_text_blob(atom))


# Structured enterprise site-code shape — when the SITE KEY ITSELF
# matches this pattern (region-function-N / region-NN / store-N /
# bldg-N), it's intrinsically a site identifier and the candidate's
# structure IS the positive evidence. Bypass the negative-blob test
# so site IDs aren't dropped just because the surrounding atom text
# also happens to mention "contract" / "license" / a vendor name.
_STRUCTURED_SITE_KEY_RE = re.compile(
    r"^(?:"
    r"[a-z]{2,5}_[a-z0-9]{1,8}(?:_[a-z0-9]{1,6}){0,3}"  # atl_hq_01, nyc_dc_12
    r"|s\d{2,4}|site_?\d{1,4}"                          # s001, site_12
    r"|store_?\d{1,4}|loc_?\d{1,4}"                     # store_142
    r"|bldg_?[a-z0-9]{1,4}|b\d{1,4}"                    # bldg_12
    r"|mdc_?\d{1,4}|idc_?\d{1,4}|dc\d{1,4}"             # mdc_01
    r")$",
    re.IGNORECASE,
)


def _filter_with_audit(
    atom: Any, keys: Iterable[str], *, blob: str
) -> tuple[list[str], list[dict[str, Any]]]:
    kept: list[str] = []
    dropped: list[dict[str, Any]] = []
    atom_id = getattr(atom, "id", None)

    for key in keys:
        if not isinstance(key, str):
            continue
        if not key.startswith("site:"):
            kept.append(key)
            continue

        slug = key[len("site:"):]
        candidate = slug.replace("_", " ")

        # Structured site IDs (atl_hq_01, store_142, bldg_a2, ...) ARE
        # the positive evidence. Bypass the negative-blob test so a
        # canonical site code isn't dropped just because the same
        # sentence mentions "contract" / "license" / a vendor name.
        if _STRUCTURED_SITE_KEY_RE.match(slug):
            kept.append(key)
            continue

        cand_neg = bool(_SITE_NEGATIVE_RE.search(candidate))
        cand_pos = bool(_SITE_POSITIVE_RE.search(candidate))

        if cand_neg and not cand_pos:
            dropped.append(
                {
                    "atom_id": atom_id,
                    "dropped_site_candidate": key,
                    "reason": "candidate_name_negative_match",
                    "negative_terms": _matched(_SITE_NEGATIVE_RE, candidate),
                    "positive_terms": [],
                    "source_atom_id": atom_id,
                }
            )
            continue

        test_blob = f"{candidate} {blob}"
        if _SITE_NEGATIVE_RE.search(test_blob) and not _SITE_POSITIVE_RE.search(
            test_blob
        ):
            dropped.append(
                {
                    "atom_id": atom_id,
                    "dropped_site_candidate": key,
                    "reason": "evidence_blob_negative_dominated",
                    "negative_terms": _matched(_SITE_NEGATIVE_RE, test_blob),
                    "positive_terms": [],
                    "source_atom_id": atom_id,
                }
            )
            continue

        kept.append(key)

    return sorted(set(kept)), dropped


def _matched(pattern: re.Pattern[str], text: str) -> list[str]:
    seen: list[str] = []
    for m in pattern.finditer(text):
        v = m.group(0).strip()
        if v and v not in seen:
            seen.append(v)
        if len(seen) >= 6:
            break
    return seen


__all__ = ["filter_entity_keys_for_atom", "filter_entity_keys_with_audit"]
