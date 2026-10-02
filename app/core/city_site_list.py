"""A list of "City, ST" lines is a list of sites.

Live 000132: a HubSpot note ends

    Locations
    Delphos, OH
    Hudson, WI
    ...
    Wilmington, DE

Every site reader in the parsers wants a street before it will call a line a
place (``find_us_addresses_in_text`` needs a house number or a ZIP), so the six
cities came out as untyped prose -- or, once a copy of the same lines elsewhere
won a dedup, as nothing at all in the note. The deal's job sites were in the
file and nobody could label them as sites.

Shape only, no vocabulary: two or more consecutive lines that each split
cleanly into a city and a US state code (``split_city_state_strict``, which
abstains on anything ambiguous) are a site list. One such line alone is not --
"Thanks, Al" style coincidences need company to look like a list. The short
line directly above the run, when there is one, is the list's label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.core.address_parse import split_city_state_strict

_BULLET_RE = re.compile(r"^\s*(?:[-*•–·]|\d{1,3}[.)])\s+")

#: A run must be at least this long to be a list rather than a coincidence.
MIN_RUN = 2


@dataclass(frozen=True)
class CitySite:
    line_index: int  # 0-based, into the lines given
    text: str  # the line as written, bullet stripped
    city: str
    state: str
    label: str  # the list's heading ("Locations"), or ""

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", f"{self.city}_{self.state}".lower()).strip("_")

    @property
    def entity_key(self) -> str:
        return f"site:{self.slug}"


def _unquote(line: str) -> str:
    """A quoted mail line (">> Dallas, TX") without its quote marks."""
    return re.sub(r"^\s*(?:>\s?)+", "", str(line or ""))


def _city_state(line: str) -> tuple[str, str] | None:
    s = _BULLET_RE.sub("", _unquote(line)).strip().rstrip(";.")
    if not s or len(s) > 60 or not re.search(r"[,/]", s):
        return None
    city, state = split_city_state_strict(s)
    if not city or not state:
        return None
    return city, state


def find_city_site_lists(lines: Iterable[str]) -> list[CitySite]:
    """Every line of every "City, ST" run, in file order."""
    seq = [str(x) for x in lines]
    out: list[CitySite] = []
    i = 0
    while i < len(seq):
        if _city_state(seq[i]) is None:
            i += 1
            continue
        j = i
        run: list[tuple[int, str, str]] = []
        while j < len(seq) and (cs := _city_state(seq[j])) is not None:
            run.append((j, cs[0], cs[1]))
            j += 1
        if len(run) >= MIN_RUN:
            label = ""
            k = i - 1
            if k >= 0 and _unquote(seq[k]).strip():
                head = _unquote(seq[k]).strip()
                if len(head.split()) <= 5 and len(head) <= 60:
                    label = head.rstrip(":").strip()
            for idx, city, state in run:
                out.append(CitySite(idx, _BULLET_RE.sub("", _unquote(seq[idx])).strip(), city, state, label))
        i = j
    return out


#: "Supported Locations: Delphos, OH, Hudson, WI" -- a short lead, a colon,
#: then the list on the same line.
_INLINE_LEAD_RE = re.compile(r"^\s*([^:\n]{1,60}?)\s*:\s*(.+?)\s*$", re.S)


def _is_lead(head: str) -> bool:
    head = head.strip()
    return bool(head) and len(head) <= 60 and len(head.split()) <= 5 and _city_state(head) is None


def split_city_list_paragraph(text: str) -> tuple[str, list[CitySite]] | None:
    """A paragraph that is a lead and a list of "City, ST" places.

    Two shapes, both from a SOW (live 000132 v1)::

        Supported Locations: Delphos, OH, Hudson, WI, ..., Wilmington, DE

        Supported Locations:
        Delphos, OH
        Hudson, WI

    Returns the lead as written (colon kept) and one ``CitySite`` per place,
    each labelled with the lead, or None when the paragraph is anything else:
    every token after the lead must pair into a city and a state, and there
    must be at least ``MIN_RUN`` of them.
    """
    raw = str(text or "")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 1:
        head, rest = lines[0], lines[1:]
        if not head.endswith(":") or not _is_lead(head.rstrip(":")) or len(rest) < MIN_RUN:
            return None
        found = find_city_site_lists(rest)
        if len(found) != len(rest):
            return None
        label = head.rstrip(":").strip()
        return head, [CitySite(i, cs.text, cs.city, cs.state, label) for i, cs in enumerate(found)]
    m = _INLINE_LEAD_RE.match(" ".join(raw.split()))
    if not m or not _is_lead(m.group(1)):
        return None
    label, body = m.group(1).strip(), m.group(2).rstrip(";.").strip()
    toks = [t.strip() for t in re.split(r"\s*[,;]\s*", body)]
    if toks and toks[-1].lower().startswith("and "):
        toks[-1] = toks[-1][4:].strip()
    if len(toks) < 2 * MIN_RUN or len(toks) % 2:
        return None
    sites: list[CitySite] = []
    for i in range(0, len(toks), 2):
        city_tok = re.sub(r"^and\s+", "", toks[i], flags=re.I)
        written = f"{city_tok}, {toks[i + 1]}"
        cs = _city_state(written)
        if cs is None:
            return None
        sites.append(CitySite(i // 2, written, cs[0], cs[1], label))
    return f"{label}:", sites


def city_site_value(site: CitySite, **extra) -> dict:
    """The ``physical_site`` value for one list line."""
    name = f"{site.city}, {site.state}"
    val = {
        "kind": "physical_site",
        "id": site.slug,
        "site_id": site.slug,
        "name": name,
        "names": list(dict.fromkeys([name, site.city])),
        "aliases": [],
        "city": site.city,
        "state": site.state,
        "city_state": name,
        "street_address": "",
        "address": "",
        "zip": "",
        "from_city_list": True,
    }
    if site.label:
        val["list_label"] = site.label
    val.update(extra)
    return val


__all__ = ["CitySite", "find_city_site_lists", "split_city_list_paragraph", "city_site_value", "MIN_RUN"]
