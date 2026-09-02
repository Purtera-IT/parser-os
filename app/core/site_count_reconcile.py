"""Does the number of sites we RESOLVED match the number the deal SAYS it has?

An as-of envelope can be internally consistent and still wrong in the one way
that matters: the documents state a count, and the site layer resolves a
different one, and nothing compares them.

Measured on deal 010215 (2026-09-02). The emails say "10" nine separate times --
"We need to have 10 timeclocks installed", "SOW's for each of the ten
locations", "$305 per site x 10 sites = $3,050" -- and a quantity entity of 10
was extracted and kept. The site layer resolved SIX addresses, four of which
were two addresses fused together ("601 gurley street 1205 south main street"),
one truncated ("1123 Sandy Bluff Rd" -> "123 sandy bluff road"), and three sites
missing entirely.

Both facts sat in the same envelope. Nothing asked the question.

That is the failure this module exists to make impossible: not to fix the count,
but to refuse to let a contradiction pass silently. A stated count that does not
match a resolved count is exactly the kind of thing a PM must be told about
rather than left to discover from a technician standing at the wrong address.
"""

from __future__ import annotations

import re
from typing import Any

# "10 sites", "10 locations", "ten timeclocks", "10 separate locations"
# The lookarounds are the whole safety of this pattern. Without them
# "$3,050 for the site" reads as "50 sites" -- a dollar amount becoming a site
# count, which is a confident wrong answer rather than a missing one. A digit,
# comma, period or currency symbol on either side means the number is part of a
# larger figure and is not a count of anything.
_COUNT_NEAR_SITE = re.compile(
    r"(?<![\d.,$])"
    r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
    r"(?![\d.,])"
    r"(?:\s+\w+){0,3}?\s+"
    r"(sites?|locations?|timeclocks?|time\s+clocks?|schools?|buildings?|stores?)\b",
    re.I,
)

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def _as_int(token: str) -> int | None:
    t = str(token).strip().lower()
    if t.isdigit():
        n = int(t)
        # A "count" in the hundreds is a part number or a dollar figure, not sites.
        return n if 1 <= n <= 200 else None
    return _WORDS.get(t)


def stated_site_counts(atoms: list[Any]) -> list[tuple[int, str]]:
    """Every explicit site count the documents assert, with the sentence saying it.

    Returns (count, evidence) pairs — the evidence is the point. A bare number a
    PM cannot trace back to a sentence is a claim, not a finding.
    """
    out: list[tuple[int, str]] = []
    for a in atoms or []:
        text = str(getattr(a, "raw_text", None) or (a.get("raw_text") if isinstance(a, dict) else "") or "")
        if not text:
            continue
        for m in _COUNT_NEAR_SITE.finditer(text):
            n = _as_int(m.group(1))
            if n is not None:
                out.append((n, text.strip()[:240]))
    return out


def reconcile_site_count(atoms: list[Any], resolved_sites: int) -> dict[str, Any]:
    """Compare what the deal says against what we resolved.

    `agrees` is deliberately three-valued via `stated`: None means the documents
    never stated a count, which is not the same as agreeing. Collapsing those two
    would reproduce the silence this exists to break.
    """
    stated = stated_site_counts(atoms)
    if not stated:
        return {
            "stated": None,
            "resolved": int(resolved_sites),
            "agrees": None,
            "reason": "no explicit site count found in the documents",
            "evidence": [],
        }

    # The most-repeated assertion wins; a number said nine times outranks one
    # said once, and ties break toward the larger claim so we under-promise
    # coverage rather than over-promise it.
    counts: dict[int, int] = {}
    ev: dict[int, str] = {}
    for n, text in stated:
        counts[n] = counts.get(n, 0) + 1
        ev.setdefault(n, text)
    best = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]

    agrees = best == int(resolved_sites)
    return {
        "stated": best,
        "stated_mentions": counts[best],
        "resolved": int(resolved_sites),
        "agrees": agrees,
        "reason": (
            "resolved site count matches what the documents state"
            if agrees
            else f"documents state {best} sites; {resolved_sites} resolved"
        ),
        "evidence": [ev[best]],
        "all_counts_seen": dict(sorted(counts.items())),
    }
