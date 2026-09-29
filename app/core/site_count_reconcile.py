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

# "10 sites", "10 locations", "10 separate locations"
#
# A DEVICE NOUN IS NOT A PLACE NOUN. This list used to include "timeclocks"
# and "time clocks", because on deal 010215 -- nine sites, one clock each --
# the two were interchangeable. On 010264 they are not: 1,500 clocks stand
# at ~795 locations, because roughly 600 addresses carry more than one. A
# pattern that reads "1,500 time clocks" as a site count reproduces, in
# code, the exact confusion that put "one thousand five hundred (795)
# locations" into a customer-facing proposal twice.
#
# 010215 still resolves, on the sentence this module's own docstring quotes:
# "SOW's for each of the ten locations".
# The lookarounds are the whole safety of this pattern. Without them
# "$3,050 for the site" reads as "50 sites" -- a dollar amount becoming a site
# count, which is a confident wrong answer rather than a missing one. A digit,
# comma, period or currency symbol on either side means the number is part of a
# larger figure and is not a count of anything.
_COUNT_NEAR_SITE = re.compile(
    r"(?<![\d.,$])"
    # 10 | 795 | 1500 | 1,400 -- a national estate is four digits, and the
    # old three-digit limit could not express one.
    r"\b(\d{1,3}(?:,\d{3})+|\d{1,5}|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|eleven|twelve)\b"
    r"(?![\d.,]\d)"
    # A closing bracket may sit between the number and the noun: the
    # proposal writes "one thousand five hundred (795) United States
    # locations", where the only machine-readable count is the one in
    # brackets.
    r"[)\]]?"
    r"(?:\s+\w+){0,3}?\s+"
    r"(sites?|locations?|schools?|buildings?|stores?|facilities|"
    r"premises|campuses|branches)\b",
    re.I,
)

#: A second count has to be asserted at least this many times before it is
#: treated as a rival rather than a slip of the tongue.
MIN_RIVAL_MENTIONS = 3

#: ...and differ from the leader by at least this share. Two counts within a
#: few percent are the same claim rounded differently; 795 against 1,500 is
#: two different things being counted.
RIVAL_GAP = 0.2

#: ...and be the same order of magnitude. "Four locations" said nine times
#: is a day's schedule on a nine-site job, not a rival claim about a
#: nationwide estate. A competing count of the same thing lands within a
#: quarter of the leader; 795 against 1,500 does, 4 against 1,500 does not.
RIVAL_FLOOR = 0.25

_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


#: The largest number this will believe is a site count. It has to be big
#: enough for a national estate: deal 010264 states "795 locations" in four
#: documents and "1,500" in several more, and the old ceiling of 200 threw
#: every one of them away -- so a rollout across forty states reported
#: `stated: 1` and called it agreed, which is the exact silence this module
#: was written to break. The lookarounds, not the ceiling, are what stop a
#: dollar figure being read as a count.
_MAX_PLAUSIBLE_SITES = 100_000


def _as_int(token: str) -> int | None:
    t = str(token).strip().lower().replace(",", "")
    if t.isdigit():
        n = int(t)
        return n if 1 <= n <= _MAX_PLAUSIBLE_SITES else None
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
        # Prefer a sentence that shows the number as DIGITS. Both "10
        # timeclocks" and "each of the ten locations" assert ten, and a
        # reader scanning a finding sees the digit; the spelled-out form
        # reads as prose and the number disappears into it. First writer
        # wins only among equals.
        prior = ev.get(n)
        if prior is None or (str(n) not in prior and str(n) in text):
            ev[n] = text
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    best = ranked[0][0]

    # ...unless the documents are arguing with each other. Picking the mode
    # when two counts are both asserted repeatedly and differ materially
    # reports a winner where there is a dispute, which is the failure this
    # module exists to prevent -- it just moves it one level up. On deal
    # 010264 the corpus says "1500" nine times and "795" six, and those are
    # not two estimates of one thing: 1,500 is the CLOCK count and 795 the
    # LOCATION count, with ~600 addresses carrying more than one clock.
    # Saying "stated: 1500" would launder that into a fact.
    rival = next(((n, k) for n, k in ranked[1:]
                  if k >= MIN_RIVAL_MENTIONS
                  and abs(n - best) > max(best, n) * RIVAL_GAP
                  and min(n, best) >= max(n, best) * RIVAL_FLOOR), None)

    if rival is not None:
        # The documents do not agree with each other, so there is nothing to
        # reconcile a resolved count against yet. Say that, and hand back
        # both claims with the sentence asserting each.
        n, k = rival
        lo, hi = sorted((best, n))
        return {
            "stated": None,
            "stated_mentions": counts[best],
            "resolved": int(resolved_sites),
            "agrees": None,
            "reason": (
                f"the documents disagree with themselves: {best} stated "
                f"{counts[best]} time(s) and {n} stated {k} time(s). "
                f"Resolve {lo} against {hi} before comparing either to the "
                f"{resolved_sites} resolved."
            ),
            "rival_counts": {best: counts[best], n: k},
            "evidence": [ev[best], ev[n]],
            "all_counts_seen": dict(sorted(counts.items())),
        }

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
