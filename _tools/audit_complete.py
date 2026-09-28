"""Is a label complete, or does it merely exist?

The 100%-labelled claim of 27 September counted rows carrying a label_type. On
the one document we had worked hardest on, 10 of 11 atoms had no pointer and
none carried about, wants or reads -- so the number was true and meant nothing.
This grades the fields a head actually trains on, and validates every value
against the registry rather than trusting that I typed the key correctly.

    PG=... python audit_complete.py            # whole deal
    PG=... python audit_complete.py 114272444842   # one document
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import psycopg2

sys.path.insert(0, r"C:\Users\lilli\parser-os-registry")

#: The deal being labelled. An env var, not a constant: these scripts were
#: written for 010180 and a hardcoded id here would silently write another
#: deal's labels under this deal's rows, which no audit would catch.
DEAL = os.environ.get("DEAL") or "c79db726-323e-41f8-899d-1d8ca29a579a"
#: The labeler whose rows this reads. An env var, not a constant: it was hardcoded,
#: so a second labeler auditing their own deal graded someone else's rows and saw
#: 0% forever -- or, worse, saw 100% on a deal they had not touched.
MINE = os.environ.get("LABELER", "developer@purtera-it.com")
HERE = Path(__file__).parent
REG = json.loads(Path(r"C:\Users\lilli\parser-os-registry\app\core\atom_types.json")
                 .read_text(encoding="utf-8"))

VALID_ABOUT = {x["key"] for x in REG["about"]}
VALID_WANTS = {x["key"] for x in REG["wants"]}
VALID_READS = {x["key"] for x in REG["reads"]}
VALID_HINTS = {x["key"] for x in REG["context_hints"]}
VALID_SUPP = {x["key"] for x in REG["suppliers"]}
def _names(node):
    """The registry is not uniform: `types` entries key on `name`, the facet and
    hint lists on `key`, and `keep` is a bare object rather than a list."""
    if isinstance(node, dict):
        node = [node]
    out = set()
    for x in node or []:
        if isinstance(x, dict):
            v = x.get("name") or x.get("key")
            if v:
                out.add(v)
        elif isinstance(x, str):
            out.add(x)
    return out


VALID_TYPES = _names(REG["types"]) | _names(REG["keep"])

#: A label that only says what something IS teaches a type head and nothing
#: else. These are the fields the other heads read.
REQUIRED = ("label_type", "weight_tier", "note", "hint_refs", "entity_keys", "about", "wants")


def _j(v, default):
    if isinstance(v, (list, dict)):
        return v
    if not v:
        return default
    try:
        return json.loads(v)
    except Exception:
        return default


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    walk = json.loads((HERE / os.environ.get("WALK", "walk_180.json")).read_text(encoding="utf-8"))["atoms"]
    if only:
        walk = [a for a in walk if only in a["filename"]]
    # A dedup loser is in the walk to be ACCOUNTED FOR, not to be judged: it is
    # a copy the parser folded into a richer atom, and the winner carries its
    # source_refs. Counting 206 of those against completeness would make a
    # fully-labelled deal read as half-done and push toward labelling
    # duplicates -- which is the opposite of what surfacing them was for.
    informational = [a for a in walk if "dedup" in (a.get("suppressedBy") or "")]
    walk = [a for a in walk if "dedup" not in (a.get("suppressedBy") or "")]
    if informational:
        print(f"(excluding {len(informational)} dedup losers -- folded copies, "
              f"listed for audit, not for judgement)")
    live = {a["labelKey"] for a in walk}

    cn = psycopg2.connect(os.environ["PG"], connect_timeout=25)
    cur = cn.cursor()
    cur.execute("SELECT label_key, text, label_type, weight_tier, note, hint_refs, "
                "entity_keys, about, wants, reads_set, hints, supplier "
                "FROM public.atom_labels WHERE deal_id=%s AND labeler=%s", (DEAL, MINE))
    rows = {r[0]: r for r in cur.fetchall() if r[0] in live}

    missing = Counter()
    invalid = []
    complete = 0
    unlabelled = [a for a in walk if a["labelKey"] not in rows]

    for key, r in rows.items():
        (_, text, ltype, tier, note, hrefs, ekeys, about, wants, reads, hints, supp) = r
        hrefs, ekeys = _j(hrefs, []), _j(ekeys, [])
        reads, hints = _j(reads, {}), _j(hints, [])
        gaps = []
        if not ltype: gaps.append("label_type")
        if not tier: gaps.append("weight_tier")
        if not (note or "").strip(): gaps.append("note")
        if not hrefs: gaps.append("hint_refs")
        # A rejection has no entities to key. `_keep` means "not a fact worth
        # typing", so demanding that it name the things it is about asks for
        # something that by definition is not there -- and inventing keys to
        # satisfy an audit is how a metric starts measuring itself.
        if not ekeys and ltype != "_keep": gaps.append("entity_keys")
        if not about: gaps.append("about")
        if not wants: gaps.append("wants")
        for g in gaps:
            missing[g] += 1
        if not gaps:
            complete += 1
        # Values must be in the registry -- the `_reject` mistake, generalised.
        for name, val, ok in (("about", about, VALID_ABOUT), ("wants", wants, VALID_WANTS),
                              ("supplier", supp, VALID_SUPP), ("label_type", ltype, VALID_TYPES)):
            if val and val not in ok:
                invalid.append((name, val, text[:44]))
        for k in (reads or {}):
            if k not in VALID_READS:
                invalid.append(("reads", k, text[:44]))
        for h in (hints or []):
            if h not in VALID_HINTS:
                invalid.append(("hints", h, text[:44]))

    n = len(live)
    scope = only or "whole deal"
    print(f"=== completeness audit: {scope} ===")
    print(f"atoms                 : {n}")
    print(f"has a label           : {len(rows)}  ({100 * len(rows) // max(n, 1)}%)")
    print(f"COMPLETE (all 7 fields): {complete}  ({100 * complete // max(n, 1)}%)")
    print(f"unlabelled            : {len(unlabelled)}")
    print("\nmissing field counts (of labelled):")
    for f in REQUIRED:
        c = missing.get(f, 0)
        bar = "!" * min(c // 4, 30)
        print(f"   {f:14} missing on {c:4}  {bar}")
    print(f"\nvalues not in the registry: {len(invalid)}")
    for name, val, t in invalid[:12]:
        print(f"   {name}={val!r}  on  {t}")
    if unlabelled:
        byk = {a["labelKey"]: a for a in walk}
        print("\nunlabelled atoms:")
        for a in unlabelled[:10]:
            print(f"   {a['filename'][-24:]:26} | {a['text'][:52]}")


if __name__ == "__main__":
    main()


def invariants():
    """The checks that would have caught what review caught by hand.

    Each one exists because it was violated: a site key was reintroduced by a
    later pass after being fixed once, a pointer quoted "--" where the atom had
    an em dash, a `_keep` rendered as an ordinary row because the workspace
    styles off `rejected` and not off the type.
    """
    import re as _re
    walk = json.loads((HERE / os.environ.get("WALK", "walk_180.json")).read_text(encoding="utf-8"))["atoms"]
    live = {a["labelKey"] for a in walk if "dedup" not in (a.get("suppressedBy") or "")}
    cn = psycopg2.connect(os.environ["PG"], connect_timeout=25)
    cur = cn.cursor()
    cur.execute("SELECT label_key,text,label_type,entity_keys,hint_refs,rejected "
                "FROM public.atom_labels WHERE deal_id=%s AND labeler=%s", (DEAL, MINE))
    rows = [(k, " ".join((t or "").split()), lt, _j(ek, []), _j(hr, []), rj)
            for k, t, lt, ek, hr, rj in cur.fetchall() if k in live]
    SAYS = _re.compile(r"penn|west 3[01]|31st|30th|suite 1200|floor 12|12,?154|great neck"
                       r"|seventh avenue|7th avenue", _re.I)
    fails = {
        "pointer not literally in its atom": sum(
            1 for _, t, _, _, hr, _ in rows for p in hr
            if p.get("hint") == "own_words"
            and " ".join((p.get("text") or "").split()).lower() not in t.lower()),
        "site key the atom never names": sum(
            1 for _, t, _, ek, _, _ in rows for x in ek
            if x.startswith("site:") and not SAYS.search(t)),
        "_keep without rejected (renders as a normal row)": sum(
            1 for _, _, lt, _, _, rj in rows
            if lt == "_keep" and str(rj) not in ("true", "t")),
        "rejected without _keep (red but claims a type)": sum(
            1 for _, _, lt, _, _, rj in rows
            if lt != "_keep" and str(rj) in ("true", "t")),
        # The drawing writes schedule counts zero-padded ("PRIVATE OFFICE 01"),
        # so compare as integers. Comparing the literal strings flagged two
        # correct labels, and the fix for a wrong check is the check.
        "numeric key naming a number not in the atom": sum(
            1 for _, t, _, ek, _, _ in rows for x in ek
            if _re.match(r"^(quantity|price):\d+$", x)
            and int(x.split(":")[1]) not in {int(n) for n in
                                             _re.findall(r"\d+", t.replace(",", ""))}),
    }
    print("\n=== invariants ===")
    for name, n in fails.items():
        print(f"   {'OK  ' if n == 0 else 'FAIL'} {name}: {n}")
    return sum(fails.values())


if __name__ == "__main__" and "--invariants" in sys.argv:
    raise SystemExit(1 if invariants() else 0)
