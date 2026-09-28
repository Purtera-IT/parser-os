# -*- coding: utf-8 -*-
"""Check a hand-written pass file before it is applied.

audit_complete.py grades what is already in the database. By then a fabricated
pointer or an off-registry type is gold that trains nothing, and finding it costs
a patch pass. This runs the same rules against the JSON, so the cost of a mistake
is an edit instead.

It checks more than the five invariants, because two of those cannot protect a
new deal:

  * The site-key invariant in audit_complete.py matches 010180's vocabulary
    (penn|great neck|seventh avenue|...). On any other deal it passes vacuously.
    Here, ANY `site:` key is checked the only way that generalises: does the atom
    contain a recognisable piece of the key's own words?
  * The numeric check is applied to every `<prefix>:<number>` key, not only
    quantity and price, since a fabricated count is just as wrong under any name.

    DEAL=... python validate_pass.py p1.json
"""
import json, os, re, sys
from pathlib import Path

HERE = Path(__file__).parent
REG = json.loads(Path(r"C:\Users\lilli\parser-os-registry\app\core\atom_types.json")
                 .read_text(encoding="utf-8"))
WALK = json.loads((HERE / os.environ.get("WALK", "walk_180.json")).read_text(encoding="utf-8"))
BY_KEY = {a["labelKey"]: a for a in WALK["atoms"]}

VALID_TYPES = {t["name"] for t in REG["types"]} | {REG["keep"]["name"]}
VALID_ABOUT = {x["key"] for x in REG["about"]}
VALID_WANTS = {x["key"] for x in REG["wants"]}
VALID_HINTS = {x["key"] for x in REG["context_hints"]}
VALID_SUPP = {x["key"] for x in REG["suppliers"]}
VALID_TIERS = {"load_bearing", "ordinary", "slight"}
REQUIRED = ("label_type", "weight_tier", "note", "hint_refs", "entity_keys", "about", "wants")

_norm = lambda s: " ".join(str(s or "").split()).lower()


def main() -> None:
    payload = json.loads((HERE / sys.argv[1]).read_text(encoding="utf-8"))
    problems: list[str] = []
    n = 0

    for item in payload["labels"]:
        key = item.get("labelKey")
        atom = BY_KEY.get(key)
        where = f"[{item.get('_i', '?')}] {_norm(item.get('_text'))[:52]!r}"
        if atom is None:
            problems.append(f"{where}: labelKey is not in the walk -- stale or mistyped")
            continue
        f = item.get("fields") or {}
        if not any(str(v).strip() for v in f.values() if not isinstance(v, (list, dict))):
            continue                                    # untouched stub, not yet labelled
        n += 1
        text = _norm(atom["text"])

        for r in REQUIRED:
            v = f.get(r)
            empty = v in (None, "", [], {}) or (isinstance(v, str) and not v.strip())
            # A rejection has nothing to key: _keep means "not a fact worth typing",
            # so demanding entity_keys asks for something that is not there.
            if r == "entity_keys" and f.get("label_type") == "_keep":
                continue
            if empty:
                problems.append(f"{where}: missing {r}")

        for name, val, ok in (("label_type", f.get("label_type"), VALID_TYPES),
                              ("about", f.get("about"), VALID_ABOUT),
                              ("wants", f.get("wants"), VALID_WANTS),
                              ("weight_tier", f.get("weight_tier"), VALID_TIERS),
                              ("supplier", f.get("supplier"), VALID_SUPP)):
            if val and val not in ok:
                problems.append(f"{where}: {name}={val!r} is not in the registry")

        for p in f.get("hint_refs") or []:
            h = p.get("hint")
            if h and h not in VALID_HINTS:
                problems.append(f"{where}: hint={h!r} is not in the registry")
            if h == "own_words":
                quote = _norm(p.get("text"))
                if not quote:
                    problems.append(f"{where}: own_words pointer is empty")
                elif quote not in text:
                    problems.append(f"{where}: own_words {p['text']!r} is NOT in the atom "
                                    f"-- fabricated citation")

        nums = {int(x) for x in re.findall(r"\d+", atom["text"].replace(",", ""))}
        for k in f.get("entity_keys") or []:
            if ":" not in k:
                problems.append(f"{where}: entity key {k!r} has no prefix")
                continue
            prefix, _, value = k.partition(":")
            if re.fullmatch(r"\d+", value) and int(value) not in nums:
                problems.append(f"{where}: {k} names a number the atom does not state "
                                f"-- a cross-reference is a LINK, not a key")
            if prefix == "site":
                words = [w for w in re.split(r"[_\-]", value) if len(w) > 3]
                if words and not any(w in text for w in words):
                    problems.append(f"{where}: {k} claims a site the atom never names")

    print(f"{sys.argv[1]}: {n} labelled of {len(payload['labels'])} stubs")
    if not problems:
        print("OK -- nothing to fix")
        return
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print(f"   {p}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
