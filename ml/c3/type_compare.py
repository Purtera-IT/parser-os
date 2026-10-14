"""The base's line types against the parser's, on the same labeled lines.

    python -m ml.c3.type_compare --read reads/010087.json --labels atom_labels.json

``--read`` is ``ml.c3.predict``'s file for one deal. ``--labels`` is the
deal's saved label rows (a labeling snapshot list, or a training file with a
``labels`` list); each row's ``parser_type`` is what the parser's typed-atom
classifier said when the deal was compiled. Only lines that have a saved
type, a parser type and a base read are compared, so both sides are scored
on exactly the same lines. Prints counts and rates only, never line text.
Score validation deals with it, never the holdout.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import TYPES_PATH

NOT_FACT = ("_keep", "small_talk")


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("labels") or payload.get("rows") or []
    return [r for r in payload if isinstance(r, dict)]


def compare(read: dict[str, Any], labels: list[dict[str, Any]], *, focus: str = "task",
            sure: tuple[float, ...] = (0.5, 0.7, 0.9), typed: bool = False) -> dict[str, Any]:
    """``typed``: score the base's best named type (``_keep`` set aside), not its top answer."""
    coarse = {t["name"]: t["coarse"] for t in json.loads(TYPES_PATH.read_text())["types"]}
    lines = read.get("lines") or {}
    rows = []
    for lb in labels:
        gold, parser = lb.get("label_type"), lb.get("parser_type")
        got = (lines.get(lb.get("label_key")) or {}).get("reads", {}).get("col:label_type")
        if gold and parser and got:
            if typed and "typed" in got:
                rows.append((gold, parser, got["typed"], float(got.get("typed_p", 0.0))))
            else:
                rows.append((gold, parser, got["answer"], float(got.get("p", 0.0))))

    def side(pairs: list[tuple[str, str]]) -> dict[str, Any]:
        n = len(pairs)
        said = sum(p == focus for _, p in pairs)
        real = sum(g == focus for g, _ in pairs)
        both = sum(g == p == focus for g, p in pairs)
        facts = [(g, p) for g, p in pairs if g not in NOT_FACT]
        return {
            "n": n,
            "exact": sum(g == p for g, p in pairs) / n if n else None,
            "group": sum(coarse.get(g, g) == coarse.get(p, p) for g, p in pairs) / n if n else None,
            "exact_on_facts": sum(g == p for g, p in facts) / len(facts) if facts else None,
            focus: {"labeled": real, "said": said, "right": both,
                    "recall": both / real if real else None,
                    "precision": both / said if said else None},
        }

    out = {"lines": len(rows),
           "parser": side([(g, p) for g, p, _, _ in rows]),
           "base": side([(g, b) for g, _, b, _ in rows]),
           "base_when_sure": {}}
    for cut in sure:
        kept = [(g, b) for g, _, b, p in rows if p >= cut]
        out["base_when_sure"][str(cut)] = {"share_of_lines": len(kept) / len(rows) if rows else None,
                                           **side(kept)}
    out["base_misses"] = Counter(f"{g}->{b}" for g, _, b, _ in rows if g != b).most_common(10)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--read", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--focus", default="task", help="the type to report recall and precision for")
    p.add_argument("--typed", action="store_true", help="score the best named type, _keep set aside")
    args = p.parse_args()
    read = json.loads(Path(args.read).read_text(encoding="utf-8"))
    labels = _rows(json.loads(Path(args.labels).read_text(encoding="utf-8")))
    print(json.dumps(compare(read, labels, focus=args.focus, typed=args.typed), indent=2))


if __name__ == "__main__":
    main()
