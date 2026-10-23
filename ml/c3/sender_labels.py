"""The said_by / said_to / internal_only heads are labels, not leaked inputs.

The line's sender *inputs* (``speaker_side``, ``speaker_role``,
``speaker_org``) are the parser's read of who sent it (``c3_record.sender``,
the same rules in the training export and the parser's compile). The label's
``said_by`` / ``said_to`` are the person's answer, in a finer vocabulary
(customer, provider, seller, partner, ...). The column heads
(schema.SAID_COLUMNS, ``col:internal_only``) are labels only if

* featurize never feeds them back as inputs: the Batch's role / side / org
  are the atoms' own sender fields with or without the labels (tests);
* where both speak, the label's side agrees with the input's side
  (``consistent``) or a person overruled it (``overruled``: a forwarded mail,
  a call line the parser gave the wrong side), and the label says something
  the input cannot (``refines``: a role the input vocabulary has no word for);
* ``internal_only`` is the person's own mark, not a function of the parties
  (app.core.deal_parties derives a page default from them; the saved mark
  differs): counted as marked, and marked on a line the other side said.

    python -m ml.c3.sender_labels ~/c3-local/runE/train_plan/*.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

from .data import _is_a_person, excluded

#: The input vocabulary of speaker_role (c3_record.sender.ROLES_FULL).
INPUT_ROLES = frozenset({"internal", "reseller", "vendor", "unknown", ""})


def _side(v: Any) -> str:
    v = str(v or "").strip().lower()
    return "theirs" if v in ("partner", "vendor") else v


def check(payload: dict[str, Any]) -> Counter:
    """Counts over one training file's person-saved labels (see the module doc)."""
    c: Counter = Counter()
    atoms = {str(a.get("label_key") or ""): a for a in payload.get("atoms") or [] if isinstance(a, dict)}
    for r in payload.get("labels") or []:
        if not isinstance(r, dict) or not _is_a_person(r.get("labeler")) or excluded(r):
            continue
        by = r.get("said_by") if isinstance(r.get("said_by"), dict) else None
        to = [p for p in r.get("said_to") or [] if isinstance(p, dict)] if isinstance(r.get("said_to"), list) else []
        a = atoms.get(str(r.get("label_key") or ""))
        if by is not None:
            c["said_by"] += 1
            lside, iside = _side(by.get("side")), _side((a or {}).get("speaker_side"))
            if not lside:
                c["said_by_no_side"] += 1
            elif a is None or not iside:
                c["said_by_no_input"] += 1
            elif lside == iside:
                c["consistent"] += 1
            else:
                c["overruled"] += 1
            if str(by.get("role") or "").strip().lower() not in INPUT_ROLES:
                c["refines"] += 1
        if to:
            c["said_to"] += 1
        if r.get("internal_only") is True:
            c["internal_only"] += 1
            sides = [_side(by.get("side"))] if by else []
            sides += [_side(p.get("side")) for p in to]
            if not all(s == "ours" for s in sides if s):
                c["internal_only_not_from_parties"] += 1
            if by and _side(by.get("side")) == "theirs":
                c["internal_only_said_by_theirs"] += 1
    return c


def main(argv: list[str] | None = None) -> int:
    paths = list(sys.argv[1:] if argv is None else argv)
    total: Counter = Counter()
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            total.update(check(json.load(fh)))
    both = total["consistent"] + total["overruled"]
    print(json.dumps(dict(total), indent=1, sort_keys=True))
    if both:
        print(f"label side agrees with the sender input on {100 * total['consistent'] / both:.1f}% "
              f"({total['overruled']} overruled); role finer than the input on "
              f"{100 * total['refines'] / max(total['said_by'], 1):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
