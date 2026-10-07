"""Count entity keys and scope_category values off the closed lists (app/core/label_vocab.json).

Dry run by default: reads a label snapshot, prints per off-list value how many
rows carry it and the canonical value proposed, and writes nothing.

    python scripts/label_vocab_dryrun.py walk.json.gz
    python scripts/label_vocab_dryrun.py labels.json --lists labels

The snapshot is JSON or gzipped JSON: a list of label rows, or an object
holding them under ``labels``, ``atom_labels``, ``rows``, ``proposedLabels``,
``otherLabels`` or ``orphanedLabels`` (each list is counted on its own). A
row's ``entity_keys`` and ``reads_set`` may be stored as JSON strings.

The output names values only by the registry's labels: an open prefix is
counted as ``prefix:*`` and only a closed list's slug is spelled out, so no
person's or customer's name is printed.

``--apply --out FILE`` writes a copy of the snapshot with the one-to-one
renames made, and only on rows a person did not label (a labeler marked
``(assistant)``, ``(bot)`` or ``(model)``): a person's picks are never
changed, no key is removed and no row is deleted. Off-list values with no
one-to-one rename are left as they are. It never writes to a database.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import label_vocab  # noqa: E402

LISTS = ("labels", "atom_labels", "rows", "proposedLabels", "otherLabels", "orphanedLabels")
NOT_A_PERSON = ("(assistant)", "(bot)", "(model)")


def _load(path: Path) -> Any:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def row_lists(data: Any, only: tuple[str, ...] = ()) -> dict[str, list[dict[str, Any]]]:
    if isinstance(data, list):
        return {"rows": [r for r in data if isinstance(r, dict)]}
    out = {}
    for k in LISTS:
        if (not only or k in only) and isinstance(data.get(k), list):
            out[k] = [r for r in data[k] if isinstance(r, dict)]
    return out


def _json_field(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _keys(row: dict[str, Any]) -> list[str]:
    v = _json_field(row.get("entity_keys"))
    if isinstance(v, str):
        return [v] if v else []
    return [str(x) for x in (v or []) if x not in (None, "")]


def _scope_category(row: dict[str, Any]) -> Any:
    reads = _json_field(row.get("reads_set"))
    return reads.get("scope_category") if isinstance(reads, dict) else None


def is_a_person(labeler: Any) -> bool:
    s = str(labeler or "").strip().lower()
    return bool(s) and not any(m in s for m in NOT_A_PERSON)


def count(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per off-list or renamed value: rows carrying it, keys, status, proposal."""
    ent: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "keys": 0, "status": "",
                                                          "proposed": "", "person_rows": 0})
    sc: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "status": "",
                                                         "proposed": "", "person_rows": 0})
    totals = Counter()
    for r in rows:
        person = is_a_person(r.get("labeler"))
        seen: set[str] = set()
        for k in _keys(r):
            c = label_vocab.entity_tag(k)
            if c.status == label_vocab.OK:
                continue
            slot = ent[c.label]
            slot["keys"] += 1
            slot["status"] = c.status
            slot["proposed"] = (c.label.split(" -> ", 1)[1] if c.status == label_vocab.ALIAS
                                else c.proposal or "(none: keep, a person decides)")
            if c.label not in seen:
                seen.add(c.label)
                slot["rows"] += 1
                slot["person_rows"] += person
        statuses = {ent[x]["status"] for x in seen}
        totals["rows"] += 1
        totals["rows_with_alias_key"] += label_vocab.ALIAS in statuses
        totals["rows_with_off_list_key"] += label_vocab.OFF_LIST in statuses
        v = _scope_category(r)
        if v not in (None, "", []):
            c = label_vocab.scope_category(v)
            totals["rows_with_scope_category"] += 1
            if c.status != label_vocab.OK:
                slot = sc[c.label]
                slot["rows"] += 1
                slot["person_rows"] += person
                slot["status"] = c.status
                slot["proposed"] = c.canonical if c.status == label_vocab.ALIAS else "(none: keep, a person decides)"
                totals[f"rows_scope_category_{c.status}"] += 1
    return {"totals": dict(totals), "entity_keys": dict(ent), "scope_category": dict(sc)}


def report(name: str, res: dict[str, Any]) -> str:
    t = res["totals"]
    lines = [f"== {name}: {t.get('rows', 0)} rows; "
             f"{t.get('rows_with_alias_key', 0)} with a renamable entity key, "
             f"{t.get('rows_with_off_list_key', 0)} with an off-list entity key; "
             f"{t.get('rows_with_scope_category', 0)} with scope_category, "
             f"{t.get('rows_scope_category_alias', 0)} respellable, "
             f"{t.get('rows_scope_category_off_list', 0)} off the list"]
    if res["entity_keys"]:
        lines.append(f"  {'entity key':<52} {'rows':>5} {'keys':>5} {'person':>6}  {'status':<8} proposed")
        for label, s in sorted(res["entity_keys"].items(), key=lambda kv: (-kv[1]["rows"], kv[0])):
            lines.append(f"  {label:<52} {s['rows']:>5} {s['keys']:>5} {s['person_rows']:>6}  "
                         f"{s['status']:<8} {s['proposed']}")
    if res["scope_category"]:
        lines.append(f"  {'scope_category':<52} {'rows':>5} {'':>5} {'person':>6}  {'status':<8} proposed")
        for label, s in sorted(res["scope_category"].items(), key=lambda kv: (-kv[1]["rows"], kv[0])):
            lines.append(f"  {label:<52} {s['rows']:>5} {'':>5} {s['person_rows']:>6}  "
                         f"{s['status']:<8} {s['proposed']}")
    return "\n".join(lines)


def apply(data: Any, only: tuple[str, ...] = ()) -> int:
    """Rename one-to-one aliases in place on rows no person labeled. Returns rows changed."""
    changed = 0
    for rows in row_lists(data, only).values():
        for r in rows:
            if is_a_person(r.get("labeler")):
                continue
            touched = False
            keys = _keys(r)
            canon = label_vocab.canonical_keys(keys)
            if canon != keys:
                was = r.get("entity_keys")
                r["entity_keys"] = json.dumps(canon) if isinstance(was, str) else canon
                touched = True
            reads = _json_field(r.get("reads_set"))
            if isinstance(reads, dict) and reads.get("scope_category") not in (None, "", []):
                c = label_vocab.scope_category(reads["scope_category"])
                if c.status == label_vocab.ALIAS:
                    reads = {**reads, "scope_category": c.canonical}
                    r["reads_set"] = json.dumps(reads) if isinstance(r.get("reads_set"), str) else reads
                    touched = True
            changed += touched
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("snapshot", type=Path, nargs="+")
    ap.add_argument("--lists", default="", help="comma list of the snapshot's row lists to read")
    ap.add_argument("--apply", action="store_true",
                    help="write a renamed copy to --out (rows no person labeled only)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    only = tuple(x.strip() for x in a.lists.split(",") if x.strip())
    if a.apply and (a.out is None or len(a.snapshot) != 1
                    or a.out.resolve() == a.snapshot[0].resolve()):
        ap.error("--apply needs one snapshot and an --out file other than it")
    for path in a.snapshot:
        data = _load(path)
        for name, rows in row_lists(data, only).items():
            print(report(f"{path.name} {name}", count(rows)))
        if a.apply:
            n = apply(data, only)
            a.out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            print(f"applied: {n} rows renamed -> {a.out}")
        else:
            print("dry run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
