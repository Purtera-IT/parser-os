"""Move a deal's saved labels onto the head format (app/learning/label_format.py).

Dry run by default: reads a snapshot, prints what would move and what still
needs a person, per deal, and writes nothing.

    python scripts/labels_to_heads.py --labels 000132-labels.json [--links 000132-links.json]
    python scripts/labels_to_heads.py --labels ... --report report.json --sql apply.sql

The snapshot is ``SELECT * FROM atom_labels WHERE deal_id = $1`` as JSON: a
list of rows, or an object holding them under ``labels``/``atom_labels``/
``rows`` (links under ``links``/``atom_label_links``). ``--sql`` writes the
UPDATEs for review; run them yourself. Each one is guarded on the row's
current reads and note, so a value someone saved after the snapshot is left
alone, and none of them changes a type, a reject flag or a note's words.
The report names rows by label_key only; it carries no source text.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.learning.label_format import (  # noqa: E402
    coverage, format_checks, transform_link, transform_row,
)


def _rows(path: Path | None, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    for k in keys:
        if isinstance(data.get(k), list):
            return [r for r in data[k] if isinstance(r, dict)]
    return []


def _q(v: Any) -> str:
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"


def _reads_json(row: dict[str, Any]) -> Any:
    r = row.get("reads_set")
    if isinstance(r, str):
        try:
            return json.loads(r)
        except ValueError:
            return r
    return r


def label_sql(old: dict[str, Any], new: dict[str, Any]) -> str | None:
    sets = []
    if new.get("about") != old.get("about"):
        sets.append(f"about = {_q(new.get('about'))}")
    if _reads_json(new) != _reads_json(old):
        sets.append(f"reads_set = {_q(json.dumps(_reads_json(new), ensure_ascii=False, sort_keys=True))}::jsonb")
    if new.get("note") != old.get("note"):
        sets.append(f"note = {_q(new.get('note'))}")
    for col in ("supplier", "entity_keys"):
        if new.get(col) != old.get(col) and col not in old:
            raise ValueError(f"{old.get('label_key')}: the snapshot has no {col} column; export it with every column")
    if new.get("supplier") != old.get("supplier"):
        sets.append(f"supplier = {_q(new.get('supplier'))}")
    if (new.get("entity_keys") or []) != (old.get("entity_keys") or []):
        sets.append(f"entity_keys = {_q(json.dumps(new.get('entity_keys') or [], ensure_ascii=False))}::jsonb")
    if not sets:
        return None
    old_reads = _reads_json(old)
    guard_reads = ("reads_set IS NULL" if old_reads is None else
                   f"reads_set = {_q(json.dumps(old_reads, ensure_ascii=False))}::jsonb")
    return (f"UPDATE public.atom_labels SET {', '.join(sets)}\n"
            f" WHERE deal_id = {_q(old.get('deal_id'))} AND label_key = {_q(old.get('label_key'))}"
            f" AND labeler = {_q(old.get('labeler'))}\n"
            f"   AND {guard_reads} AND note IS NOT DISTINCT FROM {_q(old.get('note'))}"
            f" AND about IS NOT DISTINCT FROM {_q(old.get('about'))}{_column_guards(old)};")


def _column_guards(old: dict[str, Any]) -> str:
    """Guards on the columns the snapshot carried. A snapshot without the
    column cannot vouch for it, and a write to it is refused below."""
    out = ""
    if "supplier" in old:
        out += f" AND supplier IS NOT DISTINCT FROM {_q(old.get('supplier'))}"
    if "entity_keys" in old:
        out += f" AND entity_keys = {_q(json.dumps(old.get('entity_keys') or [], ensure_ascii=False))}::jsonb"
    return out


def link_sql(old: dict[str, Any], new: dict[str, Any]) -> str | None:
    if new == old or not old.get("id"):
        return None
    return (f"UPDATE public.atom_label_links SET relation = {_q(new['relation'])}, note = {_q(new.get('note'))}\n"
            f" WHERE id = {_q(old['id'])} AND relation = {_q(old.get('relation'))}"
            f" AND note IS NOT DISTINCT FROM {_q(old.get('note'))};")


def run(labels: list[dict[str, Any]], links: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    per: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "rows": 0, "rows_changed": 0, "moves": Counter(), "checks": Counter(),
        "check_rows": defaultdict(list), "links": 0, "links_changed": 0, "after": []})
    sql: list[str] = []
    for row in labels:
        d = per[str(row.get("deal_id") or "")]
        d["rows"] += 1
        new, moved = transform_row(row)
        d["after"].append(new)
        if moved:
            d["rows_changed"] += 1
            d["moves"].update(moved)
            if not (row.get("deal_id") and row.get("label_key") and row.get("labeler")):
                d["moves"]["(no UPDATE: the row lacks deal_id, label_key or labeler)"] += 1
            else:
                try:
                    if s := label_sql(row, new):
                        sql.append(s)
                except ValueError as e:
                    d["moves"][f"(no UPDATE: {str(e).split(': ', 1)[-1]})"] += 1
        for c in format_checks(new):
            d["checks"][c["check"]] += 1
            if len(d["check_rows"][c["check"]]) < 25:
                d["check_rows"][c["check"]].append(str(row.get("label_key") or ""))
    for link in links:
        d = per[str(link.get("deal_id") or "")]
        d["links"] += 1
        new, moved = transform_link(link)
        if moved:
            d["links_changed"] += 1
            d["moves"].update(moved)
            s = link_sql(link, new)
            if s:
                sql.append(s)
    report = {}
    for deal, d in sorted(per.items()):
        report[deal] = {
            "rows": d["rows"], "rows_changed": d["rows_changed"],
            "links": d["links"], "links_changed": d["links_changed"],
            "moves": dict(d["moves"].most_common()),
            "checks": dict(d["checks"].most_common()),
            "check_rows": {k: v for k, v in d["check_rows"].items()},
            "rows_per_head": coverage(d["after"]),
        }
    return report, sql


def _print(report: dict[str, Any]) -> None:
    for deal, r in report.items():
        print(f"deal {deal or '(none)'}: {r['rows']} labels, {r['rows_changed']} would change; "
              f"{r['links']} links, {r['links_changed']} would change")
        for k, n in r["moves"].items():
            print(f"  move   {n:>4}  {k}")
        for k, n in r["checks"].items():
            print(f"  check  {n:>4}  {k}")
        print("  rows per head:")
        for k, n in r["rows_per_head"].items():
            print(f"         {n:>4}  {k}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--links", type=Path)
    ap.add_argument("--report", type=Path, help="write the report as JSON")
    ap.add_argument("--sql", type=Path, help="write the guarded UPDATEs for review (nothing is run)")
    a = ap.parse_args(argv)
    labels = _rows(a.labels, ("labels", "atom_labels", "rows"))
    links = _rows(a.links, ("links", "atom_label_links")) or _rows(a.labels, ("links", "atom_label_links"))
    report, sql = run(labels, links)
    _print(report)
    if a.report:
        a.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.sql:
        a.sql.write_text("BEGIN;\n" + "\n".join(sql) + ("\n" if sql else "") + "COMMIT;\n", encoding="utf-8")
        print(f"{len(sql)} statements -> {a.sql}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
