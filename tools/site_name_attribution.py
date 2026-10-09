"""Attribute every site/display name in compiled envelopes to the code path
that produced it (PUR-49).

Reads envelope JSON files from a LOCAL path only (a file or a directory of
``*.json``). It never touches blob storage, a database or Azure; fetching
envelopes is a separate, human-run step.

For each name on each atom it reports:

* ``rule`` -- ``structured.name_source.rule`` when the compile stamped it
  (see ``app.core.site_facility_head``); for envelopes compiled before that
  stamp existed, a ``legacy:`` rule inferred from the name's shape.
* ``class`` -- one of
    - ``verbatim``     the name is in the atom's own source text
    - ``wrong_field``  a real string, but read from an address/locality field
                       (the city or street used as the site's name)
    - ``invented``     the text appears nowhere in the source
    - ``undecidable``  the atom's text is a composed field summary, so there
                       is no verbatim source to check against (same exclusion
                       as ``tools/base_health.py``)

Output ranks rules by fabricated count (invented + wrong_field) with the
number of deals each rule hits, and states the as-of date.

Usage (synthetic fixture)::

    python tools/site_name_attribution.py tests/fixtures/site_names/envelopes
    python tools/site_name_attribution.py <dir> --json --as-of 2026-09-17

Real run (human, after downloading envelopes to a local directory with the
usual approved tooling)::

    python tools/site_name_attribution.py /path/to/local/envelopes --json-out attribution.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from tools.base_health import DISPLAY_NAME_FIELDS, is_composed_field_summary, normalize

FABRICATED_CLASSES = ("invented", "wrong_field")
_LOCALITY_FIELDS = ("city", "street_address", "address", "state", "zip")


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle) and f" {needle} " in f" {haystack} "


def infer_rule(atom: dict[str, Any], name: str) -> str:
    """The code path behind ``name`` on ``atom``."""
    st = atom.get("structured") if isinstance(atom.get("structured"), dict) else {}
    ns = st.get("name_source")
    if isinstance(ns, dict) and ns.get("rule"):
        return str(ns["rule"])
    city = normalize(st.get("city"))
    n = normalize(name)
    if city and n == f"{city} office":
        return "legacy:city_office_composed"
    if city and n == city:
        return "legacy:city_as_name"
    if n == "site 1":
        return "legacy:site_1_placeholder"
    fl = st.get("facility_label")
    if isinstance(fl, dict) and fl.get("source"):
        return f"legacy:{fl.get('source')}:{fl.get('label')}"
    return f"unattributed:{atom.get('atom_type') or '?'}"


def classify_name(atom: dict[str, Any], name: str) -> str:
    text = str(atom.get("text") or "")
    if is_composed_field_summary(text):
        return "undecidable"
    st = atom.get("structured") if isinstance(atom.get("structured"), dict) else {}
    n = normalize(name)
    for f in _LOCALITY_FIELDS:
        if n and n == normalize(st.get(f)):
            return "wrong_field"
    return "verbatim" if _contains(normalize(text), n) else "invented"


def iter_envelopes(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        deal = str(data.get("deal_id") or data.get("project_id") or f.stem) if isinstance(data, dict) else f.stem
        if isinstance(data, dict):
            yield deal, data


def attribute(envelopes: Iterable[tuple[str, dict[str, Any]]], as_of: str) -> dict[str, Any]:
    per_rule: dict[str, Counter] = defaultdict(Counter)
    deals_by_rule: dict[str, set[str]] = defaultdict(set)
    totals: Counter = Counter()
    unnamed_sites = 0
    site_atoms = 0
    envelopes_read = 0
    for deal, env in envelopes:
        envelopes_read += 1
        for atom in env.get("atoms") or []:
            if not isinstance(atom, dict):
                continue
            st = atom.get("structured") if isinstance(atom.get("structured"), dict) else {}
            if str(atom.get("atom_type") or "") == "physical_site":
                site_atoms += 1
                if not any(isinstance(st.get(f), str) and st.get(f).strip() for f in DISPLAY_NAME_FIELDS):
                    unnamed_sites += 1
            seen: set[str] = set()
            for f in DISPLAY_NAME_FIELDS:
                v = st.get(f)
                if not (isinstance(v, str) and v.strip()) or normalize(v) in seen:
                    continue
                seen.add(normalize(v))
                rule = infer_rule(atom, v)
                cls = classify_name(atom, v)
                per_rule[rule][cls] += 1
                totals[cls] += 1
                if cls in FABRICATED_CLASSES:
                    deals_by_rule[rule].add(deal)
    rows = []
    for rule, c in per_rule.items():
        fab = sum(c[k] for k in FABRICATED_CLASSES)
        rows.append({
            "rule": rule,
            "fabricated": fab,
            "invented": c["invented"],
            "wrong_field": c["wrong_field"],
            "verbatim": c["verbatim"],
            "undecidable": c["undecidable"],
            "deals_hit": len(deals_by_rule[rule]),
        })
    rows.sort(key=lambda r: (-r["fabricated"], -r["deals_hit"], r["rule"]))
    fab_total = sum(totals[k] for k in FABRICATED_CLASSES)
    return {
        "as_of": as_of,
        "envelopes": envelopes_read,
        "site_atoms": site_atoms,
        "unnamed_site_atoms": unnamed_sites,
        "totals": dict(totals),
        "fabricated_total": fab_total,
        "rules": rows,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        f"Site-name attribution as of {report['as_of']}: {report['envelopes']} envelope(s), "
        f"{report['site_atoms']} site atom(s), {report['unnamed_site_atoms']} unnamed",
        f"fabricated={report['fabricated_total']} totals={report['totals']}",
        f"{'rule':45} {'fab':>5} {'invent':>6} {'wrong':>6} {'verb':>5} {'undec':>5} {'deals':>5}",
    ]
    for r in report["rules"]:
        lines.append(
            f"{r['rule'][:45]:45} {r['fabricated']:>5} {r['invented']:>6} {r['wrong_field']:>6} "
            f"{r['verbatim']:>5} {r['undecidable']:>5} {r['deals_hit']:>5}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="local envelope JSON file or directory")
    p.add_argument("--as-of", default=_dt.date.today().isoformat())
    p.add_argument("--json", action="store_true")
    p.add_argument("--json-out", type=Path)
    a = p.parse_args(argv)
    if not a.path.exists():
        print(f"ERROR: {a.path} does not exist", file=sys.stderr)
        return 2
    report = attribute(iter_envelopes(a.path), a.as_of)
    if report["envelopes"] == 0:
        print("ERROR: no envelopes read -- not a clean result", file=sys.stderr)
        return 2
    if a.json_out:
        a.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2) if a.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
