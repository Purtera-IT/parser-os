#!/usr/bin/env python3
"""Per rule, over many compiles: fires, head agreement, deals, and the stage it is ready for.

Reads ``rule_fires`` (app/core/rule_ledger.py) from compile results: local ``result.json`` files, or the N latest
dev results on blob (``deals/<uuid>/parser-os/latest/result.json`` in purpulsedevstg01/orbitbrief-artifacts, read
with ``az login``: ``--auth-mode login``, no keys). Prints, per rule: fires, head agreement % (over the fires a head
answered), no-head fires, deals, mode, and a suggested stage:

- **Active**: under the bar, or too few deals.
- **Active (no head answers it)**: no fire had a head answer (a same-site merge, a serve fallback): a head must be
  built or read before the rule can move.
- **ready for Shadow**: the head agrees on >= ``--agree`` (0.95) of the fires it answered, over >= ``--min-deals``
  (20) deals, and the rule is active.
- **Retire**: the same bar, measured while the rule was already in shadow (the head decided, the rule only logged).

Each rule also has a ``stage`` (rule_ledger.RULES): ``downstream`` rules edit a fact the heads already decided
(merging, vetoing, retyping, dropping sites after the heads). They are listed first: the owner's rule is that they
go -- their fires become refit labels, or the rule moves upstream. ``upstream`` rules act before or at the heads'
decision, on parser output.

A result written before the ledger has no ``rule_fires``. ``--infer`` (on by default) then rebuilds approximate
fires from what the rules leave on the atoms (review flags, ``value.kind``, ``c3_label``) and from the quote's
``by`` fields; those deals are counted apart and marked ``inferred``. What that cannot see: shadow mode, the
overwrite-bar refusals and authority restores (no mark is left), semantic_dedup alias / ghost / merge drops
(the suppressed atom does not say which), and the head's answer on any line whose label a rule rewrote.

    python scripts/rule_ledger_report.py path/to/*.result.json
    python scripts/rule_ledger_report.py --blob 40
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

# This checkout's app (the stage of each rule lives in app/core/rule_ledger.py), not another one on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ACCOUNT = "purpulsedevstg01"
CONTAINER = "orbitbrief-artifacts"
SUFFIX = "/parser-os/latest/result.json"


def _stage_of(rid: str) -> str:
    try:
        from app.core.rule_ledger import stage as _st

        return _st(rid)
    except Exception:  # run outside the repo: quote picks fill gaps, everything else edits after the heads
        return "upstream" if rid.startswith(("quote.", "overwrite_bar.")) and rid != "quote.site_row_hours" else "downstream"


# ── reading results ──────────────────────────────────────────────────────────

def _az(*args: str) -> str:
    out = subprocess.run(["az", *args, "--account-name", ACCOUNT, "--auth-mode", "login", "-o", "json"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"az failed ({out.returncode}): {out.stderr.strip()[:400]}")
    return out.stdout


def blob_results(n: int, dest: Path) -> list[Path]:
    """Download the ``n`` most recently written dev results."""
    names = json.loads(_az("storage", "blob", "list", "-c", CONTAINER, "--prefix", "deals/", "--num-results", "*",
                           "--query", f"[?ends_with(name, '{SUFFIX}')].{{name: name, t: properties.lastModified}}"))
    names.sort(key=lambda x: x.get("t") or "", reverse=True)
    paths = []
    for row in names[:n]:
        deal = row["name"].split("/")[1]
        p = dest / f"{deal}.result.json"
        _az("storage", "blob", "download", "-c", CONTAINER, "-n", row["name"], "-f", str(p), "--overwrite")
        paths.append(p)
    return paths


# ── legacy results: fires rebuilt from the marks rules leave ────────────────

def _head(atom: dict[str, Any]) -> str | None:
    v = atom.get("value") if isinstance(atom.get("value"), dict) else {}
    for k in ("c3_label", "c3_base_says"):
        lab = v.get(k)
        if isinstance(lab, dict) and lab.get("type") and lab.get("act") != "follows_copy":
            return str(lab["type"])
    return None


def _agree(head: str | None, rule: str) -> bool | None:
    if head is None:
        return None
    return head != rule[4:] if rule.startswith("not:") else head == rule


def infer_fires(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def fire(rid: str, aid: str, head: str | None, rule: str) -> None:
        r = out.setdefault(rid, {"fires": 0, "agree_with_head": 0, "disagree": 0, "no_head": 0,
                                 "mode": "active", "examples": []})
        ok = _agree(head, rule)
        r["fires"] += 1
        r["agree_with_head" if ok else "disagree" if ok is False else "no_head"] += 1
        if aid and len(r["examples"]) < 5:
            r["examples"].append(aid)

    for a in d.get("atoms") or []:
        aid = str(a.get("id") or "")
        flags = set(a.get("review_flags") or [])
        v = a.get("value") if isinstance(a.get("value"), dict) else {}
        lab = v.get("c3_label") if isinstance(v.get("c3_label"), dict) else {}
        t = str(a.get("atom_type") or "")
        if "party_address" in flags and "site_kind_before_veto" in v:
            fire("party_address_veto.signature_page", aid, _head(a), "not:physical_site")
        if lab.get("act") == "follows_copy":
            # The label is the rule's; the head's own read of this copy is gone (it said physical_site to be here).
            fire("party_address_veto.follow_party_copies", aid, "physical_site", "stakeholder")
        if "geo_fallback_site" in flags:
            fire("site_geo_fallback.address", aid, None, "physical_site")
        if "geo_mention_site" in flags:
            fire("site_geo_fallback.mention", aid, None, "physical_site")
        if "site_not_backed_by_heads" in flags:
            fire("site_keys.drop_unbacked", aid, _head(a), "not:physical_site")
        if "address_lines_joined" in flags:
            fire("site_keys.join_split_address", aid, None, "same_site")
        if "street_copy_of_site" in flags:
            fire("site_keys.street_copy", aid, None, "same_site")
        if "site_key_from_head" in flags:
            fire("site_keys.mint_from_head", aid, _head(a) or "physical_site", "physical_site")
        if "kept_over_site_dedup" in flags and t != "physical_site":
            fire("semantic_dedup.site_demote", aid, _head(a), "not:physical_site")
    for a in d.get("suppressed_atoms") or []:
        sup = ((a.get("value") or {}).get("_suppression") or {}) if isinstance(a.get("value"), dict) else {}
        if sup.get("stage") == "site_geo_fallback":
            fire("site_geo_fallback.vendor_suppress", str(a.get("id") or ""), _head(a), "not:physical_site")
    q = ((d.get("c3_brief") or {}).get("quote")) or {}
    if isinstance(q, dict) and not q.get("degraded"):
        def note(field: str, by: Any) -> None:
            by = str(by or "")
            if by.startswith(("fallback:", "rule:")):
                fire(f"quote.{field}.{by}", "", None, "")
        note("billing", (q.get("billing") or {}).get("by"))
        note("job_type", q.get("job_type_by") or ("fallback:embedding_family" if q.get("job_type") else ""))
        note("sites", (q.get("sites") or {}).get("by"))
        for ln in q.get("lines") or []:
            for f, by in ((ln or {}).get("by") or {}).items():
                if by == "sheet:site_rows":
                    fire("quote.site_row_hours", "", None, "")
                else:
                    note(str(f), by)
    return out


def fires_of(d: dict[str, Any], infer: bool) -> tuple[dict[str, dict[str, Any]], str]:
    got = d.get("rule_fires")
    if not isinstance(got, dict):
        got = ((d.get("c3_brief") or {}).get("decided_by") or {}).get("rule_fires")
    if isinstance(got, dict):
        return got, "ledger"
    if infer:
        return infer_fires(d), "inferred"
    return {}, "none"


# ── summing and staging ──────────────────────────────────────────────────────

def stage(r: dict[str, Any], min_deals: int, agree: float) -> str:
    answered = r["agree_with_head"] + r["disagree"]
    if not answered:
        return "Shadow (no head answers it)" if r["modes"] == {"shadow"} else "Active (no head answers it)"
    ok = r["deals"] >= min_deals and r["agree_with_head"] / answered >= agree
    if r["modes"] == {"shadow"}:
        return "Retire" if ok else "Shadow"
    return "ready for Shadow" if ok else "Active"


def summarize(paths: list[Path], infer: bool = True) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rules: dict[str, dict[str, Any]] = defaultdict(lambda: {"fires": 0, "agree_with_head": 0, "disagree": 0,
                                                             "no_head": 0, "deals": 0, "inferred_deals": 0,
                                                             "modes": set(), "examples": []})
    sources: dict[str, int] = defaultdict(int)
    for p in paths:
        try:
            d = json.loads(Path(p).read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"skip {p}: {type(exc).__name__}: {exc}", file=sys.stderr)
            sources["unreadable"] += 1
            continue
        got, src = fires_of(d, infer)
        sources[src] += 1
        deal = str(d.get("project_id") or Path(p).name.split(".")[0])
        for rid, r in got.items():
            s = rules[rid]
            for k in ("fires", "agree_with_head", "disagree", "no_head"):
                s[k] += int(r.get(k) or 0)
            s["deals"] += 1
            s["inferred_deals"] += src == "inferred"
            s["modes"].add(str(r.get("mode") or "active").split(" ", 1)[0])
            s["stage"] = str(r.get("stage") or _stage_of(rid))
            for ex in r.get("examples") or []:
                if len(s["examples"]) < 3:
                    s["examples"].append(f"{deal}:{ex}")
    return dict(rules), dict(sources)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="local result.json files")
    ap.add_argument("--blob", type=int, default=0, help="also read the N latest dev results from blob (az login)")
    ap.add_argument("--min-deals", type=int, default=20)
    ap.add_argument("--agree", type=float, default=0.95)
    ap.add_argument("--no-infer", action="store_true", help="skip results written before the ledger")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = ap.parse_args(argv)
    paths = [Path(p) for p in a.paths]
    tmp = None
    if a.blob:
        tmp = tempfile.TemporaryDirectory(prefix="rule_ledger_")
        paths += blob_results(a.blob, Path(tmp.name))
    rules, sources = summarize(paths, infer=not a.no_infer)
    rows = []
    for rid, r in sorted(rules.items(), key=lambda kv: (kv[1].get("stage") != "downstream", -kv[1]["fires"])):
        answered = r["agree_with_head"] + r["disagree"]
        rows.append({"rule": rid, "fires": r["fires"], "agree_pct": round(100 * r["agree_with_head"] / answered, 1)
                     if answered else None, "answered": answered, "no_head": r["no_head"], "deals": r["deals"],
                     "inferred_deals": r["inferred_deals"], "mode": "/".join(sorted(r["modes"])),
                     "where": r.get("stage") or _stage_of(rid),
                     "stage": stage(r, a.min_deals, a.agree), "examples": r["examples"]})
    if a.json:
        print(json.dumps({"results": sources, "rules": rows}, indent=2))
        return 0
    print(f"results: {len(paths)} ({', '.join(f'{k} {v}' for k, v in sorted(sources.items()))})")
    print(f"stage bar: head agrees on >= {a.agree:.0%} of answered fires over >= {a.min_deals} deals\n")
    w = max([len(r["rule"]) for r in rows] + [4])
    for where, title in (("downstream", "DOWNSTREAM: edits a fact the heads decided -- to remove (refit labels, or "
                                        "move the rule upstream)"),
                         ("upstream", "UPSTREAM: acts before or at the heads' decision")):
        part = [r for r in rows if r["where"] == where]
        if not part:
            continue
        print(title)
        print(f"  {'rule':<{w}}  {'fires':>6}  {'agree%':>6}  {'n_head':>6}  {'no_head':>7}  {'deals':>5}  {'infer':>5}  "
              f"{'mode':<7}  stage")
        for r in part:
            pct = "-" if r["agree_pct"] is None else f"{r['agree_pct']:.1f}"
            print(f"  {r['rule']:<{w}}  {r['fires']:>6}  {pct:>6}  {r['answered']:>6}  {r['no_head']:>7}  "
                  f"{r['deals']:>5}  {r['inferred_deals']:>5}  {r['mode']:<7}  {r['stage']}")
        print()
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
