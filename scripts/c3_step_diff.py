"""Did turning a C3 step on lose anything? One deal, step off vs on.

    python scripts/c3_step_diff.py --off off.json --on on.json \\
        [--off-again off2.json] [--out diff.json]

Each file is a compile output (``CompileResult`` JSON) of the same deal and
the same documents. Atoms are matched by document, location and normalized
text, not by id, so two compiles of one deal line up. The report:

* ``by_type``: atom counts per type, off and on.
* ``lost`` / ``gained``: atoms in one compile and not the other, per type.
* ``retyped``: matched atoms whose type changed, as ``off->on`` pairs.
* ``protected_lost``: lost atoms of a type a PM would miss (``--protected``).

The LLM does not give the same answer twice, so a second compile with the
step off (``--off-again``) measures that churn. The step passes when it loses
no more protected atoms than the off compiles lose to each other (none, with
no second compile). Counts and atom ids only, never text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROTECTED = ("requirement", "task", "exclusion", "physical_site", "bom_line", "constraint",
             "dependency", "acceptance_criterion", "contract_term", "submission_req")


def _key(atom: dict[str, Any]) -> str:
    refs = sorted((str(r.get("filename") or ""), json.dumps(r.get("locator") or {}, sort_keys=True))
                  for r in atom.get("source_refs") or [])
    raw = json.dumps([refs, " ".join(str(atom.get("normalized_text") or "").split()).lower()])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def atoms_of(compile_json: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """match key -> (atom type, atom id)."""
    out: dict[str, tuple[str, str]] = {}
    for a in compile_json.get("atoms") or []:
        out.setdefault(_key(a), (str(a.get("atom_type") or ""), str(a.get("id") or "")))
    return out


def diff(off: dict[str, tuple[str, str]], on: dict[str, tuple[str, str]],
         protected: tuple[str, ...] = PROTECTED) -> dict[str, Any]:
    lost = [off[k] for k in off.keys() - on.keys()]
    gained = [on[k] for k in on.keys() - off.keys()]
    retyped = Counter(f"{off[k][0]}->{on[k][0]}" for k in off.keys() & on.keys() if off[k][0] != on[k][0])
    plost = [(t, i) for t, i in lost if t in protected]
    return {
        "atoms": {"off": len(off), "on": len(on), "matched": len(off.keys() & on.keys())},
        "by_type": _by_type(off, on),
        "lost": dict(Counter(t for t, _ in lost)), "gained": dict(Counter(t for t, _ in gained)),
        "retyped": dict(retyped),
        "protected_lost": len(plost), "protected_lost_ids": sorted(i for _, i in plost)[:200],
    }


def _by_type(off, on) -> dict[str, dict[str, int]]:
    a, b = Counter(v[0] for v in off.values()), Counter(v[0] for v in on.values())
    return {t: {"off": a[t], "on": b[t]} for t in sorted(a.keys() | b.keys())}


def verdict(step: dict[str, Any], churn: dict[str, Any] | None) -> tuple[bool, str]:
    allowed = churn["protected_lost"] if churn else 0
    ok = step["protected_lost"] <= allowed
    return ok, (f"protected atoms lost with the step on: {step['protected_lost']}, "
                f"lost between two off compiles: {allowed if churn else 'not measured (0 allowed)'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--off", type=Path, required=True)
    p.add_argument("--on", type=Path, required=True)
    p.add_argument("--off-again", type=Path, help="a second compile with the step off, for LLM churn")
    p.add_argument("--protected", default=",".join(PROTECTED))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    load = lambda f: atoms_of(json.loads(f.read_text(encoding="utf-8")))  # noqa: E731
    protected = tuple(t.strip() for t in args.protected.split(",") if t.strip())
    off, on = load(args.off), load(args.on)
    step = diff(off, on, protected)
    churn = diff(off, load(args.off_again), protected) if args.off_again else None
    ok, why = verdict(step, churn)
    report = {"step": step, "churn": churn, "pass": ok, "why": why}
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: step[k] for k in ("atoms", "lost", "gained", "retyped", "protected_lost")}))
    print(("PASS " if ok else "FAIL ") + why)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
