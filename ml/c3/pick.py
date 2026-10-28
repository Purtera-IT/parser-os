"""The checkpoint pick: a macro over the gated heads, against the live model's
own scores.

Every in-loop check (``loop.py --check-every``) writes one row per checkpoint
to ``<ck>.checks.json``: ``valid_<key>`` (exact) and ``valid_<key>_n`` for each
head in ``loop.CHECK_HEADS``, read on the validation views. The pick:

* **composite**: the unweighted mean of the gated heads' validation exact,
  over the heads measured with at least ``MIN_N`` answers in *every* row (so
  every checkpoint is averaged over the same heads). Line type counts once,
  like admission or supplier, so a checkpoint cannot buy type with the rest.
* **floors**: each gated head must hold the live checkpoint's real score (the
  live code reading the live inputs: ``runE/headscore``), less its noise
  (``Z`` binomial standard errors at the validation n, at least ``TOL_MIN``).
  A head with fewer than ``MIN_N`` answers on either side is reported, not
  gated. The floors never come from the run's own ``start`` check: that is
  run C read under the new texts and tags, which sits below live-as-served.
* **fit gap**: ``fit_exact - valid_exact`` may not grow more than ``GAP`` over
  the start's gap (an over-fit pick).

The best passing row with a file on disk wins; the start (the live model) is
the fallback. Counts and rates only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

#: The heads the pick averages and floors (the gated heads). The hours heads
#: count where the validation views carry enough answers; the site heads have
#: a handful of cards per deal and are usually reported only.
PICK_HEADS = ("col:admission", "col:supplier", "col:about", "col:label_type",
              "read:sow_section", "read:sow_coverage", "read:co_action",
              "read:site", "jdg:site", "jdg:site_role", "jdg:facility",
              "jdg:task_tier", "jdg:hours:per", "jdg:hours:qty", "read:co_hours_estimate",
              # Run O (planning-heads contract): the deal's primary service, one card per validation view.
              "deal:primary_service")
MIN_N = 20
#: A head measured on fewer rows by nature (one deal card per validation view file) joins the composite
#: at its own minimum; it is never floored (floors keep MIN_N).
PICK_MIN_N = {"deal:primary_service": 2}
#: Composite weight per head (1 if not listed): a handful of deal cards moves the pick at most this much.
PICK_WEIGHTS = {"deal:primary_service": 0.5}
Z = 2.0
TOL_MIN = 0.005
GAP = 0.05


def col(key: str) -> str:
    return "valid_" + key.replace(":", "_")


def read_floors(path: str | Path | None) -> dict[str, dict[str, float]]:
    """``{key: {"exact", "n"}}``: a headscore ``scores.json`` (or the same shape)."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = raw.get("heads", raw) if isinstance(raw, dict) else {}
    return {k: {"exact": float(v["exact"]), "n": float(v.get("n") or 0)}
            for k, v in raw.items() if isinstance(v, dict) and v.get("exact") is not None}


def composite_heads(checks: dict[str, dict], heads=PICK_HEADS, min_n: int = MIN_N) -> list[str]:
    """Heads measured (n >= min_n) in every check row."""
    rows = list(checks.values())
    return [k for k in heads
            if rows and all(row.get(col(k)) is not None
                            and row.get(col(k) + "_n", 0) >= min(min_n, PICK_MIN_N.get(k, min_n)) for row in rows)]


def composite(row: dict, heads: list[str]) -> float:
    """The weighted mean (PICK_WEIGHTS, else 1) of the heads' validation exact on one check row."""
    got = [(PICK_WEIGHTS.get(k, 1.0), row[col(k)]) for k in heads if row.get(col(k)) is not None]
    den = sum(w for w, _ in got)
    return sum(w * v for w, v in got) / den if den else 0.0


def judge(row: dict, floors: dict, heads: list[str], start: dict | None = None, *,
          min_n: int = MIN_N, z: float = Z, tol_min: float = TOL_MIN,
          gap: float = GAP) -> tuple[bool, float, list[str]]:
    """(passes, composite, reasons it failed)."""
    comp = composite(row, heads)
    bad = []
    for k in heads:
        f = floors.get(k)
        n = row.get(col(k) + "_n", 0)
        if f is None or f["n"] < min_n or n < min_n:
            continue
        tol = max(tol_min, z * math.sqrt(max(f["exact"] * (1 - f["exact"]), 1e-6) / n))
        if row[col(k)] < f["exact"] - tol:
            bad.append(f"{k} {row[col(k)]:.3f}<{f['exact']:.3f}-{tol:.3f}")
    if start and gap is not None:
        g0 = start.get("fit_exact", 0) - start.get("valid_exact", 0)
        g = row.get("fit_exact", 0) - row.get("valid_exact", 0)
        if "fit_exact" in row and g > g0 + gap:
            bad.append(f"fit gap {g:.3f}>{g0:.3f}+{gap}")
    return not bad, comp, bad


def best_at(checks: dict[str, dict], floors: dict, exists=lambda at: True, **kw) -> str | None:
    """The passing check with a file and the highest composite; ties go to the later check
    (``at_step``, the step it was read at). None when nothing passes."""
    heads = composite_heads(checks, min_n=kw.get("min_n", MIN_N))
    start = checks.get("start")
    best, key = None, None
    for at, row in checks.items():
        ok, comp, _ = judge(row, floors, heads, start, **kw)
        k = (comp, row.get("at_step", 0.0))
        if ok and exists(at) and (key is None or k >= key):
            best, key = at, k
    return best


def pick(ck: str, init: str, floors_path: str | None, **kw) -> dict:
    checks = json.loads(Path(f"{ck}.checks.json").read_text(encoding="utf-8"))
    if "start" not in checks:
        raise SystemExit(f"pick: no start check in {ck}.checks.json")
    floors = read_floors(floors_path)
    if not floors:
        raise SystemExit("pick: no floors (FLOORS: the live checkpoint's headscore scores.json)")
    path = lambda at: init if at == "start" else f"{ck}.{at}"  # noqa: E731
    heads = composite_heads(checks, min_n=kw.get("min_n", MIN_N))
    print("pick composite over " + " ".join(heads))
    reported = [k for k in PICK_HEADS if k not in heads]
    if reported:
        print("pick reported only (n<%d somewhere, or not measured): %s" % (kw.get("min_n", MIN_N),
                                                                           " ".join(reported)))
    for k in heads:
        f = floors.get(k)
        print(f"pick floor {k:22} " + (f"{f['exact']:.4f} (live n={int(f['n'])})" if f else "none (not gated)"))
    start = checks["start"]
    rows = []
    for at, row in sorted(checks.items(), key=lambda kv: kv[1].get("at_step", 0.0)):
        ok, comp, bad = judge(row, floors, heads, start, **kw)
        here = os.path.exists(path(at))
        rows.append((at, ok, comp, here))
        print(f"pick {at:16} composite={comp:.4f} " + " ".join(
            f"{k.split(':')[-1]}={row[col(k)]:.3f}" for k in heads if row.get(col(k)) is not None)
            + (" pass" if ok else " fail: " + "; ".join(bad)) + ("" if here else " (no file)"))
    at = best_at(checks, floors, exists=lambda a: os.path.exists(path(a)), **kw) or "start"
    comp = next(r[2] for r in rows if r[0] == at)
    print(f"picked {at} {path(at)} composite={comp:.4f}" + (" (fallback: nothing passed)"
                                                              if not next(r[1] for r in rows if r[0] == at) else ""))
    out = {"at": at, "path": path(at), "composite": comp, "heads": heads,
           "floors": {k: floors[k] for k in heads if k in floors}, "floors_file": str(floors_path),
           "rows": {r[0]: {"pass": r[1], "composite": r[2], "file": r[3]} for r in rows}}
    Path(f"{ck}.pick.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    link = f"{ck}.best"
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.realpath(path(at)), link)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True, help="<ck>: reads <ck>.checks.json, writes <ck>.pick.json")
    ap.add_argument("--init", required=True, help="the start checkpoint (the fallback)")
    ap.add_argument("--floors", required=True, help="the live model's per-head scores (scores.json shape)")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--z", type=float, default=Z)
    ap.add_argument("--gap", type=float, default=GAP)
    a = ap.parse_args()
    pick(a.checkpoint, a.init, a.floors, min_n=a.min_n, z=a.z, gap=a.gap)


if __name__ == "__main__":
    main()
