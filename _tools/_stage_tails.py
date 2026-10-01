"""Per-stage: how much of a stage the counter cannot see, and how well it predicts.

`compileEta.ts` carries ONE tail allowance, 0.143, and it was measured on
`typed_atom_classification` alone -- the only stage that had an interior
counter. Seven stages have one now, and there is no reason their uncounted
fractions should match:

  * `enrich_entities` builds the authoritative-site catalog over the whole
    corpus BEFORE its loop starts, so its uncounted work is at the FRONT.
  * `pdf_image_vision` has a time budget and a host breaker that can cut its
    loop short, so its counter can stop well below its total.
  * `open_question_resolution` gathers candidates first and then does one
    decide() each -- its head and tail should both be small.
  * `document_job_scope` counts bundles, which are coarse: a deal of three
    documents has three ticks, and one of them is most of the stage.

A single constant across all of those is the same mistake as a single
denominator across two phases, which #258 measured at 156% error. So this
reports, per stage:

    head      the fraction of the stage BEFORE the counter's first tick
    tail      the fraction AFTER the counter reached its total
    reach     how close to its total the counter actually got
    error     the panel's own estimate scored, as compileEta.ts computes it

Run after a sampling pass:

    ETA_SAMPLES=D:/temp/claude/eta_counters.json python _tools/_stage_tails.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

OUT = Path(os.environ.get("ETA_SAMPLES", r"D:/temp/claude/eta_counters.json"))

# compileEta.ts, mirrored. A score computed under different gates is a score of
# something nobody is shown.
RATE_WINDOW_MS = 45_000
MIN_WINDOW_MS = 4_000
MIN_ITEMS_IN_WINDOW = 2
NEARLY_DONE_ITEMS = 2
NEARLY_DONE_FRACTION = 0.05
TAIL_ALLOWANCE = 0.143


def _load() -> list[dict]:
    return json.loads(OUT.read_text(encoding="utf-8"))


def main() -> None:
    log = _load()
    by_compile: dict[str, list[dict]] = {}
    for s in sorted(log, key=lambda x: x["t"]):
        if s.get("compile"):
            by_compile.setdefault(s["compile"], []).append(s)

    # Readings that carry a counter, grouped per (compile, stage).
    runs: dict[tuple, list[dict]] = {}
    for s in log:
        if s.get("total") and s.get("done") is not None:
            runs.setdefault((s["compile"], s["stage"]), []).append(s)

    per_stage: dict[str, list[dict]] = {}
    for (cid, stage), samples in runs.items():
        samples.sort(key=lambda x: x["t"])
        series = by_compile.get(cid, ())
        # The stage's boundaries, OBSERVED: the last reading before this stage
        # was first seen, and the first reading after it was last seen. Derived
        # stage times are wrong -- elapsed_ms is wall clock while
        # stages[].duration_ms excludes the gaps between stages.
        first_t = samples[0]["t"]
        exit_t = next((x["t"] for x in series if x["t"] > samples[-1]["t"] and x.get("stage") != stage), None)
        enter_t = None
        for x in series:
            if x["t"] >= first_t:
                break
            if x.get("stage") == stage:
                enter_t = min(enter_t, x["t"]) if enter_t else x["t"]
        if exit_t is None:
            continue  # never seen to finish: nothing to measure against

        total = samples[0]["total"]
        # When the counter first MOVED, and when it reached its total.
        moved_t = next((x["t"] for x in samples if x["done"] > 0), None)
        full_t = next((x["t"] for x in samples if x["done"] >= total), None)
        reached = max(x["done"] for x in samples) / total if total else 0.0

        counted_ms = (full_t - (moved_t or first_t)) if full_t else (samples[-1]["t"] - (moved_t or first_t))
        head_ms = (moved_t - (enter_t or first_t)) if moved_t else 0
        tail_ms = (exit_t - full_t) if full_t else 0

        # Score the estimate exactly as the panel would, including its silence.
        errs = []
        for i, now in enumerate(samples):
            done, tot = now["done"], now["total"]
            left = tot - done
            if left <= NEARLY_DONE_ITEMS or left < tot * NEARLY_DONE_FRACTION:
                continue
            w = [x for x in samples[: i + 1] if now["t"] - x["t"] <= RATE_WINDOW_MS]
            if len(w) < 2:
                continue
            d_items = w[-1]["done"] - w[0]["done"]
            d_ms = w[-1]["t"] - w[0]["t"]
            if d_items < MIN_ITEMS_IN_WINDOW or d_ms < MIN_WINDOW_MS:
                continue
            rate_rem = left * (d_ms / d_items)
            counted_phase = (now["t"] - samples[0]["t"]) + rate_rem
            pred = rate_rem + TAIL_ALLOWANCE * counted_phase
            actual = exit_t - now["t"]
            if actual < 3_000:
                continue
            errs.append(abs(pred - actual) / actual)

        per_stage.setdefault(stage, []).append({
            "compile": cid, "total": total, "reached": reached,
            "counted_s": counted_ms / 1000.0, "head_s": head_ms / 1000.0, "tail_s": tail_ms / 1000.0,
            "head_frac": (head_ms / counted_ms) if counted_ms > 0 else None,
            "tail_frac": (tail_ms / counted_ms) if counted_ms > 0 else None,
            "errs": errs,
        })

    if not per_stage:
        print("no stage finished within the sample -- nothing to measure")
        return

    rows = []
    print(f"{'stage':30} {'n':>2} {'counted_s':>9} {'head':>7} {'tail':>7} {'reach':>6} {'err%':>6} {'est':>4}")
    print("-" * 82)
    for stage, obs in sorted(per_stage.items(), key=lambda kv: -statistics.median(o["counted_s"] for o in kv[1])):
        heads = [o["head_frac"] for o in obs if o["head_frac"] is not None]
        tails = [o["tail_frac"] for o in obs if o["tail_frac"] is not None]
        errs = [e for o in obs for e in o["errs"]]
        med = lambda v: statistics.median(v) if v else float("nan")  # noqa: E731
        print(f"{stage:30} {len(obs):>2} {med([o['counted_s'] for o in obs]):>9.1f} "
              f"{med(heads):>7.3f} {med(tails):>7.3f} {med([o['reached'] for o in obs]):>6.2f} "
              f"{(statistics.median(errs)*100 if errs else float('nan')):>6.1f} {len(errs):>4}")
        rows.append((stage, med(tails), len(errs)))
    print("-" * 82)
    print(f"\nthe panel's single allowance is {TAIL_ALLOWANCE}. Per stage, the measured tail is:")
    for stage, tail, n in rows:
        if tail != tail:  # nan
            continue
        verdict = "about right" if abs(tail - TAIL_ALLOWANCE) < 0.05 else ("TOO LOW" if tail > TAIL_ALLOWANCE else "too high")
        print(f"   {stage:30} {tail:5.3f}   ({verdict}, {n} scored readings)")
    allerrs = [e for obs in per_stage.values() for o in obs for e in o["errs"]]
    if allerrs:
        f = lambda p: sum(1 for e in allerrs if e <= p) * 100 / len(allerrs)  # noqa: E731
        print(f"\nall stages together: n={len(allerrs)} median={statistics.median(allerrs)*100:.1f}% "
              f"within25={f(0.25):.0f}% within50={f(0.5):.0f}% within2x={f(1.0):.0f}%")


if __name__ == "__main__":
    sys.exit(main())
