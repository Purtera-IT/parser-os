"""Offline measurement for the work_order stage (PUR-31, PUR-46).

Nothing here talks to a database, blob store or cloud resource: it compiles
local deal directories through a caller-supplied ``compile_fn`` and compares
the results. The scripts ``scripts/work_order_repro.py`` and
``scripts/work_order_corpus_diff.py`` are thin CLIs over these functions, and
the tests drive them with a fake compile so they run without an LLM.

Two questions it answers:

* **Reproducibility (PUR-31).** Compile the same deal N times with the flag on;
  how much do the work lines agree, exactly (``work_line_agreement``) and by
  meaning (``soft_work_line_agreement``, which stops counting a paraphrase such
  as "Update documentation" / "Update Access One documentation" as a
  difference)? Aggregated per corpus by ``summarize_repro``.
* **Flag off vs on (PUR-46).** Compile each deal once with the flag off and N
  times with it on; which atoms appear or disappear, how many work lines are
  minted, how long it takes, and does every minted line still point at an
  artifact that exists in the compile (``diff_deal``).
"""
from __future__ import annotations

import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from app.core import work_order

FLAG = "SOWSMITH_WORK_ORDER"
CompileFn = Callable[[Path], Any]

# A deal whose atom count moves by more than this fraction is flagged for a
# closer look (PUR-46: "look hardest at deals where the count moves a lot").
BIG_DELTA_FRACTION = 0.25
# Many atoms reduced to at most this many work lines is the Barton Malow shape
# (1,173 atoms -> one line): either the point of the stage or a bug.
COLLAPSE_MIN_ATOMS = 200
COLLAPSE_MAX_LINES = 1
SOFT_MATCH_THRESHOLD = 0.5


# ---------------------------------------------------------------- discovery

def list_deals(corpus: Path, include: Iterable[str] = (), exclude: Iterable[str] = ()) -> list[Path]:
    """Deal directories directly under ``corpus``, sorted by name.

    A directory counts as a deal if it contains at least one file. Hidden
    directories are skipped. ``include`` / ``exclude`` match directory names.
    """
    inc = {x for x in include if x}
    exc = {x for x in exclude if x}
    deals = []
    for p in sorted(Path(corpus).iterdir(), key=lambda q: q.name):
        if not p.is_dir() or p.name.startswith((".", "_")):
            continue
        if inc and p.name not in inc:
            continue
        if p.name in exc:
            continue
        if any(f.is_file() for f in p.rglob("*")):
            deals.append(p)
    return deals


# ---------------------------------------------------------------- flag

class flag_env:
    """Context manager: set ``SOWSMITH_WORK_ORDER`` for one compile, then restore."""

    def __init__(self, on: bool):
        self.on = on
        self._prev: str | None = None

    def __enter__(self):
        self._prev = os.environ.get(FLAG)
        os.environ[FLAG] = "1" if self.on else "0"
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = self._prev
        return False


# ---------------------------------------------------------------- stub LLM

def stub_complete(prompt: str, **_: object) -> str:
    """Deterministic stand-in for the extraction call: one line per kept document."""
    docs = re.findall(r"^\[([^\]]+)\]", prompt, flags=re.M)
    return json.dumps(
        {
            "work_lines": [
                {"work": f"deliver the work described in {d}", "object": d, "count": None, "unit": None}
                for d in docs
            ],
            "site_count": None,
            "after_hours": False,
            "no_onsite_hands": False,
            "customer_supplies_equipment": False,
            "one_line_summary": "stub",
        }
    )


def install_stub_llm() -> None:
    from app.core import llm_client

    llm_client.complete = stub_complete  # type: ignore[assignment]
    os.environ.setdefault("SOWSMITH_WORK_ORDER_MIN_BODY", "1")


# ---------------------------------------------------------------- agreement

def _norm(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).lower()).strip()


def _tok(label: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(label)))


def _soft_pair(a: set[str], b: set[str], threshold: float) -> float:
    """Jaccard over lines where near-paraphrases count as the same line.

    Greedy one-to-one matching of lines whose token Jaccard is >= threshold.
    """
    if not a and not b:
        return 1.0
    a_l, b_l = sorted(a), sorted(b)
    pairs = []
    for i, x in enumerate(a_l):
        tx = _tok(x)
        for j, y in enumerate(b_l):
            ty = _tok(y)
            u = tx | ty
            s = len(tx & ty) / len(u) if u else 1.0
            if s >= threshold:
                pairs.append((-s, i, j))
    pairs.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = 0
    for _, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched += 1
    return matched / (len(a_l) + len(b_l) - matched)


def soft_work_line_agreement(runs: list[list[str]], threshold: float = SOFT_MATCH_THRESHOLD) -> float:
    """Mean pairwise agreement where paraphrased lines match (see ``_soft_pair``)."""
    sets = [{_norm(x) for x in r} for r in runs]
    if len(sets) < 2:
        return 1.0
    scores = [
        _soft_pair(sets[i], sets[j], threshold)
        for i in range(len(sets))
        for j in range(i + 1, len(sets))
    ]
    return sum(scores) / len(scores)


def repro_row(deal_id: str, runs: list[list[str]], seconds: list[float] | None = None) -> dict[str, Any]:
    """``work_order.deal_agreement`` plus meaning-level agreement and line-count spread."""
    row = work_order.deal_agreement(deal_id, runs)
    counts = [len(set(_norm(x) for x in r)) for r in runs]
    row["soft_agreement"] = round(soft_work_line_agreement(runs), 4)
    row["line_counts"] = counts
    row["line_count_spread"] = (max(counts) - min(counts)) if counts else 0
    if seconds is not None:
        row["seconds"] = [round(s, 2) for s in seconds]
    return row


def summarize_repro(rows: list[dict[str, Any]], threshold: float = 0.9) -> dict[str, Any]:
    """Corpus-level spread: the number PUR-47 needs to set its gate."""
    if not rows:
        return {"deals": 0}
    agr = [r["agreement"] for r in rows]
    soft = [r.get("soft_agreement", r["agreement"]) for r in rows]
    return {
        "deals": len(rows),
        "runs_per_deal": max(r["runs"] for r in rows),
        "mean_agreement": round(statistics.fmean(agr), 4),
        "median_agreement": round(statistics.median(agr), 4),
        "min_agreement": round(min(agr), 4),
        "mean_soft_agreement": round(statistics.fmean(soft), 4),
        "min_soft_agreement": round(min(soft), 4),
        "max_line_count_spread": max(r.get("line_count_spread", 0) for r in rows),
        "threshold": threshold,
        "deals_below_threshold": sorted(r["deal_id"] for r in rows if r["agreement"] < threshold),
        "deals_below_threshold_soft": sorted(
            r["deal_id"] for r in rows if r.get("soft_agreement", r["agreement"]) < threshold
        ),
    }


def run_repro(deals: list[Path], compile_fn: CompileFn, runs: int = 5) -> list[dict[str, Any]]:
    rows = []
    for deal in deals:
        labels, secs = [], []
        for _ in range(max(1, runs)):
            with flag_env(True):
                t0 = time.perf_counter()
                result = compile_fn(deal)
                secs.append(time.perf_counter() - t0)
            labels.append(work_order.work_line_labels(list(result.atoms)))
        rows.append(repro_row(deal.name, labels, secs))
    return rows


# ---------------------------------------------------------------- off vs on

def _is_work_line(atom: Any) -> bool:
    return (getattr(atom, "value", None) or {}).get("backfill_reason") == "work_order"


def _atom_type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "")


def atom_fingerprint(atom: Any) -> tuple[str, str, str]:
    """Identity that survives id churn: artifact, type, normalized text."""
    text = getattr(atom, "normalized_text", None) or getattr(atom, "raw_text", "") or ""
    return (str(getattr(atom, "artifact_id", "")), _atom_type(atom), _norm(text))


def _type_counts(atoms: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in atoms:
        t = _atom_type(a)
        out[t] = out.get(t, 0) + 1
    return dict(sorted(out.items()))


def provenance_problems(atoms: list[Any], known_artifacts: set[str]) -> list[dict[str, Any]]:
    """Minted work lines whose provenance does not resolve to a real artifact."""
    problems = []
    for a in atoms:
        if not _is_work_line(a):
            continue
        refs = list(getattr(a, "source_refs", None) or [])
        ref_ids = {str(getattr(r, "artifact_id", "")) for r in refs}
        own = str(getattr(a, "artifact_id", ""))
        bad = sorted(x for x in (ref_ids | {own}) if x and x not in known_artifacts)
        if not refs or bad:
            problems.append(
                {"label": _norm(getattr(a, "raw_text", "")), "no_source_refs": not refs, "unknown_artifacts": bad}
            )
    return problems


def diff_deal(
    deal_id: str,
    off_atoms: list[Any],
    on_runs: list[list[Any]],
    off_seconds: float | None = None,
    on_seconds: list[float] | None = None,
    sample_lost: int = 20,
) -> dict[str, Any]:
    """Per-deal flag-off vs flag-on report. ``on_runs`` are the atoms of each on-run."""
    off_fp = {atom_fingerprint(a) for a in off_atoms}
    known = {str(getattr(a, "artifact_id", "")) for a in off_atoms}
    first_on = on_runs[0] if on_runs else []
    on_non_minted = [a for a in first_on if not _is_work_line(a)]
    on_fp = {atom_fingerprint(a) for a in on_non_minted}
    known |= {str(getattr(a, "artifact_id", "")) for a in on_non_minted}

    lost = sorted(off_fp - on_fp)
    gained_non_minted = sorted(on_fp - off_fp)
    lines = [work_order.work_line_labels(r) for r in on_runs]
    minted = len(lines[0]) if lines else 0
    n_off, n_on = len(off_atoms), len(first_on)
    delta = n_on - n_off
    frac = (abs(delta) / n_off) if n_off else (1.0 if n_on else 0.0)
    prov = provenance_problems(first_on, known)
    agreement = repro_row(deal_id, lines) if len(lines) > 1 else None

    flags = []
    if lost:
        flags.append("lost_atoms")
    if frac > BIG_DELTA_FRACTION:
        flags.append("big_atom_delta")
    if prov:
        flags.append("bad_provenance")
    if n_off >= COLLAPSE_MIN_ATOMS and minted <= COLLAPSE_MAX_LINES:
        flags.append("collapse")
    if agreement and agreement["agreement"] < 1.0:
        flags.append("unstable_work_lines")

    return {
        "deal_id": deal_id,
        "atoms_off": n_off,
        "atoms_on": n_on,
        "atom_delta": delta,
        "atom_delta_fraction": round(frac, 4),
        "atom_types_off": _type_counts(off_atoms),
        "atom_types_on": _type_counts(first_on),
        "work_lines_added": lines[0] if lines else [],
        "work_lines_added_count": minted,
        "atoms_lost_count": len(lost),
        "atoms_lost_sample": [list(x) for x in lost[:sample_lost]],
        "non_minted_atoms_gained_count": len(gained_non_minted),
        "provenance_problems": prov,
        "seconds_off": round(off_seconds, 2) if off_seconds is not None else None,
        "seconds_on": [round(s, 2) for s in on_seconds] if on_seconds is not None else None,
        "agreement": agreement,
        "flags": flags,
    }


def run_corpus_diff(
    deals: list[Path],
    compile_fn: CompileFn,
    on_runs: int = 2,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for deal in deals:
        try:
            with flag_env(False):
                t0 = time.perf_counter()
                off = list(compile_fn(deal).atoms)
                t_off = time.perf_counter() - t0
            ons, t_ons = [], []
            for _ in range(max(1, on_runs)):
                with flag_env(True):
                    t0 = time.perf_counter()
                    ons.append(list(compile_fn(deal).atoms))
                    t_ons.append(time.perf_counter() - t0)
            row = diff_deal(deal.name, off, ons, t_off, t_ons)
        except Exception as exc:  # one broken deal must not stop the corpus
            row = {"deal_id": deal.name, "error": f"{type(exc).__name__}: {exc}", "flags": ["error"]}
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def summarize_diff(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    by_flag: dict[str, list[str]] = {}
    for r in rows:
        for f in r.get("flags", []):
            by_flag.setdefault(f, []).append(r["deal_id"])
    agr = [r["agreement"]["agreement"] for r in ok if r.get("agreement")]
    return {
        "deals": len(rows),
        "errors": len(rows) - len(ok),
        "atoms_off_total": sum(r["atoms_off"] for r in ok),
        "atoms_on_total": sum(r["atoms_on"] for r in ok),
        "work_lines_total": sum(r["work_lines_added_count"] for r in ok),
        "deals_with_work_lines": sum(1 for r in ok if r["work_lines_added_count"]),
        "additive": not by_flag.get("lost_atoms"),
        "mean_agreement": round(statistics.fmean(agr), 4) if agr else None,
        "seconds_off_total": round(sum(r["seconds_off"] or 0 for r in ok), 1),
        "seconds_on_mean_total": round(
            sum(statistics.fmean(r["seconds_on"]) for r in ok if r.get("seconds_on")), 1
        ),
        "flagged": {k: sorted(v) for k, v in sorted(by_flag.items())},
    }


def render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    out = ["# work_order corpus diff (flag off vs on)", "", "## Summary", "", "```json",
           json.dumps(summary, indent=2), "```", "", "## Per deal", "",
           "| deal | atoms off | atoms on | delta | lines | lost | prov | agreement | s off | s on | flags |",
           "| -- | --: | --: | --: | --: | --: | --: | --: | --: | --: | -- |"]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['deal_id']} | | | | | | | | | | error: {r['error']} |")
            continue
        agr = r["agreement"]["agreement"] if r.get("agreement") else ""
        s_on = statistics.fmean(r["seconds_on"]) if r.get("seconds_on") else ""
        s_on = f"{s_on:.1f}" if s_on != "" else ""
        out.append(
            f"| {r['deal_id']} | {r['atoms_off']} | {r['atoms_on']} | {r['atom_delta']:+d} | "
            f"{r['work_lines_added_count']} | {r['atoms_lost_count']} | {len(r['provenance_problems'])} | "
            f"{agr} | {r['seconds_off'] if r['seconds_off'] is not None else ''} | {s_on} | "
            f"{', '.join(r['flags'])} |"
        )
    out += ["", "## Work lines by deal", ""]
    for r in rows:
        if r.get("work_lines_added"):
            out.append(f"### {r['deal_id']}")
            out += [f"- {x}" for x in r["work_lines_added"]]
            out.append("")
    return "\n".join(out) + "\n"
