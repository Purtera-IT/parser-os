"""PUR-15 gate: a correction on deal A must change deal B and leave deal C alone.

For every A/B/C triple (``tests/fixtures/paired_deals``):

1. start from an empty store (no training deals are added -- standing rule),
2. propose A, apply the single-field correction the triple names, store the
   lesson,
3. propose B and C before and after.

A triple PASSES when B's corrected field changed AND C's did not. Both halves
matter: a key loose enough to change everything passes the first half and fails
the second, which is the failure the wording key has.

The gate reports each key mode side by side so the wording key's number is the
baseline the work-shape key has to beat. The embedder is the offline hashed
bag-of-words (:mod:`app.eval.offline_embedder`) unless one is injected: on
synthetic triples this measures the KEY, not the production embedder. Real
cross-deal transfer must be re-measured on real pairs (format:
``docs/LESSON_KEYS.md``) with the pipeline embedder.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.core import estimator_head as eh
from app.core import work_shape as ws

DEFAULT_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "paired_deals" / "triples.v1.json"

_REQUIRED_DEAL_KEYS = ("deal_id", "wording", "work_order")


@dataclass
class TripleResult:
    id: str
    field: str
    key_mode: str
    b_before: float
    b_after: float
    c_before: float
    c_after: float
    b_changed: bool
    c_changed: bool
    #: Deals from OTHER triples the lesson changed. Not part of pass/fail per
    #: triple (another triple's deal can genuinely share the shape), but a
    #: long list here is a key loose enough to change everything.
    collateral: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.b_changed and not self.c_changed


@dataclass
class GateReport:
    key_mode: str
    results: list[TripleResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def transfer_rate(self) -> float:
        return sum(r.b_changed for r in self.results) / self.n if self.n else 0.0

    @property
    def leak_rate(self) -> float:
        return sum(r.c_changed for r in self.results) / self.n if self.n else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / self.n if self.n else 0.0

    @property
    def collateral(self) -> int:
        return sum(len(r.collateral) for r in self.results)

    @property
    def passed(self) -> bool:
        return self.n > 0 and all(r.passed for r in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key_mode": self.key_mode,
            "n": self.n,
            "transfer_rate": round(self.transfer_rate, 4),
            "leak_rate": round(self.leak_rate, 4),
            "pass_rate": round(self.pass_rate, 4),
            "collateral_changes": self.collateral,
            "passed": self.passed,
            "triples": [{**asdict(r), "passed": r.passed} for r in self.results],
        }


def validate_triples(doc: dict) -> list[str]:
    """Problems with a fixture document; empty means usable. Same checks apply
    to real pairs."""
    problems: list[str] = []
    triples = doc.get("triples")
    if not isinstance(triples, list) or not triples:
        return ["no triples"]
    seen: set[str] = set()
    for t in triples:
        tid = str(t.get("id") or "")
        if not tid or tid in seen:
            problems.append(f"missing or duplicate id {tid!r}")
        seen.add(tid)
        corr = t.get("correction") or {}
        if corr.get("field") not in eh.FIELDS:
            problems.append(f"{tid}: correction.field must be one of {eh.FIELDS}")
        for role in ("a", "b", "c"):
            d = t.get(role) or {}
            for k in _REQUIRED_DEAL_KEYS:
                if not d.get(k):
                    problems.append(f"{tid}.{role}: missing {k}")
        a, b, c = (t.get(r) or {} for r in ("a", "b", "c"))
        ids = {a.get("deal_id"), b.get("deal_id"), c.get("deal_id")}
        if len(ids) != 3:
            problems.append(f"{tid}: A, B and C must be three different deals")
        if a.get("wording") and a.get("wording") == b.get("wording"):
            problems.append(f"{tid}: B must not share A's wording")
        if b.get("customer") and b.get("customer") == a.get("customer"):
            problems.append(f"{tid}: B must not share A's customer")
    return problems


def load_triples(path: str | Path | None = None) -> dict:
    doc = json.loads(Path(path or DEFAULT_FIXTURES).read_text())
    problems = validate_triples(doc)
    if problems:
        raise ValueError("invalid triples: " + "; ".join(problems))
    return doc


def run_triple(
    triple: dict,
    key_mode: str,
    store_factory: Callable[[], Any],
    corpus: list[dict] | None = None,
) -> TripleResult:
    store = store_factory()
    a, b, c = triple["a"], triple["b"], triple["c"]
    corr = triple["correction"]
    fld, li = corr["field"], int(corr.get("line_index", 0))
    b0 = eh.propose(b, store, key_mode=key_mode).field_value(li, fld)
    c0 = eh.propose(c, store, key_mode=key_mode).field_value(li, fld)
    prop_a = eh.propose(a, store, key_mode=key_mode)
    lesson = eh.lesson_from_override(
        a, prop_a, line_index=li, field_name=fld, accepted=float(corr["accepted"]),
        reason_code=str(corr.get("reason_code") or "other"),
        reason_text=str(corr.get("reason_text") or ""), key_mode=key_mode,
    )
    collateral: list[str] = []
    if corpus:
        own = {a["deal_id"], b["deal_id"], c["deal_id"]}
        collateral = sorted({
            ch["deal_id"] for ch in eh.preview_transfer(store, lesson, corpus, key_mode=key_mode)
            if ch["deal_id"] not in own
        })
    store.add(lesson)
    b1 = eh.propose(b, store, key_mode=key_mode).field_value(li, fld)
    c1 = eh.propose(c, store, key_mode=key_mode).field_value(li, fld)
    return TripleResult(
        id=triple["id"], field=fld, key_mode=key_mode,
        b_before=b0, b_after=b1, c_before=c0, c_after=c1,
        b_changed=abs(b1 - b0) > 1e-9, c_changed=abs(c1 - c0) > 1e-9,
        collateral=collateral,
    )


def run_gate(
    doc: dict | None = None,
    *,
    key_modes: tuple[str, ...] = (ws.MODE_WORDING, ws.MODE_WORK_SHAPE),
    store_factory: Callable[[], Any] | None = None,
) -> dict[str, GateReport]:
    if doc is None:
        doc = load_triples()
    if store_factory is None:
        from app.eval.offline_embedder import offline_store

        store_factory = offline_store
    corpus = [t[r] for t in doc["triples"] for r in ("a", "b", "c")]
    return {
        mode: GateReport(mode, [run_triple(t, mode, store_factory, corpus) for t in doc["triples"]])
        for mode in key_modes
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="PUR-15 cross-deal transfer gate (offline, fixtures only)")
    ap.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    ap.add_argument("--gate-mode", default=ws.MODE_WORK_SHAPE, choices=[ws.MODE_WORDING, ws.MODE_WORK_SHAPE],
                    help="the key mode whose result decides the exit code")
    ap.add_argument("--min-pass-rate", type=float, default=1.0,
                    help="exit 0 only when the gate mode passes at least this share of triples "
                         "with zero leaks and zero collateral (default: all)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    reports = run_gate(load_triples(args.fixtures))
    if args.json:
        print(json.dumps({m: r.as_dict() for m, r in reports.items()}, indent=2))
    else:
        for mode, r in reports.items():
            print(f"{mode:>10}: transfer {r.transfer_rate:.0%}  leak {r.leak_rate:.0%}  "
                  f"pass {r.pass_rate:.0%} ({sum(x.passed for x in r.results)}/{r.n})  "
                  f"collateral {r.collateral}")
            for x in r.results:
                if not x.passed:
                    print(f"            FAIL {x.id}: B {x.b_before:g}->{x.b_after:g}  C {x.c_before:g}->{x.c_after:g}")
                if x.collateral:
                    print(f"            COLLATERAL {x.id}: {', '.join(x.collateral)}")
    g = reports[args.gate_mode]
    ok = g.n > 0 and g.pass_rate >= args.min_pass_rate and g.leak_rate == 0 and g.collateral == 0
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
