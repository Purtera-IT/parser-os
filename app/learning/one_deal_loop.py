"""One deal at a time: predict, correct like a PM, say why, learn, re-predict.

PUR-16 / PUR-57. One named deal, one field at a time, no bulk runs:

    predict   show every estimate field with the evidence behind it
              (work shape, the wording it came from, baseline or which lesson)
    correct   take one field's accepted value plus a typed reason, turn it into
              a lesson, and PREVIEW which other deals it would change. Nothing
              is written unless --commit is given. With --commit: store the
              lesson, record the override against the proposal id/version
              (PUR-54), refresh the store's cached heads/prototypes, and
              re-predict the deal.

The preview is the point: an empty or absurd list is the cheapest check that
the key is wrong.

Runs offline on fixtures (``tests/fixtures/paired_deals``) with the hashed
embedder; it never touches a remote database, blob storage or the network.

    python -m app.learning.one_deal_loop predict --deal SYN-A01
    python -m app.learning.one_deal_loop correct --deal SYN-A01 \\
        --field hours_per_visit --value 18 --reason-code other \\
        --reason-text "lift and home run per camera" --key-mode work_shape
    ... --commit --store /tmp/x/lessons.sqlite --overrides /tmp/x/overrides.sqlite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.core import estimator_head as eh
from app.core import work_shape as ws
from app.core.override_log import REASON_CODES, OverrideLog, OverrideRecord


def load_corpus(path: str | Path | None) -> list[dict]:
    """Deals from a triples fixture file, a JSON list of deals, or one deal."""
    from app.eval.transfer_gate import DEFAULT_FIXTURES

    doc = json.loads(Path(path or DEFAULT_FIXTURES).read_text())
    if isinstance(doc, dict) and "triples" in doc:
        deals: list[dict] = []
        for t in doc["triples"]:
            deals.extend(t[r] for r in ("a", "b", "c"))
        return deals
    if isinstance(doc, list):
        return doc
    return [doc]


def find_deal(ref: str, corpus: list[dict]) -> dict:
    p = Path(ref)
    if p.suffix == ".json" and p.exists():
        return json.loads(p.read_text())
    for d in corpus:
        if str(d.get("deal_id")) == ref:
            return d
    raise SystemExit(f"deal {ref!r} not found (pass a deal JSON path or a deal_id in --corpus)")


def open_store(path: str | None):
    from app.eval.offline_embedder import offline_store

    return offline_store(path or ":memory:")


def refresh(store: Any) -> None:
    """Drop every cached head and prototype so the next resolve refits on the
    store as it now is. This is the 'retrain' of a store-backed head: the lesson
    is live on the next prediction, no nightly run needed."""
    try:
        store._heads.clear()
        store._proto_dirty = True
    except AttributeError:  # pragma: no cover
        pass


def render(proposal: eh.Proposal) -> dict[str, Any]:
    out = proposal.as_dict()
    return out


def cmd_predict(args, corpus) -> dict:
    store = open_store(args.store)
    deal = find_deal(args.deal, corpus)
    return render(eh.propose(deal, store, key_mode=args.key_mode))


def cmd_correct(args, corpus) -> dict:
    store = open_store(args.store)
    deal = find_deal(args.deal, corpus)
    before = eh.propose(deal, store, key_mode=args.key_mode)
    proposed = before.field_value(args.line, args.field)
    lesson = eh.lesson_from_override(
        deal, before, line_index=args.line, field_name=args.field, accepted=args.value,
        reason_code=args.reason_code, reason_text=args.reason_text or "",
        actor=args.actor, key_mode=args.key_mode,
    )
    record = OverrideRecord(
        deal_id=before.deal_id,
        proposal_id=before.proposal_id,
        proposal_version=before.version,
        line_index=args.line,
        field=args.field,
        proposed_value=proposed,
        accepted_value=float(args.value),
        inputs={**before.lines[args.line].inputs, "shape": before.lines[args.line].shape},
        reason_code=args.reason_code,
        reason_text=args.reason_text or "",
        actor=args.actor,
        lesson_id=lesson.id,
        key_mode=before.key_mode,
    )
    from app.core.override_log import validate

    problems = validate(record)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    preview = eh.preview_transfer(store, lesson, corpus, key_mode=args.key_mode)
    result: dict[str, Any] = {
        "deal_id": before.deal_id,
        "proposal_id": before.proposal_id,
        "proposal_version": before.version,
        "field": args.field,
        "proposed": proposed,
        "accepted": float(args.value),
        "lesson": {"id": lesson.id, "relation": lesson.relation, "verdict": lesson.verdict,
                   "key": lesson.exemplars[0]},
        "would_also_change": preview,
        "committed": False,
    }
    if args.commit:
        store.add(lesson)
        override_id = OverrideLog(args.overrides or ":memory:").record(record)
        refresh(store)
        after = eh.propose(deal, store, key_mode=args.key_mode)
        result.update({
            "committed": True,
            "override_id": override_id,
            "repredicted": render(after),
        })
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="one_deal_loop", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--deal", required=True, help="deal_id in the corpus, or a deal JSON path")
        p.add_argument("--corpus", default=None, help="triples fixture or JSON list of deals")
        p.add_argument("--store", default=None, help="local SQLite lesson store (default: in memory)")
        p.add_argument("--key-mode", default=None, choices=[ws.MODE_WORDING, ws.MODE_WORK_SHAPE],
                       help=f"lesson key (default: ${ws.KEY_MODE_ENV} or wording)")

    common(sub.add_parser("predict", help="show the estimate for one deal"))
    c = sub.add_parser("correct", help="correct one field, preview transfer, optionally commit")
    common(c)
    c.add_argument("--line", type=int, default=0)
    c.add_argument("--field", required=True, choices=list(eh.FIELDS))
    c.add_argument("--value", required=True, type=float)
    c.add_argument("--reason-code", required=True, choices=list(REASON_CODES))
    c.add_argument("--reason-text", default="")
    c.add_argument("--actor", default="pm")
    c.add_argument("--overrides", default=None, help="local SQLite override log (default: in memory)")
    c.add_argument("--commit", action="store_true", help="store the lesson and record the override")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = load_corpus(args.corpus)
    out = cmd_predict(args, corpus) if args.cmd == "predict" else cmd_correct(args, corpus)
    json.dump(out, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
