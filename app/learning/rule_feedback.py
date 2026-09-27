"""Join what a rule DECIDED to what a reviewer said about the atom it made.

A SemanticRule fires on a line while the document is being cut into atoms. The
labels we collect are about atoms that already exist. Those are two different
questions at two different moments, and they meet at exactly one point: when a
reviewer accepts or rejects an atom, they are also saying whether the decision
that created it was right.

That join is the only route from labelling to a trained parser rule, and it has
two halves that neither side can supply alone:

  the LOG   every decision the rule made, INCLUDING the ones where it did not
            fire, with the margins it scored (`best_pos`, `best_neg`). Written
            by `semantic_rules._log_decision` when `SOWSMITH_RULE_LOG` is set.
  the LABELS whether the atom that resulted is a real fact or junk.

Measured on 010180 + 010288 before this existed: joining the labels ALONE to
the `list_item_under_label` rule gives 65 rows and every one of them is a
positive. That is structural, not bad luck -- an atom only exists when the rule
fired, so a line the rule wrongly SKIPPED leaves nothing on screen to label and
no negative to learn from. A threshold fitted on 65 positives and no negatives
collapses to zero and fires on everything. It is the same thing the corpus loss
audit reports every run: "40 of 65 atoms carry no `rejected`, so they teach a
point and not a boundary."

So: the log supplies the candidates including the ones that got away, the
labels supply the truth, and `_train_semantic_rules.py` reads the result.

    python -m app.learning.rule_feedback --deal <uuid> [--deal <uuid> ...]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

#: Where `semantic_rules._log_decision` writes and `_train_semantic_rules.py`
#: reads. One JSON object per line.
LOG_PATH_ENV = "SOWSMITH_RULE_LOG"


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def log_path() -> Path | None:
    raw = os.environ.get(LOG_PATH_ENV)
    return Path(raw) if raw else None


def read_decisions(path: Path | None = None) -> list[dict[str, Any]]:
    """Every decision the rules logged, oldest first. Missing log -> []."""
    path = path or log_path()
    if path is None or not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict) and rec.get("rule") and rec.get("text"):
            out.append(rec)
    return out


def truth_from_labels(labels: Iterable[dict[str, Any]]) -> dict[str, bool]:
    """text -> was this atom accepted as a real fact?

    Keyed on the atom's words because that is what both sides have: the log
    records the line it judged, the label records the atom's text, and for a
    rule that admits a line as an atom those are the same string.
    """
    truth: dict[str, bool] = {}
    for row in labels:
        text = _norm(row.get("text"))
        if not text:
            continue
        rejected = str(row.get("rejected")).strip().lower() in ("true", "t", "1")
        # A reject is decisive; two labels on one string, one of them a reject,
        # means somebody looked and said no.
        truth[text] = truth.get(text, True) and not rejected
    return truth


def join(decisions: Iterable[dict[str, Any]],
         truth: dict[str, bool]) -> Iterator[dict[str, Any]]:
    """Emit the trainer's shape: the logged row plus a ground-truth `label`.

    A decision is labelled only where a reviewer actually saw the result:

      fired + atom accepted  -> 1   the rule was right to fire
      fired + atom rejected  -> 0   it made junk
      did not fire + the words turn up as an accepted atom anyway
                             -> 1   something else rescued it; the rule
                                    should have fired
      did not fire + nothing -> unlabelled. We do not know whether the line was
                                correctly skipped or silently lost, and guessing
                                here would manufacture the negatives the whole
                                exercise is short of.
    """
    for rec in decisions:
        text = _norm(rec.get("text"))
        seen = truth.get(text)
        fired = bool(rec.get("decision"))
        if seen is None:
            if fired:
                continue          # it made an atom nobody has judged yet
            continue              # the silent skip; see the docstring
        label = 1 if seen else 0
        if not fired and seen:
            label = 1             # it should have fired
        yield {**rec, "label": label}


def write_labelled(rows: Iterable[dict[str, Any]], path: Path) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


__all__ = ["LOG_PATH_ENV", "log_path", "read_decisions", "truth_from_labels",
           "join", "write_labelled"]
