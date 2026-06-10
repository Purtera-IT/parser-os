"""Canonical train/holdout split — ONE source of truth, read by every script.

Why this exists (the bug it fixes):
  Four scripts (clean_labels_universal.py, rubric_relabel_deepseek.py,
  train_contrastive_encoder_gpu.py, type_head.py) each re-derived the split with
  a copy-pasted `sha256(deal_id) % 100 < HOLDOUT`, IGNORING the `split` column the
  training_rows already carry. Measured disagreement vs the recorded split:
    _training_coarse.db    17.9%   (3,773 recorded-HOLDOUT rows the hash calls train)
    _training_deepseek.db  17.6%   (3,773)
    _training_cloud.db     97.2%   (2,903)
  Consequences: (1) results across scripts and vs label_stats are not on the same
  split; (2) clean_labels_universal.py DELETES rows its own hash calls "train" —
  some of which are recorded-holdout, so cleaning silently mutated the eval set
  while printing "held-out rows UNTOUCHED"; (3) gate train pool and head holdout
  could overlap. Every metric measured on the hash split is contaminated.

Contract:
  - Prefer the recorded `split` column (normalized: test/holdout -> "holdout").
  - Fall back to the deterministic hash ONLY for deals with no recorded split
    (e.g. fresh flywheel rows not yet assigned), so nothing is left unsplit.
  - Split is per-DEAL: all rows of a deal share one split (asserted by
    audit_split_consistency); this prevents row-level leakage within a deal.
"""
from __future__ import annotations

import hashlib

DEFAULT_HOLDOUT = 0.25


def hash_split(deal_id: str, holdout: float = DEFAULT_HOLDOUT) -> str:
    """Deterministic by-deal fallback. Returns 'train' | 'holdout'."""
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16)
    return "holdout" if (h % 100) / 100.0 < holdout else "train"


def _norm(s: str | None) -> str | None:
    if not s:
        return None
    return "holdout" if s in ("test", "holdout") else "train"


def load_split_map(con) -> dict[str, str]:
    """deal_id -> recorded split ('train'|'holdout') from training_rows.

    HOLDOUT WINS: if any row of a deal is recorded holdout, the whole deal is
    holdout. This is the conservative resolution for the ~10 deals that carry
    BOTH splits (a logging bug); it guarantees we never train on a deal whose
    rows also appear in eval. Use audit_split_consistency() to surface them.
    """
    m: dict[str, str] = {}
    for deal_id, split in con.execute(
        "SELECT deal_id, split FROM training_rows WHERE split IS NOT NULL AND split != ''"
    ):
        s = _norm(split)
        if not (deal_id and s):
            continue
        if m.get(deal_id) == "holdout":
            continue  # already locked to holdout
        m[deal_id] = "holdout" if s == "holdout" else m.get(deal_id, "train")
        if s == "holdout":
            m[deal_id] = "holdout"
    return m


def split_of(deal_id: str, split_map: dict[str, str], holdout: float = DEFAULT_HOLDOUT) -> str:
    """Recorded split if known, else deterministic hash fallback. Never returns None."""
    return split_map.get(deal_id) or hash_split(deal_id, holdout)


def audit_split_consistency(con) -> dict:
    """Report deals that carry >1 distinct recorded split (a data bug)."""
    by_deal: dict[str, set] = {}
    for deal_id, split in con.execute(
        "SELECT deal_id, split FROM training_rows WHERE split IS NOT NULL AND split != ''"
    ):
        s = _norm(split)
        if deal_id and s:
            by_deal.setdefault(deal_id, set()).add(s)
    conflicts = {d: sorted(v) for d, v in by_deal.items() if len(v) > 1}
    return {"deals": len(by_deal), "conflicting_deals": conflicts}


if __name__ == "__main__":  # quick audit over the local DBs
    import sqlite3
    import sys

    for db in sys.argv[1:] or [
        "_training_coarse.db",
        "_training_deepseek.db",
        "_training_cloud.db",
    ]:
        try:
            con = sqlite3.connect(db)
        except Exception as e:  # noqa: BLE001
            print(db, "skip", e)
            continue
        sm = load_split_map(con)
        rows = con.execute(
            "SELECT deal_id, split FROM training_rows WHERE relation='atom_type'"
        ).fetchall()
        recorded = sum(1 for _, s in rows if _norm(s))
        disagree = sum(
            1 for d, s in rows if _norm(s) and _norm(s) != hash_split(d)
        )
        cons = audit_split_consistency(con)
        print(
            f"{db}: rows={len(rows)} recorded={recorded} "
            f"hash_disagree={disagree} ({100*disagree/max(1,recorded):.1f}%) "
            f"conflicting_deals={len(cons['conflicting_deals'])}"
        )
