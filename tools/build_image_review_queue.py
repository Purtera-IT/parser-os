"""Build the PARSER REVIEW QUEUE for suspicious image-skip verdicts.

Standalone tool (NOT wired into the pipeline or FE). Ranks image candidates by
"culprit likelihood" — how bad it would be if the image gate said "skip":

  * PM-critical vocabulary hits (zero_miss.PM_CRITICAL_TERMS) — strongest,
  * quantity signals (counts, ports/outlets/drops, money) — a skipped image
    with quantities is lost scope,
  * OCR density — the more real text an image carries, the less likely "skip"
    is the right verdict.
  * Fired skip-vetoes (``relation='pdf_image_veto'``), band-aware — the
    binary head disagreed with a vlm/store skip at >=0.88 (band=hard) or sat
    in its uncertain zone 0.70-0.88 (band=soft, active-learning harvest).
    Hard floats above soft; both float above ordinary candidates.

Inputs:
  * a harvest JSONL from tools/harvest_image_silver.py (--candidates), and/or
  * a TrainingLog SQLite db (--training-db) whose logged pdf_image_kind
    verdicts (feature text + label) are folded in — logged "skip" rows are the
    queue's primary audit target — plus any ``pdf_image_veto`` rows, which
    outrank ordinary skips.

Output: one CSV a human can walk top-to-bottom. The verdict column is EMPTY —
this tool never guesses a label. Valid verdicts (the FE image-kind set):
skip / photo / diagram / chart / table_image / screenshot.

Usage:
  python tools/build_image_review_queue.py \
      --candidates <scratch>/image_silver_candidates.jsonl \
      --training-db <scratch>/_training_deepseek.db \
      --out <scratch>/image_review_queue.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.harvest_image_silver import suspicious_hits  # noqa: E402

# The FE's closed verdict set for this queue (documented in the CSV header).
REVIEW_VERDICTS = ["skip", "photo", "diagram", "chart", "table_image", "screenshot"]

# Quantity signals: "18 Total Data Outlets", "4 x AP", "qty 12", "$4,500",
# "15 - Duplex Data Outlets", "(2) cameras" — the shapes deal PDFs use.
QUANTITY_RE = re.compile(
    r"(?ix)"
    r"\$\s?\d[\d,]*"                                  # money
    r"|\b\d+\s*(?:x|ea|each|units?|pcs?)\b"           # 4 x / 12 ea
    r"|\bqty\.?\s*:?\s*\d+"                           # qty 12
    r"|\(\s*\d+\s*\)"                                 # (2)
    r"|\b\d+\s*(?:-|–)?\s*(?:[a-z]+\s+){0,2}"         # 18 Total Data Outlets
    r"(?:ports?|outlets?|drops?|cameras?|cables?|jacks?|racks?|panels?|"
    r"aps?\b|switch(?:es)?|phones?|licenses?|devices?)"
)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


# ── pure scoring functions (unit-tested) ────────────────────────────


def quantity_signals(text: str) -> list[str]:
    """Distinct quantity-looking matches in the text (empty = none found)."""
    return sorted({m.group(0).strip() for m in QUANTITY_RE.finditer(text or "")})


def ocr_token_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def culprit_score(
    *, pm_hit_count: int, quantity_count: int, tokens: int,
    logged_skip: bool = False, veto_fired: bool = False, veto_band: str = "",
) -> float:
    """Rank key: higher = review sooner.

    PM-critical vocab dominates (3.0 each), quantities next (1.5 each, capped),
    OCR density fills in (up to 2.0 at ~100 tokens). A row the pipeline
    actually LOGGED as "skip" gets a flat bonus: it is not hypothetical — that
    verdict already fired on a real compile. A fired skip-veto gets a larger
    bonus on top, BAND-AWARE: a HARD veto (confident disagreement, +4.0)
    outranks a SOFT veto (the head's uncertain zone — active-learning harvest,
    +2.0), and both outrank plain logged skips and harvest candidates. Rows
    without a band (pre-band vetoes) count as hard.
    """
    score = 3.0 * pm_hit_count
    score += 1.5 * min(quantity_count, 4)
    score += min(tokens / 50.0, 2.0)
    if logged_skip:
        score += 2.5
    if veto_fired:
        score += 2.0 if veto_band == "soft" else 4.0
    return round(score, 3)


def score_candidate(cand: dict[str, Any]) -> dict[str, Any]:
    """Attach score + signal columns to one candidate row (pure)."""
    feat = str(cand.get("feature_text") or "")
    hits = cand.get("pm_term_hits")
    if hits is None:  # rows from the TrainingLog carry only feature text
        hits = suspicious_hits(feat)
    qts = quantity_signals(feat)
    toks = ocr_token_count(str(cand.get("ocr_snippet") or feat))
    veto_fired = bool(cand.get("veto_fired")) or str(cand.get("source") or "") == "veto"
    veto_band = str(cand.get("veto_band") or "") if veto_fired else ""
    # Veto rows always imply the gate skipped (that's what was vetoed).
    logged_skip = str(cand.get("logged_label") or "") == "skip" or veto_fired
    return {
        **cand,
        "pm_term_kinds": ";".join(sorted({h["kind"] for h in hits})),
        "pm_hit_count": len(hits),
        "quantity_signals": ";".join(qts)[:200],
        "quantity_count": len(qts),
        "ocr_tokens": toks,
        "veto_fired": veto_fired,
        "veto_band": veto_band,
        "culprit_score": culprit_score(
            pm_hit_count=len(hits), quantity_count=len(qts), tokens=toks,
            logged_skip=logged_skip, veto_fired=veto_fired, veto_band=veto_band,
        ),
    }


def rank_queue(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score and sort (desc); stable for equal scores (pure)."""
    scored = [score_candidate(c) for c in cands]
    scored.sort(key=lambda c: -c["culprit_score"])
    for i, c in enumerate(scored, 1):
        c["rank"] = i
    return scored


# ── inputs ──────────────────────────────────────────────────────────


def load_harvest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            obj["source"] = "harvest"
            obj["logged_label"] = ""
            rows.append(obj)
    return rows


def _row_from_training(
    *, rid: str, label: str, teacher: str, conf: Any, raw: str,
    prov: str, ts: str, deal_id: str, source: str, veto_fired: bool,
) -> dict[str, Any]:
    pdf = page = region = veto_band = crop_ref = ""
    if prov:
        try:
            p = json.loads(prov) if isinstance(prov, str) and prov.startswith("{") else {}
            if isinstance(p, dict):
                pdf = str(p.get("pdf") or "")
                page = str(p.get("page") if p.get("page") is not None else "")
                region = str(p.get("region_ref") or p.get("image_ref") or "")
                veto_band = str(p.get("band") or "")
                crop_ref = str(p.get("crop_ref") or "")
        except Exception:
            pass
    if veto_fired and not veto_band:
        veto_band = "hard"  # pre-band veto rows were all hard by construction
    return {
        "source": source,
        "deal_id": deal_id or "",
        "pdf": pdf,
        "page": page,
        "image_ref": region or rid,
        "caption": "",
        "ocr_snippet": "",
        "feature_text": raw or "",
        "logged_label": label or "",
        "logged_teacher": teacher or "",
        "logged_confidence": conf,
        "logged_at": ts,
        "logged_provenance": prov or "",
        "veto_fired": veto_fired,
        "veto_band": veto_band if veto_fired else "",
        "crop_ref": crop_ref,
    }


def load_training_log_rows(db_path: Path) -> list[dict[str, Any]]:
    """pdf_image_kind + pdf_image_veto rows from the TrainingLog.

    Kind rows: all labels; 'skip' gets the logged-skip score bonus.
    Veto rows: top-tier, band-aware — hard vetoes (confident disagreement)
    above soft vetoes (uncertain-zone harvest), both above plain skips.
    ``crop_ref`` (blob path of the persisted disputed crop) is carried into
    the CSV when the pipeline stamped one.
    """
    rows: list[dict[str, Any]] = []
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT id, relation, label, teacher, confidence, raw_text, "
            "provenance, deal_id, datetime(created_at,'unixepoch') "
            "FROM training_rows "
            "WHERE relation IN ('pdf_image_kind', 'pdf_image_veto')"
        )
        for rid, relation, label, teacher, conf, raw, prov, deal_id, ts in cur.fetchall():
            veto = relation == "pdf_image_veto"
            rows.append(_row_from_training(
                rid=rid, label=("skip" if veto else (label or "")),
                teacher=teacher or "", conf=conf, raw=raw or "",
                prov=prov or "", ts=ts or "", deal_id=deal_id or "",
                source="veto" if veto else "training_log",
                veto_fired=veto,
            ))
    finally:
        conn.close()
    return rows


# ── output ──────────────────────────────────────────────────────────

_QUEUE_FIELDS = [
    "rank", "culprit_score", "source", "logged_label", "veto_fired",
    "veto_band", "crop_ref", "deal_id",
    "pdf", "page", "image_ref", "caption", "ocr_snippet", "feature_text",
    "pm_term_kinds", "pm_hit_count", "quantity_signals", "quantity_count",
    "ocr_tokens", "verdict",
]

_HEADER_DOC = (
    "# PARSER REVIEW QUEUE — suspicious image-skip candidates.\n"
    "# Walk top-to-bottom; fill the LAST column (verdict) per row.\n"
    "# Valid verdicts: skip | photo | diagram | chart | table_image | screenshot\n"
    "#   skip        = truly no PM-relevant content (logo/decorative/signature/empty)\n"
    "#   photo       = site/equipment photo worth describing\n"
    "#   diagram     = network/wiring/floor-plan style diagram\n"
    "#   chart       = graph/plot with values\n"
    "#   table_image = a table rendered as an image (BOM, pricing, port map)\n"
    "#   screenshot  = software/UI capture (configs, consoles, tickets)\n"
    "# Leave verdict empty if undecidable from the text alone (open the image).\n"
    "# rows with logged_label=skip were ACTUALLY skipped by the pipeline gate.\n"
    "# source=veto / veto_fired=1: skip-veto head flagged the skip — review first.\n"
    "#   veto_band=hard: head confidently disagreed (possible lost evidence).\n"
    "#   veto_band=soft: head was UNCERTAIN (0.70-0.88) — active-learning\n"
    "#   harvest; your grade here teaches the most per label.\n"
    "# crop_ref (when set) = blob path of the persisted crop pixels:\n"
    "#   container orbitbrief-artifacts, deals/<deal>/orbitbrief/disputed_crops/.\n"
)


def write_queue(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        f.write(_HEADER_DOC)
        w = csv.DictWriter(f, fieldnames=_QUEUE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["verdict"] = ""
            r["veto_fired"] = "1" if r.get("veto_fired") else ""
            r["feature_text"] = str(r.get("feature_text") or "")[:600]
            w.writerow({k: r.get(k, "") for k in _QUEUE_FIELDS})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", type=Path, default=None,
                    help="harvest JSONL from tools/harvest_image_silver.py")
    ap.add_argument("--training-db", type=Path, default=None,
                    help="TrainingLog sqlite db with logged pdf_image_kind rows")
    ap.add_argument("--out", type=Path, required=True,
                    help="review CSV path (keep OUTSIDE the repo)")
    ap.add_argument("--top", type=int, default=0,
                    help="cap the queue at the N highest-scoring rows")
    args = ap.parse_args(argv)

    rows: list[dict[str, Any]] = []
    if args.candidates:
        rows.extend(load_harvest(args.candidates))
    if args.training_db:
        rows.extend(load_training_log_rows(args.training_db))
    if not rows:
        print("No input rows (--candidates and/or --training-db).", file=sys.stderr)
        return 1

    ranked = rank_queue(rows)
    if args.top:
        ranked = ranked[: args.top]
    write_queue(ranked, args.out)
    n_susp = sum(1 for r in ranked if r["pm_hit_count"] > 0)
    n_logged = sum(1 for r in ranked if r.get("logged_label"))
    n_veto = sum(1 for r in ranked if r.get("veto_fired"))
    print(
        f"Review queue: {len(ranked)} rows ({n_logged} from the TrainingLog, "
        f"{n_veto} veto-fired, {n_susp} with PM-critical hits) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
