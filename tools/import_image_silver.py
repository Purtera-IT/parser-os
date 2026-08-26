"""Import hand-graded pdf_image_kind rows into a TrainingLog sqlite.

Takes a graded JSONL (one object per line) produced by the silver-audit
grading pass and appends ``relation='pdf_image_kind'`` rows to the given
TrainingLog database, matching the existing schema exactly (same table the
pipeline's ``_log_gate_silver`` writes).

Contract:
  * ``teacher='silver_audit'`` — distinct from 'llm' (VLM silver) and 'pm'
    (human gold) so these rows are filterable and reversible:
    ``DELETE FROM training_rows WHERE teacher='silver_audit'`` undoes an import.
  * Idempotent by content hash: the row id is derived from
    (relation | label | feature_text), so re-running the import is a no-op
    (the log's ``INSERT OR REPLACE`` keyed on id absorbs repeats).
  * Guess-free: rows must carry a label from the closed gate set
    (skip / photo / diagram / chart / table_image / screenshot); anything else
    is rejected loudly, never coerced.
  * Split assignment stays with the log (deal-id hash), so graded rows from
    held-out deals land in holdout exactly like every other relation.

Input JSONL fields (per row):
  required: label, feature_text  (or caption+ocr_snippet to rebuild it)
  optional: meaningful, variety, rationale, confidence, deal_id, pdf, page,
            image_ref, image_sha16

Usage:
  python tools/import_image_silver.py \
      --graded <scratch>/image_silver_graded.jsonl \
      --db     <scratch>/_training_deepseek.db      # LOCAL COPY, never blob-direct
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.pdf_image_gate import gate_feature_text  # noqa: E402
from app.core.training_log import TrainingLog, TrainingRow  # noqa: E402

RELATION = "pdf_image_kind"
TEACHER_SILVER_AUDIT = "silver_audit"
# The closed label set the gate trains on — pdf_image_vision's
# _IMAGE_KIND_CANDIDATES with the finer skip kinds (logo/decorative/signature/
# empty) collapsed to "skip", exactly mirroring _log_gate_silver's labels.
# ("instructions"/"label"/"map" are runtime kinds even though the FE review
# verdict set exposes only six.)
ALLOWED_LABELS = frozenset({
    "skip", "photo", "diagram", "chart", "table_image", "screenshot",
    "instructions", "label", "map",
})
# Hand-graded from feature text by a careful reader: above VLM silver (0.7),
# below PM gold. Overridable per row.
DEFAULT_CONFIDENCE = 0.85
# Above llm silver (1.0), below pm gold (5.0) — same reasoning as confidence.
WEIGHT = 2.0


# ── pure functions (unit-tested) ────────────────────────────────────


def content_row_id(label: str, feature_text: str, *, relation: str = RELATION) -> str:
    """Deterministic id from row CONTENT — the idempotency key."""
    h = hashlib.sha256(
        f"{relation}|{label}|{feature_text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"trn_sa_{h}"


def graded_to_training_row(obj: dict[str, Any]) -> TrainingRow:
    """Validate + convert one graded JSONL object. Raises ValueError on any
    row that cannot be imported verbatim (no coercion, no guessing)."""
    label = str(obj.get("label") or "").strip().lower()
    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"label {label!r} not in {sorted(ALLOWED_LABELS)} "
            f"(row image_ref={obj.get('image_ref')!r})"
        )
    feat = str(obj.get("feature_text") or "").strip()
    if not feat:
        feat = gate_feature_text(
            str(obj.get("caption") or ""), str(obj.get("ocr_snippet") or "")
        )
    if not feat or feat == "no context":
        raise ValueError(
            f"empty feature text — nothing for the gate to learn from "
            f"(row image_ref={obj.get('image_ref')!r})"
        )
    try:
        confidence = float(obj.get("confidence") or DEFAULT_CONFIDENCE)
    except (TypeError, ValueError):
        confidence = DEFAULT_CONFIDENCE
    deal_id = str(obj.get("deal_id") or "")
    if deal_id.startswith("local:"):
        deal_id = ""  # local corpus rows have no real deal — never fake one
    provenance = {
        "stage": "silver_audit_import",
        "variety": str(obj.get("variety") or ""),
        "rationale": str(obj.get("rationale") or ""),
        "meaningful": bool(obj.get("meaningful", label != "skip")),
        "pdf": str(obj.get("pdf") or ""),
        "page": obj.get("page", ""),
        "image_ref": str(obj.get("image_ref") or ""),
        "image_sha16": str(obj.get("image_sha16") or ""),
    }
    return TrainingRow(
        id=content_row_id(label, feat),
        relation=RELATION,
        label=label,
        raw_text=feat,
        masked_text=feat,  # mirror _log_gate_silver: feature text IS the feature
        label_kind="judgment",
        teacher=TEACHER_SILVER_AUDIT,
        weight=WEIGHT,
        confidence=confidence,
        deal_id=deal_id,
        provenance=provenance,
    )


def load_graded(path: Path) -> tuple[list[TrainingRow], list[str]]:
    """Parse the graded JSONL. Returns (rows, per-line error messages)."""
    rows: list[TrainingRow] = []
    errors: list[str] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(graded_to_training_row(obj))
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"line {lineno}: {exc}")
    return rows, errors


def relation_census(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM training_rows WHERE relation=?", (RELATION,)
        ).fetchone()[0]
        by = {}
        for label, teacher, split, n in conn.execute(
            "SELECT label, teacher, split, COUNT(*) FROM training_rows "
            "WHERE relation=? GROUP BY label, teacher, split", (RELATION,)
        ):
            by[f"{label}/{teacher}/{split}"] = n
        return {"total": int(total), "by_label_teacher_split": by}
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graded", type=Path, required=True,
                    help="graded JSONL (label + feature_text per line)")
    ap.add_argument("--db", type=Path, required=True,
                    help="TrainingLog sqlite path — a LOCAL COPY, never a "
                         "blob-mounted or production file")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate + report, write nothing")
    args = ap.parse_args(argv)

    if not args.graded.is_file():
        print(f"graded file not found: {args.graded}", file=sys.stderr)
        return 1
    rows, errors = load_graded(args.graded)
    for e in errors:
        print(f"  ! skipped {e}", file=sys.stderr)
    if not rows:
        print("No importable rows.", file=sys.stderr)
        return 1

    before = relation_census(args.db) if args.db.is_file() else {"total": 0}
    print(f"Before: {json.dumps(before, indent=2)}")
    if args.dry_run:
        print(f"Dry run: {len(rows)} rows would be imported "
              f"({len(errors)} rejected).")
        return 0

    log = TrainingLog(str(args.db))
    written = log.add_many(rows)
    after = relation_census(args.db)
    print(f"Imported {written} rows ({len(errors)} rejected as unimportable).")
    print(f"After: {json.dumps(after, indent=2)}")
    print(
        "\nNOT uploaded anywhere. To publish (after human confirmation):\n"
        f"  az storage blob upload --account-name purpulsedevstg01 "
        f"--container-name ml-artifacts --name _training_deepseek.db "
        f"--file {args.db} --auth-mode login --overwrite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
