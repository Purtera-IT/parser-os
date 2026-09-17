"""Fallthrough census + labelling set for worksheets (PUR-51 / PUR-52).

Walks a LOCAL directory of workbooks (.xlsx / .csv), classifies every sheet
with ``app.parsers.sheet_classifier.classify_sheet`` and records whether the
decision was a positive match or a fallthrough. Fallthrough sheets are written
as labelling records carrying structure only:

    sheet_id, workbook_hash, workbook_file, sheet_index, customer_key,
    headers, header_row, column_types, row_count, nonblank_rows, max_width,
    mean_width, fill_density, rule_reason, label (empty; one of LABEL_CLASSES)

Raw cell values beyond the header row are NOT written unless
``include_samples`` is set. The sheet name is written for the labeller's
convenience only (``sheet_name``) and is never a training feature.

This module performs no network, database or cloud access.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app.core.sheet_structure_head import LABEL_CLASSES, profile_sheet
from app.parsers.sheet_classifier import MATCH_FALLTHROUGH, classify_sheet

WORKBOOK_SUFFIXES = (".xlsx", ".xlsm", ".csv")
CUSTOMER_PLACEHOLDER = "UNKNOWN_CUSTOMER"

LABEL_FIELDS = (
    "sheet_id", "workbook_hash", "workbook_file", "sheet_index", "sheet_name",
    "customer_key", "rule_reason", "headers", "header_row", "column_types",
    "row_count", "nonblank_rows", "max_width", "mean_width", "fill_density",
    "label", "label_options",
)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def iter_workbook_sheets(path: Path, *, max_rows: int = 5000) -> Iterator[tuple[int, str, list[list[Any]]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            rows = [list(r) for _, r in zip(range(max_rows), csv.reader(fh))]
        yield 0, "csv", rows
        return
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for i, ws in enumerate(wb.worksheets):
            try:
                ws.reset_dimensions()
            except Exception:
                pass
            rows: list[list[Any]] = []
            for r in ws.iter_rows(values_only=True):
                rows.append(list(r))
                if len(rows) >= max_rows:
                    break
            yield i, ws.title, rows
    finally:
        wb.close()


def _customer_key(path: Path, root: Path, mode: str) -> str:
    if mode == "parent_dir":
        rel = path.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else CUSTOMER_PLACEHOLDER
    return CUSTOMER_PLACEHOLDER


@dataclass
class CensusResult:
    total_sheets: int = 0
    fallthrough_rules_only: int = 0
    fallthrough_with_head: int = 0
    by_reason: Counter = field(default_factory=Counter)
    records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def report(self) -> dict[str, Any]:
        n = max(1, self.total_sheets)
        return {
            "total_sheets": self.total_sheets,
            "fallthrough_rules_only": self.fallthrough_rules_only,
            "fallthrough_rate_rules_only": round(self.fallthrough_rules_only / n, 4),
            "fallthrough_with_head": self.fallthrough_with_head,
            "fallthrough_rate_with_head": round(self.fallthrough_with_head / n, 4),
            "by_reason": dict(self.by_reason.most_common()),
            "labelling_records": len(self.records),
            "errors": self.errors,
        }


def run_census(
    root: str | Path,
    *,
    customer_key_mode: str = "placeholder",
    include_samples: int = 0,
    include_positive: bool = False,
) -> CensusResult:
    root = Path(root)
    res = CensusResult()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in WORKBOOK_SUFFIXES)
    for path in files:
        try:
            wb_hash = _file_hash(path)
            for idx, name, rows in iter_workbook_sheets(path):
                res.total_sheets += 1
                # Rules only (the pre-head world); learned PM store is not
                # consulted so the census is reproducible offline.
                rules = classify_sheet(name, rows, use_structural_head=False, use_learned_store=False)
                withhead = classify_sheet(name, rows, use_structural_head=True, use_learned_store=False)
                res.by_reason[rules.reason.split(":")[0]] += 1
                ft = rules.match == MATCH_FALLTHROUGH
                res.fallthrough_rules_only += int(ft)
                res.fallthrough_with_head += int(withhead.match == MATCH_FALLTHROUGH)
                if not (ft or include_positive):
                    continue
                prof = profile_sheet(rows)
                rec: dict[str, Any] = {
                    "sheet_id": f"{wb_hash}:{idx}",
                    "workbook_hash": wb_hash,
                    "workbook_file": path.name,
                    "sheet_index": idx,
                    "sheet_name": name,
                    "customer_key": _customer_key(path, root, customer_key_mode),
                    "rule_reason": rules.reason,
                    "match": rules.match,
                    **prof.to_dict(),
                    "label": "",
                    "label_options": "|".join(LABEL_CLASSES),
                }
                if include_samples:
                    start = (prof.header_row + 1) if prof.header_row is not None else 0
                    rec["sample_rows"] = [
                        ["" if c is None else str(c) for c in r] for r in rows[start : start + include_samples]
                    ]
                res.records.append(rec)
        except Exception as exc:  # a bad file must not stop the census
            res.errors.append(f"{path.name}:{type(exc).__name__}:{exc}")
    return res


def write_outputs(res: CensusResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "sheet_labels.jsonl"
    with jsonl.open("w") as fh:
        for r in res.records:
            fh.write(json.dumps(r, default=str) + "\n")
    csv_path = out / "sheet_labels.csv"
    with csv_path.open("w", newline="") as fh:
        extra = ["sample_rows"] if any("sample_rows" in r for r in res.records) else []
        w = csv.DictWriter(fh, fieldnames=list(LABEL_FIELDS) + ["match"] + extra, extrasaction="ignore")
        w.writeheader()
        for r in res.records:
            row = dict(r)
            for k in ("headers", "column_types", "sample_rows"):
                if k in row:
                    row[k] = json.dumps(row[k], default=str)
            w.writerow(row)
    report = out / "fallthrough_report.json"
    report.write_text(json.dumps(res.report(), indent=2))
    return {"jsonl": jsonl, "csv": csv_path, "report": report}


__all__ = ["run_census", "write_outputs", "CensusResult", "iter_workbook_sheets", "CUSTOMER_PLACEHOLDER"]
