from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpyxl import Workbook

from app.eval.parser_metrics import (
    ParserBenchmarkReport,
    ParserCaseExpectation,
    aggregate_parser_metrics,
    evaluate_case_metrics,
    parser_threshold_failures,
)
from app.parsers.docx_parser import DocxParser
from app.parsers.email_parser import EmailParser
from app.parsers.quote_parser import QuoteParser
from app.parsers.transcript_parser import TranscriptParser
from app.parsers.xlsx_parser import XlsxParser
from app.testing.mutators import write_docx_fixture


def _aggregate_reports(reports):
    if not reports:
        return {}
    keys = [
        "atom_recall_by_type",
        "source_ref_coverage",
        "entity_key_accuracy",
        "quantity_accuracy",
        "authority_class_accuracy",
        "review_flag_accuracy",
        "parse_crash_rate",
    ]
    return {
        key: round(sum(float(getattr(report, key)) for report in reports) / len(reports), 4)
        for key in keys
    } | {
        "unsupported_feature_warnings": int(sum(report.unsupported_feature_warnings for report in reports)),
    }


def _xlsx_cases(tmp: Path):
    path = tmp / "xlsx_adv.xlsx"
    wb = Workbook()
    ws = wb.active
    for _ in range(6):
        ws.append(["title", "", "", "", "", ""])
    ws.append(["Site", "Floor", "Device", "QTY.", "Access Window", "Scope"])
    ws.append(["Main Campus", "1", "IP Camera", "50 EA", "Weekdays", "Install"])
    ws.append(["Bldg A West", "2", "IP Camera", "1,200", "Escort", "Install"])
    ws.append(["Subtotal", "", "", "1250", "", ""])
    wb.save(path)
    atoms = XlsxParser().parse_artifact("proj", "art_xlsx", path)
    expected = ParserCaseExpectation(
        expected_atom_types={"quantity", "scope_item", "constraint"},
        expected_entity_keys={"site:west_wing", "site:main_campus"},
        expected_quantities={50.0, 1200.0},
        expected_authorities={"approved_site_roster"},
    )
    return [evaluate_case_metrics(atoms, expected)]


def _email_cases(tmp: Path):
    path = tmp / "email_adv.txt"
    path.write_text(
        "From: jane.customer@example.com\n"
        "Sent: 2026-01-01 10:00\n"
        "Subject: scope\n\n"
        "Do not proceed at West Wing.\n\n"
        "On 2025-12-20, Jane wrote:\n"
        "> Include West Wing in scope.\n",
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact("proj", "art_email", path)
    expected = ParserCaseExpectation(
        expected_atom_types={"exclusion", "customer_instruction"},
        expected_authorities={"customer_current_authored"},
    )
    return [evaluate_case_metrics(atoms, expected)]


def _docx_cases(tmp: Path):
    path = tmp / "docx_adv.docx"
    write_docx_fixture(path, included_site="Main Campus", excluded_site="West Wing", scoped_device="IP Camera", mutation="scope_in_table")
    atoms = DocxParser().parse_artifact("proj", "art_docx", path)
    expected = ParserCaseExpectation(
        expected_atom_types={"scope_item", "exclusion", "constraint"},
        expected_authorities={"deleted_text", "meeting_note"},
        unsupported_warning_count=1,
    )
    return [evaluate_case_metrics(atoms, expected)]


def _transcript_cases(tmp: Path):
    path = tmp / "transcript_adv.txt"
    path.write_text(
        "[00:00:01] Jane Customer: Please remove West Wing from scope.\n"
        "[00:00:42] Unknown: maybe add 5 cameras at Main Campus.\n"
        "Decisions:\n- Move ahead with baseline.\n"
        "Open Questions:\n- MDF badge access?\n",
        encoding="utf-8",
    )
    atoms = TranscriptParser().parse_artifact("proj", "art_tx", path)
    expected = ParserCaseExpectation(
        expected_atom_types={"customer_instruction", "open_question", "quantity"},
        expected_entity_keys={"site:west_wing"},
        expected_authorities={"meeting_note"},
        expected_review_flags={"verbal_commitment_requires_confirmation"},
    )
    return [evaluate_case_metrics(atoms, expected)]


def _quote_cases(tmp: Path):
    path = tmp / "quote_adv.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["Part Number", "Description", "Quantity", "Unit Price", "Lead Time"])
    ws.append(["cam-ip-001", "IP Cam", "72", "$300.00", ""])
    ws.append(["TOTAL", "", "72", "", ""])
    wb.save(path)
    atoms = QuoteParser().parse_artifact("proj", "art_quote", path)
    expected = ParserCaseExpectation(
        expected_atom_types={"vendor_line_item", "quantity"},
        expected_entity_keys={"device:ip_camera"},
        expected_quantities={72.0},
        expected_authorities={"vendor_quote"},
    )
    txt = tmp / "quote_adv.txt"
    txt.write_text("Part Number|Description|Quantity|Unit Price|Lead Time\nCAM-IP-002|IP Cam|10|$280.00|1 week\n", encoding="utf-8")
    txt_atoms = QuoteParser().parse_artifact("proj", "art_quote_txt", txt)
    txt_expected = ParserCaseExpectation(expected_atom_types={"vendor_line_item", "quantity"})
    return [evaluate_case_metrics(atoms, expected), evaluate_case_metrics(txt_atoms, txt_expected)]


def run_parser_benchmark(out: Path) -> ParserBenchmarkReport:
    tmp = Path(tempfile.mkdtemp(prefix="parser_bench_"))
    reports = [
        aggregate_parser_metrics("xlsx", _xlsx_cases(tmp)),
        aggregate_parser_metrics("email", _email_cases(tmp)),
        aggregate_parser_metrics("docx", _docx_cases(tmp)),
        aggregate_parser_metrics("transcript", _transcript_cases(tmp)),
        aggregate_parser_metrics("quote", _quote_cases(tmp)),
    ]
    aggregate = _aggregate_reports(reports)
    report = ParserBenchmarkReport(
        aggregate_metrics=aggregate,
        parser_reports=reports,
        threshold_failures=[],
    )
    report.threshold_failures = parser_threshold_failures(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parser-specific adversarial benchmark")
    parser.add_argument("--out", type=Path, required=True, help="Output benchmark report JSON path")
    parser.add_argument("--allow-fail", action="store_true", help="Exit zero even when thresholds fail")
    args = parser.parse_args()
    report = run_parser_benchmark(args.out)
    print(
        json.dumps(
            {
                "aggregate_metrics": report.aggregate_metrics,
                "threshold_failures": report.threshold_failures,
            }
        )
    )
    if report.threshold_failures and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
