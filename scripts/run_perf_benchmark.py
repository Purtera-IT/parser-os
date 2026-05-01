from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.compiler import compile_project


def _build_synthetic_project(base_dir: Path, sites: int, devices_per_site: int) -> Path:
    project_dir = base_dir / "perf_project"
    project_dir.mkdir(parents=True, exist_ok=True)

    site_wb = Workbook()
    site_ws = site_wb.active
    site_ws.title = "site_roster"
    site_ws.append(["Site", "Floor", "Device", "Quantity", "Access Window", "Scope"])
    total_qty = 0
    for site_index in range(1, sites + 1):
        for device_index in range(1, devices_per_site + 1):
            qty = device_index + 1
            total_qty += qty
            site_ws.append(
                [
                    f"Site {site_index}",
                    str((site_index % 5) + 1),
                    f"IP Camera {device_index}",
                    qty,
                    "Weekdays 8am-5pm",
                    "Install",
                ]
            )
    site_ws.append(["TOTAL", "", "", total_qty, "", ""])
    site_wb.save(project_dir / "site_list.xlsx")
    site_wb.close()

    quote_wb = Workbook()
    quote_ws = quote_wb.active
    quote_ws.title = "quote"
    quote_ws.append(["Part Number", "Description", "Quantity", "Unit Price", "Lead Time"])
    quote_ws.append(["CAM-IP-001", "IP Camera", total_qty, "300.00", "2 weeks"])
    quote_wb.save(project_dir / "vendor_quote.xlsx")
    quote_wb.close()

    (project_dir / "customer_email.txt").write_text(
        "From: perf@example.com\n"
        "Subject: Performance test\n\n"
        "Main Campus requires escort access after 5pm.\n",
        encoding="utf-8",
    )
    return project_dir


def run_benchmark(sites: int, devices_per_site: int) -> dict[str, object]:
    temp_dir = Path(tempfile.mkdtemp(prefix="purtera_perf_"))
    project_dir = _build_synthetic_project(temp_dir, sites=sites, devices_per_site=devices_per_site)
    result = compile_project(
        project_dir=project_dir,
        project_id=f"perf_sites_{sites}_devices_{devices_per_site}",
        allow_unverified_receipts=True,
    )

    total_ms = result.trace.total_duration_ms if result.trace is not None else 0.0
    seconds = max(total_ms / 1000.0, 0.001)
    report = {
        "artifacts": len(result.manifest.artifact_fingerprints) if result.manifest is not None else 0,
        "atoms": len(result.atoms),
        "edges": len(result.edges),
        "packets": len(result.packets),
        "total_duration_ms": round(total_ms, 3),
        "atoms_per_second": round(len(result.atoms) / seconds, 3),
        "packets_per_second": round(len(result.packets) / seconds, 3),
        "sites": sites,
        "devices_per_site": devices_per_site,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compile performance benchmark against synthetic fixtures.")
    parser.add_argument("--sites", type=int, default=100, help="Number of synthetic sites")
    parser.add_argument("--devices", type=int, default=1, help="Devices per site")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    report = run_benchmark(sites=args.sites, devices_per_site=args.devices)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
