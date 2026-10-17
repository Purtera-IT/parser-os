"""Score dev's current brief for each golden deal against its known facts.

    python scripts/check_golden.py [deal ...]     # default: every deal in golden.json

Reads deals/<uuid>/orbitbrief/latest/PM_HANDOFF.json and envelope.json from the
dev storage account (connection string from AZURE_STORAGE_CONNECTION_STRING),
and checks what a PM sees: the overview's sections, the sites, the contacts and
the questions. Each fact in tests/golden_deals/golden.json came from a graded
review of the deal's own documents; a build that breaks one fails here.
Exit code 1 when any check fails.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden_deals" / "golden.json"


def _blob(conn: str, name: str) -> dict:
    from azure.storage.blob import BlobClient

    data = BlobClient.from_connection_string(conn, "orbitbrief-artifacts", name).download_blob(
        max_concurrency=4).readall()
    return json.loads(data)


def _section(overview: str, title: str) -> str:
    m = re.search(rf"\*\*{re.escape(title)}\.\*\*(.*?)(?=\n\n\*\*|\Z)", overview or "", re.S | re.I)
    return (m.group(1) if m else "").strip()


def check(deal: str, facts: dict, conn: str) -> list[tuple[bool, str]]:
    base = f"deals/{facts['uuid']}/orbitbrief/latest/"
    pmh = _blob(conn, base + "PM_HANDOFF.json")
    env = _blob(conn, base + "envelope.json")
    es = pmh.get("executive_summary") or {}
    overview = str(es.get("overview") or "")
    low = overview.lower()
    out: list[tuple[bool, str]] = []
    out.append((es.get("source") == "c3_brief", "page shows the heads' brief"))
    fees = _section(overview, "Fees").lower()
    for s in (facts.get("fee") or {}).get("must_include", []):
        out.append((s.lower() in fees, f"fees say {s}"))
    for s in (facts.get("fee") or {}).get("must_not_include", []):
        out.append((s.lower() not in fees, f"fees do not say {s.strip()}"))
    page_sites = " | ".join(f"{s.get('name') or ''} {s.get('address') or ''}"
                            for s in pmh.get("sites") or []).lower()
    roster = " | ".join(f"{s.get('display_name') or ''} {s.get('street_address') or ''} {s.get('site') or ''}"
                        for s in (env.get("site_readiness") or {}).get("sites") or []).lower().replace("_", " ")
    for alts in (facts.get("sites") or {}).get("must_have_any_of", []):
        out.append((any(a in roster for a in alts), f"roster has {alts[0]}"))
        out.append((any(a in page_sites for a in alts), f"PM page sites have {alts[0]}"))
    for s in (facts.get("sites") or {}).get("must_not_have", []):
        out.append((s not in roster, f"roster has no {s}"))
        out.append((s not in page_sites, f"PM page sites have no {s}"))
    contacts = _section(overview, "Contacts").lower()
    for s in (facts.get("contacts") or {}).get("must_include", []):
        out.append((s.lower() in contacts, f"contacts name {s}"))
    for s in (facts.get("overview") or {}).get("must_not_include", []):
        out.append((s.lower() not in low, f"overview does not say '{s}'"))
    cust = _section(overview, "Customer responsibilities").lower()
    for s in (facts.get("customer_responsibilities") or {}).get("must_not_include", []):
        out.append((s.lower() not in cust, f"customer responsibilities do not include '{s}'"))
    qs = " | ".join(str(q.get("suggested_open_question") or "") for q in pmh.get("customer_questions") or []).lower()
    for s in (facts.get("questions") or {}).get("must_not_include", []):
        out.append((s.lower() not in qs, f"questions do not mention '{s}'"))
    return out


def main() -> int:
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn:
        print("set AZURE_STORAGE_CONNECTION_STRING (dev storage)", file=sys.stderr)
        return 2
    golden = {k: v for k, v in json.loads(GOLDEN.read_text()).items() if not k.startswith("_")}
    deals = sys.argv[1:] or list(golden)
    failed = total = 0
    for deal in deals:
        results = check(deal, golden[deal], conn)
        bad = [m for ok, m in results if not ok]
        total += len(results)
        failed += len(bad)
        print(f"{deal}: {len(results) - len(bad)}/{len(results)} pass")
        for m in bad:
            print(f"   FAIL {m}")
    print(f"ALL: {total - failed}/{total} pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
