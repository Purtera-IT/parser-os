# -*- coding: utf-8 -*-
"""Impact run for "a signature does not swallow a table": same emails, twice.

A deploy is not needed to know whether a parser change loses content. Parse the
real .eml artifacts with the change absent, then present, and diff the atoms.
Anything the second run does not produce that the first did is a LOSS, named,
before it reaches a deal.

Tests written alongside a fix share its blind spot. The documents do not.

    python _tools/_impact_signature_table.py dump  > out.json
"""
import json, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from azure.storage.blob import BlobServiceClient  # noqa: E402

CONN = (HERE / ".bloburl").read_text().strip()
CONTAINER = "orbitbrief-artifacts"
MAX_DEALS = int(os.environ.get("IMPACT_DEALS", "14"))
MAX_EMAILS_PER_DEAL = int(os.environ.get("IMPACT_EMAILS", "4"))


def dump() -> None:
    from app.parsers.email_parser import EmailParser

    cc = BlobServiceClient.from_connection_string(CONN).get_container_client(CONTAINER)
    # Deals named explicitly always go in. A lexical walk from the top never
    # reaches a deal whose uuid starts with 'd', and an impact run that cannot
    # reach the document the fix was written for proves only that nothing else
    # broke.
    deals: list[str] = [d for d in os.environ.get("IMPACT_INCLUDE", "").split(",") if d.strip()]
    for p in cc.walk_blobs(name_starts_with="deals/", delimiter="/"):
        name = getattr(p, "name", "")
        parts = name.split("/")
        if len(parts) > 1 and parts[1]:
            deals.append(parts[1])
        if len(deals) >= MAX_DEALS:
            break

    out: dict[str, list[str]] = {}
    tmp = Path(tempfile.mkdtemp())
    for deal in deals:
        seen = 0
        for b in cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"):
            if not b.name.lower().endswith(".eml"):
                continue
            if seen >= MAX_EMAILS_PER_DEAL:
                break
            seen += 1
            local = tmp / f"{deal[:8]}-{seen}.eml"
            local.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
            try:
                res = EmailParser().parse(local)
                atoms = getattr(res, "atoms", res) or []
                out[f"{deal[:8]}/{Path(b.name).name}"] = sorted(
                    {" ".join(str(getattr(a, "raw_text", "") or getattr(a, "text", "") or "").split())
                     for a in atoms}
                )
            except Exception as exc:  # a parser that raises is its own finding
                out[f"{deal[:8]}/{Path(b.name).name}"] = [f"__PARSE_ERROR__ {type(exc).__name__}: {exc}"]
    json.dump(out, sys.stdout)


def compare(before_path: str, after_path: str) -> int:
    before = json.load(open(before_path, encoding="utf-8"))
    after = json.load(open(after_path, encoding="utf-8"))
    lost_total = gained_total = 0
    lost_examples: list[str] = []
    for key, old_atoms in before.items():
        new_atoms = set(after.get(key, []))
        old_set = set(old_atoms)
        lost = old_set - new_atoms
        gained = new_atoms - old_set
        lost_total += len(lost)
        gained_total += len(gained)
        for t in sorted(lost)[:3]:
            lost_examples.append(f"  {key}: {t[:90]}")
    print(f"documents compared : {len(before)}")
    print(f"atoms LOST         : {lost_total}")
    print(f"atoms GAINED       : {gained_total}")
    if lost_examples:
        print("\nLOST (the gate):")
        print("\n".join(lost_examples[:25]))
    errs = [k for k, v in after.items() if v and str(v[0]).startswith("__PARSE_ERROR__")]
    if errs:
        print(f"\nparse errors after: {len(errs)} -> {errs[:5]}")
    return 1 if lost_total else 0


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump()
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
