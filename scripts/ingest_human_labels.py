"""Pull human atom labels from blob (or a local dir) into _training_human.db.

    python scripts/ingest_human_labels.py                 # from blob
    python scripts/ingest_human_labels.py --dir labels/   # from local JSON files
    python scripts/ingest_human_labels.py --gold-export ~/Downloads/gold_labels.json  # + old zip exports

Blob: ``orbitbrief-artifacts/_labeling/labels/<deal_id>.json``, written by the
purpulse atom labeler on every save. Uses AZURE_STORAGE_CONNECTION_STRING, or
falls back to DefaultAzureCredential (``az login``) on purpulsedevstg01.
Then rebuild the multitask table as usual (``python -m
app.learning.multitask_table``); it globs ``_training_*.db``.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.learning.human_labels import docs_from_gold_export, write_db  # noqa: E402

PREFIX = "_labeling/labels/"


def _blob_docs(container: str, account: str):
    from azure.storage.blob import BlobServiceClient

    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if conn:
        svc = BlobServiceClient.from_connection_string(conn)
    else:
        from azure.identity import DefaultAzureCredential

        svc = BlobServiceClient(f"https://{account}.blob.core.windows.net", DefaultAzureCredential())
    cc = svc.get_container_client(container)
    for b in cc.list_blobs(name_starts_with=PREFIX):
        if b.name.endswith(".json"):
            yield json.loads(cc.download_blob(b.name).readall().decode("utf-8"))


def _dir_docs(path: Path):
    for p in sorted(path.glob("*.json")):
        yield json.load(io.open(p, encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--gold-export", type=Path, action="append", default=[],
                    help="also fold in an offline zip-labeler gold_labels*.json (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("_training_human.db"))
    ap.add_argument("--container", default="orbitbrief-artifacts")
    ap.add_argument("--account", default="purpulsedevstg01")
    a = ap.parse_args()
    docs = list(_dir_docs(a.dir) if a.dir else _blob_docs(a.container, a.account))
    for p in a.gold_export:
        docs.extend(docs_from_gold_export(json.load(io.open(p, encoding="utf-8")), labeler=p.stem))
    rep = write_db(docs, a.out)
    print(f"deals={rep.deals} labels={rep.labels} rows={rep.rows} "
          f"deal_answers={rep.deal_answers} -> {a.out}")
    for why, n in sorted(rep.skipped.items(), key=lambda kv: -kv[1]):
        print(f"  skipped {n}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
