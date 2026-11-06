"""Publish human atom labels from Postgres to the blob the trainer reads.

    python scripts/export_atom_labels.py <deal_id> [--apply]

The atom labeler writes two places when a person saves in the UI: the
``atom_labels`` table, and ``orbitbrief-artifacts/_labeling/labels/<deal>.json``
in blob. ``scripts/ingest_human_labels.py`` reads ONLY the blob, so a label
that reaches Postgres by any other route never becomes a training row.

That is not hypothetical. Labelling deal 010264 in bulk -- 3,303 labels
across 43 documents, written straight to Postgres -- left the blob holding
549 of them, the last state the UI had saved. Eighty-three percent of a
night's labelling was invisible to the trainer, and nothing said so: the
table was right, the deal looked labelled in the workspace, and the rows
simply never existed downstream.

So this rebuilds the blob document from the table, in the shape
Platform-infra's ``shared/atom-labeling.js`` writes and
``app.learning.human_labels`` parses. Dry run unless ``--apply``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONTAINER = "orbitbrief-artifacts"
PREFIX = "_labeling/labels/"

#: Columns copied straight through. Anything the table has and the document
#: does not is simply not part of the contract the trainer reads.
_FIELDS = (
    "label_key", "atom_id", "compile_id", "text", "filename", "doc_type",
    "page", "section", "lead_in", "neighbors_above", "neighbors_below",
    "table_ref", "decide_text", "origin", "link", "decide_text_version",
    "parser_type", "label_type", "is_new_type", "coarse", "hints",
    "hint_refs", "about", "wants", "consumer", "rejected", "decided_by",
    "weight_tier", "rejected_reads", "reads_kept", "reads_shown",
    "reads_set", "said_by", "said_to", "internal_only", "supplier",
    "entity_keys", "note",
)


def rows_for(deal_id: str) -> list[dict]:
    import psycopg2
    import psycopg2.extras

    dsn = os.environ.get("PG")
    if not dsn:
        raise SystemExit("set PG to the Postgres connection string")
    conn = psycopg2.connect(dsn)
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "select * from public.atom_labels where deal_id = %s "
            "order by filename, label_key", (deal_id,))
        raw = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    out = []
    for r in raw:
        row = {}
        for k in _FIELDS:
            v = r.get(k)
            if isinstance(v, _dt.datetime):
                v = v.isoformat()
            row[k] = v
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("deal_id")
    ap.add_argument("--purpose", default="train",
                    help="'eval' forces the whole deal to the holdout split")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    labels = rows_for(args.deal_id)
    if not labels:
        raise SystemExit(f"no labels in atom_labels for {args.deal_id}")

    doc = {
        "deal_id": args.deal_id,
        "purpose": args.purpose,
        "registry_version": _dt.date.today().isoformat(),
        "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="milliseconds").replace("+00:00", "Z"),
        "labels": labels,
        "deal_answers": [],
        "judgments": [],
        "links": [],
    }
    kept = sum(1 for x in labels if x.get("label_type") != "_keep")
    name = f"{PREFIX}{args.deal_id}.json"
    print(f"{len(labels)} labels ({kept} kept, {len(labels)-kept} rejected) "
          f"-> {CONTAINER}/{name}")

    if not args.apply:
        print("dry run -- pass --apply to publish")
        return

    from azure.storage.blob import BlobServiceClient

    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if conn:
        svc = BlobServiceClient.from_connection_string(conn)
    else:
        from azure.identity import DefaultAzureCredential
        svc = BlobServiceClient(
            "https://purpulsedevstg01.blob.core.windows.net",
            DefaultAzureCredential())
    cc = svc.get_container_client(CONTAINER)
    before = 0
    try:
        before = len(json.loads(
            cc.download_blob(name).readall()).get("labels") or [])
    except Exception:
        pass
    cc.upload_blob(name, json.dumps(doc).encode("utf-8"), overwrite=True)
    print(f"published. the document held {before} labels before, "
          f"{len(labels)} now.")


if __name__ == "__main__":
    main()
