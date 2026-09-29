# -*- coding: utf-8 -*-
"""Back up every human label on every labelled deal, and restore it.

The labels are the only thing in this system that cannot be regenerated. A
compile can be re-run, an envelope rebuilt, a manifest repaired -- the judgement
in a note took a person an hour and exists nowhere else. Before any change to
the parser or to how atoms are keyed, that judgement gets copied somewhere a
mistake cannot reach.

Backs up all four label tables, writes a timestamped snapshot locally AND to
blob (so it survives this machine), and can restore any snapshot.

Only rows written by a person are backed up. `human_labels` skips any other
labeler, so rows under a tool name are not gold; keeping them would inflate the
counts and make a restore look successful when the gold was lost.

    PG=... python backup_labels.py                 # snapshot every labelled deal
    PG=... python backup_labels.py --list          # what snapshots exist
    PG=... python backup_labels.py --restore <name> --apply
"""
import json, os, sys, datetime as dt
from pathlib import Path

import psycopg2
import psycopg2.extras

HERE = Path(__file__).parent
OUT = HERE / "_label_backups"
CONTAINER = "orbitbrief-artifacts"
BLOB_PREFIX = "label-backups/"

TABLES = ("atom_labels", "atom_label_links", "atom_label_judgments",
          "atom_label_deal_answers")
#: Rows whose labeler is not a person are not gold -- `human_labels._is_a_person`
#: skips them -- so they are excluded rather than restored as if they counted.
HUMAN = "labeler LIKE '%@%' AND labeler NOT ILIKE '%claude%' AND labeler NOT ILIKE '%assistant%'"


def _blob():
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    return BlobServiceClient.from_connection_string(conn).get_container_client(CONTAINER)


def snapshot() -> None:
    cn = psycopg2.connect(os.environ["PG"], connect_timeout=40)
    cur = cn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    snap = {"taken_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tables": {}}

    for t in TABLES:
        try:
            cur.execute(f"SELECT * FROM public.{t} WHERE {HUMAN}")
            rows = [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            cn.rollback()
            print(f"   {t:28} SKIPPED ({type(exc).__name__})")
            continue
        snap["tables"][t] = rows
        deals = {r.get("deal_id") for r in rows}
        print(f"   {t:28} {len(rows):>5} rows across {len(deals)} deal(s)")

    cur.close(); cn.close()

    name = f"labels_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.json"
    OUT.mkdir(exist_ok=True)
    body = json.dumps(snap, indent=1, ensure_ascii=False, default=str).encode("utf-8")
    (OUT / name).write_bytes(body)
    print(f"\n   local  {OUT / name}  ({len(body):,} bytes)")

    # Blob too: a backup that only exists on the machine doing the risky work is
    # not a backup.
    try:
        _blob().get_blob_client(BLOB_PREFIX + name).upload_blob(body, overwrite=False)
        print(f"   blob   {CONTAINER}/{BLOB_PREFIX}{name}")
    except Exception as exc:
        print(f"   blob   UPLOAD FAILED ({type(exc).__name__}: {exc}) -- local copy only")

    by_deal = {}
    for r in snap["tables"].get("atom_labels", []):
        by_deal[r.get("deal_id")] = by_deal.get(r.get("deal_id"), 0) + 1
    print("\n   labels per deal:")
    for d, n in sorted(by_deal.items(), key=lambda kv: -kv[1]):
        print(f"      {n:>5}  {d}")


def list_snaps() -> None:
    if OUT.is_dir():
        print("local:")
        for p in sorted(OUT.glob("labels_*.json")):
            print(f"   {p.name}  {p.stat().st_size:,} bytes")
    try:
        print("blob:")
        for b in sorted(_blob().list_blobs(name_starts_with=BLOB_PREFIX),
                        key=lambda b: b.name):
            print(f"   {b.name.split('/')[-1]}  {b.size:,} bytes  {b.last_modified:%Y-%m-%d %H:%M}")
    except Exception as exc:
        print(f"   (blob unreadable: {type(exc).__name__})")


def restore(name: str, apply: bool) -> None:
    path = OUT / name
    if path.is_file():
        snap = json.loads(path.read_text(encoding="utf-8"))
    else:
        snap = json.loads(_blob().download_blob(BLOB_PREFIX + name).readall())
    print(f"snapshot {name}  taken {snap.get('taken_at')}")

    cn = psycopg2.connect(os.environ["PG"], connect_timeout=40)
    cur = cn.cursor()
    total = 0
    for t, rows in snap["tables"].items():
        if not rows:
            continue
        cols = list(rows[0].keys())
        # Re-insert only what is missing. A restore must never overwrite work
        # done since the snapshot -- that would turn a safety net into the
        # accident it exists to undo.
        sql = (f"INSERT INTO public.{t} ({', '.join(cols)}) "
               f"VALUES ({', '.join(['%s'] * len(cols))}) ON CONFLICT DO NOTHING")
        print(f"   {t:28} {len(rows):>5} rows -> insert-if-missing")
        if apply:
            for r in rows:
                cur.execute(sql, [r[c] for c in cols])
        total += len(rows)
    if apply:
        cn.commit()
        print(f"\nrestored {total} row(s) (existing rows untouched)")
    else:
        print(f"\ndry run -- would restore {total} row(s); pass --apply")
    cur.close(); cn.close()


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_snaps()
    elif "--restore" in sys.argv:
        i = sys.argv.index("--restore")
        restore(sys.argv[i + 1], "--apply" in sys.argv)
    else:
        snapshot()
