# -*- coding: utf-8 -*-
"""Rebuild a deal's newest manifest against the LIVE attachment set.

The failure this repairs: HubSpot re-fetched six emails, each came back a
different size, so each got a new content_sha256 and a new attachments row while
the old row was soft-deleted. The newest manifest was written before those rows
landed, so it still points at the six DELETED blobs -- and because the filenames
are identical on both sides, every filename-based check says the manifest is
complete. It is not: six of its sixty-eight artifacts are stale shas.

Re-compiling on that manifest parses the deleted copies and never sees the live
ones, which is what "the reparse is broken" looks like from the outside.

Only the six stale entries are touched, and only their identity fields. The
derived fields -- authored_at, direction, sender_domain -- are NOT columns on
`attachments`; they are computed by the finalize function. Carrying them over
from the stale entry keeps them, because a re-fetch of the same HubSpot message
is the same message: same author, same timestamp, same direction. Rebuilding the
manifest from the database instead would silently drop all three and degrade
dedup, ordering and document_copy_of.

    DEAL=<uuid> PG=... python _fix_manifest.py          # dry run
    DEAL=<uuid> PG=... python _fix_manifest.py --apply
"""
import json, os, sys, uuid
from pathlib import Path

import psycopg2
from azure.storage.blob import BlobServiceClient

DEAL = os.environ["DEAL"]
HERE = Path(__file__).parent
CONTAINER = "orbitbrief-artifacts"


def main() -> None:
    apply = "--apply" in sys.argv
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(CONTAINER)

    blobs = [b for b in cc.list_blobs(name_starts_with=f"deals/{DEAL}/parser-manifests/")
             if b.name.endswith(".json")]
    newest = max(blobs, key=lambda b: b.last_modified)
    man = json.loads(cc.get_blob_client(newest.name).download_blob().readall())
    print(f"base manifest: {newest.name.split('/')[-1]}  "
          f"({newest.last_modified:%Y-%m-%d %H:%M:%S}, {len(man.get('artifacts') or [])} artifacts)")

    cn = psycopg2.connect(os.environ["PG"], connect_timeout=30)
    cur = cn.cursor()
    cur.execute("""SELECT attachment_id, file_name, azure_blob_url, content_sha256,
                          file_size_bytes, mime_type, source, external_id, created_at, updated_at,
                          parse_status
                     FROM public.attachments
                    WHERE deal_id::text=%s AND deleted_at IS NULL""", (DEAL,))
    live = [dict(zip(("attachment_id", "filename", "blob_url", "content_sha256", "size_bytes",
                      "mime_type", "source", "external_id", "created_at", "updated_at",
                      "parse_status"), r)) for r in cur.fetchall()]
    cur.close(); cn.close()

    live_by_sha = {r["content_sha256"]: r for r in live}
    live_by_name: dict[str, list] = {}
    for r in live:
        live_by_name.setdefault(r["filename"], []).append(r)

    arts = man.get("artifacts") or []
    swapped, unresolved = [], []
    for a in arts:
        if a.get("content_sha256") in live_by_sha:
            continue                                   # already current
        cands = [r for r in live_by_name.get(a.get("filename"), [])
                 if r["content_sha256"] not in {x.get("content_sha256") for x in arts}]
        if len(cands) != 1:
            unresolved.append((a.get("filename"), len(cands)))
            continue
        new = cands[0]
        old_sha, old_size = a.get("content_sha256"), a.get("size_bytes")
        # Identity fields move; authored_at / direction / sender_domain stay.
        a["attachment_id"] = str(new["attachment_id"])
        a["blob_url"] = new["blob_url"]
        a["content_sha256"] = new["content_sha256"]
        a["size_bytes"] = new["size_bytes"]
        if new["mime_type"]:
            a["mime_type"] = new["mime_type"]
        for k, v in (("created_at", new["created_at"]), ("updated_at", new["updated_at"])):
            if v is not None:
                a[k] = v.isoformat() if hasattr(v, "isoformat") else str(v)
        swapped.append((a.get("filename"), old_sha, new["content_sha256"], old_size,
                        new["size_bytes"]))

    print(f"\nstale artifacts swapped to the live blob: {len(swapped)}")
    for fn, o, n, os_, ns in swapped:
        print(f"   {fn[:46]:48} {str(o)[:10]} -> {str(n)[:10]}  {os_:>9,} -> {ns:>9,} bytes")
    if unresolved:
        print(f"\nCOULD NOT RESOLVE {len(unresolved)} (not swapped):")
        for fn, n in unresolved:
            print(f"   {fn[:52]}  candidates={n}")

    live_shas = {r["content_sha256"] for r in live}
    man_shas = {a.get("content_sha256") for a in arts}
    print(f"\nafter fix -- live not in manifest: {len(live_shas - man_shas)}   "
          f"manifest not live: {len(man_shas - live_shas)}")
    pend = [r['filename'] for r in live if r['parse_status'] == 'pending']
    print(f"pending attachments now carried by the manifest: "
          f"{sum(1 for r in live if r['parse_status']=='pending' and r['content_sha256'] in man_shas)}"
          f"/{len(pend)}")

    if not apply:
        print("\ndry run -- pass --apply to upload a new manifest")
        return
    if unresolved:
        raise SystemExit("refusing to upload while artifacts are unresolved")

    new_id = str(uuid.uuid4())
    man["compile_id"] = new_id
    name = f"deals/{DEAL}/parser-manifests/{new_id}.json"
    cc.get_blob_client(name).upload_blob(
        json.dumps(man, indent=1, ensure_ascii=False).encode("utf-8"), overwrite=False)
    print(f"\nuploaded {name}")
    print(f"manifest id: {new_id}")


if __name__ == "__main__":
    main()
