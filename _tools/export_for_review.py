# -*- coding: utf-8 -*-
"""Export one deal as a self-contained review bundle -- no credentials needed to read it.

Written because a reviewer working in an editor-based agent has the repository
but not the Azure connection: the code is local, the deal is not. Telling them to
"set up API access" is the wrong answer when the thing they need is a file.

The bundle answers the three questions a parse review actually asks, per document:

  * what was extracted        -> the live atoms, with type, section and entity keys
  * what was read and dropped -> suppressed atoms, each carrying the stage that
                                 took it (substance gate, one of the dedup passes)
  * what was never read at all -> `text_coverage.unclaimed`: the lines that
                                 produced no atom, with line numbers. This is the
                                 recall side, and it is the half a reviewer cannot
                                 reconstruct from the output alone -- a miss leaves
                                 no trace unless it is reported.

Writes both a JSON bundle (for a tool to read) and a Markdown review file (for a
person or an agent to read straight through, document by document).

Reads blob ONLY -- the envelope and the parser result. It needs no database
connection, so a reviewer can be given the artifacts container and nothing else:
they can read what the parser produced and cannot read or write a single label.

    DEAL=<uuid> python export_for_review.py [outdir]
"""
import json, os, sys, collections
from pathlib import Path

from azure.storage.blob import BlobServiceClient

HERE = Path(__file__).parent


#: The storage account the artifacts container lives in, for the signed-in path.
ACCOUNT_URL = os.environ.get(
    "BLOB_ACCOUNT_URL", "https://purpulsedevstg01.blob.core.windows.net")
CONTAINER = "orbitbrief-artifacts"


def _container():
    """Where to read blobs from, in order of least privilege held by the reader.

    1. ``BLOB_SAS_URL`` -- a container SAS. What a reviewer should be given: it
       can be issued read-only, given an expiry, and revoked.
    2. An Azure identity (``az login``, managed identity, service principal) via
       ``DefaultAzureCredential``. No secret changes hands at all.
    3. ``BLOB_CONN`` or the local ``.bloburl`` -- the account connection string,
       which carries the account key. Full rights, no expiry, cannot be revoked
       without rotating the key. Last resort, and never given to a reviewer.

    On (2): a subscription role such as Contributor is NOT enough. Blob reads go
    through the data plane and need a data role -- Storage Blob Data Reader. An
    identity holding only Contributor authenticates fine and then fails every
    read with 403, which reads like a bad credential rather than a missing role.
    That has already cost this project a day once.
    """
    sas = os.environ.get("BLOB_SAS_URL")
    if sas:
        from azure.storage.blob import ContainerClient
        return ContainerClient.from_container_url(sas)

    conn = os.environ.get("BLOB_CONN")
    if not conn and (HERE / ".bloburl").is_file():
        conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    if conn:
        return BlobServiceClient.from_connection_string(conn).get_container_client(
            CONTAINER)

    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise SystemExit(
            "No blob credential. Set BLOB_SAS_URL to a container SAS, or "
            "`pip install azure-identity` and `az login` with the Storage Blob "
            "Data Reader role on the storage account.")
    return BlobServiceClient(
        ACCOUNT_URL, credential=DefaultAzureCredential()).get_container_client(
            CONTAINER)


DEAL = os.environ.get("DEAL") or ""
if not DEAL:
    raise SystemExit("set DEAL=<uuid>")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else HERE)

_DEDUP = ("quoted_history_dedup", "semantic_dedup", "pasted_note_dedup",
          "duplicate_atom_collapse", "pre_classify_dedup")


def _stage(flags):
    for f in flags or []:
        if f.startswith("suppressed:"):
            return f.split(":", 1)[1]
    return ""


def main() -> None:
    cc = _container()
    env = json.loads(cc.download_blob(
        f"deals/{DEAL}/orbitbrief/latest/envelope.json").readall())
    res = json.loads(cc.download_blob(
        f"deals/{DEAL}/parser-os/latest/result.json").readall())

    docs = {d.get("artifact_id"): d for d in (env.get("documents") or [])}
    name = lambda aid: (docs.get(aid) or {}).get("filename") or aid or "(unknown)"

    try:
        sys.path.insert(0, str(HERE.parent))
        from app.core.label_key import label_key
    except Exception:                                   # bundle is still usable
        label_key = lambda *a: ""

    live = collections.defaultdict(list)
    for a in env.get("atoms") or []:
        loc = a.get("locator") or {}
        text = " ".join(str(a.get("text") or "").split())
        live[a.get("artifact_id")].append({
            "atom_id": a.get("id"),
            "label_key": label_key(DEAL, name(a.get("artifact_id")), loc.get("page"), text),
            "type": a.get("atom_type"),
            "text": text,
            "section": a.get("section_path") or [],
            "entity_keys": a.get("entity_keys") or [],
            "page": loc.get("page"),
        })

    dropped = collections.defaultdict(list)
    for a in res.get("suppressed_atoms") or []:
        text = " ".join(str(a.get("raw_text") or "").split())
        if not text:
            continue
        st = _stage(a.get("review_flags"))
        dropped[a.get("artifact_id")].append({
            "type": a.get("atom_type"),
            "text": text,
            "dropped_by": st,
            "kind": "duplicate" if st in _DEDUP else "judged not a fact",
        })

    cover = {c.get("artifact_id"): c for c in (res.get("text_coverage") or [])}

    bundle, md = {"deal_id": DEAL, "documents": []}, []
    md.append(f"# Parse review bundle — deal `{DEAL}`\n")
    md.append(f"Compile `{res.get('compile_id')}` · generated from the envelope and the "
              f"parser result. Every line below came out of the parser's own ledger.\n")
    md.append("For each document: what was extracted, what was read and dropped, and "
              "what was never read at all. The third list is the one that matters most "
              "— a line that produced no atom leaves no other trace.\n")

    order = sorted(docs, key=lambda k: -len(live.get(k, [])))
    for aid in order:
        d = docs.get(aid) or {}
        fn = name(aid)
        cv = cover.get(aid) or {}
        unclaimed = [u for u in (cv.get("unclaimed") or [])
                     if str(u.get("text") or "").strip()]
        entry = {
            "artifact_id": aid, "filename": fn,
            "parser": d.get("parser_name"), "direction": d.get("direction"),
            "authored_at": d.get("authored_at"),
            "coverage": {k: cv.get(k) for k in
                         ("lines_total", "lines_claimed", "unread_count", "image_count")},
            "atoms": live.get(aid, []),
            "dropped": dropped.get(aid, []),
            "unclaimed_lines": [{"line": u.get("line"), "state": u.get("state"),
                                 "text": " ".join(str(u.get("text")).split())}
                                for u in unclaimed],
        }
        bundle["documents"].append(entry)

        tot, cl = cv.get("lines_total"), cv.get("lines_claimed")
        md.append(f"\n---\n\n## {fn}\n")
        md.append(f"`{d.get('parser_name')}` · {d.get('direction') or 'n/a'} · "
                  f"{str(d.get('authored_at'))[:16]}  \n"
                  f"**{len(entry['atoms'])} atoms** · {len(entry['dropped'])} dropped · "
                  f"{len(unclaimed)} lines unclaimed"
                  + (f" · {cl}/{tot} lines claimed" if tot else "") + "\n")

        if entry["atoms"]:
            md.append("\n### Extracted\n")
            for a in entry["atoms"]:
                sec = " > ".join(a["section"]) if a["section"] else ""
                md.append(f"- **`{a['type']}`**{f' _(§{sec})_' if sec else ''} — {a['text']}"
                          + (f"  \n  `keys:` {', '.join(a['entity_keys'])}"
                             if a["entity_keys"] else ""))
        if entry["dropped"]:
            md.append("\n### Read, then dropped\n")
            for a in entry["dropped"]:
                md.append(f"- _{a['kind']}_ (`{a['dropped_by']}`) — {a['text'][:300]}")
        if unclaimed:
            md.append("\n### Never read — produced no atom\n")
            for u in entry["unclaimed_lines"]:
                md.append(f"- L{u['line']} _({u['state']})_ — {u['text'][:300]}")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"deal_{DEAL[:8]}_review"
    (OUT / f"{stem}.json").write_text(
        json.dumps(bundle, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / f"{stem}.md").write_text("\n".join(md), encoding="utf-8")

    na = sum(len(e["atoms"]) for e in bundle["documents"])
    nd = sum(len(e["dropped"]) for e in bundle["documents"])
    nu = sum(len(e["unclaimed_lines"]) for e in bundle["documents"])
    print(f"{len(bundle['documents'])} documents · {na} atoms · {nd} dropped · "
          f"{nu} unclaimed lines")
    for p in (OUT / f"{stem}.json", OUT / f"{stem}.md"):
        print(f"   {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
