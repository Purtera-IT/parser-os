# -*- coding: utf-8 -*-
"""Does a warm cache make the embedder being DOWN stop changing the atoms?

That is the whole claim behind shipping the embedding cache. Parse a deal
twice in one process -- once normally, once with `_reachable()` forced False,
which is exactly what a dead endpoint looks like to a SemanticRule -- and
compare the text of every atom.

Before the ordering fix this could only fail: `fires()` asked reachability
BEFORE consulting the cache, so a dead endpoint sent every rule down the regex
path no matter what was on disk.

    DEAL=<uuid> python _embedder_down_test.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")

# The cache the warm pass filled.
os.environ.setdefault(
    "SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "embed_cache.db"))

DEAL = os.environ["DEAL"]


def main() -> None:
    from azure.storage.blob import BlobServiceClient

    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    tmp = Path(tempfile.mkdtemp(prefix="down_"))
    paths = []
    for b in list(cc.list_blobs(name_starts_with="deals/%s/artifacts/" % DEAL))[:15]:
        p = tmp / b.name.split("/")[-1]
        p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
        paths.append(p)

    from app.parsers import _ocr_chain
    _ocr_chain.ocr_pdf_page = lambda *a, **k: {"text": "", "lines": [], "tables": []}
    from app.parsers import email_parser as ep
    ep._ocr_text_from_cid_inline = lambda payload, content_type="": ""

    from app.core.semantic_rules import SemanticRule, reset_semantic_backend
    from app.parsers.registry import choose_parser

    def run() -> tuple[str, int]:
        h = hashlib.sha256()
        n = 0
        for p in sorted(paths, key=lambda x: x.name):
            try:
                parser, _m, _a = choose_parser(p)
                if parser is None:
                    continue
                out = parser.parse_artifact_full(
                    project_id=DEAL, artifact_id="a", path=p, domain_pack=None)
                for a in getattr(out, "atoms", out):
                    t = str(getattr(a, "raw_text", "") or getattr(a, "text", "") or "")
                    h.update(b"\0" + t.encode("utf-8", "replace"))
                    n += 1
            except Exception:
                pass
        return h.hexdigest()[:12], n

    print("cache: %s" % os.environ["SOWSMITH_EMBED_CACHE_DB"])

    reset_semantic_backend()
    up_digest, up_n = run()
    print("embedder UP   : %5d atoms  %s" % (up_n, up_digest))

    real = SemanticRule._reachable
    SemanticRule._reachable = lambda self: False
    try:
        reset_semantic_backend()
        down_digest, down_n = run()
    finally:
        SemanticRule._reachable = real
    print("embedder DOWN : %5d atoms  %s" % (down_n, down_digest))

    same = up_digest == down_digest
    print("\n%s" % (
        "IDENTICAL -- the cache carries the decision, the network does not."
        if same else
        "*** DIFFERENT -- the embedder's state is still moving the atoms. ***"))
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
