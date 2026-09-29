# -*- coding: utf-8 -*-
"""Determinism ACROSS PROCESSES, which the in-process harness cannot test.

`_determinism.py` runs every width in one interpreter, so every run shares one
hash seed. Python randomises string hashing per process, so any code that
iterates a `set` of strings to produce output order is stable inside that
harness and unstable in production -- where every compile is a new process.

That is exactly how `email_parser` emitted its OCR atoms: `targets` is a set of
content-ids, and `for cid in targets` put the same email's atoms in a different
order in every process. Two runs agreed only when PYTHONHASHSEED was pinned.

This forks a child per run, with the seed left random, and compares the full
atom SEQUENCE. Set-order bugs show as a reordering with an identical multiset,
which this reports separately from genuine content loss -- they are different
faults and need different fixes.

    DEAL=<uuid> RUNS=3 python _determinism_xproc.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = r"C:\Users\lilli\parser-os-labeling"

CHILD = r'''
import hashlib, json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, r"{root}")
DEAL = os.environ["DEAL"]
from azure.storage.blob import BlobServiceClient
conn = Path(r"{here}").joinpath(".bloburl").read_text(encoding="utf-8").strip()
cc = BlobServiceClient.from_connection_string(conn).get_container_client("orbitbrief-artifacts")
tmp = Path(tempfile.mkdtemp()); paths = []
for b in list(cc.list_blobs(name_starts_with="deals/%s/artifacts/" % DEAL))[:{limit}]:
    p = tmp / b.name.split("/")[-1]
    p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
    paths.append(p)

# A DETERMINISTIC stand-in for OCR: keyed on the image, so a reordering is
# visible but the vision bill is not paid. Stubbing it to "" (as the
# in-process harness does) hides every OCR-ordering fault.
import app.parsers._ocr_chain as oc
oc._ocr_image_bytes_uncached = lambda image_bytes, notes: {{
    "text": "OCR:" + hashlib.sha256(image_bytes).hexdigest()[:8],
    "backend": "stub", "confidence": 0.9, "notes": []}}
oc.ocr_pdf_page = lambda *a, **k: {{"text": "", "lines": [], "tables": []}}

from app.parsers.registry import choose_parser
texts = []
for p in sorted(paths, key=lambda x: x.name):
    try:
        parser, _m, _a = choose_parser(p)
        if parser is None:
            continue
        out = parser.parse_artifact_full(project_id=DEAL, artifact_id="a",
                                         path=p, domain_pack=None)
        for a in getattr(out, "atoms", out):
            texts.append(str(getattr(a, "raw_text", "") or getattr(a, "text", "") or ""))
    except Exception:
        pass
Path(sys.argv[1]).write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
'''


def run_child(deal: str, out: Path, limit: int) -> list[str]:
    src = HERE / "_xproc_child.py"
    src.write_text(CHILD.format(root=ROOT, here=str(HERE), limit=limit), encoding="utf-8")
    env = dict(os.environ, DEAL=deal)
    env.pop("PYTHONHASHSEED", None)      # the point: let it be random
    subprocess.run([sys.executable, str(src), str(out)], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if not out.exists():
        return []
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    deal = os.environ["DEAL"]
    runs = int(os.environ.get("RUNS", "3"))
    limit = int(os.environ.get("MAX_ARTIFACTS", "20"))
    tmp = Path(tempfile.mkdtemp(prefix="xproc_"))

    seqs = [run_child(deal, tmp / ("r%d.json" % i), limit) for i in range(runs)]
    ref = seqs[0]
    ok = True
    for i, s in enumerate(seqs[1:], 1):
        if s == ref:
            print("  run %d: %5d atoms  same" % (i, len(s)))
            continue
        ok = False
        same_content = Counter(s) == Counter(ref)
        kind = ("REORDERED (same atoms, different order -- a set is being "
                "iterated)" if same_content else "CONTENT DIFFERS")
        print("  run %d: %5d atoms  *** %s ***" % (i, len(s), kind))
        if not same_content:
            ca, cb = Counter(ref), Counter(s)
            for t in list((ca - cb).elements())[:3]:
                print("       only in run 0: %s" % t[:90])
            for t in list((cb - ca).elements())[:3]:
                print("       only in run %d: %s" % (i, t[:90]))
        else:
            for j, (x, y) in enumerate(zip(ref, s)):
                if x != y:
                    print("       first reorder at index %d: %r vs %r"
                          % (j, x[:60], y[:60]))
                    break
    print("  run 0: %5d atoms  <- reference" % len(ref))
    print("%s" % ("CROSS-PROCESS DETERMINISTIC" if ok
                  else "*** NOT DETERMINISTIC ACROSS PROCESSES ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
