# -*- coding: utf-8 -*-
"""How many billed OCR calls does the READ phase make, and how many are waste?

`_ocr_cid_part` OCRs every inline image when the body references no CID,
because HTML-only `cid:` refs are invisible to a plain-text scan, and dropping
that fallback would lose real content images. Two guards keep it cheap:

  floor   `SOWSMITH_EMAIL_IMAGE_MIN_BYTES` -- an image too small to hold text
          is a signature logo or a social icon, not content.
  cache   sha256 of the image bytes -- the same logo repeats down a thread.

ONE setting per process. `_OCR_MIN_BYTES` is read into a module-level constant
at import, and the registry holds a parser instance bound to that module, so
setting the variable and reloading mid-run does NOT change what the registered
parser does. An earlier version of this script did exactly that and reported a
saving for deals it had not actually re-measured.

    SOWSMITH_EMAIL_IMAGE_MIN_BYTES=0 DEAL=<uuid> python _ocr_savings.py
"""
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
HERE = Path(__file__).parent
DEAL = os.environ["DEAL"]

from azure.storage.blob import BlobServiceClient
conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
cc = BlobServiceClient.from_connection_string(conn).get_container_client("orbitbrief-artifacts")

import app.parsers._ocr_chain as oc
from app.parsers.registry import choose_parser
from app.parsers import email_parser as ep

billed = {"n": 0}
distinct: set[str] = set()


def counting_ocr(image_bytes, notes=None):
    import hashlib
    billed["n"] += 1
    distinct.add(hashlib.sha256(image_bytes).hexdigest())
    return {"text": "", "backend": "stub", "notes": []}


oc._ocr_image_bytes_uncached = counting_ocr

tmp = Path(tempfile.mkdtemp())
paths = []
for b in cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/"):
    if b.name.lower().endswith((".eml", ".msg")):
        p = tmp / b.name.split("/")[-1]
        p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
        paths.append(p)

for p in paths:
    try:
        parser, _m, _a = choose_parser(p)
        if parser is None:
            continue
        parser.parse_artifact_full(project_id="o", artifact_id="a", path=p, domain_pack=None)
    except Exception:
        pass

print("%s  floor=%s  %d emails  %d billed OCR calls  (%d distinct images)"
      % (DEAL[:8], ep._OCR_MIN_BYTES, len(paths), billed["n"], len(distinct)))
