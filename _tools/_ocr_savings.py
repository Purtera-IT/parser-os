# -*- coding: utf-8 -*-
"""How many billed OCR calls does the READ phase make, and how many are waste?

`_ocr_cid_part` OCRs every inline image when the plain-text body references no
CID, because HTML-only `cid:` refs are invisible to a text scan and dropping
the fallback would lose real content images. Two guards keep that from being
expensive, and this counts what each one removes on live mail:

  floor   `SOWSMITH_EMAIL_IMAGE_MIN_BYTES` -- an image too small to hold text
          is a signature logo or a social icon, not content.
  cache   sha256 of the image bytes -- the same logo repeats on every message
          in a thread, and eight parse threads used to each pay for it.
"""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
HERE = Path(__file__).parent
DEALS = sys.argv[1:]

from azure.storage.blob import BlobServiceClient
conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
cc = BlobServiceClient.from_connection_string(conn).get_container_client("orbitbrief-artifacts")

import app.parsers._ocr_chain as oc
from app.parsers import email_parser as ep
from app.parsers.registry import choose_parser

billed = {"n": 0, "bytes": 0}
seen: set[str] = set()
def counting_ocr(image_bytes, notes=None):
    import hashlib
    billed["n"] += 1
    billed["bytes"] += len(image_bytes)
    seen.add(hashlib.sha256(image_bytes).hexdigest())
    return {"text": "", "backend": "stub", "notes": []}
oc._ocr_image_bytes_uncached = counting_ocr

tot = {"calls": 0, "distinct": 0, "no_floor": 0}
for deal in DEALS:
    tmp = Path(tempfile.mkdtemp()); paths = []
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
        if not b.name.lower().endswith((".eml", ".msg")):
            continue
        p = tmp / b.name.split("/")[-1]
        p.write_bytes(cc.get_blob_client(b.name).download_blob().readall()); paths.append(p)
    if not paths:
        continue
    for floor, label in ((0, "no_floor"), (3000, "with_floor")):
        os.environ["SOWSMITH_EMAIL_IMAGE_MIN_BYTES"] = str(floor)
        import importlib; importlib.reload(ep)
        oc._OCR_CACHE.clear()
        billed["n"] = 0; seen.clear()
        for p in paths:
            try:
                parser, _m, _a = choose_parser(p)
                if parser is None: continue
                parser.parse_artifact_full(project_id="o", artifact_id="a", path=p, domain_pack=None)
            except Exception:
                pass
        if label == "no_floor":
            tot["no_floor"] += billed["n"]
        else:
            tot["calls"] += billed["n"]; tot["distinct"] += len(seen)
    print("  %s: %d emails, %d billed calls without the floor, %d with it"
          % (deal[:8], len(paths), tot["no_floor"], tot["calls"]))

print("\nacross these deals: %d billed OCR calls without the byte floor, %d with it"
      % (tot["no_floor"], tot["calls"]))
if tot["no_floor"]:
    print("the floor removes %.0f%% of the billed calls" %
          (100.0 * (tot["no_floor"] - tot["calls"]) / tot["no_floor"]))
