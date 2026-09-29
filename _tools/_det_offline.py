"""Determinism with every NETWORK path removed.

Splits the question in two: state shared between threads (our bug, fixable)
versus a model or endpoint answering differently (inherent, needs caching or
pinning). With OCR stubbed, the LLM disabled and semantic rules forced to the
lexical fallback, nothing here leaves the process -- so anything that still
varies is shared state.
"""
import os, sys, tempfile, hashlib
os.environ["SOWSMITH_DISABLE_LLM"] = "1"
os.environ["SOWSMITH_SEMANTIC_RULES"] = "0"
from pathlib import Path
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor

DEAL = os.environ["DEAL"]
conn = Path(__file__).parent.joinpath(".bloburl").read_text(encoding="utf-8").strip()
cc = BlobServiceClient.from_connection_string(conn).get_container_client("orbitbrief-artifacts")
tmp = Path(tempfile.mkdtemp()); paths = []
for b in list(cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/"))[:20]:
    p = tmp / b.name.split("/")[-1]
    p.write_bytes(cc.get_blob_client(b.name).download_blob().readall()); paths.append(p)

from app.parsers import _ocr_chain
_ocr_chain.ocr_pdf_page = lambda *a, **k: {"text": "", "lines": [], "tables": []}
from app.parsers import email_parser as ep
ep._ocr_text_from_cid_inline = lambda payload, content_type="": ""

from app.parsers.registry import choose_parser
def parse(p):
    parser, _m, _a = choose_parser(p)
    if parser is None: return []
    out = parser.parse_artifact_full(project_id="det", artifact_id="a", path=p, domain_pack=None)
    return [repr(a) for a in getattr(out, "atoms", out)]
def digest(res):
    h = hashlib.sha256()
    for n in sorted(res):
        h.update(n.encode())
        for t in res[n]: h.update(b"\0" + t.encode("utf-8", "replace"))
    return h.hexdigest()[:12]

ref = None; ok = True
plan = [(w, r) for w in (1, 4, 8) for r in range(3)]
for w, r in plan:
    if w == 1: res = {p.name: parse(p) for p in paths}
    else:
        with ThreadPoolExecutor(max_workers=w) as pool:
            res = dict(zip([p.name for p in paths], pool.map(parse, paths)))
    d = digest(res); n = sum(len(v) for v in res.values())
    if ref is None: ref = d
    same = d == ref; ok &= same
    print("  width %d run %d: %5d atoms  %s  %s" % (w, r, n, d, "same" if same else "*** DIFFERS ***"))
print("OFFLINE DETERMINISM:", "clean" if ok else "*** SHARED STATE REMAINS ***")
