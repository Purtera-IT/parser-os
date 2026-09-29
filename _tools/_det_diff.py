import os, sys, tempfile, difflib
from pathlib import Path
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
HERE = Path(__file__).parent
DEAL = os.environ["DEAL"]

from azure.storage.blob import BlobServiceClient
conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
cc = BlobServiceClient.from_connection_string(conn).get_container_client("orbitbrief-artifacts")
tmp = Path(tempfile.mkdtemp(prefix="dd_")); paths = []
for b in list(cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/"))[:20]:
    p = tmp / b.name.split("/")[-1]
    p.write_bytes(cc.get_blob_client(b.name).download_blob().readall()); paths.append(p)

from app.parsers import email_parser as ep
ep._ocr_text_from_cid_inline = lambda payload, content_type="": ""
from app.parsers.registry import choose_parser
from concurrent.futures import ThreadPoolExecutor

def one(p):
    parser, _m, _a = choose_parser(p)
    if parser is None: return p.name, []
    out = parser.parse_artifact_full(project_id="det", artifact_id="a", path=p, domain_pack=None)
    return p.name, [repr(a) for a in getattr(out, "atoms", out)]

def run(w):
    if w <= 1: return dict(one(p) for p in paths)
    with ThreadPoolExecutor(max_workers=w) as pool: return dict(pool.map(one, paths))

a, b = run(1), run(8)
for name in sorted(a):
    if a[name] != b[name]:
        print("FILE", name, len(a[name]), "->", len(b[name]))
        for x, y in zip(a[name], b[name]):
            if x != y:
                sm = difflib.SequenceMatcher(None, x, y)
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag != "equal":
                        print("   serial[%s]: %r" % (tag, x[max(0,i1-40):i2+40]))
                        print("   par   [%s]: %r" % (tag, y[max(0,j1-40):j2+40]))
                print()
                break
