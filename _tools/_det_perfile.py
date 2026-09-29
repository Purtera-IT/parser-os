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
tmp = Path(tempfile.mkdtemp()); paths=[]
for b in list(cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/"))[:20]:
    p = tmp / b.name.split("/")[-1]
    p.write_bytes(cc.get_blob_client(b.name).download_blob().readall()); paths.append(p)
from app.parsers import _ocr_chain
_ocr_chain.ocr_pdf_page = lambda *a, **k: {"text": "", "lines": [], "tables": []}
from app.parsers import email_parser as ep
ep._ocr_text_from_cid_inline = lambda payload, content_type="": ""
from app.parsers.registry import choose_parser
def parse(p):
    parser,_m,_a = choose_parser(p)
    if parser is None: return []
    out = parser.parse_artifact_full(project_id="det", artifact_id="a", path=p, domain_pack=None)
    return [repr(a) for a in getattr(out,"atoms",out)]
def fh(rs): return hashlib.sha256("\0".join(rs).encode("utf-8","replace")).hexdigest()[:10]

alone = {p.name: fh(parse(p)) for p in paths}
varies = {}
for trial in range(4):
    with ThreadPoolExecutor(max_workers=8) as pool:
        res = dict(zip([p.name for p in paths], pool.map(parse, paths)))
    for n, rs in res.items():
        varies.setdefault(n, set()).add(fh(rs))
print("%-58s %-12s %s" % ("file", "alone", "distinct hashes over 4 pooled runs"))
for n in sorted(varies):
    hs = varies[n]
    flag = "" if len(hs) == 1 and alone[n] in hs else "   <<< VARIES"
    print("%-58s %-12s %d%s" % (n[:56], alone[n], len(hs), flag))
