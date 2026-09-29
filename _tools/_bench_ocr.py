"""Worker widths when OCR actually costs what Document Intelligence costs.

The CPU-only benchmark says threads buy nothing on 4 cores. But the reason
`PARSE_WORKERS` exists is the OCR round trip, so stub it with a real latency
instead of zero and measure again.
"""
import os, statistics, sys, tempfile, time
from pathlib import Path
import psutil

sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
HERE = Path(__file__).parent
DEAL = os.environ["DEAL"]
LAT = float(os.environ.get("OCR_LATENCY", "1.5"))
PIN = int(os.environ.get("PIN_CPUS", "4"))
psutil.Process().cpu_affinity(list(range(PIN)))

from azure.storage.blob import BlobServiceClient
conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
cc = BlobServiceClient.from_connection_string(conn).get_container_client("orbitbrief-artifacts")
tmp = Path(tempfile.mkdtemp(prefix="ocrb_")); paths = []
for b in list(cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/"))[:24]:
    p = tmp / b.name.split("/")[-1]
    p.write_bytes(cc.get_blob_client(b.name).download_blob().readall()); paths.append(p)

calls = {"n": 0}
from app.parsers import email_parser as ep
def fake_ocr(payload, content_type=""):
    calls["n"] += 1
    time.sleep(LAT)          # a Document Intelligence round trip
    return "ocr text"
ep._ocr_text_from_cid_inline = fake_ocr

from app.parsers.registry import choose_parser
from concurrent.futures import ThreadPoolExecutor

def one(p):
    try:
        parser, _m, _a = choose_parser(p)
        if parser is None: return 0
        out = parser.parse_artifact_full(project_id="b", artifact_id="a", path=p, domain_pack=None)
        return len(getattr(out, "atoms", out))
    except Exception:
        return 0

def run(w):
    t0 = time.perf_counter()
    if w <= 1: r = [one(p) for p in paths]
    else:
        with ThreadPoolExecutor(max_workers=w) as pool: r = list(pool.map(one, paths))
    return time.perf_counter() - t0, sum(r)

print("%d artifacts, OCR stubbed at %.1fs, pinned to %d CPUs\n" % (len(paths), LAT, PIN))
print("%8s%11s%9s%8s%9s" % ("workers", "median s", "atoms", "vs 1", "ocr calls"))
base = None
for w in [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]:
    runs = []
    for _ in range(int(os.environ.get("REPEATS", "2"))):
        calls["n"] = 0
        runs.append(run(w))
    med = statistics.median(r[0] for r in runs)
    if base is None: base = med
    print("%8d%11.2f%9d%8s%9d" % (w, med, runs[0][1], "%.2fx" % (base/med), calls["n"]))
