import sys, tempfile, time, hashlib, json
from pathlib import Path
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
from azure.storage.blob import BlobServiceClient
conn = Path(__file__).parent.joinpath(".bloburl").read_text(encoding="utf-8").strip()
cc = BlobServiceClient.from_connection_string(conn).get_container_client("orbitbrief-artifacts")
FILES = [
 ("deals/abcfdf7a-a3b3-47e2-9f58-718cadd96a4f/artifacts/cf59f5f7563c561c0f82386087d50608735cbeda1a4e852835f2dc3f38acc15a/010319-CDW Pricing Sheet 09-22-26.xls","010319-CDW.xls"),
 ("deals/1cf3e058-e3c0-4598-ab36-9abb28a05330/artifacts/37f402c0214060fd0558e50311fbc55aee53d9869531e0add1f0dede1f6baae8/The Collgiate Pricing Sheet ADJ 09-14-26-1.xls","Collegiate.xls"),
 ("deals/eb4568c4-d424-4a5b-817f-cfbce6c85c5e/artifacts/4e61338346baf57fb00a72f76af2529abead8ca07bf98047fe9d50a30c32128a/The Holbrok Pricing Sheet  09-15-26.xls","Holbrook.xls"),
]
tmp = Path(tempfile.mkdtemp())
from app.parsers.registry import choose_parser
out = {}
for blobname, local in FILES:
    p = tmp / local
    p.write_bytes(cc.get_blob_client(blobname).download_blob().readall())
    parser,_m,_a = choose_parser(p)
    t0 = time.perf_counter()
    res = parser.parse_artifact_full(project_id="p", artifact_id="a", path=p, domain_pack=None)
    dt = time.perf_counter() - t0
    atoms = list(getattr(res, "atoms", res))
    h = hashlib.sha256()
    for a in atoms: h.update(repr(a).encode("utf-8","replace"))
    out[local] = {"secs": round(dt,2), "atoms": len(atoms), "sha": h.hexdigest()[:16]}
    print("%-16s %6.2fs  %5d atoms  %s" % (local, dt, len(atoms), h.hexdigest()[:16]))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
