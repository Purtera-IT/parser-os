# -*- coding: utf-8 -*-
"""Same artifacts, several widths, several runs: do the atoms come out identical?

Compares atom TEXT, not just counts -- ``label_key`` is a hash of the text, so
two runs that agree on the count and disagree on a boundary still detach gold.

    DEAL=<uuid> python _determinism.py [widths...]      default 1 4 8
"""
import hashlib, os, sys, tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
HERE = Path(__file__).parent
DEAL = os.environ["DEAL"]
REPEATS = int(os.environ.get("REPEATS", "2"))
MAX_ARTIFACTS = int(os.environ.get("MAX_ARTIFACTS", "24"))


def fetch() -> list[Path]:
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    tmp = Path(tempfile.mkdtemp(prefix="det_"))
    out = []
    for b in list(cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/"))[:MAX_ARTIFACTS]:
        p = tmp / b.name.split("/")[-1]
        p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
        out.append(p)
    return out


def run(paths: list[Path], workers: int) -> dict[str, list[str]]:
    from concurrent.futures import ThreadPoolExecutor
    from app.parsers.registry import choose_parser

    def one(p: Path) -> tuple[str, list[str]]:
        try:
            parser, _m, _a = choose_parser(p)
            if parser is None:
                return p.name, []
            out = parser.parse_artifact_full(
                project_id="det", artifact_id="a", path=p, domain_pack=None)
            atoms = getattr(out, "atoms", out)
            return p.name, [str(getattr(a, "text", a) or "") for a in atoms]
        except Exception as exc:
            return p.name, ["!!ERROR %s" % type(exc).__name__]

    if workers <= 1:
        pairs = [one(p) for p in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(one, paths))
    return dict(pairs)


def digest(res: dict[str, list[str]]) -> str:
    h = hashlib.sha256()
    for name in sorted(res):
        h.update(name.encode())
        for t in res[name]:
            h.update(b"\x00" + t.encode("utf-8", "replace"))
    return h.hexdigest()[:12]


def main() -> None:
    from app.parsers import email_parser as ep
    ep._ocr_text_from_cid_inline = lambda payload, content_type="": ""

    widths = [int(w) for w in sys.argv[1:]] or [1, 4, 8]
    paths = fetch()
    print("%d artifacts from %s, %d runs per width\n" % (len(paths), DEAL[:8], REPEATS))

    ref = None
    ok = True
    for w in widths:
        for r in range(REPEATS):
            res = run(paths, w)
            n = sum(len(v) for v in res.values())
            d = digest(res)
            if ref is None:
                ref, ref_res = d, res
                print("  width %2d run %d: %5d atoms  %s   <- reference" % (w, r, n, d))
                continue
            same = d == ref
            ok &= same
            print("  width %2d run %d: %5d atoms  %s   %s"
                  % (w, r, n, d, "same" if same else "*** DIFFERS ***"))
            if not same:
                for name in sorted(res):
                    a, b = Counter(ref_res.get(name, [])), Counter(res[name])
                    if a != b:
                        print("      %s: %d -> %d" % (name, sum(a.values()), sum(b.values())))
                        for t in list((a - b))[:3]:
                            print("         only reference: %s" % t[:90])
                        for t in list((b - a))[:3]:
                            print("         only this run : %s" % t[:90])
    print("\n%s" % ("DETERMINISTIC: every width and run produced identical atoms."
                    if ok else "*** NONDETERMINISTIC - do not ship ***"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
