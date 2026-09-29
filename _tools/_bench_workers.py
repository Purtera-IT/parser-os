# -*- coding: utf-8 -*-
"""Does SOWSMITH_PARSE_WORKERS=8 beat 4 on a 4-CPU container?

#247 made `parse_artifacts` run a ThreadPoolExecutor and defaulted it to 8. The
dev worker has 4 CPUs and does not set the variable, so it runs 8 threads on 4
cores and nobody has measured whether that is faster than 4, or than 1.

Parsing is a mix: mostly CPU (zip inflate, XML, regex) with some I/O (blob is
already local by this point, but OCR calls out). Threads help the I/O and fight
over the GIL for the rest, so oversubscription is a real possibility rather than
a theoretical one.

Runs the SAME artifacts at each width, several times, and reports the median.
OCR is stubbed out: this measures the readers, not the vision bill.

    DEAL=<uuid> python _bench_workers.py [widths...]     default 1 2 4 8 12
"""
import os, statistics, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
HERE = Path(__file__).parent

DEAL = os.environ["DEAL"]
REPEATS = int(os.environ.get("REPEATS", "3"))
MAX_ARTIFACTS = int(os.environ.get("MAX_ARTIFACTS", "24"))


def fetch() -> list[Path]:
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    tmp = Path(tempfile.mkdtemp(prefix="bench_"))
    out = []
    for b in list(cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/"))[:MAX_ARTIFACTS]:
        p = tmp / b.name.split("/")[-1]
        p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
        out.append(p)
    return out


def run_once(paths: list[Path], workers: int) -> tuple[float, int]:
    from concurrent.futures import ThreadPoolExecutor
    from app.parsers.registry import choose_parser

    def one(p: Path) -> int:
        try:
            parser, _m, _a = choose_parser(p)
            if parser is None:
                return 0
            out = parser.parse_artifact_full(
                project_id="bench", artifact_id="a", path=p, domain_pack=None)
            atoms = getattr(out, "atoms", out)
            return len(atoms)
        except Exception:
            return 0

    t0 = time.perf_counter()
    if workers <= 1:
        counts = [one(p) for p in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            counts = list(pool.map(one, paths))
    return time.perf_counter() - t0, sum(counts)


def main() -> None:
    # Measure the readers, not Document Intelligence.
    from app.parsers import email_parser as ep
    ep._ocr_text_from_cid_inline = lambda payload, content_type="": ""

    widths = [int(w) for w in sys.argv[1:]] or [1, 2, 4, 8, 12]
    paths = fetch()
    print(f"{len(paths)} artifacts from {DEAL[:8]}, {REPEATS} runs per width, "
          f"{os.cpu_count()} CPUs visible\n")

    baseline = None
    print(f"{'workers':>8}{'median s':>11}{'best s':>9}{'atoms':>9}{'vs 1':>8}")
    for w in widths:
        runs = [run_once(paths, w) for _ in range(REPEATS)]
        secs = [r[0] for r in runs]
        med = statistics.median(secs)
        if baseline is None:
            baseline = med
        speed = f"{baseline / med:.2f}x" if med else "-"
        print(f"{w:>8}{med:>11.2f}{min(secs):>9.2f}{runs[0][1]:>9}{speed:>8}")


if __name__ == "__main__":
    main()
