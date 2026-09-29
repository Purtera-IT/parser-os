# -*- coding: utf-8 -*-
"""Impact run for the email OCR byte floor: same files twice, floor off vs on.

A deploy is not needed to know whether a parser change loses content. Run the
real parser over the real documents with the change disabled, then enabled, and
diff the atoms. Anything the second run does not produce that the first did is a
loss, named, before it reaches a deal.

This is the gate the project already learned to insist on -- tests written
alongside a fix share its blind spot; the documents do not.

    DEAL=<uuid> python impact_ocr_floor.py
"""
import collections, json, os, sys, tempfile
from pathlib import Path

from azure.storage.blob import BlobServiceClient

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

DEAL = os.environ["DEAL"]
MAX_EMAILS = int(os.environ.get("MAX_EMAILS", "25"))


def atoms_for(paths, floor: int) -> dict[str, list]:
    """Parse every .eml with the floor set to `floor` bytes."""
    os.environ["SOWSMITH_EMAIL_IMAGE_MIN_BYTES"] = str(floor)
    for mod in [m for m in list(sys.modules) if "email_parser" in m or "_ocr_chain" in m]:
        del sys.modules[mod]
    from app.parsers.email_parser import EmailParser

    # The OCR backend must be the REAL one, or this measures the harness.
    #
    # The first version of this script stubbed OCR with a marker string. Because
    # the marker was always truthy, every skipped image looked like a lost atom
    # and the run reported CONTENT LOST on a change that loses nothing. A stub
    # that cannot return what the real service returns cannot answer the
    # question the run exists to ask.
    #
    # Live Document Intelligence on 010094's sub-3KB inline images returns 'in',
    # 'f', 'X' and two empty strings -- three social icon glyphs. That is the
    # content at stake, and only the real backend could say so.
    from app.parsers import email_parser as ep
    from app.core.doc_intel_ocr import doc_intel_available

    if not doc_intel_available():
        raise SystemExit(
            "AZURE_DOC_INTEL_ENDPOINT / _KEY are not set. This run must use the "
            "real OCR backend; a stub reports losses that do not exist.")

    seen: list[int] = []
    _real = ep._ocr_text_from_cid_inline

    def _counting(payload, *, content_type=""):
        seen.append(len(payload))
        return _real(payload, content_type=content_type)

    ep._ocr_text_from_cid_inline = _counting

    out: dict[str, list] = {}
    for p in paths:
        try:
            atoms = EmailParser().parse(p)
        except Exception as exc:
            out[p.name] = [f"__PARSE_ERROR__ {type(exc).__name__}: {exc}"]
            continue
        out[p.name] = sorted(
            " ".join(str(getattr(a, "raw_text", "") or "").split()) for a in atoms
        )
    return {"atoms": out, "ocr_calls": seen}


def main() -> None:
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    blobs = [b for b in cc.list_blobs(name_starts_with=f"deals/{DEAL}/artifacts/")
             if b.name.endswith(".eml")][:MAX_EMAILS]
    tmp = Path(tempfile.mkdtemp(prefix="impact_ocr_"))
    paths = []
    for b in blobs:
        p = tmp / b.name.split("/")[-1]
        p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
        paths.append(p)
    print(f"{len(paths)} emails from {DEAL[:8]}\n")

    before = atoms_for(paths, floor=0)        # every image OCR'd, today's behaviour
    after = atoms_for(paths, floor=3000)      # the change

    nb = sum(len(v) for v in before["atoms"].values())
    na = sum(len(v) for v in after["atoms"].values())
    print(f"OCR calls : {len(before['ocr_calls']):>5}  ->  {len(after['ocr_calls']):>5}   "
          f"({len(before['ocr_calls']) - len(after['ocr_calls'])} skipped, "
          f"{100 * (len(before['ocr_calls']) - len(after['ocr_calls'])) // max(len(before['ocr_calls']), 1)}%)")
    print(f"atoms     : {nb:>5}  ->  {na:>5}")

    lost, gained = [], []
    for fn in sorted(set(before["atoms"]) | set(after["atoms"])):
        b = collections.Counter(before["atoms"].get(fn, []))
        a = collections.Counter(after["atoms"].get(fn, []))
        for t, n in (b - a).items():
            lost += [(fn, t)] * n
        for t, n in (a - b).items():
            gained += [(fn, t)] * n

    print(f"\nLOST   {len(lost)}")
    for fn, t in lost[:25]:
        print(f"   {fn[-26:]:28} {t[:96]}")
    print(f"GAINED {len(gained)}")
    for fn, t in gained[:10]:
        print(f"   {fn[-26:]:28} {t[:96]}")

    skipped = sorted(set(before["ocr_calls"]) - set(after["ocr_calls"]))
    if skipped:
        print(f"\nimage sizes no longer sent to OCR: "
              f"min {min(skipped)}B  max {max(skipped)}B  ({len(skipped)} distinct)")

    print("\n" + ("*** CONTENT LOST -- DO NOT SHIP ***" if lost
                  else "no atom lost; every skipped image was chrome"))
    raise SystemExit(1 if lost else 0)


if __name__ == "__main__":
    main()
