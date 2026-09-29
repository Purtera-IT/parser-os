# -*- coding: utf-8 -*-
"""Parse one real file of each type with TODAY's parser, and time it.

`attachments.atoms_count` is a record of what some past compile produced, not a
measure of what the parser produces now. A large zero-atom email in that table
parses to 106 atoms in 5.5s on the current code, so the column reports history
and was read here as loss.

This asks the only question that settles it: hand the file to the parser the
registry picks for it, today, and see what comes back and how long it takes.
"""
import sys, tempfile, time
from pathlib import Path

sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
HERE = Path(__file__).parent

SAMPLES = [
    ("pptx", "c392d960dc0d6ce18dd6a2097a9337f9225ea3bfa4498c9ca885a06830edbd58", "Compute_Storage_ Backup Standards.pptx"),
    ("xls",  "cf59f5f7563c561c0f82386087d50608735cbeda1a4e852835f2dc3f38acc15a", "010319-CDW Pricing Sheet 09-22-26.xls"),
    ("dotx", "1d72a226bfec1876ba2061cd018fc071f3166cf724ab7e99a596e207668734f2", "010195- TV Install Change Order 8.20 v1.dotx"),
    ("docx", "9ca74250d4c52b3be3d506eede1ed59811339627c48aac44bb2a7fd74752ca18", "vSOW PureTeraIT EQX Program Deployment.docx"),
    ("pdf",  "14d0d036bf4940d94cbc5886eee02a078fb4274a5ebd39e88a7ce6e1b2bb7556", "Springfield Floor Plan 2026.pdf"),
    ("xlsx", "cf80318cff12cb9318cce91cbb13f170a04902ce6d54d3d9beccd2b98233d21b", "000089  anywAIR CALC.xlsx"),
    ("msg",  "351f7d26eede6a766e146a0cec7005e22a55faa190655e33ad40fe96526ce974", "Re_ Balata Intro.msg"),
]


def main() -> None:
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")

    # No billed OCR while probing: this measures the reader, not the vision bill.
    try:
        from app.parsers import email_parser as ep
        ep._ocr_text_from_cid_inline = lambda payload, content_type="": ""
    except Exception:
        pass

    from app.parsers.registry import choose_parser  # type: ignore
    tmp = Path(tempfile.mkdtemp(prefix="probe_"))

    print(f"{'ext':6}{'MB':>7}  {'parser':18}{'atoms':>7}{'secs':>8}  note")
    for ext, sha, name in SAMPLES:
        blobs = [b for b in cc.list_blobs(name_starts_with="deals/") if sha[:20] in b.name]
        if not blobs:
            print(f"{ext:6}{'':>7}  {'(blob missing)':18}")
            continue
        b = blobs[0]
        p = tmp / name
        p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
        mb = p.stat().st_size / 1e6

        try:
            parser, match, _all = choose_parser(p)
            if parser is None:
                print(f"{ext:6}{mb:>7.1f}  {'NO PARSER MATCHED':18}")
                continue
            pname = getattr(match, "parser_name", None) or type(parser).__name__
        except Exception as exc:
            print(f"{ext:6}{mb:>7.1f}  ROUTING FAILED: {type(exc).__name__}: {exc}")
            continue

        t0 = time.time()
        note = ""
        try:
            atoms = parser.parse(p)
            n = len(atoms)
        except Exception as exc:
            n = -1
            note = f"RAISED {type(exc).__name__}: {str(exc)[:60]}"
        el = time.time() - t0
        print(f"{ext:6}{mb:>7.1f}  {str(pname)[:18]:18}{n:>7}{el:>8.1f}  {note}")


if __name__ == "__main__":
    main()
