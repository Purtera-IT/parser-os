# -*- coding: utf-8 -*-
"""Phase 6 (GRAPH + PACKETIZE): which surviving atoms never reach the PM?

The funnel ends here and it is the narrowest part of it: live 01491cca is 830
atoms -> 3751 edges -> 58 packets, and a PACKET is what a person reads. So an
atom can survive every gate in phases 1-5 -- present in the envelope, unflagged,
uncited -- and still never reach anybody.

No audit before this one could see that. The suppression ledger says an atom was
kept. `confidence_recalibration` says it may govern. Neither says a packet ever
mentioned it.

This reports, per atom_type, how many surviving atoms are cited by NO packet, in
any role (governing, supporting or contradicting), and highlights the ones that
carry figures -- because an uncited number is a price, a quantity or a date that
the compile read, kept, and then did not show.

    DEALS="uuid,uuid" python _phase6_orphans.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "p6_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "p6_o.db"))

FIG = re.compile(r"\d[\d,.]*")
MONEY = re.compile(r"[$£€]\s?\d|\b\d{3,}\b")

CITED = Counter()
ORPHAN = Counter()
ORPHAN_FIG = Counter()
SAMPLES: list = []
TOTALS = Counter()


def run(deal: str):
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="p6_")) / "d"
    (proj / "artifacts").mkdir(parents=True)
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
        try:
            (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                cc.get_blob_client(b.name).download_blob().readall())
        except Exception:
            pass
    return C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                             allow_errors=True, allow_unverified_receipts=True,
                             use_cache=False)


def main() -> None:
    for deal in [d.strip() for d in os.environ.get("DEALS", "").split(",") if d.strip()]:
        try:
            res = run(deal)
        except Exception as exc:
            print("  %s: %s: %s" % (deal[:8], type(exc).__name__, exc)); continue

        atoms = list(res.atoms or [])
        packets = list(res.packets or [])
        used: set[str] = set()
        for p in packets:
            for field in ("governing_atom_ids", "supporting_atom_ids",
                          "contradicting_atom_ids", "atom_ids"):
                for aid in (getattr(p, field, None) or []):
                    used.add(str(aid))

        TOTALS["atoms"] += len(atoms)
        TOTALS["packets"] += len(packets)
        TOTALS["edges"] += len(res.edges or [])
        for a in atoms:
            at = str(getattr(a, "atom_type", "")).split(".")[-1]
            aid = str(getattr(a, "id", ""))
            text = getattr(a, "raw_text", "") or ""
            if aid in used:
                CITED[at] += 1
                continue
            ORPHAN[at] += 1
            if MONEY.search(text):
                ORPHAN_FIG[at] += 1
                if len(SAMPLES) < 22:
                    SAMPLES.append((deal[:8], at, " ".join(text.split())[:78]))

    tot_o = sum(ORPHAN.values())
    tot_c = sum(CITED.values())
    print("\n=== phase 6: %d atoms, %d edges, %d packets ===" % (
        TOTALS["atoms"], TOTALS["edges"], TOTALS["packets"]))
    print("cited by at least one packet : %d" % tot_c)
    print("cited by NO packet           : %d  (%.0f%% of survivors)" % (
        tot_o, 100 * tot_o / (tot_o + tot_c) if (tot_o + tot_c) else 0))
    print("\n%-26s %8s %8s %9s %s" % ("atom_type", "cited", "orphan", "orphan%", "with a figure"))
    for at in sorted(set(CITED) | set(ORPHAN), key=lambda k: -ORPHAN[k]):
        c, o = CITED[at], ORPHAN[at]
        print("%-26s %8d %8d %8.0f%% %9d" % (at, c, o, 100*o/(c+o) if c+o else 0, ORPHAN_FIG[at]))
    print("\n=== uncited atoms carrying a figure -- read these ===")
    for d, at, t in SAMPLES:
        print("  %-9s %-20s %s" % (d, at[:20], t))


if __name__ == "__main__":
    main()
