# -*- coding: utf-8 -*-
"""Every cross-type fold that deleted something only the loser held.

`cross_type_dedup_atoms` groups atoms by a QUANTITY-STRIPPED text key and keeps
the highest-priority type. The losers' `_merge_atom_metadata` carries
source_refs, receipts, entity_keys and review_flags -- and NOT `value`. So when
a `quantity` atom (priority 3, the default: it is not in the table at all)
loses its cell to a `scope_item` or a `site_attribute` saying the same words,
the structured number it was the only carrier of leaves the compile.

Live 010180's WiFi survey tiers are the case that found this:

    Site type: Large  | Survey type: Predictive WiFi Survey - Upto 150k sq m
    Site type: Medium | Survey type: Predictive WiFi Survey - Upto  75k sq m

carrying quantity=500 and quantity=250 access points. Both atoms are deleted
and no surviving atom states either number -- the key STRIPS the digits, so the
quantity atom and its prose twin key identically by construction.

This is the phase-2 invariant again: atoms stating different numbers are
different facts. Here it is sharper -- the fold is between types, so the
question is not "are these the same fact" but "does the survivor still say
everything the loser said".

    DEALS="uuid,uuid" python _crosstype_loss.py
"""
from __future__ import annotations

import os
import sys
import traceback
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "xt_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "xt_o.db"))

#: Keys that are bookkeeping rather than a fact the atom is the carrier of.
BORING = {"raw", "text", "source", "confidence", "notes", "id", "kind", "label",
          "atom_type", "raw_text", "span", "page", "line"}

FOLDS: list[dict] = []
#: Every quantity an atom carried at fold time, and whether it was the loser.
QTYS: list[tuple] = []


def _val(a):
    v = getattr(a, "value", None)
    return v if isinstance(v, dict) else {}


def install() -> None:
    import app.core.semantic_dedup as SD
    real = SD._merge_atom_metadata

    def spy(winner, loser):
        wv, lv = _val(winner), _val(loser)
        blank = (None, "", [], {})
        lost = {k: lv[k] for k in lv
                if k not in BORING and lv.get(k) not in blank
                and wv.get(k) in blank}
        # Worse than deletion: BOTH carry the key and they disagree. The
        # survivor then states a number the document never put on that line.
        clash = {k: (wv[k], lv[k]) for k in lv
                 if k not in BORING and lv.get(k) not in blank
                 and wv.get(k) not in blank and wv[k] != lv[k]}
        for label, a in (("winner", winner), ("loser", loser)):
            q = _val(a).get("quantity")
            if q not in (None, ""):
                QTYS.append((label, q, _val(a).get("noun"),
                             (getattr(a, "raw_text", "") or "")[:70]))
        if lost or clash:
            stack = [f.name for f in traceback.extract_stack()[-6:-1]]
            FOLDS.append({
                "caller": " < ".join(reversed([n for n in stack if n != "spy"])),
                "winner_type": str(getattr(winner, "atom_type", "") or ""),
                "loser_type": str(getattr(loser, "atom_type", "") or ""),
                "lost": lost,
                "clash": clash,
                "loser_text": (getattr(loser, "raw_text", "") or "")[:110],
                "wv": {k: repr(v)[:70] for k, v in wv.items()},
                "lv": {k: repr(v)[:70] for k, v in lv.items()},
                "winner_text": (getattr(winner, "raw_text", "") or "")[:110],
            })
        return real(winner, loser)

    SD._merge_atom_metadata = spy


def run(deal: str, max_artifacts: int = 25):
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="xt_")) / "d"
    (proj / "artifacts").mkdir(parents=True)
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:max_artifacts]:
        try:
            (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                cc.get_blob_client(b.name).download_blob().readall())
        except Exception:
            pass
    return C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                             allow_errors=True, allow_unverified_receipts=True,
                             use_cache=False)


def main() -> None:
    install()
    deals = [d.strip() for d in os.environ.get("DEALS", "").split(",") if d.strip()]
    pairs: Counter = Counter()
    fields: Counter = Counter()
    clashes: Counter = Counter()
    callers: Counter = Counter()
    for deal in deals:
        FOLDS.clear(); QTYS.clear()
        try:
            res = run(deal)
        except Exception as exc:
            print("  %s: %s: %s" % (deal[:8], type(exc).__name__, exc))
            continue
        print("\n=== %s: %d fold(s) deleted a field ===" % (deal[:8], len(FOLDS)))
        for f in FOLDS:
            pairs["%-34s %s <- %s" % (f["caller"].split(" < ")[0],
                                      f["winner_type"].split(".")[-1],
                                      f["loser_type"].split(".")[-1])] += 1
            for k in f["lost"]:
                if not k.startswith("_"):
                    fields[k] += 1
            for k in f["clash"]:
                clashes["%-34s %s.%s" % (f["caller"].split(" < ")[0],
                                         f["loser_type"].split(".")[-1], k)] += 1
            if f["clash"] or any(not k.startswith("_") for k in f["lost"]):
                callers[f["caller"].split(" < ")[0]] += 1
        # Did every quantity that entered a fold come out the other side?
        kept = {}
        for a in (res.atoms or []):
            q = _val(a).get("quantity")
            if q not in (None, ""):
                kept.setdefault(str(q), []).append(_val(a).get("noun"))
        gone = [(l, q, n, t) for (l, q, n, t) in QTYS
                if l == "loser" and str(q) not in kept]
        print("  quantities that entered a fold as the LOSER and survive nowhere: %d" % len(gone))
        for (_l, q, n, t) in gone[:10]:
            print("      %s %s   <- %s" % (q, n, t))
        import json as _j
        want = [f for f in FOLDS
                if f["loser_type"].split(".")[-1] in ("commercial_total", "deal_metadata")]
        _j.dump(want[:400], open(r"D:/temp/claude/folddump_%s.json" % deal[:8], "w"),
                indent=1, default=str)
        WATCH = ("value", "metric", "name", "date", "quantity", "hubspot_note_id")
        for f in [x for x in FOLDS
                  if any(k in WATCH for k in x["clash"])][:8]:
            print("  CHAIN %s" % f["caller"])
            print("     %s <- %s" % (f["winner_type"], f["loser_type"]))
            for k in f["clash"]:
                if k in WATCH:
                    print("       %-12s winner=%r  loser=%r"
                          % (k, f["clash"][k][0], f["clash"][k][1]))
            print("       loser : %s" % f["loser_text"])
            print("       winner: %s" % f["winner_text"])
        for f in [x for x in FOLDS if x["clash"]][:0]:
            print("  CLASH %-16s <- %-16s %s" % (f["winner_type"], f["loser_type"], f["clash"]))
            print("      loser : %s" % f["loser_text"])
            print("      winner: %s" % f["winner_text"])
        for f in [x for x in FOLDS
                  if any(not k.startswith("_") for k in x["lost"])][:10]:
            print("  %-16s <- %-16s lost %s" % (f["winner_type"], f["loser_type"], f["lost"]))
            print("      loser : %s" % f["loser_text"])
            print("      winner: %s" % f["winner_text"])

    print("\n\n=== ACROSS %d DEALS ===" % len(deals))
    print("-- winner <- loser --")
    for k, n in pairs.most_common(18):
        print("  %-40s %4d" % (k, n))
    print("-- which FUNCTION did the damage --")
    for k, n in callers.most_common(15):
        print("  %-40s %4d" % (k, n))
    print("-- field CLASHED (survivor now states a different value) --")
    for k, n in clashes.most_common(24):
        print("  %-40s %4d" % (k, n))
    print("-- real field deleted (plumbing excluded) --")
    for k, n in fields.most_common(25):
        print("  %-28s %4d" % (k, n))


if __name__ == "__main__":
    main()
