# -*- coding: utf-8 -*-
"""Which cross-type GROUP swallowed this atom, and who won it.

    DEAL=<uuid> NEEDLE=2701149 python _why_dropped.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "wy_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "wy_o.db"))

NEEDLE = os.environ.get("NEEDLE", "2701149")


def install() -> None:
    import app.core.semantic_dedup as SD

    # Spy on the REAL typed-twice set. Recomputing it afterwards lies: the
    # function merges metadata as it goes, so a second call sees atoms the
    # first one already changed.
    real_tt = SD._same_utterance_as_a_question
    last_tt = {}

    def tt(atoms):
        out = real_tt(atoms)
        last_tt["ids"] = set(out)
        last_tt["by_id"] = {id(a): a for a in atoms}
        return out

    SD._same_utterance_as_a_question = tt
    real = SD.cross_type_dedup_atoms
    seen = {"n": 0}

    def wrapped(atoms):
        targets = [a for a in atoms if NEEDLE in (getattr(a, "raw_text", "") or "")]
        # Keys BEFORE the call: the function merges provenance as it goes, so
        # recomputing a key afterwards can describe a group that never existed.
        keys_before = {id(a): SD._cross_type_text_key(a) for a in atoms}
        snapshot = list(atoms)
        out = real(atoms)
        seen["n"] += 1
        kept = {id(a) for a in out}
        for t in targets:
            if id(t) in kept:
                continue
            key = keys_before.get(id(t))
            group = [a for a in snapshot if keys_before.get(id(a)) == key]
            print("\n### call #%d dropped it. key=%r" % (seen["n"], key))
            print("   group of %d:" % len(group))
            for g in group:
                print("     %-9s %-24s prio=%-2s %s"
                      % ("TARGET" if g is t else "",
                         str(getattr(g, "atom_type", ""))[:24],
                         SD._cross_type_priority(g),
                         " ".join((getattr(g, "raw_text", "") or "").split())[:78]))
            winners = [g for g in group if id(g) in kept]
            print("   survived: %d" % len(winners))
            for w in winners:
                print("     WINNER  %-24s figs=%s"
                      % (str(getattr(w, "atom_type", ""))[:24],
                         sorted(SD._figures_stated(w))[:8]))
            print("   target figs: %s" % sorted(SD._figures_stated(t))[:8])
            hit = id(t) in last_tt.get("ids", set())
            print("   killed as typed-twice-question: %s" % hit)
            if hit:
                uk = SD._utterance_key(t)
                print("   utterance key: %r" % (uk,))
                for o in atoms:
                    if SD._utterance_key(o) == uk:
                        print("     %-8s %-20s %s"
                              % ("TARGET" if o is t else "",
                                 str(getattr(o, "atom_type", ""))[:20],
                                 " ".join((getattr(o, "raw_text", "") or "").split())[:76]))
        return out

    SD.cross_type_dedup_atoms = wrapped
    import app.core.compiler as C
    C.cross_type_dedup_atoms = wrapped


def main() -> None:
    install()
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    deal = os.environ["DEAL"]
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="wy_")) / "d"
    (proj / "artifacts").mkdir(parents=True)
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
        try:
            (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                cc.get_blob_client(b.name).download_blob().readall())
        except Exception:
            pass
    C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                      allow_errors=True, allow_unverified_receipts=True, use_cache=False)


if __name__ == "__main__":
    main()
