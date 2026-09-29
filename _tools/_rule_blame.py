# -*- coding: utf-8 -*-
"""Which extractor made each atom, and what happened to it afterwards.

The question "which rules are bad" has never been answerable because nothing
connected a rule to the fate of what it produced. Two things now exist that
did not:

* a COMPLETE suppression ledger -- as of the phase-3 pass every stage that
  deletes an atom files a receipt, so "this rule's atoms keep getting deleted"
  is a measurable statement
* PM corrections, which say "this atom was wrong" in a person's words

So: group atoms by the extractor that made them, and report the share that
survived, the share a later stage deleted, and which stage did the deleting.
A rule whose output is mostly deleted downstream is either wrong or redundant,
and either way it is the first thing worth labelling.

    DEALS="uuid,uuid" python _rule_blame.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "rb_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "rb_o.db"))


def _maker(atom) -> str:
    """The most specific thing that claims to have produced this atom."""
    for ref in (getattr(atom, "receipts", None) or []):
        name = str(getattr(ref, "extractor_name", "") or "")
        method = str(getattr(ref, "extraction_method", "") or "")
        if name:
            return f"{name} [{method}]" if method else name
    for ref in (getattr(atom, "source_refs", None) or []):
        name = str(getattr(ref, "extraction_method", "") or "")
        if name:
            return f"<source_ref> {name}"
    return "<unattributed>"


def run(deal: str):
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="rb_")) / "d"
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
    kept: Counter = Counter()
    dropped: Counter = Counter()
    by_stage: dict[str, Counter] = defaultdict(Counter)
    for deal in [d.strip() for d in os.environ.get("DEALS", "").split(",") if d.strip()]:
        try:
            res = run(deal)
        except Exception as exc:
            print("  %s: %s: %s" % (deal[:8], type(exc).__name__, exc)); continue
        for a in (res.atoms or []):
            kept[_maker(a)] += 1
        for a in (getattr(res, "suppressed_atoms", None) or []):
            m = _maker(a)
            dropped[m] += 1
            for f in (getattr(a, "review_flags", None) or []):
                if str(f).startswith("suppressed:"):
                    by_stage[m][str(f).split(":", 1)[1]] += 1
                    break

    makers = set(kept) | set(dropped)
    print("\n%-46s %7s %8s %7s   %s" % ("extractor", "kept", "dropped", "drop%", "top stage"))
    rows = []
    for m in makers:
        k, d = kept.get(m, 0), dropped.get(m, 0)
        tot = k + d
        if tot < 5:
            continue
        rows.append((d / tot, tot, m, k, d, by_stage[m].most_common(1)))
    for pct, tot, m, k, d, top in sorted(rows, key=lambda r: (-r[0], -r[1])):
        stage = ("%s x%d" % top[0]) if top else "-"
        print("%-46s %7d %8d %6.0f%%   %s" % (m[:46], k, d, pct * 100, stage))


if __name__ == "__main__":
    main()
