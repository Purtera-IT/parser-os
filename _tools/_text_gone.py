# -*- coding: utf-8 -*-
"""What TEXT a stage removed that no surviving atom says, in its own words.

Figures are the sharp instrument; text is the blunt one, and it needs reading
rather than counting. A quoted "Good morning." leaving the compile is not the
same event as a quoted specification leaving it.

    DEAL=<uuid> STAGE=quoted_history_dedup python _text_gone.py
"""
from __future__ import annotations

import json as _j
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "tg_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "tg_o.db"))

norm = lambda t: " ".join((t or "").split()).lower()
STAGE = os.environ.get("STAGE", "quoted_history_dedup")


def main() -> None:
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    deal = os.environ["DEAL"]
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="tg_")) / "d"
    (proj / "artifacts").mkdir(parents=True)
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
        try:
            (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                cc.get_blob_client(b.name).download_blob().readall())
        except Exception:
            pass
    res = C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                            allow_errors=True, allow_unverified_receipts=True,
                            use_cache=False)

    kept = {norm(getattr(x, "raw_text", "")) for x in (res.atoms or [])}
    gone = []
    for a in (getattr(res, "suppressed_atoms", None) or []):
        flags = [str(f) for f in (getattr(a, "review_flags", None) or [])
                 if str(f).startswith("suppressed:")]
        if not flags or flags[0].split(":", 1)[1] != STAGE:
            continue
        t = norm(getattr(a, "raw_text", ""))
        if t and t not in kept:
            gone.append((str(getattr(a, "atom_type", "")).split(".")[-1],
                         " ".join((getattr(a, "raw_text", "") or "").split())))

    print("\n=== %s on %s: %d atom(s) whose TEXT no survivor states ===" % (
        STAGE, deal[:8], len(gone)))
    words = Counter(len(t.split()) for _, t in gone)
    short = sum(n for w, n in words.items() if w <= 6)
    print("    %d of %d are six words or fewer" % (short, len(gone)))
    seen: Counter = Counter(t for _, t in gone)
    print("\n-- most repeated --")
    for t, n in seen.most_common(12):
        print("   x%-3d %s" % (n, t[:96]))
    print("\n-- the longest, which is where a real loss would hide --")
    for at, t in sorted(set(gone), key=lambda r: -len(r[1]))[:10]:
        print("   %-16s %s" % (at[:16], t[:120]))


if __name__ == "__main__":
    main()
