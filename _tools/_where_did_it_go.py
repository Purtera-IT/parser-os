# -*- coding: utf-8 -*-
"""Does ANY surviving atom still say this, in text OR in value?

`_figures_gone.py` compares raw_text, which cannot see a fact that survived
into a structured field under different words. Before calling an address lost,
look everywhere.

    DEAL=<uuid> NEEDLES="amber park,11720" python _where_did_it_go.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "wd_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "wd_o.db"))


def blob_of(atom) -> str:
    try:
        v = json.dumps(getattr(atom, "value", None) or {}, default=str)
    except Exception:
        v = str(getattr(atom, "value", ""))
    return ((getattr(atom, "raw_text", "") or "") + " " + v).lower()


def main() -> None:
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    deal = os.environ["DEAL"]
    needles = [n.strip().lower() for n in os.environ["NEEDLES"].split(",") if n.strip()]

    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="wd_")) / "d"
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

    kept = list(res.atoms or [])
    dropped = list(getattr(res, "suppressed_atoms", None) or [])
    for needle in needles:
        k = [a for a in kept if needle in blob_of(a)]
        d = [a for a in dropped if needle in blob_of(a)]
        print("\n=== %r : %d surviving, %d dropped ===" % (needle, len(k), len(d)))
        for a in k[:6]:
            print("  KEPT    %-26s %s" % (str(getattr(a, "atom_type", ""))[:26],
                                          " ".join((getattr(a, "raw_text", "") or "").split())[:90]))
        for a in d[:8]:
            flags = [f for f in (getattr(a, "review_flags", None) or [])
                     if str(f).startswith("suppressed:")]
            print("  DROPPED %-26s %-24s %s"
                  % (str(getattr(a, "atom_type", ""))[:26],
                     (flags[0] if flags else "?")[:24],
                     " ".join((getattr(a, "raw_text", "") or "").split())[:70]))


if __name__ == "__main__":
    main()
