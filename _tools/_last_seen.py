# -*- coding: utf-8 -*-
"""The last compile stage at which an atom still existed.

The suppression ledger is only a guarantee if every departure is recorded in
it. An atom that is in no output AND in no ledger left without a receipt, and
no audit that reads the ledger can see it go. This walks the stages and names
the one after which the needle stops appearing.

    DEAL=<uuid> NEEDLE=2701149 python _last_seen.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "ls_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "ls_o.db"))

NEEDLE = os.environ.get("NEEDLE", "2701149")
TRAIL: list[tuple] = []


def install() -> None:
    """Count matching atoms at every stage boundary, from the caller's frame."""
    import inspect
    import app.core.telemetry as TEL
    real_stage, real_end = TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage

    def _count():
        # `atoms` is the compile's working list; read it off the calling frame.
        for f in inspect.stack()[1:14]:
            atoms = f.frame.f_locals.get("atoms")
            if isinstance(atoms, list):
                hits = [a for a in atoms
                        if NEEDLE in (getattr(a, "raw_text", "") or "")]
                return len(atoms), [str(getattr(a, "atom_type", ""))[:22] for a in hits]
        return None, None

    open_names: list[str] = []

    def stage(self, name, **kw):
        n, hits = _count()
        TRAIL.append(("in ", name, n, hits))
        open_names.append(name)
        return real_stage(self, name, **kw)

    def end_stage(self, st, **kw):
        name = (getattr(st, "stage", None) or getattr(st, "name", None)
                or (open_names.pop() if open_names else "?"))
        n, hits = _count()
        TRAIL.append(("out", name, n, hits))
        return real_end(self, st, **kw)

    TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage = stage, end_stage


def main() -> None:
    install()
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    deal = os.environ["DEAL"]
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="ls_")) / "d"
    (proj / "artifacts").mkdir(parents=True)
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
        try:
            (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                cc.get_blob_client(b.name).download_blob().readall())
        except Exception:
            pass
    C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                      allow_errors=True, allow_unverified_receipts=True, use_cache=False)

    print("\n\n=== where %r stopped appearing ===" % NEEDLE)
    prev = None
    for where, name, n, hits in TRAIL:
        cur = len(hits or [])
        if cur != prev:
            print("  %-3s %-32s atoms=%-6s matches=%d %s"
                  % (where, name, n, cur, hits))
            prev = cur


if __name__ == "__main__":
    main()
