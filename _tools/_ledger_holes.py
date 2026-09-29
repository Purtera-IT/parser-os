# -*- coding: utf-8 -*-
"""Which stages delete atoms WITHOUT a suppression receipt.

The suppression ledger is the instrument every content-loss audit in this
repo reads, including `_phase3_audit.py`. It is only a guarantee if every
departure is written to it. A stage whose output is smaller than its input by
more atoms than it filed is deleting silently, and no ledger-based audit can
see it go -- which is why thirteen of sixteen phase-3 stages looked innocent.

    DEALS="uuid,uuid" python _ledger_holes.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "lh_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "lh_o.db"))


def run(deal: str):
    import app.core.telemetry as TEL
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient

    import inspect

    seen: dict[str, dict] = {}
    order: list[str] = []

    def _live_atom_count():
        """The length of the compile's working `atoms` list, read off the
        calling frame. `output_count` cannot be used for this: for most stages
        it reports what the stage DID (packets built, atoms tagged, gates run),
        not how many atoms came out, so a shrink in it is not a deletion."""
        for f in inspect.stack()[2:16]:
            atoms = f.frame.f_locals.get("atoms")
            if isinstance(atoms, list):
                return len(atoms)
        return None
    real_stage, real_end = TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage
    open_names: list[str] = []

    def stage(self, name, **kw):
        if name not in seen:
            order.append(name)
        seen.setdefault(name, {})["in"] = _live_atom_count()
        open_names.append(name)
        return real_stage(self, name, **kw)

    def end_stage(self, st, **kw):
        name = (getattr(st, "stage", None) or getattr(st, "name", None)
                or (open_names.pop() if open_names else "?"))
        if name in seen:
            seen[name]["out"] = _live_atom_count()
        return real_end(self, st, **kw)

    TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage = stage, end_stage
    try:
        conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
        cc = BlobServiceClient.from_connection_string(conn).get_container_client(
            "orbitbrief-artifacts")
        proj = Path(tempfile.mkdtemp(prefix="lh_")) / "d"
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
        return seen, order, res
    finally:
        TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage = real_stage, real_end


def main() -> None:
    holes: Counter = Counter()
    for deal in [d.strip() for d in os.environ.get("DEALS", "").split(",") if d.strip()]:
        try:
            seen, order, res = run(deal)
        except Exception as exc:
            print("  %s: %s: %s" % (deal[:8], type(exc).__name__, exc))
            continue
        filed: Counter = Counter()
        for a in (getattr(res, "suppressed_atoms", None) or []):
            for f in (getattr(a, "review_flags", None) or []):
                if str(f).startswith("suppressed:"):
                    filed[str(f).split(":", 1)[1]] += 1
                    break

        print("\n=== %s ===" % deal[:8])
        print("%-32s %7s %7s %7s %7s  %s" % ("stage", "in", "out", "lost", "filed", ""))
        for name in order:
            i, o = seen[name].get("in"), seen[name].get("out")
            if not isinstance(i, int) or not isinstance(o, int):
                continue
            # A stage whose output_count is not an atom count (packets,
            # findings, gates) reports a larger or unrelated number; only a
            # SHRINK against its own input is a deletion.
            lost = i - o
            if lost <= 0:
                continue
            # A sub-step may file under its OWN name rather than the stage's
            # -- open_question_resolution's filter files as
            # "open_question_quality_filter" -- which is finer attribution,
            # not a lost atom, but a stage-level audit has to know it.
            ALIAS = {"open_question_resolution": ("open_question_quality_filter",)}
            f = filed.get(name, 0) + sum(filed.get(a, 0) for a in ALIAS.get(name, ()))
            gap = lost - f
            if gap > 0:
                holes[name] += gap
            print("%-32s %7d %7d %7d %7d  %s"
                  % (name, i, o, lost, f, ("<== %d with NO receipt" % gap) if gap > 0 else "ok"))

    print("\n\n=== atoms deleted with no suppression receipt ===")
    for k, n in holes.most_common():
        print("  %-32s %5d" % (k, n))


if __name__ == "__main__":
    main()
