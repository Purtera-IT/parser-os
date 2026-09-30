# -*- coding: utf-8 -*-
"""Phase 4 (HEADS + BACKFILL): what does the compile INVENT?

Phases 1-3 asked what a stage deleted, and the suppression ledger answers that.
Phase 4 mostly ADDS -- twelve stages that backfill atoms and run heads -- and the
ledger says nothing about an atom that appeared. The failure mode inverts: not
"the deal lost its margin" but "the deal asserts something no document says".

For every atom a stage ADDS, and every value a stage writes onto an atom that
already existed, this records whether it is GROUNDED:

  no source_ref        nothing says where the claim came from
  dangling source_ref  it names an artifact this compile never read
  no receipt           nothing a PM can be shown to justify it
  value written        a field appeared on an existing atom; which stage, which
                       field, and whether that atom has any receipt at all

None of that is proof of fabrication on its own -- a backfill that copies a
value from a sibling atom on the same page is legitimate and may carry the
sibling's refs. It is the list worth reading, which does not exist today.

    DEALS="uuid,uuid" python _phase4_invent.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "p4_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "p4_o.db"))

#: The twelve stages between "atoms worth gating" and "atoms worth resolving".
PHASE4 = [
    "task_atom_backfill", "site_task_anchor", "task_tier_classification",
    "task_admission", "quote_context_head", "quote_line_head", "task_hours",
    "commercial_terms", "hardware_evidence_backfill", "site_facility_head",
    "drawing_pairs", "deal_state",
]

ADDED: dict[str, list] = defaultdict(list)
VALUES: dict[str, Counter] = defaultdict(Counter)
UNGROUNDED: dict[str, Counter] = defaultdict(Counter)


def _val_keys(a) -> frozenset:
    v = getattr(a, "value", None)
    return frozenset(v.keys()) if isinstance(v, dict) else frozenset()


def _snap(atoms):
    """id -> (value keys, receipt count) for every atom, cheaply."""
    out = {}
    for a in atoms:
        aid = id(a)
        out[aid] = (_val_keys(a), len(getattr(a, "receipts", None) or []))
    return out


def install(artifact_ids: set) -> None:
    import inspect
    import app.core.telemetry as TEL
    real_stage, real_end = TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage
    names: list[str] = []
    before: list[dict] = []

    def _atoms():
        for f in inspect.stack()[2:16]:
            got = f.frame.f_locals.get("atoms")
            if isinstance(got, list):
                return got
        return None

    def stage(self, name, **kw):
        a = _atoms()
        before.append(_snap(a) if a is not None else {})
        names.append(name)
        return real_stage(self, name, **kw)

    def end_stage(self, st, **kw):
        name = (getattr(st, "stage", None) or getattr(st, "name", None)
                or (names.pop() if names else "?"))
        prev = before.pop() if before else {}
        a = _atoms()
        if name in PHASE4 and a is not None:
            for atom in a:
                aid = id(atom)
                if aid not in prev:
                    ADDED[name].append(atom)
                    refs = getattr(atom, "source_refs", None) or []
                    if not refs:
                        UNGROUNDED[name]["no_source_ref"] += 1
                    else:
                        art = str(getattr(refs[0], "artifact_id", "") or "")
                        if art and artifact_ids and art not in artifact_ids:
                            UNGROUNDED[name]["dangling_source_ref"] += 1
                    if not (getattr(atom, "receipts", None) or []):
                        UNGROUNDED[name]["no_receipt"] += 1
                else:
                    old_keys, old_rcpt = prev[aid]
                    for k in (_val_keys(atom) - old_keys):
                        VALUES[name][k] += 1
                    if len(getattr(atom, "receipts", None) or []) > old_rcpt:
                        VALUES[name]["+receipt"] += 1
        return real_end(self, st, **kw)

    TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage = stage, end_stage


def main() -> None:
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    deals = [d.strip() for d in os.environ.get("DEALS", "").split(",") if d.strip()]
    for deal in deals:
        proj = Path(tempfile.mkdtemp(prefix="p4_")) / "d"
        (proj / "artifacts").mkdir(parents=True)
        for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
            try:
                (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                    cc.get_blob_client(b.name).download_blob().readall())
            except Exception:
                pass
        install(set())
        try:
            res = C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                                    allow_errors=True, allow_unverified_receipts=True,
                                    use_cache=False)
            del res
        except Exception as exc:
            print("  %s: %s: %s" % (deal[:8], type(exc).__name__, exc))

    print("\n=== phase 4: what each stage ADDED ===")
    print("%-30s %7s %s" % ("stage", "added", "ungrounded"))
    for s in PHASE4:
        n = len(ADDED[s])
        u = dict(UNGROUNDED[s])
        print("%-30s %7d %s" % (s, n, u or ""))
    print("\n=== values written onto atoms that already existed ===")
    for s in PHASE4:
        if VALUES[s]:
            print("  %-28s %s" % (s, VALUES[s].most_common(8)))
    print("\n=== a sample of what was added ===")
    for s in PHASE4:
        for atom in ADDED[s][:2]:
            print("  %-26s %-16s %s" % (
                s, str(getattr(atom, "atom_type", ""))[:16],
                " ".join((getattr(atom, "raw_text", "") or "").split())[:70]))


if __name__ == "__main__":
    main()
