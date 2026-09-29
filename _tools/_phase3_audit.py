# -*- coding: utf-8 -*-
"""Phase 3 (ENRICH + CLASSIFY), every stage, across several deals.

32.5% of the compile and sixteen stages, of which two have been looked at:
`enrich_entity_keys` (profiled, 13.84s -> 11.52s with a substring prefilter)
and `pre_classify_dedup` (audited, correct). Fourteen have not.

Same method as phase 2, because it kept working:

* run a REAL compile per deal -- stages only fire on the material they act on,
  and COPPER never exercised the mail stages
* record what every stage did from the telemetry
* for each atom the compile DROPPED, ask what left entirely -- text and
  figures, whitespace-normalised, because an exact-string compare once
  reported 59 losses that were all double spaces
* several deals, because one cannot tell a dead stage from an inapplicable one

    DEALS="uuid,uuid,uuid" python _phase3_audit.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "p3_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "p3_o.db"))

#: The sixteen stages between "atoms worth enriching" and "atoms worth gating".
PHASE3 = [
    "enrich_entities", "pre_classify_dedup", "document_job_scope",
    "typed_atom_classification", "work_order", "atom_type_sanity",
    "bom_owner", "span_admission", "open_question_resolution",
    "site_geo_fallback", "receipt_backfill", "semantic_dedup",
    "cross_document_conflicts", "site_provenance_join", "stakeholder_dedup",
    "note_provenance_backfill",
]

norm = lambda t: " ".join((t or "").split()).lower()
figs = lambda t: set(re.findall(r"\d+(?:[.,]\d+)*", t or ""))


def run_deal(deal: str, max_artifacts: int = 25) -> dict:
    import app.core.telemetry as TEL
    import app.core.compiler as C

    seen: dict[str, dict] = {}
    real_stage, real_end = TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage

    def stage(self, name, **kw):
        seen.setdefault(name, {})["in"] = kw.get("input_count")
        return real_stage(self, name, **kw)

    def end_stage(self, st, **kw):
        name = getattr(st, "stage", None) or getattr(st, "name", None)
        if name in seen:
            seen[name]["out"] = kw.get("output_count")
            seen[name]["ms"] = getattr(st, "duration_ms", None)
        return real_end(self, st, **kw)

    TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage = stage, end_stage
    try:
        from azure.storage.blob import BlobServiceClient
        conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
        cc = BlobServiceClient.from_connection_string(conn).get_container_client(
            "orbitbrief-artifacts")
        proj = Path(tempfile.mkdtemp(prefix="p3_")) / "d"
        (proj / "artifacts").mkdir(parents=True)
        n = 0
        for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:max_artifacts]:
            try:
                (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                    cc.get_blob_client(b.name).download_blob().readall())
                n += 1
            except Exception:
                pass
        res = C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                                allow_errors=True, allow_unverified_receipts=True,
                                use_cache=False)
        return {"stages": seen, "result": res, "artifacts": n}
    finally:
        TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage = real_stage, real_end


def main() -> None:
    deals = [d.strip() for d in os.environ.get("DEALS", "").split(",") if d.strip()]
    if not deals:
        print("set DEALS=uuid,uuid"); return

    ran: Counter = Counter()
    drops: Counter = Counter()
    tgone: Counter = Counter()
    fgone: Counter = Counter()

    for deal in deals:
        try:
            got = run_deal(deal)
        except Exception as exc:
            print("  %s: compile failed: %s: %s" % (deal[:8], type(exc).__name__, exc))
            continue
        st, res = got["stages"], got["result"]

        by_stage: Counter = Counter()
        texts: dict[str, list] = defaultdict(list)
        for a in (getattr(res, "suppressed_atoms", None) or []):
            for f in (getattr(a, "review_flags", None) or []):
                if str(f).startswith("suppressed:"):
                    s = str(f).split(":", 1)[1]
                    by_stage[s] += 1
                    texts[s].append(a)
                    break

        kept_text = {norm(getattr(x, "raw_text", "")) for x in (res.atoms or [])}
        kept_figs: set[str] = set()
        for x in (res.atoms or []):
            kept_figs |= figs(getattr(x, "raw_text", ""))

        print("\n=== %s  (%d artifacts) ===" % (deal[:8], got["artifacts"]))
        print("%-30s %8s %8s %9s %10s %9s" % ("stage", "in", "out", "dropped", "text gone", "figs gone"))
        for name in PHASE3:
            info = st.get(name)
            if info is None:
                print("%-30s %8s" % (name, "DID NOT RUN"))
                continue
            ran[name] += 1
            d = by_stage.get(name, 0)
            drops[name] += d
            tg = fg = 0
            for a in texts.get(name, []):
                t = norm(getattr(a, "raw_text", ""))
                if t and t not in kept_text:
                    tg += 1
                    fg += len(figs(getattr(a, "raw_text", "")) - kept_figs)
            tgone[name] += tg
            fgone[name] += fg
            print("%-30s %8s %8s %9d %10d %9d"
                  % (name, info.get("in"), info.get("out"), d, tg, fg))

    print("\n\n=== ACROSS %d DEALS ===" % len(deals))
    print("%-30s %6s %10s %11s %11s" % ("stage", "ran", "dropped", "text gone", "figs gone"))
    for name in PHASE3:
        print("%-30s %6d %10d %11d %11d"
              % (name, ran[name], drops[name], tgone[name], fgone[name]))
    silent = [n for n in PHASE3 if ran[n] and not drops[n]]
    print("\nran and never suppressed anything: %s" % (", ".join(silent) or "none"))
    loss = [n for n in PHASE3 if fgone[n]]
    print("stages where a FIGURE left the compile: %s" % (", ".join(loss) or "none"))


if __name__ == "__main__":
    main()
