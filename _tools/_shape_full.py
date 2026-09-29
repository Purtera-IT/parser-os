# -*- coding: utf-8 -*-
"""Every SHAPE stage, on several deals, in one pass.

The phase-2 audit so far has been stage-by-stage on whichever deal happened to
exercise it. That leaves two questions unanswered:

  * a stage that produced nothing on COPPER -- is it DEAD, or was that deal
    simply not the kind it acts on? One deal cannot tell those apart.
  * the stages nobody has looked at yet: email_threading, prose_list_split,
    confidence_floor, candidate_adjudication, supply_conflicts.

This runs a real compile per deal, records what every stage did from the
telemetry, and for the stages that DROP atoms reports what left the compile
entirely -- text and figures both, whitespace-normalised, because an earlier
pass reported 59 losses that were only double spaces.

    DEALS="uuid,uuid,uuid" python _shape_full.py
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
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "full_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "full_o.db"))

#: The eleven stages between "the compile has atoms" and "worth enriching".
SHAPE = [
    "email_threading", "pasted_note_dedup", "quoted_history_dedup",
    "candidate_adjudication", "source_replay", "supply_conflicts",
    "confidence_floor", "prose_list_split", "duplicate_atom_collapse",
    "execution_boilerplate_drop", "table_rollup",
]

norm = lambda t: " ".join((t or "").split()).lower()
figs = lambda t: set(re.findall(r"\d+(?:[.,]\d+)*", t or ""))


def _blob(atom) -> str:
    """An atom's words AND its structured value.

    Checking `raw_text` alone reports losses that never happened: a quoted
    message header is dropped as a duplicate while its timestamp lives on in
    the survivor's `value.email_thread`, and a site's address survives into
    `value` under different words. Both showed up as "a figure left the
    compile" until the value was read too.
    """
    import json as _j
    try:
        v = _j.dumps(getattr(atom, "value", None) or {}, default=str)
    except Exception:
        v = str(getattr(atom, "value", "") or "")
    return (getattr(atom, "raw_text", "") or "") + " " + v



def stage_table(deal: str) -> dict:
    """Run one compile and record every SHAPE stage's in/out from telemetry."""
    import app.core.telemetry as TEL
    import app.core.compiler as C

    seen: dict[str, dict] = {}
    real_stage = TEL.CompileTelemetry.stage
    real_end = TEL.CompileTelemetry.end_stage

    def stage(self, name, **kw):
        seen.setdefault(name, {})["in"] = kw.get("input_count")
        return real_stage(self, name, **kw)

    def end_stage(self, st, **kw):
        name = getattr(st, "stage", None) or getattr(st, "name", None)
        if name in seen:
            seen[name]["out"] = kw.get("output_count")
        return real_end(self, st, **kw)

    TEL.CompileTelemetry.stage = stage
    TEL.CompileTelemetry.end_stage = end_stage
    try:
        from azure.storage.blob import BlobServiceClient
        conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
        cc = BlobServiceClient.from_connection_string(conn).get_container_client(
            "orbitbrief-artifacts")
        proj = Path(tempfile.mkdtemp(prefix="full_")) / "d"
        (proj / "artifacts").mkdir(parents=True)
        n = 0
        for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
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
        TEL.CompileTelemetry.stage = real_stage
        TEL.CompileTelemetry.end_stage = real_end


def main() -> None:
    deals = [d.strip() for d in os.environ.get("DEALS", "").split(",") if d.strip()]
    if not deals:
        print("set DEALS=uuid,uuid"); return

    fired: dict[str, int] = defaultdict(int)
    rows: list[tuple] = []
    for deal in deals:
        try:
            got = stage_table(deal)
        except Exception as exc:
            print("  %s: compile failed: %s: %s" % (deal[:8], type(exc).__name__, exc))
            continue
        st = got["stages"]
        # what the compile threw away, by stage
        sup = list(getattr(got["result"], "suppressed_atoms", None) or [])
        by_stage = Counter()
        texts: dict[str, list] = defaultdict(list)
        for a in sup:
            for f in (getattr(a, "review_flags", None) or []):
                if str(f).startswith("suppressed:"):
                    s = str(f).split(":", 1)[1]
                    by_stage[s] += 1
                    texts[s].append(a)
                    break
        kept_text = {norm(getattr(x, "raw_text", "")) for x in (got["result"].atoms or [])}
        kept_figs: set[str] = set()
        for x in (got["result"].atoms or []):
            kept_figs |= figs(_blob(x))

        print("\n=== %s  (%d artifacts) ===" % (deal[:8], got["artifacts"]))
        print("%-30s %8s %8s %9s %10s %9s" % ("stage", "in", "out", "suppressed", "text gone", "figs gone"))
        for name in SHAPE:
            info = st.get(name)
            if info is None:
                print("%-30s %8s" % (name, "DID NOT RUN"))
                continue
            fired[name] += 1
            drop = by_stage.get(name, 0)
            tgone = fgone = 0
            for a in texts.get(name, []):
                t = norm(getattr(a, "raw_text", ""))
                if t and t not in kept_text:
                    tgone += 1
                    fgone += len(figs(getattr(a, "raw_text", "")) - kept_figs)
            print("%-30s %8s %8s %9d %10d %9d"
                  % (name, info.get("in"), info.get("out"), drop, tgone, fgone))
            rows.append((deal[:8], name, drop, tgone, fgone))

    print("\n\n=== ACROSS %d DEALS ===" % len(deals))
    print("%-30s %7s %10s %10s %10s" % ("stage", "ran", "suppressed", "text gone", "figs gone"))
    for name in SHAPE:
        rs = [r for r in rows if r[1] == name]
        print("%-30s %7d %10d %10d %10d"
              % (name, fired[name], sum(r[2] for r in rs),
                 sum(r[3] for r in rs), sum(r[4] for r in rs)))
    dead = [n for n in SHAPE if fired[n] and not any(r[1] == n and r[2] for r in rows)]
    print("\nran on every deal and never suppressed anything: %s" % (", ".join(dead) or "none"))


if __name__ == "__main__":
    main()
