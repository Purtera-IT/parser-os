# -*- coding: utf-8 -*-
"""Phase 5 (GATE + RESOLVE): which atoms lose the right to govern?

Phases 1-3 asked what a stage DELETED and the suppression ledger answers it.
Phase 4 asks what a stage INVENTED. Phase 5 asks neither, because its two
biggest stages do not add or remove anything:

  confidence_floor          does not drop an atom -- it forces needs_review, so
                            the atom survives and stops governing
  confidence_recalibration  rewrites every atom's confidence from provenance
                            defaults (0.82/0.85) to content-aware scoring

A wrong confidence does not lose a fact. It demotes one out of the brief, and
no content-loss audit can see that because the atom is still in the envelope.

So: record every atom's confidence before and after recalibration, and count the
ones that CROSS the floor -- above it going in, below it coming out. Those are
atoms the compile decided, silently, are not worth governing.

`LOW_CONFIDENCE_FLOOR = 0.50` is a bare constant in compiler.py and is not in
`app/core/calibration.py`, the registry whose whole rule is "small is fine;
hidden is not".

    DEALS="uuid,uuid" python _phase5_authority.py
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
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "p5_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "p5_o.db"))

FLOOR = 0.50
WATCH = ("confidence_recalibration", "confidence_floor", "substance_gate",
         "entity_resolution", "noise_suppression")

MOVED: Counter = Counter()
CROSSED_DOWN: list = []
CROSSED_UP: Counter = Counter()
SHIFT: list = []


def install() -> None:
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

    def _conf(atoms):
        out = {}
        for a in atoms:
            try:
                out[id(a)] = float(getattr(a, "confidence", 0.0) or 0.0)
            except Exception:
                out[id(a)] = 0.0
        return out

    def stage(self, name, **kw):
        a = _atoms()
        before.append(_conf(a) if a is not None else {})
        names.append(name)
        return real_stage(self, name, **kw)

    def end_stage(self, st, **kw):
        name = (getattr(st, "stage", None) or getattr(st, "name", None)
                or (names.pop() if names else "?"))
        prev = before.pop() if before else {}
        a = _atoms()
        if name in WATCH and a is not None:
            for atom in a:
                aid = id(atom)
                if aid not in prev:
                    continue
                was = prev[aid]
                try:
                    now = float(getattr(atom, "confidence", 0.0) or 0.0)
                except Exception:
                    continue
                if abs(now - was) < 1e-9:
                    continue
                MOVED[name] += 1
                SHIFT.append((name, round(now - was, 4)))
                if was >= FLOOR > now:
                    CROSSED_DOWN.append((
                        name, round(was, 3), round(now, 3),
                        str(getattr(atom, "atom_type", ""))[:20],
                        " ".join((getattr(atom, "raw_text", "") or "").split())[:74],
                    ))
                elif now >= FLOOR > was:
                    CROSSED_UP[name] += 1
        return real_end(self, st, **kw)

    TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage = stage, end_stage


def main() -> None:
    install()
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    deals = [d.strip() for d in os.environ.get("DEALS", "").split(",") if d.strip()]
    ok = 0
    for deal in deals:
        proj = Path(tempfile.mkdtemp(prefix="p5_")) / "d"
        (proj / "artifacts").mkdir(parents=True)
        for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
            try:
                (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                    cc.get_blob_client(b.name).download_blob().readall())
            except Exception:
                pass
        try:
            C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                              allow_errors=True, allow_unverified_receipts=True,
                              use_cache=False)
            ok += 1
        except Exception as exc:
            print("  %s: %s: %s" % (deal[:8], type(exc).__name__, exc))

    print("\n=== phase 5 over %d deal(s): who moved a confidence ===" % ok)
    print("%-30s %9s %11s %9s" % ("stage", "moved", "crossed DOWN", "up"))
    for s in WATCH:
        down = sum(1 for r in CROSSED_DOWN if r[0] == s)
        print("%-30s %9d %11d %9d" % (s, MOVED[s], down, CROSSED_UP[s]))
    if SHIFT:
        ups = [d for _, d in SHIFT if d > 0]
        downs = [d for _, d in SHIFT if d < 0]
        print("\nshifts: %d up (mean +%.3f), %d down (mean %.3f)" % (
            len(ups), sum(ups)/len(ups) if ups else 0,
            len(downs), sum(downs)/len(downs) if downs else 0))
    print("\n=== atoms that lost the right to govern (above %.2f -> below) ===" % FLOOR)
    print("total: %d" % len(CROSSED_DOWN))
    for stage_, was, now, at, text in CROSSED_DOWN[:20]:
        print("  %-24s %.3f -> %.3f  %-18s %s" % (stage_, was, now, at, text))


if __name__ == "__main__":
    main()
