# -*- coding: utf-8 -*-
"""Which stages change nothing at all.

A compile runs 51 stages. Twenty-one of them average under five milliseconds,
which is suggestive and not evidence: `table_rollup` is 1.5ms and does real
work, and a stage can be slow and still be a no-op.

The honest question is whether the atom list came out DIFFERENT. This hashes
every atom's id, text, type, value, entity keys and review flags before and
after each stage. A stage whose hash is unchanged did nothing on this deal --
not "little", nothing -- and a stage whose hash never changes on any deal is a
line in the pipeline that could be a function call or deleted.

Reports three classes:

  no effect     the atom list is byte-identical afterwards
  tagged only   same atoms, but some atom's value/flags changed
  structural    atoms were added or removed

Run it models-ON as well before deleting anything: several stages only act when
a model answers, and with `SOWSMITH_NO_MODELS` they are switched off rather than
idle. That distinction is exactly what made the phase-3 audit wrong the first
time.

    DEALS="uuid,uuid" python _stage_effect.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "se_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "se_o.db"))

EFFECT: dict[str, Counter] = {}


def _atom_sig(a) -> str:
    try:
        v = json.dumps(getattr(a, "value", None) or {}, sort_keys=True, default=str)
    except Exception:
        v = str(getattr(a, "value", ""))
    parts = [
        str(getattr(a, "id", "")),
        str(getattr(a, "atom_type", "")),
        str(getattr(a, "raw_text", "") or ""),
        v,
        ",".join(sorted(str(k) for k in (getattr(a, "entity_keys", None) or []))),
        ",".join(sorted(str(f) for f in (getattr(a, "review_flags", None) or []))),
        str(getattr(a, "confidence", "")),
        # Receipts and source_refs MUST be here. Without them
        # `source_replay` -- whose entire job is attaching receipts --
        # reads as a no-op, and so does every backfill that only adds
        # provenance. The first run of this tool reported 29 dead stages
        # on exactly that error.
        str(len(getattr(a, "receipts", None) or [])),
        str(len(getattr(a, "source_refs", None) or [])),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8", "replace")).hexdigest()[:12]


def _list_sig(atoms) -> tuple[str, int]:
    sigs = [_atom_sig(a) for a in atoms]
    return hashlib.sha1("".join(sigs).encode()).hexdigest()[:12], len(sigs)


def install() -> None:
    import inspect
    import app.core.telemetry as TEL
    real_stage, real_end = TEL.CompileTelemetry.stage, TEL.CompileTelemetry.end_stage
    open_names: list[str] = []
    before: list[tuple] = []

    def _atoms():
        for f in inspect.stack()[2:16]:
            got = f.frame.f_locals.get("atoms")
            if isinstance(got, list):
                return got
        return None

    def stage(self, name, **kw):
        a = _atoms()
        before.append(_list_sig(a) if a is not None else (None, None))
        open_names.append(name)
        return real_stage(self, name, **kw)

    def end_stage(self, st, **kw):
        name = (getattr(st, "stage", None) or getattr(st, "name", None)
                or (open_names.pop() if open_names else "?"))
        b_sig, b_n = before.pop() if before else (None, None)
        a = _atoms()
        a_sig, a_n = _list_sig(a) if a is not None else (None, None)
        bucket = EFFECT.setdefault(name, Counter())
        if b_sig is None or a_sig is None:
            bucket["unknown"] += 1
        elif b_n != a_n:
            bucket["structural"] += 1
        elif b_sig != a_sig:
            bucket["tagged"] += 1
        else:
            bucket["none"] += 1
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
    for deal in deals:
        proj = Path(tempfile.mkdtemp(prefix="se_")) / "d"
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
        except Exception as exc:
            print("  %s: %s: %s" % (deal[:8], type(exc).__name__, exc))

    print("\n=== stage effect over %d deal(s) ===" % len(deals))
    dead = [n for n, c in EFFECT.items() if c["structural"] == 0 and c["tagged"] == 0
            and c["none"] > 0]
    tagged = [n for n, c in EFFECT.items() if c["tagged"] and not c["structural"]]
    struct = [n for n, c in EFFECT.items() if c["structural"]]
    print("stages observed          : %d" % len(EFFECT))
    print("changed the list         : %d" % len(struct))
    print("tagged without adding    : %d" % len(tagged))
    print("changed NOTHING          : %d" % len(dead))
    print("\n-- changed nothing, every deal --")
    for n in sorted(dead):
        print("   %s" % n)
    print("\n-- tagged only --")
    for n in sorted(tagged):
        print("   %s" % n)


if __name__ == "__main__":
    main()
