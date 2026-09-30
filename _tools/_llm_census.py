# -*- coding: utf-8 -*-
"""Which STAGE reaches for a model, how often, and for how long.

`SOWSMITH_DISABLE_LLM=1` is set by every audit tool in this directory, so
every measurement taken with them describes a compile with the model paths
off. That is the right default for a determinism or content-loss audit -- a
remote model is not reproducible -- but it means "this stage never suppressed
anything" may mean "this stage did not run".

This counts, per stage, every call that reaches a model entry point, with the
kill-switch OFF, so the two configurations can be compared honestly.

    DEAL=<uuid> python _llm_census.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")
# Deliberately NOT setting SOWSMITH_DISABLE_LLM.
os.environ.setdefault("SOWSMITH_EMBED_CACHE_DB", str(HERE / "_rule_harvest" / "lc_e.db"))
os.environ.setdefault("SOWSMITH_OCR_CACHE_DB", str(HERE / "_rule_harvest" / "lc_o.db"))
os.environ.setdefault("SOWSMITH_LLM_TIMEOUT", os.environ.get("LLM_TIMEOUT", "8"))

CALLS: Counter = Counter()
SECS: Counter = Counter()
CURRENT = {"stage": "<before any stage>"}

#: Every module-level entry point that can reach a model.
TARGETS = [
    ("app.core.llm_client", "call_llm"),
    ("app.core.llm_client", "call_teacher"),
    ("app.core.ollama_host", "ollama_reachable"),
    ("app.core.vision_extraction", "vision_endpoint_reachable"),
    ("app.core.semantic_role", "classify_role"),
    ("app.core.typed_atom_classifier", "classify_promotable"),
    ("app.core.multi_entity_llm", "extract_entities_llm"),
    ("app.core.site_llm_verify", "verify_sites"),
    ("app.core.reranker", "rerank"),
]


def install() -> None:
    import importlib
    import app.core.telemetry as TEL

    real_stage = TEL.CompileTelemetry.stage

    def stage(self, name, **kw):
        CURRENT["stage"] = name
        return real_stage(self, name, **kw)

    TEL.CompileTelemetry.stage = stage

    for modname, fn in TARGETS:
        try:
            mod = importlib.import_module(modname)
            real = getattr(mod, fn)
        except Exception:
            continue

        def make(real=real, label=f"{modname.split('.')[-1]}.{fn}"):
            def wrapped(*a, **kw):
                t0 = time.perf_counter()
                try:
                    return real(*a, **kw)
                finally:
                    key = f"{CURRENT['stage']} -> {label}"
                    CALLS[key] += 1
                    SECS[key] += time.perf_counter() - t0
            return wrapped

        try:
            setattr(mod, fn, make())
        except Exception:
            continue


def main() -> None:
    install()
    import app.core.compiler as C
    from azure.storage.blob import BlobServiceClient
    deal = os.environ["DEAL"]
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")
    proj = Path(tempfile.mkdtemp(prefix="lc_")) / "d"
    (proj / "artifacts").mkdir(parents=True)
    for b in list(cc.list_blobs(name_starts_with=f"deals/{deal}/artifacts/"))[:25]:
        try:
            (proj / "artifacts" / b.name.split("/")[-1]).write_bytes(
                cc.get_blob_client(b.name).download_blob().readall())
        except Exception:
            pass
    t0 = time.perf_counter()
    res = C.compile_project(project_dir=proj / "artifacts", project_id=deal[:8],
                            allow_errors=True, allow_unverified_receipts=True,
                            use_cache=False)
    wall = time.perf_counter() - t0

    print("\n=== compile %.1fs, %d atoms, kill-switch OFF ==="
          % (wall, len(res.atoms or [])))
    print("%-58s %7s %9s" % ("stage -> model entry point", "calls", "sec"))
    for key, n in CALLS.most_common(24):
        print("%-58s %7d %9.2f" % (key[:58], n, SECS[key]))
    print("\ntotal model-path time: %.1fs of %.1fs (%.0f%%)"
          % (sum(SECS.values()), wall, 100 * sum(SECS.values()) / wall if wall else 0))


if __name__ == "__main__":
    main()
