# -*- coding: utf-8 -*-
"""Warm the persistent embedding cache, and harvest the rule-decision corpus.

ONE pass over the artifacts does both jobs, because they want the same thing:
the set of lines the 14 SemanticRules actually get asked about.

  warming     `embedding_cache` is content-addressed sqlite keyed by
              sha256(model || text) and `embed_texts` already routes through
              it -- but nothing sets `SOWSMITH_EMBED_CACHE_DB`, so it defaults
              to ~/.parseros/embed_cache.db and dies with the container. Every
              cold start re-embeds from the network, and whether the network
              answers is what makes the same deal parse differently on
              different days. Fill the file, ship the file, and the embedder
              being down stops moving the atoms.

  harvesting  `SemanticRule.fires` logs (rule, text, best_pos, best_neg,
              threshold, decision) to `SOWSMITH_RULE_LOG`. Those rows, once a
              person says which decisions were right, are what the threshold
              trainer re-fits on -- and `_trained_threshold` already reads the
              result back from `SOWSMITH_RULE_THRESHOLDS`. So the same parse
              that warms the cache produces the labelling queue.

Only the AMBIGUOUS decisions are worth a person's time: a line whose nearest
positive prototype sits far from the threshold was never in doubt. This ranks
by |best_pos - threshold| so the labelling queue starts where the rule is
actually unsure.

    DEAL_LIMIT=20 python warm_embed_cache.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, r"C:\Users\lilli\parser-os-labeling")

OUT = HERE / "_rule_harvest"
OUT.mkdir(exist_ok=True)
RULE_LOG = OUT / "rule_decisions.jsonl"
QUEUE = OUT / "labelling_queue.jsonl"

# Must be set BEFORE app.core.semantic_rules is imported anywhere.
os.environ["SOWSMITH_RULE_LOG"] = str(RULE_LOG)
os.environ.setdefault(
    "SOWSMITH_EMBED_CACHE_DB", str(OUT / "embed_cache.db"))

DEAL_LIMIT = int(os.environ.get("DEAL_LIMIT", "20"))
ARTIFACT_LIMIT = int(os.environ.get("ARTIFACT_LIMIT", "25"))
#: How close to the threshold a decision has to be to be worth labelling.
BAND = float(os.environ.get("AMBIGUITY_BAND", "0.08"))


def deals(cc, limit: int) -> list[str]:
    out = []
    for b in cc.walk_blobs(name_starts_with="deals/", delimiter="/"):
        out.append(b.name.split("/")[1])
        if len(out) >= limit:
            break
    return out


def main() -> None:
    from azure.storage.blob import BlobServiceClient

    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    cc = BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")

    if RULE_LOG.exists():
        RULE_LOG.unlink()

    from app.core.embedding_cache import get_cache
    from app.parsers import _ocr_chain
    from app.parsers.registry import choose_parser

    # OCR is a separate determinism problem and costs real money; it has
    # nothing to teach the rules, so it is stubbed out of the warm pass.
    _ocr_chain.ocr_pdf_page = lambda *a, **k: {"text": "", "lines": [], "tables": []}
    from app.parsers import email_parser as ep
    ep._ocr_text_from_cid_inline = lambda payload, content_type="": ""

    cache = get_cache()
    print("embed cache: %s" % os.environ["SOWSMITH_EMBED_CACHE_DB"])

    picked = deals(cc, DEAL_LIMIT)
    print("warming over %d deals\n" % len(picked))

    t0 = time.perf_counter()
    parsed = 0
    for i, deal in enumerate(picked, 1):
        tmp = Path(tempfile.mkdtemp(prefix="warm_"))
        paths = []
        for b in list(cc.list_blobs(
                name_starts_with="deals/%s/artifacts/" % deal))[:ARTIFACT_LIMIT]:
            p = tmp / b.name.split("/")[-1]
            try:
                p.write_bytes(cc.get_blob_client(b.name).download_blob().readall())
                paths.append(p)
            except Exception:
                pass
        for p in paths:
            try:
                parser, _m, _a = choose_parser(p)
                if parser is None:
                    continue
                parser.parse_artifact_full(
                    project_id=deal, artifact_id="a", path=p, domain_pack=None)
                parsed += 1
            except Exception:
                pass
        print("  [%2d/%2d] %s  %d artifacts  (%.0fs elapsed)"
              % (i, len(picked), deal[:8], len(paths), time.perf_counter() - t0))

    print("\nparsed %d artifacts in %.0fs" % (parsed, time.perf_counter() - t0))

    if not RULE_LOG.exists():
        print("NO rule decisions logged -- the rules never fired. Either the "
              "embedder is unreachable (they took the lexical path, which does "
              "not log) or SOWSMITH_SEMANTIC_RULES is 0.")
        return

    rows = []
    with RULE_LOG.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    by_rule: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r.get("rule", ""), r.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        by_rule[r["rule"]].append(r)

    fired = Counter(r["rule"] for r in rows if r.get("decision"))
    asked = Counter(r["rule"] for r in rows)

    print("\n%-32s %8s %8s %8s %10s" % ("rule", "asked", "distinct", "fired", "ambiguous"))
    queue = []
    for rule in sorted(by_rule):
        rs = by_rule[rule]
        amb = [r for r in rs
               if abs(float(r["best_pos"]) - float(r["threshold"])) <= BAND]
        amb.sort(key=lambda r: abs(float(r["best_pos"]) - float(r["threshold"])))
        queue.extend({**r, "margin": round(
            abs(float(r["best_pos"]) - float(r["threshold"])), 4)} for r in amb)
        print("%-32s %8d %8d %8d %10d"
              % (rule, asked[rule], len(rs), fired[rule], len(amb)))

    queue.sort(key=lambda r: r["margin"])
    with QUEUE.open("w", encoding="utf-8") as fh:
        for r in queue:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\ntotal decisions logged : %d" % len(rows))
    print("distinct (rule, text)  : %d" % len(seen))
    print("worth labelling        : %d  (within %.2f of the threshold)"
          % (len(queue), BAND))
    print("labelling queue        : %s" % QUEUE)

    db = Path(os.environ["SOWSMITH_EMBED_CACHE_DB"])
    if db.exists():
        print("embed cache on disk    : %.1f MB" % (db.stat().st_size / 1e6))
    if cache is None:
        print("WARNING: no cache object -- nothing was persisted.")


if __name__ == "__main__":
    main()
