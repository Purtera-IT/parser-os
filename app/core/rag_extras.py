"""v42 — the six "insane" RAG upgrades stacked on top of v41:

  1. HyDE (Hypothetical Document Embeddings) — at first run, LLM
     generates 30 hypothetical sentences per entity type. Cached to
     disk so subsequent runs reuse them. Expands exemplar set ~3.5x
     for broader semantic coverage.

  2. Iterative SICRL — runs SICRL 2-3 passes with the bigger model
     (qwen3:32b when SOWSMITH_BIG_MODEL is set). Each pass refines
     section coverage until convergence.

  3. Self-bootstrapping negative exemplars — each canonicalize
     rejection appends to .parser_os/negatives/<entity_type>.json.
     Each compile makes the negatives sharper without curation.

  4. Tournament canonicalization — pairs items with cosine_sim > 0.85
     and asks LLM "same canonical? if yes, which is better?"
     Eliminates "RFP says X" + "Response says X" duplication.

  5. Cross-document contradiction detection — for entities found in
     multiple artifacts with different values, LLM judges compatibility.
     Auto-emits reconciliation_flag records for the PM.

  6. Multi-document graph RAG — builds entity co-occurrence graph,
     expands sparse-entity-type search via graph neighbors.

Tunable via env:
  SOWSMITH_HYDE_DISABLE=1            disable HyDE augmentation
  SOWSMITH_HYDE_PER_TYPE=30          how many HyDE examples per type
  SOWSMITH_TOURNAMENT_DISABLE=1      disable tournament canon
  SOWSMITH_CONTRADICTION_DISABLE=1   disable cross-doc contradiction
  SOWSMITH_GRAPHRAG_DISABLE=1        disable graph expansion
  SOWSMITH_BIG_MODEL=qwen3:32b       use bigger model for canon/SICRL
  SOWSMITH_NEG_BOOTSTRAP_DISABLE=1   disable self-learning negatives
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 1. HyDE — Hypothetical Document Embeddings
# ════════════════════════════════════════════════════════════════════


_HYDE_CACHE_DIR = Path.home() / ".parser_os" / "hyde"


def _hyde_cache_path(entity_type: str) -> Path:
    _HYDE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _HYDE_CACHE_DIR / f"{entity_type}.json"


_HYDE_PROMPTS: dict[str, str] = {
    "requirement": (
        "Generate 30 DIVERSE hypothetical sentences that would appear\n"
        "in a real RFP / bid document as REQUIREMENTS (obligations on\n"
        "the contractor / vendor / district). Cover ALL these flavors:\n"
        "  - shall / must / will / agrees to / is required to\n"
        "  - covenants / warrants / undertakes / commits to\n"
        "  - shall not / must not / may not (negative obligations)\n"
        "  - district reserves the right / may terminate / may require\n"
        "  - insurance / indemnification / compliance / financial / labor\n"
        "  - technical compliance / certification / interoperability\n\n"
        "Each sentence should be SPECIFIC and realistic (no placeholders\n"
        'like "the contractor shall X"). Vary structure, vocabulary,\n'
        "and topic. Output ONE PER LINE, no numbering, no JSON.\n\n"
        "/no_think"
    ),
    "stakeholder": (
        "Generate 30 DIVERSE hypothetical sentences from bid docs\n"
        "that name real people (stakeholders / contacts / project team\n"
        "members). Include these patterns:\n"
        "  - bid-contact lines (Please contact / All questions to)\n"
        "  - email signature blocks (Name, Title, email@domain)\n"
        "  - team rosters (Front of the House: Name1/Role, Name2/Role)\n"
        "  - sign-off lines (Authorized signature: Name)\n"
        "  - role-titled introductions (Name will serve as Title)\n"
        "  - project staff lists (Engineering: Name1, Field: Name2)\n\n"
        "Use diverse names (different ethnicities + first/last orderings).\n"
        "Output ONE PER LINE, no numbering, no JSON.\n\n"
        "/no_think"
    ),
    "site": (
        "Generate 30 DIVERSE hypothetical sentences from bid docs\n"
        "that name PHYSICAL SITES (specific buildings, codes, addresses,\n"
        "campuses). Include:\n"
        "  - site codes (ATL-HQ-01, STORE-142, MDF-3A, BCSD-CO)\n"
        "  - named buildings (Beaufort Elementary School, Innovation Tower)\n"
        "  - full street addresses with city/state/zip\n"
        "  - named campuses (Wesley School campus, Atlanta corporate campus)\n"
        "  - sub-buildings with site context\n\n"
        "Use diverse customer types (school district, hospital, airport,\n"
        "retail, corporate office). Output ONE PER LINE, no JSON.\n\n"
        "/no_think"
    ),
    "quantity": (
        "Generate 30 DIVERSE hypothetical sentences with structural\n"
        "QUANTITIES from bid docs: SLAs, counts, durations, percentages,\n"
        "commercial terms. Include:\n"
        "  - uptime / availability (99.999% / 99.95% / 99.5%)\n"
        "  - response times (within 2 hours, 5 minute failover)\n"
        "  - counts (X schools, Y access points, Z stores)\n"
        "  - help desk hours (M-F 8a-5p EST, 24/7 monitoring)\n"
        "  - payment terms (Net-30, Net-45)\n"
        "  - durations (5-year contract, 3-year warranty)\n"
        "  - lead times (6-8 weeks, 14-day TSA badging)\n\n"
        "Output ONE PER LINE, no numbering, no JSON.\n\n"
        "/no_think"
    ),
}


def _parse_hyde_response(text: str) -> list[str]:
    """Parse newline-separated sentences from LLM HyDE response.
    Strips numbering, bullets, code-block markers."""
    if not text:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering / bullets / quotes
        line = re.sub(r"^\s*(?:\d+[\.)]\s*|[-*•]\s*|>\s*)", "", line).strip()
        line = line.strip('"\'')
        # Skip code-block fences, markdown headers, JSON brackets
        if line.startswith(("```", "#", "{", "}", "[", "]")):
            continue
        if len(line) < 20 or len(line) > 400:
            continue
        out.append(line)
    return out


def load_or_generate_hyde(
    entity_type: str,
    *,
    llm_call: Callable[[str, int], str],
    force_regenerate: bool = False,
    per_type: int = 30,
) -> list[str]:
    """Load HyDE examples from disk cache, or generate via LLM if missing.

    Returns list of hypothetical sentences to AUGMENT the exemplar set.
    """
    if os.environ.get("SOWSMITH_HYDE_DISABLE"):
        return []
    cache = _hyde_cache_path(entity_type)
    if not force_regenerate and cache.exists():
        try:
            with cache.open("r", encoding="utf-8") as f:
                data = json.load(f)
            sentences = data.get("sentences", [])
            if isinstance(sentences, list) and len(sentences) >= 5:
                return [s for s in sentences if isinstance(s, str)]
        except Exception:
            pass

    prompt = _HYDE_PROMPTS.get(entity_type)
    if not prompt:
        return []
    logger.info("HyDE generating %d examples for %s ...", per_type, entity_type)
    t0 = time.time()
    text = llm_call(prompt, 2048)
    parsed = _parse_hyde_response(text)[:per_type]
    elapsed = time.time() - t0
    logger.info("HyDE %s: %d examples in %.1fs", entity_type, len(parsed), elapsed)

    if parsed:
        try:
            with cache.open("w", encoding="utf-8") as f:
                json.dump(
                    {"entity_type": entity_type, "sentences": parsed,
                     "generated_at": time.time()},
                    f, indent=2,
                )
        except Exception as e:
            logger.warning("HyDE cache write failed: %s", e)

    return parsed


# ════════════════════════════════════════════════════════════════════
# 3. Self-Bootstrapping Negative Exemplars
# ════════════════════════════════════════════════════════════════════


_NEG_CACHE_DIR = Path.home() / ".parser_os" / "negatives"


def _neg_cache_path(entity_type: str) -> Path:
    _NEG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _NEG_CACHE_DIR / f"{entity_type}.json"


def load_bootstrapped_negatives(entity_type: str, *, max_count: int = 200) -> list[str]:
    """Load accumulated negative exemplars from disk.
    Returns at most `max_count` (most recent if file is bigger)."""
    if os.environ.get("SOWSMITH_NEG_BOOTSTRAP_DISABLE"):
        return []
    p = _neg_cache_path(entity_type)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("rejected", [])
        if isinstance(items, list):
            return [s for s in items[-max_count:] if isinstance(s, str)]
    except Exception:
        pass
    return []


def append_bootstrapped_negative(
    entity_type: str,
    rejected_sentence: str,
    *,
    max_total: int = 500,
) -> None:
    """Append a sentence to the persistent negative-exemplar store.
    Used after canonicalize returns keep=False with HIGH confidence
    (clear noise). Caps at max_total most-recent entries.
    """
    if os.environ.get("SOWSMITH_NEG_BOOTSTRAP_DISABLE"):
        return
    if not rejected_sentence or len(rejected_sentence) < 20:
        return
    if len(rejected_sentence) > 300:
        return  # too long, likely false positive
    p = _neg_cache_path(entity_type)
    items: list[str] = []
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("rejected", [])
        except Exception:
            items = []
    # Dedup before append (case-insensitive trim)
    sig = re.sub(r"\s+", " ", rejected_sentence.lower()).strip()
    existing_sigs = {re.sub(r"\s+", " ", s.lower()).strip() for s in items}
    if sig in existing_sigs:
        return
    items.append(rejected_sentence.strip())
    if len(items) > max_total:
        items = items[-max_total:]
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump({"entity_type": entity_type, "rejected": items}, f, indent=2)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════
# 4. Tournament Canonicalization
# ════════════════════════════════════════════════════════════════════


_TOURNAMENT_PROMPT = (
    "TASK: Are these two items the SAME {entity_type} in this bid?\n\n"
    "ITEM A: {a}\n\nITEM B: {b}\n\n"
    "Output exactly one JSON object on one line:\n"
    '  {{"same": true, "canonical": "<merged canonical form>"}} if same\n'
    '  {{"same": false}} if distinct\n\n'
    "Merge rules when same:\n"
    "  - Prefer the more specific / longer canonical form\n"
    "  - Drop boilerplate prefixes ('The contractor shall')\n"
    "  - Keep numeric values verbatim\n"
    "  - Don't fabricate information\n\n"
    "/no_think"
)


def run_tournament(
    items: list[dict[str, Any]],
    item_embeddings: np.ndarray,  # (N, D) L2-normalized
    *,
    entity_type: str,
    canonical_key: str,
    llm_call: Callable[[str, int], str],
    parse_json: Callable[[str], Any],
    sim_threshold: float = 0.85,
    max_pairs: int = 100,
) -> list[dict[str, Any]]:
    """Pairwise tournament dedup. For each pair of items with cosine
    similarity > sim_threshold, LLM judges same/distinct and produces
    canonical form when same.

    Returns deduped items (some merged).
    """
    if os.environ.get("SOWSMITH_TOURNAMENT_DISABLE"):
        return items
    if len(items) < 2 or item_embeddings.size == 0:
        return items
    n = len(items)
    # Pairwise sims
    sims = item_embeddings @ item_embeddings.T
    # Find pairs above threshold
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= sim_threshold:
                pairs.append((i, j, float(sims[i, j])))
    # Sort by sim desc, cap
    pairs.sort(key=lambda x: -x[2])
    pairs = pairs[:max_pairs]
    if not pairs:
        return items

    # Union-find for transitive merge
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    parallel = int(os.environ.get("SOWSMITH_TOURNAMENT_PARALLEL", "8"))
    merge_canons: dict[int, str] = {}

    def judge(pair_data):
        i, j, sim = pair_data
        a_text = items[i].get(canonical_key) or items[i].get("name") or ""
        b_text = items[j].get(canonical_key) or items[j].get("name") or ""
        if not a_text or not b_text:
            return None
        prompt = _TOURNAMENT_PROMPT.format(
            entity_type=entity_type, a=a_text, b=b_text,
        )
        text = llm_call(prompt, 256)
        obj = parse_json(text)
        if not isinstance(obj, dict) or not obj.get("same"):
            return None
        canon = obj.get("canonical") or a_text
        return (i, j, canon)

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(judge, p) for p in pairs]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception:
                result = None
            if not result:
                continue
            i, j, canon = result
            union(i, j)
            # Stash the canonical form on whichever root wins
            root = find(i)
            merge_canons[root] = canon

    # Build deduped output: one item per cluster, using the canonical
    # form from merge_canons when set.
    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        r = find(idx)
        clusters.setdefault(r, []).append(idx)

    out: list[dict[str, Any]] = []
    for root, members in clusters.items():
        # Use the first member as base
        base = dict(items[members[0]])
        if root in merge_canons and merge_canons[root]:
            # Update canonical_key with merged form
            if canonical_key in base:
                base[canonical_key] = merge_canons[root]
            elif "name" in base:
                base["name"] = merge_canons[root]
            elif "text" in base:
                base["text"] = merge_canons[root]
        # Collect source attribution from all members
        sources = set()
        for m in members:
            aid = items[m].get("_source_artifact_id")
            if aid:
                sources.add(aid)
        if sources:
            base["_source_artifact_ids"] = sorted(sources)
        if len(members) > 1:
            base["_merged_from"] = len(members)
        out.append(base)
    return out


# ════════════════════════════════════════════════════════════════════
# 5. Cross-Document Contradiction Detection
# ════════════════════════════════════════════════════════════════════


_CONTRADICTION_PROMPT = (
    "TASK: Do these two statements about the same topic CONTRADICT\n"
    "each other? They appeared in different documents in this bid.\n\n"
    "STATEMENT A (from {source_a}): {a}\n\n"
    "STATEMENT B (from {source_b}): {b}\n\n"
    "Output exactly one JSON object on one line:\n"
    '  {{"contradicts": true, "kind": "<short label>", "explanation": "<one sentence>"}}\n'
    "  or\n"
    '  {{"contradicts": false}}\n\n'
    "Examples of contradictions:\n"
    "  - Net-30 vs Net-45 → kind=net_terms_disagreement\n"
    "  - 99.5% uptime vs 99.95% uptime → kind=sla_uptime_disagreement\n"
    "  - $1M coverage vs $2M coverage → kind=coverage_limit_disagreement\n"
    "  - 52 access points vs 30 access points → kind=quantity_disagreement\n\n"
    "Output contradicts:false if A and B are about different things,\n"
    "complementary, or both true (no conflict).\n\n"
    "/no_think"
)


def detect_cross_doc_contradictions(
    items: list[dict[str, Any]],
    item_embeddings: np.ndarray,
    *,
    canonical_key: str,
    llm_call: Callable[[str, int], str],
    parse_json: Callable[[str], Any],
    sim_threshold_min: float = 0.55,
    sim_threshold_max: float = 0.92,  # below tournament dedup threshold
    max_pairs: int = 60,
) -> list[dict[str, Any]]:
    """For pairs of items from DIFFERENT artifacts with moderate
    similarity (close enough to be about the same topic, distinct
    enough not to be exact dupes), ask the LLM if they contradict.

    Returns list of contradiction flags:
      {kind, explanation, item_a, item_b, source_a, source_b}
    """
    if os.environ.get("SOWSMITH_CONTRADICTION_DISABLE"):
        return []
    if len(items) < 2 or item_embeddings.size == 0:
        return []
    n = len(items)
    sims = item_embeddings @ item_embeddings.T
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        a_src = items[i].get("_source_artifact_id")
        for j in range(i + 1, n):
            b_src = items[j].get("_source_artifact_id")
            # Only cross-document pairs
            if not a_src or not b_src or a_src == b_src:
                continue
            s = float(sims[i, j])
            if sim_threshold_min <= s < sim_threshold_max:
                pairs.append((i, j, s))
    pairs.sort(key=lambda x: -x[2])
    pairs = pairs[:max_pairs]
    if not pairs:
        return []

    parallel = int(os.environ.get("SOWSMITH_CONTRADICTION_PARALLEL", "8"))
    flags: list[dict[str, Any]] = []

    def judge(pair_data):
        i, j, sim = pair_data
        a = items[i].get(canonical_key) or items[i].get("name") or ""
        b = items[j].get(canonical_key) or items[j].get("name") or ""
        if not a or not b:
            return None
        prompt = _CONTRADICTION_PROMPT.format(
            a=a, b=b,
            source_a=items[i].get("_source_artifact_id", "?"),
            source_b=items[j].get("_source_artifact_id", "?"),
        )
        text = llm_call(prompt, 256)
        obj = parse_json(text)
        if not isinstance(obj, dict) or not obj.get("contradicts"):
            return None
        return {
            "kind": obj.get("kind", "cross_doc_contradiction"),
            "explanation": obj.get("explanation", ""),
            "item_a": a,
            "item_b": b,
            "source_a": items[i].get("_source_artifact_id", "?"),
            "source_b": items[j].get("_source_artifact_id", "?"),
            "similarity": round(sim, 4),
        }

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(judge, p) for p in pairs]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception:
                result = None
            if result:
                flags.append(result)
    return flags


# ════════════════════════════════════════════════════════════════════
# 6. Multi-Document Graph RAG (entity co-occurrence)
# ════════════════════════════════════════════════════════════════════


def build_cooccurrence_graph(
    atoms: list[Any],
) -> dict[str, set[str]]:
    """Build entity co-occurrence graph: for each entity_key, which
    other entity_keys appear on the same atom.

    Returns {entity_key: {co_occurring_keys}}.
    """
    graph: dict[str, set[str]] = {}
    for atom in atoms:
        keys = getattr(atom, "entity_keys", None) or []
        if len(keys) < 2:
            continue
        keyset = set(keys)
        for k in keys:
            graph.setdefault(k, set()).update(keyset - {k})
    return graph


def graph_expand_seeds(
    seed_keys: set[str],
    graph: dict[str, set[str]],
    *,
    target_prefix: str,
    max_expansion: int = 50,
) -> set[str]:
    """Expand a set of seed entity keys via 1-hop graph neighbors,
    keeping only neighbors with the target prefix.

    E.g. seeds = {customer:beaufort_county_school_district},
         target_prefix = "site:",
         → returns all site:* keys co-occurring with the customer.
    """
    out: set[str] = set()
    for seed in seed_keys:
        neighbors = graph.get(seed, set())
        for n in neighbors:
            if n.startswith(target_prefix):
                out.add(n)
                if len(out) >= max_expansion:
                    return out
    return out


# ════════════════════════════════════════════════════════════════════
# UNIFIED API HELPERS used by multi_entity_llm.py
# ════════════════════════════════════════════════════════════════════


def augment_exemplars_with_hyde(
    base_exemplars: list[str],
    entity_type: str,
    *,
    llm_call: Callable[[str, int], str],
) -> list[str]:
    """Combine static exemplars with HyDE-generated + bootstrapped negatives.

    Returns augmented positive exemplars (HyDE adds variety).
    Negatives are returned separately (caller picks them up).
    """
    hyde_extras = load_or_generate_hyde(entity_type, llm_call=llm_call)
    if hyde_extras:
        # Mix in HyDE without losing static — static is hand-curated
        # and high-quality; HyDE adds breadth.
        return list(base_exemplars) + hyde_extras
    return list(base_exemplars)


__all__ = [
    "load_or_generate_hyde",
    "augment_exemplars_with_hyde",
    "load_bootstrapped_negatives",
    "append_bootstrapped_negative",
    "run_tournament",
    "detect_cross_doc_contradictions",
    "build_cooccurrence_graph",
    "graph_expand_seeds",
]
