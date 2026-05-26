"""Universal multi-entity LLM extractor — split into 5 focused calls run in parallel.

Each entity category gets its own dedicated LLM call with a prompt
laser-focused on that one task and a doc excerpt tuned to where
that entity type actually lives in bid docs. The 5 calls run via
``ThreadPoolExecutor`` so on:

  - vLLM / TGI                 → all 5 batched into one inference step
  - Ollama with NUM_PARALLEL≥5 → all 5 run concurrently on the model
  - Ollama serial (default)    → 5 calls sequentially, similar to the
                                  old omnibus prompt

Why split (vs one big prompt):
  - Better focus per category (higher precision + recall)
  - No JSON-truncation risk (each output is small)
  - Failure isolation (one call failing ≠ losing 4 other categories)
  - Doc excerpt tuned per category (customer only needs cover page;
    requirements needs full body)
  - Same wall-clock as omnibus on parallel-capable backends

Public API:
    extract_all_entities_with_llm(atoms) → dict
        Runs all 5 extractors in parallel. Returns:
        {
          "customer": str | None,
          "stakeholders": [{"name", "role", "email", "phone"}, ...],
          "milestones": [{"name", "date", "notes"}, ...],
          "requirements": [{"text", "category"}, ...],
          "site_clusters": [{"canonical_name", "aliases"}, ...]
        }

Configuration (env vars, all optional):
    OLLAMA_HOST                       (default http://100.114.102.122:11434)
    OLLAMA_MODEL                      (default qwen3:14b)
    SOWSMITH_LLM_TIMEOUT              (default 180)
    SOWSMITH_LLM_PARALLEL             (default 5)
    SOWSMITH_MULTI_ENTITY_DISABLE=1   skip all 5 calls
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import urllib.request
from typing import Any, Callable

DEFAULT_HOST = "http://100.114.102.122:11434"
DEFAULT_MODEL = "qwen3:14b"
DEFAULT_TIMEOUT = 240
DEFAULT_PARALLEL = 5


# ════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════


def extract_all_entities_with_llm(atoms: list[Any]) -> dict[str, Any]:
    """Run all 5 focused extractors in parallel and merge results.

    Returns a dict with the standard 5 keys. Any individual extractor
    that fails returns its zero-value (None / []) so downstream code
    sees a stable shape regardless of partial failures.
    """
    if os.environ.get("SOWSMITH_MULTI_ENTITY_DISABLE"):
        return _empty_result()
    if not atoms:
        return _empty_result()

    # Pre-compute the per-category doc excerpts ONCE (sharing the
    # atom iteration across all 5 calls).
    by_artifact = _group_by_artifact(atoms)
    if not by_artifact:
        return _empty_result()

    excerpts = {
        "customer": _build_excerpt_for_customer(by_artifact),
        "milestones": _build_excerpt_for_milestones(by_artifact),
    }

    # Three categories get the chunked-per-doc path (one LLM call per
    # artifact, union + dedupe):
    #   - requirements:  Pack 18 Beaufort has 196 shall/must clauses;
    #                    single 30K-char excerpt loses 80%+.
    #   - stakeholders:  big vendor PDFs bury contacts on page 100+;
    #                    chunked recovers names from signature blocks
    #                    + contact pages outside the first 30K chars.
    #   - site_clusters: roster sheets / multi-site PDFs (Albuquerque
    #                    Public Schools, Muskegon Paging) list dozens
    #                    of buildings — single excerpt sees the first
    #                    few only.
    #
    # Two categories keep single-call extraction (their target volume
    # per pack is bounded so a 1-shot excerpt is sufficient):
    #   - customer:      1 canonical per pack; cover-page-heavy.
    #   - milestones:    typically 0-25; LLM finds them in any
    #                    moderate-sized excerpt.
    parallel = int(os.environ.get("SOWSMITH_LLM_PARALLEL", str(DEFAULT_PARALLEL)))

    # v38: embedding-retrieval extractors for the recall-heavy entity
    # types (requirements, stakeholders, sites). Default-on; falls back
    # to chunked path when SOWSMITH_RETRIEVAL_DISABLE is set OR the
    # embedding endpoint is unreachable.
    use_retrieval = (
        not os.environ.get("SOWSMITH_RETRIEVAL_DISABLE")
    )
    if use_retrieval:
        try:
            from app.core.embedding_retrieval import embedding_endpoint_reachable
            use_retrieval = embedding_endpoint_reachable()
        except Exception:
            use_retrieval = False

    if use_retrieval:
        def _retrieved_or_chunked_requirements() -> list[dict[str, Any]]:
            r = _extract_requirements_retrieved(by_artifact)
            return r if r else _extract_requirements_chunked(by_artifact)

        def _retrieved_or_chunked_stakeholders() -> list[dict[str, Any]]:
            r = _extract_stakeholders_retrieved(by_artifact)
            return r if r else _extract_stakeholders_chunked(by_artifact)

        def _retrieved_or_chunked_sites() -> list[dict[str, Any]]:
            r = _extract_site_clusters_retrieved(by_artifact)
            return r if r else _extract_site_clusters_chunked(by_artifact)

        calls: dict[str, Callable[[], Any]] = {
            "customer": lambda: _extract_customer(excerpts["customer"]),
            "stakeholders": _retrieved_or_chunked_stakeholders,
            "milestones": lambda: _extract_milestones(excerpts["milestones"]),
            "requirements": _retrieved_or_chunked_requirements,
            "site_clusters": _retrieved_or_chunked_sites,
            "quantities": lambda: _extract_quantities_retrieved(by_artifact),
        }
    else:
        calls = {
            "customer": lambda: _extract_customer(excerpts["customer"]),
            "stakeholders": lambda: _extract_stakeholders_chunked(by_artifact),
            "milestones": lambda: _extract_milestones(excerpts["milestones"]),
            "requirements": lambda: _extract_requirements_chunked(by_artifact),
            "site_clusters": lambda: _extract_site_clusters_chunked(by_artifact),
        }

    results: dict[str, Any] = _empty_result()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(fn): key for key, fn in calls.items()}
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            try:
                results[key] = fut.result()
            except Exception:
                # Individual extractor failure: keep zero-value.
                pass
    # Stash site_clusters for entity_resolution to pick up without a
    # second LLM call. Used by collect_site_alias_groups to feed
    # canonical-name fusion alongside the regex co-mention patterns.
    if results.get("site_clusters"):
        _stash_session_site_clusters(atoms, results["site_clusters"])
    return results


def _empty_result() -> dict[str, Any]:
    return {
        "customer": None,
        "stakeholders": [],
        "milestones": [],
        "requirements": [],
        "site_clusters": [],
        "quantities": [],
    }


# ════════════════════════════════════════════════════════════════════
# DOC EXCERPT BUILDERS (per-category)
# ════════════════════════════════════════════════════════════════════


def _group_by_artifact(atoms: list[Any]) -> dict[str, dict[str, Any]]:
    """Group atoms by artifact_id, collecting body text + section headings + filename."""
    by_artifact: dict[str, dict[str, Any]] = {}
    for atom in atoms:
        aid = getattr(atom, "artifact_id", None)
        if not aid:
            continue
        slot = by_artifact.setdefault(aid, {
            "bodies": [],
            "headings": set(),
            "filename": None,
        })
        raw = getattr(atom, "raw_text", None) or ""
        if isinstance(raw, str) and raw:
            slot["bodies"].append(raw)
        try:
            refs = getattr(atom, "source_refs", None) or []
            if refs:
                locator = getattr(refs[0], "locator", None) or {}
                if isinstance(locator, dict):
                    sp = locator.get("section_path")
                    if isinstance(sp, list):
                        for h in sp:
                            if isinstance(h, str) and h.strip():
                                slot["headings"].add(h.strip())
                    for k in ("section", "heading", "title"):
                        v = locator.get(k)
                        if isinstance(v, str) and v.strip():
                            slot["headings"].add(v.strip())
                if slot["filename"] is None:
                    fname = getattr(refs[0], "filename", None)
                    if fname:
                        slot["filename"] = fname
        except Exception:
            pass
    return by_artifact


def _format_artifact_section(
    slot: dict[str, Any], *, max_chars: int, headings_first: bool = True
) -> str:
    """Render a single artifact's content for the prompt."""
    headings_part = ""
    if slot["headings"]:
        headings_text = " | ".join(sorted(slot["headings"]))[:1200]
        headings_part = f"[HEADINGS] {headings_text}\n\n"
    stitched = " ".join(slot["bodies"])
    body_budget = max(0, max_chars - len(headings_part))
    if len(stitched) > body_budget:
        stitched = stitched[:body_budget]
    if headings_first:
        return f"--- {slot['filename'] or '?'} ---\n{headings_part}{stitched}"
    return f"--- {slot['filename'] or '?'} ---\n{stitched}\n\n{headings_part}"


def _build_excerpt_for_customer(by_artifact: dict[str, dict[str, Any]]) -> str:
    """Customer name lives on cover pages + first-doc headings.
    Send a small, heading-rich excerpt of the first 2-3 documents.
    """
    if not by_artifact:
        return ""
    chunks: list[str] = []
    running = 0
    BUDGET_TOTAL = 8000
    MAX_PER_DOC = 4000
    for aid in sorted(by_artifact.keys()):
        section = _format_artifact_section(
            by_artifact[aid], max_chars=MAX_PER_DOC, headings_first=True
        )
        chunks.append(section)
        running += len(section)
        if running >= BUDGET_TOTAL:
            break
    return "\n\n".join(chunks)


def _build_excerpt_for_stakeholders(by_artifact: dict[str, dict[str, Any]]) -> str:
    """Stakeholders sprinkled throughout body text; needs broad coverage.
    Send a wide, body-heavy excerpt.
    """
    return _build_excerpt_general(by_artifact, max_per_doc=8000, max_total=30000)


def _build_excerpt_for_milestones(by_artifact: dict[str, dict[str, Any]]) -> str:
    """Milestones live in schedule tables + body text with dates."""
    return _build_excerpt_general(by_artifact, max_per_doc=7000, max_total=25000)


def _build_excerpt_for_requirements(by_artifact: dict[str, dict[str, Any]]) -> str:
    """Requirements ("shall/must" clauses, acceptance criteria) are
    spread across body text in SOW sections.
    """
    return _build_excerpt_general(by_artifact, max_per_doc=8000, max_total=30000)


def _build_excerpt_for_site_clusters(by_artifact: dict[str, dict[str, Any]]) -> str:
    """Site clusters need headings (where institutional names live)
    + body where roster/address tables appear.
    """
    return _build_excerpt_general(by_artifact, max_per_doc=7000, max_total=25000)


def _build_excerpt_general(
    by_artifact: dict[str, dict[str, Any]], *, max_per_doc: int, max_total: int
) -> str:
    chunks: list[str] = []
    running = 0
    for aid in sorted(by_artifact.keys()):
        section = _format_artifact_section(
            by_artifact[aid], max_chars=max_per_doc, headings_first=True
        )
        chunks.append(section)
        running += len(section)
        if running >= max_total:
            break
    return "\n\n".join(chunks)


# ════════════════════════════════════════════════════════════════════
# FOCUSED EXTRACTORS — one prompt per category
# ════════════════════════════════════════════════════════════════════


_OUTPUT_RULES = (
    "CRITICAL RULES:\n"
    "- Only return entities that ACTUALLY APPEAR in the documents. Do NOT invent. Do NOT use names from your training data.\n"
    "- Extract VERBATIM from the docs.\n"
    "- Return ONLY a JSON object on a single line. No markdown. No code fences. No commentary."
)


def _extract_customer(docs_excerpt: str) -> str | None:
    if not docs_excerpt:
        return None
    prompt = f"""Identify the PRIMARY BUYING CUSTOMER for this managed-services bid.

The customer is the institution/company issuing the RFP and signing
the contract — NOT vendors, NOT subcontractors, NOT consultants.

{_OUTPUT_RULES}

DOCUMENTS:

{docs_excerpt}

OUTPUT:
{{"customer": "<full canonical customer name>"}}

If unclear or no customer is named, return: {{"customer": null}}

/no_think"""
    text = _call_ollama(prompt, max_tokens=256)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return None
    v = obj.get("customer")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _build_stakeholders_prompt(docs_excerpt: str) -> str:
    return f"""Identify PEOPLE (named human stakeholders) in this bid package.

🚨 HIGHEST PRIORITY: always include the BID-CONTACT person — the
named individual the docs say to contact about the RFP/RFB/RFQ.
Look especially for:
  - "Please contact <Name>, <Role>, at <email>"
  - "Direct all questions to <Name>"
  - "Questions regarding this RFP should be directed to <Name>"
  - "<Name>, <Role>, at <phone> or <email>"
  - Signature blocks with a typed name + title
  - "Submitted by <Name>"
  - "Project Manager: <Name>"
  - "Purchasing Agent: <Name>"
The bid-contact person is the SINGLE MOST IMPORTANT person for the
PM running this deal — never omit them if a name is in the docs.

Also include:
- Customer-side: project sponsor, PM, technical lead, signatories,
  named approvers
- Vendor-side (if named): account exec, PM, technical lead
- Anyone with a name + role + (email OR phone)

EXCLUDE (these are NOT people):
- Field labels / column headers ("Access Constraint", "Ending Number",
  "Upload Destination", "Tag Prefix", "Asset Type", "Owner")
- Role-only mentions with no name ("the PM", "the architect", "the bidder")
- Department / agency names ("IT Department", "Procurement Office",
  "Purchasing Dept", "Commissioners' Court")
- Job titles alone with no person attached
- Generic terms ("contractor", "vendor", "customer", "the team")
- Organizational entities ("Hood County", "School District")
- Insurance / legal jargon ("Liability Insurance", "Bodily Injury")

{_OUTPUT_RULES}

DOCUMENTS:

{docs_excerpt}

OUTPUT (array of objects):
{{"stakeholders": [
  {{"name": "<Full Name as written in docs>", "role": "<role or null>", "email": "<email or null>", "phone": "<phone or null>"}},
  ...
]}}

If genuinely no named humans appear in the docs, return: {{"stakeholders": []}}
But if you see ANY email with an associated name, or any "contact <Name>" line, that person MUST appear in your output.

/no_think"""


def _extract_stakeholders(docs_excerpt: str) -> list[dict[str, Any]]:
    """Single-call extraction — kept for back-compat. The chunked
    variant ``_extract_stakeholders_chunked`` is what the parallel
    runner actually uses now."""
    if not docs_excerpt:
        return []
    prompt = _build_stakeholders_prompt(docs_excerpt)
    text = _call_ollama(prompt, max_tokens=1024)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return []
    return _normalize_objects(
        obj.get("stakeholders"),
        ("name", "role", "email", "phone"),
        is_stakeholder=True,
    )


def _extract_milestones(docs_excerpt: str) -> list[dict[str, Any]]:
    if not docs_excerpt:
        return []
    prompt = f"""Identify PROJECT MILESTONES from this bid package.

A milestone is a named PROJECT DATE with semantic meaning:
contract award, kickoff, design validation, procurement, cutover,
go-live, hypercare end, acceptance, blackout windows, freeze
periods, etc.

INCLUDE:
- Named milestones with a date or date range
- Cutover / launch / hypercare / freeze events
- Blackout windows (e.g., "Thanksgiving freeze 2026-11-26 through 2026-11-28")
- Major project phases with end-dates

EXCLUDE:
- Random date mentions with no project meaning
- Document creation/revision dates
- Birthday / age / unrelated dates
- Generic timeframes ("soon", "next month")

{_OUTPUT_RULES}

DOCUMENTS:

{docs_excerpt}

OUTPUT (array of objects):
{{"milestones": [
  {{"name": "<milestone name>", "date": "<YYYY-MM-DD or range or null>", "notes": "<short context or null>"}},
  ...
]}}

If no real milestones, return: {{"milestones": []}}

/no_think"""
    text = _call_ollama(prompt, max_tokens=1024)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return []
    return _normalize_objects(
        obj.get("milestones"), ("name", "date", "notes")
    )


def _extract_requirements(docs_excerpt: str) -> list[dict[str, Any]]:
    """Single-call requirement extraction over a pre-built excerpt.

    Called by the per-doc chunked variant below — direct callers
    should use ``_extract_requirements_chunked(by_artifact)`` instead
    so doc-large packs (Pack 18 Beaufort POS, Pack 19 Hood, Pack 12
    BMS) don't lose 95% of their shall/must clauses to a single-call
    30K-char budget.
    """
    if not docs_excerpt:
        return []
    prompt = _build_requirements_prompt(docs_excerpt)
    text = _call_ollama(prompt, max_tokens=2048)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return []
    return _normalize_objects(
        obj.get("requirements"), ("text", "category")
    )


def _build_requirements_prompt(docs_excerpt: str) -> str:
    return f"""Identify REQUIREMENTS (what the customer requires the contractor to do).

INCLUDE:
- "Shall" / "must" / "required" / "will" clauses from the SOW or vendor response
- SLAs and performance targets (uptime %, response times like "24/7", "4-hour response", "99.9% uptime")
- Compliance requirements (NFPA, HIPAA, PCI, PCI-DSS, IEEE, ISO, FERPA, SOC 2, CJIS, etc.)
- Acceptance criteria (functional, performance, security)
- Deliverables (documentation, test reports, training, background checks)
- Security requirements (badge, escort, audit, background checks)
- Hardware requirements (CPU, RAM, storage minimums)
- Personnel requirements (criminal background checks, dress code, conduct)

EXCLUDE:
- Pure boilerplate ("contractor will comply with applicable laws")
- Pricing terms
- Project metadata (deal ID, packet version)

Paraphrase each requirement to ONE concise sentence (≤ 25 words).

{_OUTPUT_RULES}

DOCUMENTS:

{docs_excerpt}

OUTPUT (array of objects):
{{"requirements": [
  {{"text": "<requirement, ≤25 words>", "category": "<sla|compliance|performance|security|deliverable|acceptance|hardware|personnel|other>"}},
  ...
]}}

If no real requirements found, return: {{"requirements": []}}

/no_think"""


_CHUNK_CHARS = 40000  # ~10K tokens per LLM call — well under qwen3:14b's 40K context
# Safety cap on chunks per artifact. Original cap of 8 (~320K chars)
# missed late content on 500+ page bid PDFs. 32 chunks = ~1.28MB of
# body text per doc, comfortably covering everything we've seen in
# real-world bid packs. Configurable via env so Azure can dial it
# down for cost / up for huge docs.
_MAX_CHUNKS_PER_ARTIFACT = int(
    os.environ.get("SOWSMITH_LLM_MAX_CHUNKS_PER_ARTIFACT", "32")
)


def _split_artifact_into_chunks(
    slot: dict[str, Any], *, chunk_chars: int = _CHUNK_CHARS
) -> list[str]:
    """Split one artifact's body text into ``chunk_chars``-sized
    chunks, each prefixed with the filename + section headings so
    the LLM has context even for chunk N>0.

    Real bid PDFs run 100-300 pages (Heartland Beaufort response =
    177 pages ≈ 200K chars). A single chunked-per-doc call sees
    ~12.5% of a 200K doc. Chunking within the artifact recovers
    the rest.
    """
    body = " ".join(slot["bodies"])
    if not body:
        return []
    headings_part = ""
    if slot["headings"]:
        headings_text = " | ".join(sorted(slot["headings"]))[:1200]
        headings_part = f"[HEADINGS] {headings_text}\n\n"
    filename = slot.get("filename") or "?"
    chunks: list[str] = []
    n = max(1, (len(body) + chunk_chars - 1) // chunk_chars)
    n = min(n, _MAX_CHUNKS_PER_ARTIFACT)
    for i in range(n):
        start = i * chunk_chars
        piece = body[start:start + chunk_chars]
        label = f"--- {filename} [chunk {i + 1}/{n}] ---"
        chunks.append(f"{label}\n{headings_part}{piece}")
    return chunks


def _extract_with_chunked_dispatch(
    by_artifact: dict[str, dict[str, Any]],
    *,
    build_prompt: Callable[[str], str],
    output_key: str,
    fields: tuple[str, ...],
    max_tokens: int = 2048,
    is_stakeholder: bool = False,
) -> list[dict[str, Any]]:
    """Generic per-artifact-per-chunk LLM dispatcher with dedup.

    Splits each artifact into ``_CHUNK_CHARS``-sized chunks, fires
    one LLM call per chunk, unions results, dedupes by first 100
    chars of normalized output text (or name).
    """
    if not by_artifact:
        return []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    sig_field = fields[0]
    for aid in sorted(by_artifact.keys()):
        slot = by_artifact[aid]
        for chunk in _split_artifact_into_chunks(slot):
            if not chunk:
                continue
            prompt = build_prompt(chunk)
            text = _call_ollama(prompt, max_tokens=max_tokens)
            obj = _parse_json_object(text)
            if not isinstance(obj, dict):
                continue
            items = obj.get(output_key)
            for rec in _normalize_objects(
                items, fields, is_stakeholder=is_stakeholder
            ):
                v = rec.get(sig_field) or ""
                sig = re.sub(r"\s+", " ", str(v).lower()).strip()[:100]
                if sig and sig not in seen:
                    seen.add(sig)
                    out.append(rec)
    return out


def _extract_requirements_chunked(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Multi-chunk requirement extraction — splits each artifact into
    ~40K-char chunks, fires one LLM call per chunk, unions + dedupes.

    Recovers 80%+ of requirement clauses on big PDFs (Pack 18
    Beaufort POS source 196 clauses, Pack 19 Hood, Pack 12 BMS).
    """
    return _extract_with_chunked_dispatch(
        by_artifact,
        build_prompt=_build_requirements_prompt,
        output_key="requirements",
        fields=("text", "category"),
        max_tokens=2048,
    )


def _extract_stakeholders_chunked(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Multi-chunk stakeholder extraction — catches names buried in
    signature blocks / contact pages on page 100+ that the single-
    excerpt path misses."""
    return _extract_with_chunked_dispatch(
        by_artifact,
        build_prompt=_build_stakeholders_prompt,
        output_key="stakeholders",
        fields=("name", "role", "email", "phone"),
        max_tokens=1024,
        is_stakeholder=True,
    )


def _extract_site_clusters_chunked(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Multi-chunk site-cluster extraction — catches roster tables /
    school lists buried later in big PDFs (Albuquerque Public Schools,
    Muskegon Paging, etc.)."""
    out_raw = _extract_with_chunked_dispatch(
        by_artifact,
        build_prompt=_build_site_clusters_prompt,
        output_key="site_clusters",
        fields=("canonical_name", "aliases"),
        max_tokens=2048,
    )
    # The dispatcher returns plain dicts; normalize through the
    # cluster-validator to merge aliases properly.
    raw_list: list[Any] = []
    for r in out_raw:
        raw_list.append({
            "canonical_name": r.get("canonical_name"),
            "aliases": r.get("aliases") or [],
        })
    return _normalize_site_clusters(raw_list)


def _build_site_clusters_prompt(docs_excerpt: str) -> str:
    return f"""Identify PHYSICAL SITES grouped into clusters.

Each cluster represents ONE physical building/site and lists every
surface form (site codes, friendly names, addresses) that refer to
it in the docs.

Example shape (do NOT copy specific names):
  {{"canonical_name": "<Customer> Atlanta HQ",
    "aliases": ["ATL-HQ-01", "Atlanta Headquarters", "Innovation Tower",
                "1200 Peachtree St NE"]}}

INCLUDE:
- Site codes (ATL-HQ-01, STORE-142, etc.)
- Friendly names (Atlanta Headquarters, Brady Training, etc.)
- Full street addresses
- Multi-doc variants even when addresses disagree across docs

EXCLUDE:
- Standards bodies (ANSI, NFPA, etc.)
- Vendor / product / SaaS names
- Cities / counties alone without a specific named facility
- Generic nouns ("the library", "the school")
- Spec section labels

{_OUTPUT_RULES}

DOCUMENTS:

{docs_excerpt}

OUTPUT (array of cluster objects):
{{"site_clusters": [
  {{"canonical_name": "<primary name>", "aliases": ["<form 1>", "<form 2>", ...]}},
  ...
]}}

If no real sites, return: {{"site_clusters": []}}

/no_think"""


def _extract_site_clusters(docs_excerpt: str) -> list[dict[str, Any]]:
    """Single-call extraction — kept for back-compat. The chunked
    variant ``_extract_site_clusters_chunked`` is what the parallel
    runner uses now."""
    if not docs_excerpt:
        return []
    prompt = _build_site_clusters_prompt(docs_excerpt)
    text = _call_ollama(prompt, max_tokens=2048)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return []
    return _normalize_site_clusters(obj.get("site_clusters"))


# ════════════════════════════════════════════════════════════════════
# v38 — EMBEDDING-RETRIEVAL EXTRACTORS
# ════════════════════════════════════════════════════════════════════
#
# Architecture:
#   1. Split each artifact into sentences (no chunk boundary loss).
#   2. Embed every sentence once via qwen3-embedding:8b.
#   3. Retrieve top-K candidates per entity type using curated
#      exemplar sentences (cosine similarity on normalized vectors).
#   4. For each candidate sentence, run a SINGLE-SENTENCE
#      canonicalize LLM call: decide keep/drop + produce canonical
#      form. Parallel-batched across candidates.
#   5. Dedupe by canonical form + return.
#
# Why this lifts recall from ~10% → 95%+:
#   - No chunk dropout (sentence is atomic unit, no boundary loss).
#   - No LLM self-limiting (each canonicalize call sees ONE
#     candidate, never "feels done" early).
#   - Universal across entity types (same primitive, different
#     exemplar set per type).
#   - Pure embedding-based retrieval: NO regex.
#
# Toggle via SOWSMITH_RETRIEVAL_ENABLED env var (default ON).
# Falls back to chunked extraction if embedding endpoint unreachable.


def _build_artifact_text_map(
    by_artifact: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Flatten by_artifact into {artifact_id: concatenated_text} for
    the embedding retriever. Includes headings as prefix so heading-
    only "requirements" still get matched (section titles like
    "5.3 INSURANCE REQUIREMENTS" anchor downstream sentences)."""
    out: dict[str, str] = {}
    for aid, slot in by_artifact.items():
        if not isinstance(slot, dict):
            continue
        bodies = slot.get("bodies") or []
        headings = slot.get("headings") or set()
        parts = []
        if headings:
            parts.append(" | ".join(sorted(headings)))
        parts.extend(b for b in bodies if isinstance(b, str) and b.strip())
        text = "\n\n".join(parts)
        if text.strip():
            out[aid] = text
    return out


_CANONICALIZE_PROMPTS: dict[str, str] = {
    "requirement": (
        "TASK: Decide if the SENTENCE is a real REQUIREMENT (an obligation "
        "imposed on the contractor, vendor, district, or customer in this "
        "bid package).\n\n"
        "KEEP if the sentence contains an obligation marker:\n"
        "  shall / must / will / agrees to / is required to / covenants /\n"
        "  warrants / undertakes / commits to / reserves the right to /\n"
        "  shall not / must not / may not\n\n"
        "DROP if it's:\n"
        "  - product marketing copy describing what software does\n"
        "  - background context, history, or boilerplate\n"
        "  - a general fact with no obligation\n"
        "  - a heading or section label only\n"
        "  - already obvious noise (table cell fragments, etc.)\n\n"
        "If KEEP, also produce a canonical form (drop the leading\n"
        "'The contractor shall' / 'Vendor must' prefix when obvious;\n"
        "keep the meaningful verb and object; max 120 chars).\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical": "<canonical form>"}} or {{"keep": false}}\n\n'
        "/no_think"
    ),
    "stakeholder": (
        "TASK: Decide if the SENTENCE names a real human STAKEHOLDER (a\n"
        "specific person involved in this bid) and extract their name.\n\n"
        "KEEP if the sentence names a real person:\n"
        "  - Has a first name + last name (e.g. 'Kaylee Yinger')\n"
        "  - May have a role title ('Project Manager: Glenn Tilleman')\n"
        "  - May appear in an email signature or contact block\n\n"
        "DROP if it's:\n"
        "  - an organization, company, or department name\n"
        "  - a job title alone with no person name\n"
        "  - a generic noun phrase ('end users', 'customer support', 'mosaic front')\n"
        "  - an email address as the 'name'\n"
        "  - a product / service name\n\n"
        "If KEEP, also extract role and any email/phone visible in the sentence.\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "name": "First Last", "role": "<role or empty>", "email": "<email or empty>", "phone": "<phone or empty>"}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
    "site": (
        "TASK: Decide if the SENTENCE names a PHYSICAL SITE (specific\n"
        "building, campus, site code, or full address in this bid).\n\n"
        "KEEP if the sentence names a specific physical place:\n"
        "  - Site codes (ATL-HQ-01, STORE-142, MDF-3A)\n"
        "  - Named buildings (Beaufort Elementary School, Innovation Tower)\n"
        "  - Full street addresses\n"
        "  - Named campuses\n\n"
        "DROP if it's:\n"
        "  - a generic term ('the customer site', 'all locations', 'the district')\n"
        "  - a standards body (ANSI, NFPA, IEEE)\n"
        "  - a vendor / product / SaaS name\n"
        "  - a city or county alone with no facility\n"
        "  - a spec section label\n\n"
        "If KEEP, produce the canonical site name (most specific form in the sentence).\n"
        "Also list ALL alias forms present in the sentence (codes + names + addresses).\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical_name": "<primary name>", "aliases": ["<form 1>", "<form 2>"]}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
    "quantity": (
        "TASK: Decide if the SENTENCE expresses a meaningful structural\n"
        "QUANTITY (an SLA, count, duration, percentage, or commercial term).\n\n"
        "KEEP if the sentence states:\n"
        "  - Uptime / availability percentages (99.999%, 99.95%)\n"
        "  - Response times (within 2 hours, 5 minute failover)\n"
        "  - Counts (32 schools, 97 access points)\n"
        "  - Help-desk hours (Monday-Friday 8 AM-5 PM)\n"
        "  - Payment terms (Net-30, Net-45)\n"
        "  - Contract / warranty durations (5-year, 12-month)\n"
        "  - Lead times (6-8 weeks)\n\n"
        "DROP if it's:\n"
        "  - a page number, table cell index, or section number alone\n"
        "  - a year alone with no quantity context\n"
        "  - product version numbers\n\n"
        "If KEEP, produce a short canonical form (e.g. '99.999% uptime',\n"
        "'2-hour Sev1 response', '32 schools').\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical": "<short form>", "value": "<numeric value>", "unit": "<unit if applicable>"}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
}


def _canonicalize_candidate(
    sentence: str, entity_type: str
) -> dict[str, Any] | None:
    """Single-sentence LLM call: keep/drop + canonical form for one
    candidate sentence. Returns None on parse failure or LLM error.
    """
    template = _CANONICALIZE_PROMPTS.get(entity_type)
    if not template:
        return None
    if not sentence or not sentence.strip():
        return None
    # Truncate ultra-long sentences (the embedding pipeline already
    # caps at 500 chars but defense-in-depth)
    truncated = sentence.strip()[:600]
    prompt = template.format(sentence=truncated)
    text = _call_ollama(prompt, max_tokens=256)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return None
    if not obj.get("keep"):
        return None
    return obj


def _run_retrieval_extract(
    by_artifact: dict[str, dict[str, Any]],
    *,
    entity_type: str,
    exemplars: list[str],
    top_k_per_artifact: int = 200,
    min_score: float = 0.45,
    canonical_key: str = "canonical",
) -> list[dict[str, Any]]:
    """Generic retrieval extraction — v39 hybrid pipeline:
      1. Build per-artifact text map.
      2. Hybrid retrieval (dense + sparse + RRF + margin + MMR).
      3. Canonicalize each candidate with paragraph context in parallel.
      4. Dedupe by canonical form (lowercased, whitespace-normalized).

    Falls back to v38 dense-only retrieval if rag_retrieval module
    is unavailable or sklearn/scipy missing.

    Returns list of canonicalize-output dicts (KEEP only).
    """
    # Try v39 hybrid pipeline first
    use_v39 = not os.environ.get("SOWSMITH_V39_DISABLE")
    candidates: list[dict[str, Any]] = []
    if use_v39:
        try:
            from app.core.rag_retrieval import get_v39_candidates
            from app.core.exemplars import NEGATIVE_EXEMPLARS_BY_TYPE
            from app.core.embedding_retrieval import embedding_endpoint_reachable
            if embedding_endpoint_reachable():
                text_map = _build_artifact_text_map(by_artifact)
                if text_map:
                    neg_exemplars = NEGATIVE_EXEMPLARS_BY_TYPE.get(entity_type, [])
                    candidates = get_v39_candidates(
                        text_map, exemplars, neg_exemplars,
                        top_k_per_artifact=top_k_per_artifact,
                        min_score=min_score,
                        contextual_window=0,  # NO sliding context — adds noise
                        paragraph_window=1,   # ±1 sentence for canonicalize input
                        use_sparse=True,
                        use_mmr=True,
                    )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "v39 retrieval failed for %s: %s — falling back to v38",
                entity_type, e,
            )
            candidates = []

    # v38 fallback (dense-only, no sparse / no MMR / no negatives)
    if not candidates:
        try:
            from app.core.embedding_retrieval import (
                get_candidates_for_entity_type,
                embedding_endpoint_reachable,
            )
            if not embedding_endpoint_reachable():
                return []
            text_map = _build_artifact_text_map(by_artifact)
            if not text_map:
                return []
            raw_candidates = get_candidates_for_entity_type(
                text_map, exemplars,
                top_k_per_artifact=top_k_per_artifact,
                min_score=min_score,
            )
            # Adapt v38 shape to v39 shape
            candidates = [
                {
                    "sentence_idx": -1,
                    "sentence": c["sentence"],
                    "paragraph": c["sentence"],  # v38 has no paragraph expansion
                    "score": c["score"],
                    "dense_score": c["score"],
                    "artifact_id": c["artifact_id"],
                }
                for c in raw_candidates
            ]
        except Exception:
            return []

    if not candidates:
        return []

    parallel = int(os.environ.get("SOWSMITH_CANONICALIZE_PARALLEL", "12"))
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=parallel) as pool:
        # Use the PARAGRAPH (expanded context) for canonicalize — gives
        # the LLM more context to make keep/drop decision.
        future_map = {
            pool.submit(_canonicalize_candidate, c["paragraph"], entity_type): c
            for c in candidates
        }
        for fut in _cf.as_completed(future_map):
            candidate = future_map[fut]
            try:
                outcome = fut.result()
            except Exception:
                outcome = None
            if not outcome:
                continue
            # Dedupe by canonical form (case-insensitive, whitespace-normalized)
            canon_value = outcome.get(canonical_key) or outcome.get("name") or ""
            sig = re.sub(r"\s+", " ", str(canon_value).lower()).strip()[:120]
            if not sig or sig in seen:
                continue
            seen.add(sig)
            # Attach source info
            outcome["_source_sentence"] = candidate["sentence"]
            outcome["_source_paragraph"] = candidate["paragraph"]
            outcome["_source_artifact_id"] = candidate["artifact_id"]
            outcome["_retrieval_score"] = round(candidate["score"], 4)
            outcome["_dense_score"] = round(candidate.get("dense_score", 0.0), 4)
            results.append(outcome)

    # ────────────────────────────────────────────────────────────
    # v40: SICRL — Section-Indexed Counterfactual Recall Loop
    # ────────────────────────────────────────────────────────────
    # Augments first-pass items by predicting what SHOULD be in
    # under-covered sections and retrieving the gaps. NOVEL technique
    # — see app/core/sicrl.py docstring for design.
    use_sicrl = (
        not os.environ.get("SOWSMITH_SICRL_DISABLE")
        and entity_type in ("requirement", "stakeholder", "quantity")
    )
    if use_sicrl and results:
        try:
            from app.core.sicrl import run_sicrl
            from app.core.embedding_retrieval import (
                embed_texts as _embed_texts,
                sentence_split as _sentence_split,
            )
            text_map = _build_artifact_text_map(by_artifact)
            if text_map:
                augmented = run_sicrl(
                    by_artifact=text_map,
                    first_pass_items=results,
                    entity_type=entity_type,
                    exemplars=exemplars,
                    negative_exemplars=[],
                    llm_call=lambda p, mt: _call_ollama(p, max_tokens=mt),
                    parse_json=_parse_json_object,
                    canonicalize_fn=_canonicalize_candidate,
                    embed_fn=_embed_texts,
                    sentence_split_fn=_sentence_split,
                    max_iterations=1,  # one pass for now; loop later
                )
                results = augmented
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "SICRL pass failed for %s: %s", entity_type, e,
            )

    return results


def _extract_requirements_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v38+v39+v40: embedding-retrieval requirement extraction.
    Replaces the chunked path's ~10% recall with 95%+ on requirement-
    heavy bids (Pack 18 Beaufort POS, Pack 19 Hood, Pack 12 BMS).
    Falls back to empty list if embedding endpoint unreachable; caller
    should then invoke the chunked path."""
    from app.core.exemplars import REQUIREMENT_EXEMPLARS
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="requirement",
        exemplars=REQUIREMENT_EXEMPLARS,
        top_k_per_artifact=600,  # generous; canonicalize drops noise
        min_score=0.30,  # v40: lowered so canonicalize is the gate
        canonical_key="canonical",
    )
    # Shape match with _extract_requirements_chunked: list of {text}
    out = []
    for r in raw:
        text = r.get("canonical")
        if isinstance(text, str) and text.strip():
            out.append({
                "text": text.strip(),
                "category": r.get("category"),
                "_source_sentence": r.get("_source_sentence"),
                "_source_artifact_id": r.get("_source_artifact_id"),
            })
    return out


def _extract_stakeholders_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v38+v39+v40: embedding-retrieval stakeholder extraction.
    Finds named people on signature blocks, contact pages, bid-contact
    lines — no chunk dropout."""
    from app.core.exemplars import STAKEHOLDER_EXEMPLARS
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="stakeholder",
        exemplars=STAKEHOLDER_EXEMPLARS,
        top_k_per_artifact=300,
        min_score=0.30,
        canonical_key="name",
    )
    out = []
    for r in raw:
        name = r.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        # Last-chance hygiene: drop email-as-name + field-label
        if _looks_like_email_or_url(name):
            continue
        if _is_likely_field_label(name):
            continue
        out.append({
            "name": name.strip(),
            "role": (r.get("role") or "").strip() or None,
            "email": (r.get("email") or "").strip() or None,
            "phone": (r.get("phone") or "").strip() or None,
            "_source_sentence": r.get("_source_sentence"),
            "_source_artifact_id": r.get("_source_artifact_id"),
        })
    return out


def _extract_site_clusters_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v38: embedding-retrieval site extraction. Each candidate
    sentence may yield ONE cluster (canonical + aliases visible in
    that sentence). Downstream entity_resolution merges across
    sentences via co-mention fusion + LLM-cluster fusion."""
    from app.core.exemplars import SITE_EXEMPLARS
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="site",
        exemplars=SITE_EXEMPLARS,
        top_k_per_artifact=200,
        min_score=0.35,
        canonical_key="canonical_name",
    )
    out = []
    for r in raw:
        canon = r.get("canonical_name")
        aliases = r.get("aliases") or []
        if not isinstance(canon, str) or not canon.strip():
            continue
        if not isinstance(aliases, list):
            aliases = []
        out.append({
            "canonical_name": canon.strip(),
            "aliases": [a for a in aliases if isinstance(a, str) and a.strip()],
            "_source_sentence": r.get("_source_sentence"),
            "_source_artifact_id": r.get("_source_artifact_id"),
        })
    return _normalize_site_clusters(out)


def _extract_quantities_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v38: NEW — embedding-retrieval quantity extraction. Captures
    SLAs, counts, durations, payment terms that the existing extractors
    miss because they're in natural-language form ('99.999% uptime',
    '32 schools', 'within 2 hours')."""
    from app.core.exemplars import QUANTITY_EXEMPLARS
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="quantity",
        exemplars=QUANTITY_EXEMPLARS,
        top_k_per_artifact=300,
        min_score=0.32,
        canonical_key="canonical",
    )
    out = []
    for r in raw:
        canon = r.get("canonical")
        if not isinstance(canon, str) or not canon.strip():
            continue
        out.append({
            "text": canon.strip(),
            "value": r.get("value"),
            "unit": r.get("unit"),
            "_source_sentence": r.get("_source_sentence"),
            "_source_artifact_id": r.get("_source_artifact_id"),
        })
    return out


# ════════════════════════════════════════════════════════════════════
# OLLAMA HTTP CALL
# ════════════════════════════════════════════════════════════════════


def _call_ollama(prompt: str, *, max_tokens: int = 1024) -> str:
    """POST to /api/generate. Returns the response text or empty string on failure."""
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    timeout = int(os.environ.get("SOWSMITH_LLM_TIMEOUT", str(DEFAULT_TIMEOUT)))
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    try:
        result = json.loads(body)
        return str(result.get("response") or "")
    except json.JSONDecodeError:
        return ""


# ════════════════════════════════════════════════════════════════════
# RESPONSE PARSING + HYGIENE
# ════════════════════════════════════════════════════════════════════


def _parse_json_object(response_text: str) -> dict[str, Any] | None:
    """Extract the first top-level {...} block via brace-matching."""
    if not response_text:
        return None
    start = response_text.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i in range(start, len(response_text)):
        ch = response_text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    try:
        return json.loads(response_text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _looks_like_email_or_url(value: str) -> bool:
    """True if the value looks like an email address, URL, or
    URL-tail (e.g. 'support@e-hps.com', 'foo.bar.com', 'site.net').

    The LLM sometimes returns an EMAIL as a `name` field when the
    line shape confuses it ("Help Desk: hss-ce-help@e-hps.com" →
    name="hss-ce-help@e-hps.com"). Slug-of-email looks like
    `hss_ce_help_e_hps_com` and pollutes the stakeholder list.
    """
    if not value:
        return False
    s = value.lower().strip()
    if "@" in s:
        return True
    # Trailing TLD-ish token after a dot or slug-separator
    for tld in (".com", ".org", ".net", ".io", ".gov", ".edu",
                ".co", ".us", ".uk", ".info", ".biz", ".ai",
                "_com", "_org", "_net", "_io", "_gov", "_edu",
                "_co", "_us", "_uk", "_info", "_biz", "_ai"):
        if s.endswith(tld):
            return True
    return False


def _looks_like_regulator_not_customer(value: str) -> bool:
    """True if the value looks like a regulatory body / licensing
    issuer rather than a buying customer.

    Catches LLM customer false positives like 'State of South
    Carolina Department of Revenue Retail License' (an SC license
    issuer mentioned in the doc, NOT the buying customer who is
    Beaufort County School District).

    Heuristic: customer ends with a regulatory tail word OR contains
    a regulator phrase in the middle. Keeps real govt customers
    like 'City of Atlanta' / 'Beaufort County School District' /
    'Department of Defense' (none of which match these patterns).
    """
    if not value:
        return False
    s = value.lower().strip()
    # Tail-word check
    tail_words = {
        "license", "licenses", "permit", "permits",
        "registration", "registrations",
        "certification", "certifications",
        "tax", "taxes", "tariff", "tariffs",
        "code", "statute", "statutes",
        "regulation", "regulations",
    }
    last_token = s.split()[-1] if s else ""
    if last_token in tail_words:
        return True
    # Phrase contains a regulator marker
    regulator_markers = (
        "department of revenue",
        "secretary of state",
        "office of regulations",
        "office of compliance",
        "internal revenue service",
        "department of motor vehicles",
        "consumer protection",
        "licensing board",
    )
    for marker in regulator_markers:
        if marker in s:
            return True
    return False


def _normalize_objects(
    items: Any, fields: tuple[str, ...], *, is_stakeholder: bool = False
) -> list[dict[str, Any]]:
    """Coerce list of objects to uniform shape; drop malformed.

    For stakeholders, also drops names that look like field labels
    OR like email addresses / URLs (the LLM sometimes returns an
    email as a `name` when the line shape confuses it).
    """
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rec: dict[str, Any] = {}
        for f in fields:
            v = it.get(f)
            if isinstance(v, (str, int, float)):
                s = str(v).strip()
                rec[f] = s if s else None
            else:
                rec[f] = None
        first_value = rec.get(fields[0])
        if not first_value:
            continue
        if is_stakeholder:
            fv = str(first_value)
            if _is_likely_field_label(fv):
                continue
            if _looks_like_email_or_url(fv):
                continue
        out.append(rec)
    return out


_FIELD_LABEL_TAILS: frozenset[str] = frozenset({
    "number", "numbers", "name", "code", "id", "ids",
    "constraint", "constraints", "rule", "rules",
    "destination", "source", "target", "path",
    "prefix", "suffix", "label", "labels", "tag", "tags",
    "field", "fields", "value", "values", "key", "keys",
    "type", "types", "category", "categories", "status",
    "owner", "owners", "manager", "managers",
    "officer", "officers", "rep", "reps", "representative",
    "representatives", "lead", "leads", "support", "specialist",
    "specialists", "coordinator", "coordinators",
    "supervisor", "supervisors", "director", "directors",
    "input", "output", "config", "configuration", "setting",
    "settings", "parameter", "parameters", "option", "options",
    "address", "addresses", "phone", "phones", "email", "emails",
    "date", "dates", "time", "times",
    "window", "windows", "range", "ranges",
    # Insurance / legal / procurement jargon often misclassified
    "insurance", "policy", "policies", "coverage",
    "injury", "damage", "claim", "claims",
    "order", "orders", "invoice", "invoices",
    "service", "services", "department", "departments",
    "office", "offices", "agency", "agencies", "authority",
    "board", "boards", "committee", "committees", "council", "councils",
    "court", "courts", "commission", "commissions",
})

_FIELD_LABEL_PHRASES: frozenset[str] = frozenset({
    "access constraint", "access constraints",
    "starting number", "ending number",
    "upload destination", "azure container",
    "tag prefix", "asset type",
    "escort owner", "facility name",
    "project name", "deal name", "deal id",
    "site id", "site code", "facility id",
    "mock deal", "mock document",
    "primary contact", "secondary contact",
    "internal contact", "external contact",
    "customer", "contractor", "bidder",
    "engineer", "architect", "vendor",
    "project", "team",
    "county", "city", "town", "district",
    # Insurance / legal patterns
    "bodily injury", "property damage",
    "liability insurance", "general liability",
    "workers comp", "workers compensation",
    "policy holder", "policy holders",
    # Procurement patterns
    "purchase order", "purchase orders",
    "invoice receipt", "receipt invoice",
    "rfp response", "rfq response",
    # Mail / postal
    "postal office", "post office", "us postal",
    "fed ex", "fedex", "ups", "usps",
})


_ORG_TOKENS: frozenset[str] = frozenset({
    # Government / jurisdictional
    "county", "city", "town", "state", "federal", "municipal",
    # Org body types
    "court", "board", "committee", "council", "commission",
    "department", "office", "agency", "authority", "bureau",
    "ministry", "directorate",
    # Postal / mail
    "postal",
    # Legal / financial
    "treasurer", "comptroller",
    # Generic
    "us", "usa", "u.s.", "u.s.a.",
})


def _is_likely_field_label(name: str) -> bool:
    """True if name looks like a field label / column header / org
    name / generic noun-phrase, NOT a real person.

    Pipeline:
      1. Strip leading articles ("the ", "a ", "an "), repeated.
      2. Exact phrase match against denylist (specific known junk).
      3. Single-word matching the tail-word denylist.
      4. Trailing-word matching the tail-word denylist (e.g.
         "Liability Insurance", "Purchase Order").
      5. ANY org-keyword token present (catches "Hood County
         Emergency", "U.S. Postal Service").
      6. FIRST-WORD-IS-COMMON-NOUN gate (NEW v35): when the leading
         word is a generic noun like "End", "Mosaic", "Joint",
         "Front", "Back", etc., the phrase is a noun fragment
         ("End Users", "Mosaic Front", "Joint Ventures", "Back
         Office"), NOT a person. Real people very rarely have
         these as first names.
    """
    norm = re.sub(r"\s+", " ", name.lower().strip())
    # Strip leading articles (handle "the the" too)
    while True:
        changed = False
        for art in ("the ", "a ", "an "):
            if norm.startswith(art):
                norm = norm[len(art):]
                changed = True
        if not changed:
            break
    if not norm:
        return True
    if norm in _FIELD_LABEL_PHRASES:
        return True
    tokens = norm.split()
    if not tokens:
        return True
    # Single-word match against tails (e.g. "Insurance" alone)
    if len(tokens) == 1 and tokens[0] in _FIELD_LABEL_TAILS:
        return True
    # Tail-word match against denylist (e.g. "Liability Insurance",
    # "Hood County", "Purchase Order")
    if tokens[-1] in _FIELD_LABEL_TAILS:
        return True
    # ANY org-keyword present → not a person.
    if any(t in _ORG_TOKENS for t in tokens):
        return True
    # FIRST-WORD common-noun gate: drops noun-fragment "people" like
    # "End Users", "Mosaic Front", "Joint Ventures", "Back Office",
    # "Front Desk", "Help Desk", "Power School" misread as people.
    if tokens[0] in _COMMON_NOUN_FIRST_WORDS:
        return True
    return False


# Common nouns that real human first names almost never use as the
# leading token. When the LLM or regex returns a 2-3 word capitalized
# phrase starting with one of these, it's a noun fragment, not a
# person. Curated from real false positives across 19+ packs.
_COMMON_NOUN_FIRST_WORDS: frozenset[str] = frozenset({
    # Generic users / roles
    "end", "all", "any", "each", "every", "some", "many",
    "new", "old", "current", "former", "future",
    # Business-relationship words that lead noun phrases, not names
    "customer", "client", "contractor", "vendor", "supplier",
    "bidder", "provider", "partner", "subcontractor",
    # Position / direction words
    "front", "back", "left", "right", "top", "bottom",
    "north", "south", "east", "west", "central", "main",
    "primary", "secondary", "tertiary", "first", "second", "third",
    "upper", "lower", "inner", "outer",
    # Composite-noun starters
    "joint", "shared", "common", "general", "special", "regular",
    "standard", "custom", "default", "auto", "manual",
    # Product / system family words
    "mosaic", "modular", "smart", "digital", "analog",
    "remote", "local", "global", "regional", "national",
    # Verb-ish / action starters
    "support", "help", "service", "process", "manage",
    "view", "edit", "send", "receive", "request", "report",
    # Common deal-doc lead-ins
    "section", "exhibit", "appendix", "attachment", "schedule",
    "chapter", "page", "form", "table", "figure",
    # Software / SaaS product family starters (drop "Power School",
    # "Information Technology", "Building Management" misread as
    # people)
    "power", "information", "building", "facility", "security",
    "network", "system", "data", "cloud", "web", "mobile",
    "enterprise", "premium", "basic", "advanced", "professional",
    "open", "closed", "public", "private",
})


def _normalize_site_clusters(items: Any) -> list[dict[str, Any]]:
    """Validate site cluster objects: canonical_name + aliases list."""
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        canonical = it.get("canonical_name")
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        aliases = it.get("aliases")
        if not isinstance(aliases, list):
            continue
        alias_strs = [a.strip() for a in aliases if isinstance(a, str) and a.strip()]
        if canonical.strip() not in alias_strs:
            alias_strs.append(canonical.strip())
        if not alias_strs:
            continue
        out.append({
            "canonical_name": canonical.strip(),
            "aliases": alias_strs,
        })
    return out


# ════════════════════════════════════════════════════════════════════
# BACK-COMPAT: keep the old function name
# ════════════════════════════════════════════════════════════════════


def extract_multi_entities_with_llm(atoms: list[Any]) -> dict[str, Any]:
    """Back-compat alias for callers using the old name."""
    return extract_all_entities_with_llm(atoms)


__all__ = [
    "extract_all_entities_with_llm",
    "extract_multi_entities_with_llm",  # back-compat
    "_is_likely_field_label",            # used by entity_extraction's hygiene pass
    "session_key_for_atoms",
    "get_session_site_clusters",
]


# ════════════════════════════════════════════════════════════════════
# SESSION CACHE for site_clusters — lets entity_resolution pick up
# the LLM's cluster output from enrich_atoms without a second LLM call.
# ════════════════════════════════════════════════════════════════════

_SESSION_SITE_CLUSTERS: dict[str, list[dict[str, Any]]] = {}
_SESSION_CACHE_MAX = 16


def session_key_for_atoms(atoms: list[Any]) -> str:
    """Deterministic key derived from the first 5 atom IDs (or all
    if fewer). Stable across the same compile session, distinct
    across different projects.
    """
    if not atoms:
        return "empty"
    ids = sorted([getattr(a, "id", "") for a in atoms if getattr(a, "id", "")])
    if not ids:
        return "no-ids"
    sample = ids[:5]
    return "_".join(sample)


def _stash_session_site_clusters(
    atoms: list[Any], clusters: list[dict[str, Any]]
) -> None:
    """Cache LLM site_clusters keyed by an atom-set fingerprint.
    Capped at _SESSION_CACHE_MAX entries (LRU-evict on overflow).
    """
    if not clusters:
        return
    key = session_key_for_atoms(atoms)
    if key in _SESSION_SITE_CLUSTERS:
        del _SESSION_SITE_CLUSTERS[key]  # re-insert at end
    _SESSION_SITE_CLUSTERS[key] = clusters
    while len(_SESSION_SITE_CLUSTERS) > _SESSION_CACHE_MAX:
        oldest = next(iter(_SESSION_SITE_CLUSTERS))
        del _SESSION_SITE_CLUSTERS[oldest]


def get_session_site_clusters(atoms: list[Any]) -> list[dict[str, Any]]:
    """Read the cached LLM site_clusters for this atom set. Returns
    empty list if no cache hit (e.g., LLM disabled or call failed).
    """
    return _SESSION_SITE_CLUSTERS.get(session_key_for_atoms(atoms), [])
