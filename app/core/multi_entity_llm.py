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
DEFAULT_TIMEOUT = 360  # v44.4: bumped from 240s — ollama on Mac queues
DEFAULT_PARALLEL = 3   # v44.4: was 5 — Mac ollama 1-4 concurrent ceiling


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
            # v43 — 5 new entity-type extractors
            "certifications": lambda: _extract_certifications_retrieved(by_artifact),
            "risks": lambda: _extract_risks_retrieved(by_artifact),
            "acceptance_criteria": lambda: _extract_acceptance_retrieved(by_artifact),
            "penalties": lambda: _extract_penalties_retrieved(by_artifact),
            "compliance_obligations": lambda: _extract_compliance_obligations_retrieved(by_artifact),
            # v48 — 3 new extractors
            "lead_times": lambda: _extract_lead_times_retrieved(by_artifact),
            "electrical_acceptance": lambda: _extract_electrical_acceptance_retrieved(by_artifact),
            "payment_terms": lambda: _extract_payment_terms_retrieved(by_artifact),
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

    # ────────────────────────────────────────────────────────────
    # v42: CROSS-DOCUMENT CONTRADICTION DETECTION
    # ────────────────────────────────────────────────────────────
    # After all entity types extracted, scan for cross-doc pairs that
    # contradict each other (Net-30 vs Net-45, 99.5% vs 99.99% uptime,
    # different coverage limits, etc.). Auto-emits reconciliation_flag
    # records for PM review.
    if not os.environ.get("SOWSMITH_CONTRADICTION_DISABLE"):
        try:
            from app.core.rag_extras import detect_cross_doc_contradictions
            from app.core.embedding_retrieval import embed_texts as _embed_texts
            contradiction_flags: list[dict[str, Any]] = []
            for et_key, items in (
                ("requirements", "text"),
                ("quantities", "text"),
            ).__iter__() if False else [
                ("requirements", "text"),
                ("quantities", "text"),
            ]:
                items_list = results.get(et_key) or []
                if len(items_list) < 2:
                    continue
                # Embed canonical texts
                texts = [
                    (it.get("text") or it.get("canonical") or "")
                    for it in items_list
                ]
                texts = [t for t in texts if t]
                if len(texts) < 2:
                    continue
                vecs = _embed_texts(texts[:200])  # cap for speed
                if vecs.size == 0:
                    continue
                flags = detect_cross_doc_contradictions(
                    items_list[:len(texts[:200])],
                    vecs,
                    canonical_key="text" if et_key in ("requirements", "quantities") else "canonical",
                    llm_call=lambda p, mt: _call_ollama(p, max_tokens=mt),
                    parse_json=_parse_json_object,
                    sim_threshold_min=0.55,
                    sim_threshold_max=0.92,
                    max_pairs=30,
                )
                for f in flags:
                    f["entity_type"] = et_key
                    contradiction_flags.append(f)
            if contradiction_flags:
                results["contradiction_flags"] = contradiction_flags
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Cross-doc contradiction detection failed: %s", e,
            )

    # ────────────────────────────────────────────────────────────
    # v42+v43: MULTI-DOCUMENT GRAPH RAG with sparse-entity expansion.
    # Builds co-occurrence graph, then for under-populated entity
    # types (sites, milestones), expands via graph neighbors of the
    # well-populated anchors (customer, stakeholders).
    # ────────────────────────────────────────────────────────────
    if not os.environ.get("SOWSMITH_GRAPHRAG_DISABLE"):
        try:
            from app.core.rag_extras import (
                build_cooccurrence_graph,
                graph_expand_seeds,
            )
            graph = build_cooccurrence_graph(atoms)
            results["_cooccurrence_summary"] = {
                "node_count": len(graph),
                "edge_count": sum(len(v) for v in graph.values()) // 2,
            }

            # v43: Graph-expand sparse entity types
            # Find anchor keys (customer + stakeholder keys actually in graph)
            anchor_keys: set[str] = set()
            for k in graph:
                if k.startswith("customer:") or k.startswith("stakeholder:"):
                    anchor_keys.add(k)

            # If sites are sparse (≤3), expand via graph neighbors
            site_keys_extracted = {
                f"site:{re.sub(r'[^a-z0-9]+', '_', (c.get('canonical_name') or '').lower()).strip('_')}"
                for c in (results.get("site_clusters") or [])
                if isinstance(c, dict)
            }
            if anchor_keys and len(site_keys_extracted) <= 3:
                expanded_sites = graph_expand_seeds(
                    anchor_keys, graph,
                    target_prefix="site:",
                    max_expansion=20,
                )
                new_sites = expanded_sites - site_keys_extracted
                if new_sites:
                    # Add as additional clusters (single-alias each)
                    current = results.get("site_clusters") or []
                    for site_key in new_sites:
                        slug = site_key[len("site:"):]
                        name = slug.replace("_", " ").title()
                        current.append({
                            "canonical_name": name,
                            "aliases": [name],
                            "_via": "graph_expansion",
                        })
                    results["site_clusters"] = current

            # If milestones are sparse, same treatment
            ms_extracted = len(results.get("milestones") or [])
            if anchor_keys and ms_extracted <= 2:
                expanded_ms = graph_expand_seeds(
                    anchor_keys, graph,
                    target_prefix="milestone:",
                    max_expansion=15,
                )
                if expanded_ms:
                    current = results.get("milestones") or []
                    existing_slugs = {
                        re.sub(r"[^a-z0-9]+", "_", (m.get("name") or "").lower()).strip("_")
                        for m in current if isinstance(m, dict)
                    }
                    for ms_key in expanded_ms:
                        slug = ms_key[len("milestone:"):]
                        if slug not in existing_slugs:
                            current.append({
                                "name": slug.replace("_", " "),
                                "_via": "graph_expansion",
                            })
                    results["milestones"] = current
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────
    # v45: ZERO-MISS PASS — three stacked techniques on top of v44.5:
    #   1. PM-critical vocabulary sweep — 60+ hardcoded terms
    #      (insurance, bond, indemnify, terminate, sla, etc.) checked
    #      against raw text; if ≥3 mentions but no entity covers them,
    #      force-inject via canonicalize.
    #   2. Per-page coverage gauge — flag pages with high sentence
    #      count but low entity yield as "missed content" candidates.
    #   3. PLIR (Page-Level Iterative Recall) — for each low-coverage
    #      page, ask LLM "what did we miss on this page" with current
    #      extractions shown. Catches content buried in pages without
    #      section headings.
    # All gated by SOWSMITH_ZERO_MISS_DISABLE.
    # ────────────────────────────────────────────────────────────
    if not os.environ.get("SOWSMITH_ZERO_MISS_DISABLE"):
        try:
            from app.core.zero_miss import (
                pm_vocab_sweep,
                compute_page_coverage,
                find_low_coverage_pages,
                page_level_iterative_recall,
            )
            # Build raw_text_by_artifact + by_page for downstream use
            raw_by_artifact: dict[str, str] = {}
            raw_by_page: dict[tuple[str, int], str] = {}
            for atom in atoms:
                aid = getattr(atom, "artifact_id", None) or ""
                raw = getattr(atom, "raw_text", "") or ""
                if aid and raw:
                    raw_by_artifact[aid] = raw_by_artifact.get(aid, "") + "\n\n" + raw
                try:
                    refs = getattr(atom, "source_refs", None) or []
                    if refs:
                        ref = refs[0]
                        fname = getattr(ref, "filename", None) or ""
                        loc = getattr(ref, "locator", None) or {}
                        page = loc.get("page", 0) if isinstance(loc, dict) else 0
                        if fname:
                            raw_by_page[(fname, page)] = \
                                raw_by_page.get((fname, page), "") + "\n\n" + raw
                except Exception:
                    pass

            # 1. PM-critical vocab sweep
            try:
                all_raw = "\n\n".join(raw_by_artifact.values())
                missed_pm = pm_vocab_sweep(
                    all_raw, atoms, results,
                    canonicalize_fn=_canonicalize_candidate,
                    mention_threshold=3,
                )
                if missed_pm:
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "v45 PM-vocab sweep: %d force-injected items",
                        len(missed_pm),
                    )
                    # Inject into results by kind
                    for item in missed_pm:
                        kind = item.get("kind")
                        outcome = item.get("outcome", {})
                        if not kind or not outcome:
                            continue
                        # Map entity type to multi_result key
                        result_key = {
                            "requirement": "requirements",
                            "stakeholder": "stakeholders",
                            "milestone": "milestones",
                            "site": "site_clusters",
                            "quantity": "quantities",
                            "certification": "certifications",
                            "risk": "risks",
                            "acceptance_criteria": "acceptance_criteria",
                            "penalty": "penalties",
                            "compliance_obligation": "compliance_obligations",
                        }.get(kind)
                        if not result_key or result_key not in results:
                            continue
                        # Build the right shape per entity type
                        if kind == "requirement":
                            entry = {"text": outcome.get("canonical", ""),
                                     "kind": outcome.get("kind"),
                                     "_via": outcome.get("_via")}
                        elif kind == "stakeholder":
                            entry = {"name": outcome.get("name", ""),
                                     "role": outcome.get("role"),
                                     "_via": outcome.get("_via")}
                        elif kind == "site":
                            entry = {"canonical_name": outcome.get("canonical_name", ""),
                                     "aliases": [outcome.get("canonical_name", "")],
                                     "_via": outcome.get("_via")}
                        elif kind == "milestone":
                            entry = {"name": outcome.get("canonical", ""),
                                     "_via": outcome.get("_via")}
                        elif kind == "quantity":
                            entry = {"text": outcome.get("canonical", ""),
                                     "kind": outcome.get("kind"),
                                     "_via": outcome.get("_via")}
                        elif kind == "certification":
                            entry = {"name": outcome.get("canonical", ""),
                                     "level": outcome.get("level"),
                                     "kind": outcome.get("kind"),
                                     "_via": outcome.get("_via")}
                        elif kind == "risk":
                            entry = {"description": outcome.get("canonical", ""),
                                     "kind": outcome.get("kind"),
                                     "severity": outcome.get("severity"),
                                     "_via": outcome.get("_via")}
                        elif kind == "acceptance_criteria":
                            entry = {"criterion": outcome.get("canonical", ""),
                                     "kind": outcome.get("kind"),
                                     "_via": outcome.get("_via")}
                        elif kind == "penalty":
                            entry = {"description": outcome.get("canonical", ""),
                                     "kind": outcome.get("kind"),
                                     "magnitude": outcome.get("magnitude"),
                                     "_via": outcome.get("_via")}
                        elif kind == "compliance_obligation":
                            entry = {"obligation": outcome.get("canonical", ""),
                                     "statute_reference": outcome.get("statute_reference"),
                                     "kind": outcome.get("kind"),
                                     "_via": outcome.get("_via")}
                        else:
                            continue
                        results[result_key].append(entry)
            except Exception as e:
                import logging as _lg
                _lg.getLogger(__name__).warning("pm_vocab_sweep failed: %s", e)

            # 2 + 3. Per-page coverage + PLIR
            try:
                coverage = compute_page_coverage(atoms, raw_by_page)
                low_cov = find_low_coverage_pages(
                    coverage, min_sentences=10, max_ratio=0.05,
                )
                if low_cov:
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "v45 coverage: %d low-coverage pages flagged for PLIR",
                        len(low_cov),
                    )
                    # Build prior_extractions per page (best-effort)
                    # — we don't track exact page provenance for each
                    # extracted item, so prior_items will be empty
                    # for most pages. PLIR can still find new content.
                    page_extractions: dict[tuple[str, int], list] = {}
                    plir_added = page_level_iterative_recall(
                        raw_by_page, coverage, page_extractions,
                        llm_call=lambda p, mt: _call_ollama(p, max_tokens=mt),
                        parse_json=_parse_json_object,
                        max_pages=20,
                        parallel=3,
                    )
                    if plir_added:
                        _lg.getLogger(__name__).info(
                            "v45 PLIR: %d new items recovered",
                            len(plir_added),
                        )
                        # Inject PLIR finds into results (basic shape)
                        for item in plir_added:
                            kind = item.get("kind", "").lower()
                            text = item.get("text", "")
                            if not text:
                                continue
                            result_key = {
                                "requirement": "requirements",
                                "stakeholder": "stakeholders",
                                "milestone": "milestones",
                                "site": "site_clusters",
                                "quantity": "quantities",
                                "money": None,  # money is regex-emitted
                                "date": None,
                                "certification": "certifications",
                                "risk": "risks",
                                "acceptance_criteria": "acceptance_criteria",
                                "penalty": "penalties",
                                "compliance_obligation": "compliance_obligations",
                            }.get(kind)
                            if not result_key or result_key not in results:
                                continue
                            # Use sensible shape per type
                            if kind == "requirement":
                                results[result_key].append(
                                    {"text": text, "_via": "plir"})
                            elif kind == "stakeholder":
                                results[result_key].append(
                                    {"name": text, "_via": "plir"})
                            elif kind == "site":
                                results[result_key].append(
                                    {"canonical_name": text,
                                     "aliases": [text], "_via": "plir"})
                            elif kind == "milestone":
                                results[result_key].append(
                                    {"name": text, "_via": "plir"})
                            elif kind == "certification":
                                results[result_key].append(
                                    {"name": text, "_via": "plir"})
                            elif kind == "risk":
                                results[result_key].append(
                                    {"description": text, "_via": "plir"})
                            elif kind == "acceptance_criteria":
                                results[result_key].append(
                                    {"criterion": text, "_via": "plir"})
                            elif kind == "penalty":
                                results[result_key].append(
                                    {"description": text, "_via": "plir"})
                            elif kind == "compliance_obligation":
                                results[result_key].append(
                                    {"obligation": text, "_via": "plir"})
                            elif kind == "quantity":
                                results[result_key].append(
                                    {"text": text, "_via": "plir"})
            except Exception as e:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "v45 PLIR / coverage pass failed: %s", e,
                )
        except Exception as e:
            import logging as _lg
            _lg.getLogger(__name__).warning("zero_miss import failed: %s", e)

    # ────────────────────────────────────────────────────────────
    # v43: VISION-LLM extraction for pages flagged as visual-only.
    # Calls qwen2.5vl:7b on each PDF page where the text parser
    # reported "visual / table / diagram evidence not fully extracted".
    # Extracts structured rows (BOM lines, contact rosters, schedule
    # cells, etc.) and tags them with entity_type kind. Stashed under
    # `vision_rows` for downstream injection.
    # ────────────────────────────────────────────────────────────
    if not os.environ.get("SOWSMITH_VISION_DISABLE"):
        try:
            from app.core.vision_extraction import (
                find_all_pages_needing_vision,
                extract_visual_pages,
                vision_endpoint_reachable,
            )
            if vision_endpoint_reachable():
                # v45.1: union of parser-flagged visual pages +
                # pymupdf-detected table pages. Ensures vision-LLM
                # fires on EVERY page with structured visual content,
                # not just pages the text parser couldn't read.
                visual_pages = find_all_pages_needing_vision(atoms)
                if visual_pages:
                    import logging
                    logging.getLogger(__name__).info(
                        "v43 vision: %d visual pages identified",
                        len(visual_pages),
                    )
                    vision_results = extract_visual_pages(
                        visual_pages,
                        max_parallel=int(
                            os.environ.get("SOWSMITH_VISION_PARALLEL", "3")
                        ),
                        max_pages=int(
                            os.environ.get("SOWSMITH_VISION_MAX_PAGES", "30")
                        ),
                    )
                    if vision_results:
                        results["vision_rows"] = vision_results
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "vision-LLM pass failed: %s", e,
            )

    return results


def _empty_result() -> dict[str, Any]:
    return {
        "customer": None,
        "stakeholders": [],
        "milestones": [],
        "requirements": [],
        "site_clusters": [],
        "quantities": [],
        # v43 — 5 new entity types
        "certifications": [],
        "risks": [],
        "acceptance_criteria": [],
        "penalties": [],
        "compliance_obligations": [],
        # v48 — 3 new entity types
        "lead_times": [],
        "electrical_acceptance": [],
        "payment_terms": [],
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
  {{
    "name": "<Full Name as written in docs>",
    "title": "<job title or null>",
    "role": "<functional role in THIS project: PM | technical lead | approver | signatory | bid contact | exec sponsor | null>",
    "email": "<email or null>",
    "phone": "<phone or null>",
    "approval_domain": "<what this person approves, e.g. 'technical design', 'contracts >$1.5M', 'scope changes', 'all deliverables' — or null>",
    "org": "<'customer' | 'vendor' | 'integrator' | 'consultant' | null>",
    "signatory": <true if they sign contracts or acceptance docs, false otherwise>
  }},
  ...
]}}

If genuinely no named humans appear in the docs, return: {{"stakeholders": []}}
But if you see ANY email with an associated name, or any "contact <Name>" line, that person MUST appear in your output.

approval_domain examples:
- "technical sign-off on design docs"
- "CFO authority for contracts >$1.5M"
- "IT security approval on network changes"
- "project acceptance sign-off"

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
        ("name", "title", "role", "email", "phone", "approval_domain", "org", "signatory"),
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
        fields=("name", "title", "role", "email", "phone", "approval_domain", "org", "signatory"),
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
        "keep the meaningful verb and object; max 120 chars). Also\n"
        "classify the kind:\n"
        "  technical    — software / hardware / integration capability\n"
        "  commercial   — pricing, payment terms, commercial conduct\n"
        "  legal        — indemnification, termination, governing law\n"
        "  operational  — staffing, conduct, training, hours\n"
        "  compliance   — regulatory, certification, audit requirements\n"
        "  insurance    — coverage limits, bonds, deductibles\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical": "<canonical form>", "kind": "<one of: technical|commercial|legal|operational|compliance|insurance>"}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
    "stakeholder": (
        "TASK: Find ALL real human STAKEHOLDERS named in the SENTENCE.\n"
        "Team rosters / signature blocks often list multiple people on\n"
        "one line — extract EVERY person you find.\n\n"
        "KEEP a person if their name appears with:\n"
        "  - A first name + last name (e.g. 'Kaylee Yinger')\n"
        "  - May have a role title attached ('Lisa Brock/Implementation PM')\n"
        "  - May appear in a roster ('Front of the House: A/Role, B/Role, C/Role')\n\n"
        "DROP if the candidate is:\n"
        "  - an organization, company, or department name\n"
        "  - a job title alone with no person name\n"
        "  - a generic noun phrase ('end users', 'customer support', 'mosaic front')\n"
        "  - an email address used as a 'name'\n"
        "  - a product / service / SaaS name\n"
        "  - a FRAGMENT like 'John S' / 'Russell R' / 'Edmund G' (single-letter\n"
        "    initial as last name suggests the doc only shows redacted initials;\n"
        "    NEVER fabricate the missing surname or letter-only 'name')\n"
        "  - a single-word name with no surname (Russell / John / Edmund alone)\n\n"
        "REQUIREMENT: each kept person MUST have a full surname (≥2 chars).\n"
        "If you see 'Russell R.' with no full surname elsewhere, DROP — don't\n"
        "extract 'Russell R' as a stakeholder.\n\n"
        "For each kept person, extract role + email/phone if visible.\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "people": [{{"name": "First Last", "role": "...", "email": "...", "phone": "..."}}, ...]}}\n'
        "  or {{\"keep\": false}}\n\n"
        "If only one person, still wrap them in the people array.\n"
        "If no real people, return keep:false (don't fabricate names).\n\n"
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
        "'2-hour Sev1 response', '32 schools'). Also classify the kind:\n"
        "  sla | count | duration | payment_term | percentage | lead_time | hours\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical": "<short form>", "value": "<numeric value>", "unit": "<unit>", "kind": "<one of: sla|count|duration|payment_term|percentage|lead_time|hours>"}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
    # v43 — five new entity-type canonicalize prompts
    "certification": (
        "TASK: Decide if the SENTENCE references a CERTIFICATION,\n"
        "STANDARD, or COMPLIANCE FRAMEWORK the vendor / customer claims\n"
        "or requires.\n\n"
        "KEEP if the sentence references:\n"
        "  - Security / payment certs (PCI-DSS, SOC 2, ISO 27001, NIST 800-53)\n"
        "  - Privacy regulations (HIPAA, FERPA, GDPR, CCPA, COPPA)\n"
        "  - Audit / quality certs (SSAE 18, FedRAMP, FISMA)\n"
        "  - Education / govt certs (USDA approval, FNS-XXX forms)\n"
        "  - Industry standards (TIA-568, NFPA 70/72, IEEE 802.11, NEC)\n"
        "  - Quality systems (ISO 9001, AS9100)\n\n"
        "DROP if the sentence:\n"
        "  - is generic 'industry-leading security' marketing\n"
        "  - mentions the word 'standard' without naming a specific one\n"
        "  - is a job-title 'standards engineer'\n\n"
        "If KEEP, extract the certification's canonical name (PCI-DSS, SOC 2,\n"
        "NIST 800-53, etc.). Drop level / version suffixes into a separate\n"
        "field when present.\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical": "<cert name>", "level": "<level/version or empty>", "kind": "<security|privacy|audit|education|industry|quality>"}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
    "risk": (
        "TASK: Decide if the SENTENCE describes a RISK / DEPENDENCY /\n"
        "CONTINGENCY (something that could go wrong, cause delays, or\n"
        "require contingency planning).\n\n"
        "KEEP if the sentence describes:\n"
        "  - Schedule risks (long lead times, dependencies)\n"
        "  - Technical risks (capacity limits, compatibility issues)\n"
        "  - Commercial risks (payment-term ambiguity, fee escalation)\n"
        "  - Compliance risks (regulatory exposure, audit findings)\n"
        "  - Operational risks (single points of failure, single-source vendors)\n\n"
        "DROP if the sentence is:\n"
        "  - a marketing claim about how risks are mitigated (positive spin)\n"
        "  - generic 'we manage risk well' boilerplate\n"
        "  - a requirement (handled by separate extractor)\n\n"
        "If KEEP, summarize the risk in canonical form and classify the kind.\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical": "<risk summary>", "kind": "<schedule|technical|commercial|compliance|operational>", "severity": "<high|medium|low>"}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
    "acceptance": (
        "TASK: Decide if the SENTENCE defines ACCEPTANCE CRITERIA — what\n"
        "constitutes successful completion, deliverable approval, or\n"
        "phase signoff.\n\n"
        "KEEP if the sentence defines:\n"
        "  - Substantial completion criteria\n"
        "  - Final acceptance gates / observation periods\n"
        "  - Required deliverables (drawings, test reports, training records)\n"
        "  - Closeout artifacts (warranty registrations, as-builts)\n"
        "  - Sign-off requirements between phases\n\n"
        "DROP if the sentence is:\n"
        "  - a general requirement (handled by separate extractor)\n"
        "  - acceptance of terms in a legal sense (not project acceptance)\n\n"
        "If KEEP, summarize the acceptance criterion in canonical form.\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical": "<acceptance criterion>", "kind": "<substantial|final|phase_gate|deliverable|closeout>"}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
    "penalty": (
        "TASK: Decide if the SENTENCE defines a PENALTY, SERVICE CREDIT,\n"
        "LIQUIDATED DAMAGE, or TERMINATION TRIGGER (what happens when\n"
        "the contractor fails to meet an obligation).\n\n"
        "KEEP if the sentence defines:\n"
        "  - Service credits (X% of monthly fee per hour of downtime)\n"
        "  - Late delivery penalties (X% per business day)\n"
        "  - Late payment interest (X% per month)\n"
        "  - Liquidated damages ($X per day)\n"
        "  - Termination-for-default triggers (cure periods, material breach)\n"
        "  - Bond forfeiture conditions\n\n"
        "DROP if the sentence is:\n"
        "  - a general SLA (handled by quantity extractor)\n"
        "  - a marketing claim about how penalties are mitigated\n\n"
        "If KEEP, summarize the penalty in canonical form.\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical": "<penalty summary>", "kind": "<service_credit|late_delivery|late_payment|liquidated_damages|termination|bond_forfeiture>", "magnitude": "<numeric or empty>"}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
    "compliance_obligation": (
        "TASK: Decide if the SENTENCE references a COMPLIANCE OBLIGATION —\n"
        "a statute, regulation, code, or law the contractor must follow.\n\n"
        "KEEP if the sentence references:\n"
        "  - Labor laws (Fair Labor Standards Act, Davis-Bacon, ADA)\n"
        "  - Equal Employment Opportunity (Title VII, Section 504)\n"
        "  - State procurement codes (SC Code 11-35, Texas Govt 2252)\n"
        "  - Federal regulations (FAR Part 52, FedRAMP, FISMA)\n"
        "  - Industry codes (NEC, NFPA, IBC, IFC)\n"
        "  - Tax exemption statutes\n"
        "  - Data privacy regulations as legal requirements (HIPAA, FERPA)\n\n"
        "DROP if the sentence is:\n"
        "  - a certification claim (handled by certification extractor)\n"
        "  - a general 'shall comply with all applicable laws' boilerplate\n\n"
        "If KEEP, summarize the obligation including the statute reference.\n\n"
        "SENTENCE: {sentence}\n\n"
        "OUTPUT exactly one JSON object on one line:\n"
        '  {{"keep": true, "canonical": "<obligation summary>", "statute_reference": "<code section or name>", "kind": "<labor|equality|procurement|federal|industry_code|tax|privacy>"}}\n'
        "  or {{\"keep\": false}}\n\n"
        "/no_think"
    ),
}


def _canonicalize_candidate(
    sentence: str, entity_type: str
) -> dict[str, Any] | None:
    """Single-sentence LLM call: keep/drop + canonical form for one
    candidate sentence. Returns None on parse failure or LLM error.

    v42: when keep=false with confident rejection, append the sentence
    to the persistent negative-exemplar store so future runs learn
    from this rejection.
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
        # v42: self-bootstrap negatives — append this sentence to the
        # persistent negative-exemplar store. Future runs will down-rank
        # similar sentences automatically.
        try:
            from app.core.rag_extras import append_bootstrapped_negative
            append_bootstrapped_negative(entity_type, truncated)
        except Exception:
            pass
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
    # v42: AUGMENT exemplars with HyDE-generated examples + bootstrapped
    # negatives. HyDE is one-time cost (cached to disk); bootstrapped
    # negatives accumulate across runs to make the system self-improving.
    try:
        from app.core.rag_extras import (
            augment_exemplars_with_hyde,
            load_bootstrapped_negatives,
        )
        exemplars = augment_exemplars_with_hyde(
            exemplars, entity_type,
            llm_call=lambda p, mt: _call_ollama(p, max_tokens=mt),
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "HyDE augmentation failed for %s: %s", entity_type, e,
        )

    # Try v39 hybrid pipeline first (now with augmented exemplars)
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
                    static_neg = NEGATIVE_EXEMPLARS_BY_TYPE.get(entity_type, [])
                    # v42: combine static negatives with bootstrapped negatives
                    try:
                        from app.core.rag_extras import load_bootstrapped_negatives
                        boot_neg = load_bootstrapped_negatives(entity_type)
                    except Exception:
                        boot_neg = []
                    neg_exemplars = list(static_neg) + boot_neg
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

    # v44.4: was 12, lowered to 6 to avoid ollama saturation on Mac
    # (3 extractors x 6 canon = 18 max concurrent LLM calls, vs old
    # 5 x 12 = 60).
    parallel = int(os.environ.get("SOWSMITH_CANONICALIZE_PARALLEL", "6"))
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
            # v41: multi-entry canonicalize output (stakeholder "people"
            # array) — DON'T dedupe at this layer because we have
            # multiple people in one outcome. The downstream extractor
            # will expand + dedupe by name.
            if "people" in outcome and isinstance(outcome.get("people"), list):
                outcome["_source_sentence"] = candidate["sentence"]
                outcome["_source_paragraph"] = candidate["paragraph"]
                outcome["_source_artifact_id"] = candidate["artifact_id"]
                outcome["_retrieval_score"] = round(candidate["score"], 4)
                outcome["_dense_score"] = round(candidate.get("dense_score", 0.0), 4)
                results.append(outcome)
                continue
            # Single-canonical-value dedup (default)
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
    # v40+v42: SICRL — Section-Indexed Counterfactual Recall Loop
    # ────────────────────────────────────────────────────────────
    # v42: now ITERATIVE — runs up to 2 passes for stronger convergence.
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
                sicrl_iters = int(os.environ.get("SOWSMITH_SICRL_ITERS", "2"))
                for _ in range(sicrl_iters):
                    prev_count = len(results)
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
                        max_iterations=1,  # each call is one pass
                    )
                    if len(augmented) <= prev_count:
                        # Convergence — no new items found
                        results = augmented
                        break
                    results = augmented
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "SICRL pass failed for %s: %s", entity_type, e,
            )

    # ────────────────────────────────────────────────────────────
    # v42: TOURNAMENT canonicalization — cross-doc dedup via N²/2
    # pairwise LLM judging when items have high cosine similarity
    # ────────────────────────────────────────────────────────────
    if results and len(results) >= 2 and not os.environ.get("SOWSMITH_TOURNAMENT_DISABLE"):
        try:
            from app.core.rag_extras import run_tournament
            from app.core.embedding_retrieval import embed_texts as _embed_texts
            # Embed the canonical forms of all items
            canonical_strings = [
                (r.get(canonical_key) or r.get("name") or "")
                for r in results
            ]
            canonical_strings = [s for s in canonical_strings if s]
            if len(canonical_strings) >= 2:
                item_vecs = _embed_texts(canonical_strings[:300])  # cap to avoid runaway
                if item_vecs.size > 0:
                    deduped = run_tournament(
                        results[:len(canonical_strings[:300])],
                        item_vecs,
                        entity_type=entity_type,
                        canonical_key=canonical_key,
                        llm_call=lambda p, mt: _call_ollama(p, max_tokens=mt),
                        parse_json=_parse_json_object,
                        sim_threshold=0.85,
                        max_pairs=80,
                    )
                    # Re-attach any items beyond the 300-cap
                    if len(results) > 300:
                        deduped.extend(results[300:])
                    results = deduped
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Tournament dedup failed for %s: %s", entity_type, e,
            )

    # ────────────────────────────────────────────────────────────
    # v42: SELF-BOOTSTRAP NEGATIVES — accumulate canonicalize
    # rejections for next-run precision improvement
    # ────────────────────────────────────────────────────────────
    # (Currently piggybacks on the canonicalize step — see below
    # for where rejections are captured. Disabled here for simplicity;
    # to be activated by passing a callback through canonicalize.)

    return results


def _extract_requirements_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v38+v39+v40+v44: embedding-retrieval requirement extraction.
    v44 augments exemplars with pack-specific domain examples if the
    project name suggests a known domain (POS / ITAD / cabling /
    wireless / access / BMS / AV)."""
    from app.core.exemplars import REQUIREMENT_EXEMPLARS, detect_domain_extras
    exemplars = list(REQUIREMENT_EXEMPLARS)
    # v44: domain-aware exemplar routing
    try:
        project_dir_name = os.environ.get("SOWSMITH_PROJECT_DIR_NAME")
        if project_dir_name:
            extras = detect_domain_extras(project_dir_name)
            if extras:
                exemplars = exemplars + extras
    except Exception:
        pass
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="requirement",
        exemplars=exemplars,
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
    """v38+v39+v40+v41: embedding-retrieval stakeholder extraction.
    Finds named people on signature blocks, contact pages, bid-contact
    lines AND team-roster lines — no chunk dropout.

    v41 returns potentially-multi-person canonicalize output and
    expands each entry to its own stakeholder dict.
    """
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
    seen_names: set[str] = set()
    for r in raw:
        # v41 multi-person shape: {"keep": true, "people": [{name, role, email, phone}, ...]}
        # Back-compat with v40 single-person shape: {"keep": true, "name": "...", "role": ..., ...}
        people_list = r.get("people")
        if isinstance(people_list, list) and people_list:
            entries = people_list
        else:
            entries = [{
                "name": r.get("name"),
                "title": r.get("title"),
                "role": r.get("role"),
                "email": r.get("email"),
                "phone": r.get("phone"),
                "approval_domain": r.get("approval_domain"),
                "org": r.get("org"),
                "signatory": r.get("signatory"),
            }]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            # Last-chance hygiene: drop email-as-name + field-label
            if _looks_like_email_or_url(name):
                continue
            if _is_likely_field_label(name):
                continue
            # v44.1: drop initial-only surnames ("John S" / "Russell R" /
            # "Edmund G"). A real surname is ≥2 alphabetic chars.
            tokens = name.split()
            if len(tokens) >= 2:
                last = tokens[-1].rstrip(".")
                # Surname must be ≥2 chars AND not be a single-letter initial
                if len(last) < 2 or (len(last) == 1 and last.isalpha()):
                    continue
                # Drop "Name X." pattern where X is single uppercase letter
                if len(last) == 2 and last[1] == "." and last[0].isupper():
                    continue
            if len(tokens) == 1:
                # Single-word name without surname → drop
                continue
            # Dedupe by name across multi-person entries
            sig = re.sub(r"\s+", " ", name.lower()).strip()[:120]
            if sig in seen_names:
                continue
            seen_names.add(sig)
            sig_val = entry.get("signatory")
            out.append({
                "name": name,
                "title": (entry.get("title") or "").strip() or None,
                "role": (entry.get("role") or "").strip() or None,
                "email": (entry.get("email") or "").strip() or None,
                "phone": (entry.get("phone") or "").strip() or None,
                "approval_domain": (entry.get("approval_domain") or "").strip() or None,
                "org": (entry.get("org") or "").strip() or None,
                "signatory": bool(sig_val) if isinstance(sig_val, (bool, int)) else False,
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
            "kind": r.get("kind"),  # v43 classification
            "_source_sentence": r.get("_source_sentence"),
            "_source_artifact_id": r.get("_source_artifact_id"),
        })
    return out


# ════════════════════════════════════════════════════════════════════
# v43 — 5 new entity-type extractors: certification, risk, acceptance,
# penalty, compliance_obligation
# ════════════════════════════════════════════════════════════════════


def _extract_certifications_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v43: certifications / standards / compliance frameworks the
    vendor claims or customer requires (PCI-DSS, SOC 2, HIPAA, FERPA,
    NIST 800-53, ISO 27001, TIA-568, NFPA 72, USDA, FNS-XXX)."""
    from app.core.exemplars import CERTIFICATION_EXEMPLARS
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="certification",
        exemplars=CERTIFICATION_EXEMPLARS,
        top_k_per_artifact=300,
        min_score=0.25,  # v44.1: lowered — was 0.35, missed cabling/wireless certs
        canonical_key="canonical",
    )
    out = []
    for r in raw:
        canon = r.get("canonical")
        if not isinstance(canon, str) or not canon.strip():
            continue
        out.append({
            "name": canon.strip(),
            "level": (r.get("level") or "").strip() or None,
            "kind": r.get("kind"),  # security|privacy|audit|education|industry|quality
            "_source_sentence": r.get("_source_sentence"),
            "_source_artifact_id": r.get("_source_artifact_id"),
        })
    return out


def _extract_risks_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v43: risks / dependencies / contingencies."""
    from app.core.exemplars import RISK_EXEMPLARS
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="risk",
        exemplars=RISK_EXEMPLARS,
        top_k_per_artifact=300,
        min_score=0.25,  # v44.1: lowered
        canonical_key="canonical",
    )
    out = []
    for r in raw:
        canon = r.get("canonical")
        if not isinstance(canon, str) or not canon.strip():
            continue
        out.append({
            "description": canon.strip(),
            "kind": r.get("kind"),  # schedule|technical|commercial|compliance|operational
            "severity": r.get("severity"),  # high|medium|low
            "_source_sentence": r.get("_source_sentence"),
            "_source_artifact_id": r.get("_source_artifact_id"),
        })
    return out


def _extract_acceptance_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v43: acceptance criteria / deliverable gates / closeout artifacts."""
    from app.core.exemplars import ACCEPTANCE_EXEMPLARS
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="acceptance",
        exemplars=ACCEPTANCE_EXEMPLARS,
        top_k_per_artifact=250,
        min_score=0.25,  # v44.1: lowered
        canonical_key="canonical",
    )
    out = []
    for r in raw:
        canon = r.get("canonical")
        if not isinstance(canon, str) or not canon.strip():
            continue
        out.append({
            "criterion": canon.strip(),
            "kind": r.get("kind"),
            "_source_sentence": r.get("_source_sentence"),
            "_source_artifact_id": r.get("_source_artifact_id"),
        })
    return out


def _extract_penalties_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v43: penalties / service credits / liquidated damages /
    termination triggers."""
    from app.core.exemplars import PENALTY_EXEMPLARS
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="penalty",
        exemplars=PENALTY_EXEMPLARS,
        top_k_per_artifact=250,
        min_score=0.25,  # v44.1: lowered
        canonical_key="canonical",
    )
    out = []
    for r in raw:
        canon = r.get("canonical")
        if not isinstance(canon, str) or not canon.strip():
            continue
        out.append({
            "description": canon.strip(),
            "kind": r.get("kind"),
            "magnitude": r.get("magnitude"),
            "_source_sentence": r.get("_source_sentence"),
            "_source_artifact_id": r.get("_source_artifact_id"),
        })
    return out


def _extract_compliance_obligations_retrieved(
    by_artifact: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """v43: compliance obligations / statute references / regulatory
    requirements."""
    from app.core.exemplars import COMPLIANCE_EXEMPLARS
    raw = _run_retrieval_extract(
        by_artifact,
        entity_type="compliance_obligation",
        exemplars=COMPLIANCE_EXEMPLARS,
        top_k_per_artifact=300,
        min_score=0.25,  # v44.1: lowered
        canonical_key="canonical",
    )
    out = []
    for r in raw:
        canon = r.get("canonical")
        if not isinstance(canon, str) or not canon.strip():
            continue
        out.append({
            "obligation": canon.strip(),
            "statute_reference": (r.get("statute_reference") or "").strip() or None,
            "kind": r.get("kind"),
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
# v48 EXTRACTORS — lead times / electrical acceptance / payment terms
# ════════════════════════════════════════════════════════════════════


def _retrieve_for_v48(by_artifact: dict[str, list[Any]], queries: list[str]) -> str:
    """Shared retrieval + excerpt build for v48 extractors. Returns the
    excerpt text (one line per retrieved atom) or empty string if nothing
    relevant was found.
    """
    try:
        from app.core.embedding_retrieval import retrieve_for_query
    except ImportError:
        return ""
    atoms: list[Any] = []
    seen: set[str] = set()
    for art_atoms in by_artifact.values():
        for a in art_atoms:
            aid = getattr(a, "id", None)
            if aid and aid not in seen:
                atoms.append(a)
                seen.add(aid)
    if not atoms:
        return ""
    try:
        retrieved = retrieve_for_query(atoms=atoms, queries=queries, top_k=20, dedupe=True)
    except TypeError:
        # Fallback for retrieval implementations with a slightly different signature.
        retrieved = []
    except Exception:
        return ""
    if not retrieved:
        return ""
    return "\n".join(f"- {getattr(a, 'raw_text', '')}" for a in retrieved)


def _extract_lead_times_retrieved(by_artifact: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Extract procurement lead times and delivery windows.

    Targets: "ARO N weeks", "lead time N days/weeks", "delivery within N
    weeks of PO", "staging T+N", "ship window", "procurement timeline".
    """
    queries = [
        "lead time delivery weeks after purchase order ARO",
        "procurement timeline equipment delivery schedule",
        "staging ship window material availability",
        "weeks days after notice to proceed delivery",
    ]
    excerpt = _retrieve_for_v48(by_artifact, queries)
    if not excerpt:
        return []
    prompt = f"""Extract procurement lead times and delivery windows from these project document excerpts.

For each lead time / delivery constraint found, return:
- item: what is being delivered or procured
- duration: the time value (e.g. "14 weeks", "6-8 weeks", "21 days")
- trigger: what starts the clock (e.g. "ARO", "after PO", "after NTP", "after deposit")
- notes: any relevant context

DOCUMENT EXCERPTS:
{excerpt}

OUTPUT (JSON array, no markdown):
{{"lead_times": [
  {{"item": "...", "duration": "...", "trigger": "...", "notes": "..."}}
]}}

If none found: {{"lead_times": []}}
/no_think"""
    text = _call_ollama(prompt, max_tokens=1024)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return []
    raw = obj.get("lead_times", [])
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        t = str(item.get("item") or "").strip()
        d = str(item.get("duration") or "").strip()
        if t and d:
            out.append({
                "item": t,
                "duration": d,
                "trigger": str(item.get("trigger") or "").strip(),
                "notes": str(item.get("notes") or "").strip(),
            })
    return out


def _extract_electrical_acceptance_retrieved(by_artifact: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Extract electrical and commissioning acceptance test requirements.

    Targets: megger readings, ground resistance thresholds, burn-in
    periods, ATP checklist items, OTDR fiber test requirements, PoE
    load tests, UPS transfer-time tests, surge protection ratings.
    """
    queries = [
        "megger insulation resistance test ground bonding ATP acceptance test",
        "burn-in period commissioning test fiber OTDR attenuation",
        "acceptance test procedure electrical PoE load verification",
        "ground resistance ohm threshold UPS transfer time surge protection",
    ]
    excerpt = _retrieve_for_v48(by_artifact, queries)
    if not excerpt:
        return []
    prompt = f"""Extract electrical and commissioning acceptance test requirements from these excerpts.

For each test requirement, return:
- test: name or description of the test
- threshold: the pass/fail criterion (e.g. ">100 MΩ", "<5 Ω", "≥-3.5 dB insertion loss")
- scope: what equipment or system it applies to
- notes: timing, who performs it, documentation required

DOCUMENT EXCERPTS:
{excerpt}

OUTPUT (JSON array, no markdown):
{{"electrical_acceptance": [
  {{"test": "...", "threshold": "...", "scope": "...", "notes": "..."}}
]}}

If none found: {{"electrical_acceptance": []}}
/no_think"""
    text = _call_ollama(prompt, max_tokens=1024)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return []
    raw = obj.get("electrical_acceptance", [])
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        t = str(item.get("test") or "").strip()
        if t:
            out.append({
                "test": t,
                "threshold": str(item.get("threshold") or "").strip(),
                "scope": str(item.get("scope") or "").strip(),
                "notes": str(item.get("notes") or "").strip(),
            })
    return out


def _extract_payment_terms_retrieved(by_artifact: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Extract structured payment term schedules.

    Targets: milestone-based payment splits ("30% upon execution"),
    retainage terms, net-N payment windows, progress billing schedules,
    final acceptance payment triggers.
    """
    queries = [
        "payment schedule milestone percent upon contract execution deposit",
        "retainage final payment upon acceptance net 30 billing",
        "progress payment invoicing tranche equipment delivery",
        "30 percent 40 percent payment terms commercial financial",
    ]
    excerpt = _retrieve_for_v48(by_artifact, queries)
    if not excerpt:
        return []
    prompt = f"""Extract payment terms and schedules from these project document excerpts.

For each payment tranche or term, return:
- tranche: label (e.g. "Deposit", "Equipment Delivery", "Final Acceptance", "Retainage")
- percent: percentage (number only, no % symbol, e.g. 30)
- trigger: what event releases this payment
- notes: any conditions, net terms, or exceptions

Also extract any net payment window (e.g. "Net 30", "Net 45").

DOCUMENT EXCERPTS:
{excerpt}

OUTPUT (JSON, no markdown):
{{"payment_terms": [
  {{"tranche": "...", "percent": <number or null>, "trigger": "...", "notes": "..."}}
],
  "net_days": <integer or null>,
  "retainage_percent": <number or null>
}}

If none found: {{"payment_terms": [], "net_days": null, "retainage_percent": null}}
/no_think"""
    text = _call_ollama(prompt, max_tokens=1024)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return []
    raw = obj.get("payment_terms", [])
    if not isinstance(raw, list):
        return []
    net_days = obj.get("net_days") if isinstance(obj.get("net_days"), int) else None
    retainage = obj.get("retainage_percent")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tranche = str(item.get("tranche") or "").strip()
        if tranche:
            pct = item.get("percent")
            out.append({
                "tranche": tranche,
                "percent": pct if isinstance(pct, (int, float)) else None,
                "trigger": str(item.get("trigger") or "").strip(),
                "notes": str(item.get("notes") or "").strip(),
                "net_days": net_days,
                "retainage_percent": retainage,
            })
    return out


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
    # v44.1: form-field label leaks
    "date issue", "issue date", "issued date",
    "effective date", "expiration date", "due date",
    "page number", "page", "section number",
    "doc number", "document number", "form number",
    "revision number", "version number",
    "from", "to", "by", "via",
    "yes", "no", "n/a", "tbd",
    "true", "false",
    "name", "title", "role",
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
