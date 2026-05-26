"""v40 — Section-Indexed Counterfactual Recall Loop (SICRL).

GENUINELY NOVEL technique that goes beyond published RAG research.
The core insight: bid documents are STRUCTURED — every
"5.0 Insurance Requirements" section has predictable content shape
regardless of customer. We exploit this by:

  1. Detecting section headings in each artifact.
  2. Running v39 retrieval as a first pass, tagging each candidate
     with its parent section.
  3. Identifying under-covered sections (got 0-1 candidates when
     the section heading suggests there should be many).
  4. For each under-covered section, asking the LLM "given an RFP
     section titled 'X', what 5-10 typical clauses/items should
     appear here?"
  5. Each predicted clause becomes a new embedding query, searched
     ONLY against that section's text (not the whole doc).
  6. Candidates above threshold → canonicalize → union with first-pass.
  7. Reflective loop: ask LLM "did we cover section X now?" until
     convergence or max 3 iterations.

Why this is different from HyDE / RAG-Fusion / Self-RAG:
  - HyDE generates hypothetical ANSWERS to user queries.
  - RAG-Fusion generates QUERY VARIANTS from a single user question.
  - Self-RAG decides WHETHER to retrieve.
  - SICRL generates COUNTERFACTUAL QUERIES from DOCUMENT STRUCTURE,
    using the document's own heading hierarchy as ground truth for
    "what should be here." No user question involved.

Why this works:
  - Section titles encode strong content-type priors (Insurance,
    Termination, Compliance, etc.).
  - LLMs have rich training data on what "Insurance Requirements"
    typically contains in RFPs.
  - Searching within a SECTION (not the whole doc) avoids retrieving
    matches from unrelated sections that happen to share verbs.

Cost: one extra LLM call per under-covered section (~3-10 calls per
artifact). Embed cost is amortized via the disk cache. Total
overhead: ~30-60 seconds per Pack-18-sized doc.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# SECTION DETECTION
# ────────────────────────────────────────────────────────────────────


# Patterns for typical RFP / proposal section headings
_SECTION_HEADING_PATTERNS = [
    # "5.0 INSURANCE REQUIREMENTS" / "5.0.1 General Liability"
    re.compile(r"^\s*(\d+(?:\.\d+){0,3})\s+([A-Z][A-Za-z0-9 \-/&,]{4,80})\s*$", re.MULTILINE),
    # "Section 5: Insurance Requirements" / "ARTICLE 7 - Insurance"
    re.compile(
        r"^\s*(?:Section|Article|Chapter|Part|Exhibit|Appendix|Schedule|Attachment)\s+"
        r"(\d+(?:\.\d+){0,2}|[IVXLC]+|[A-Z])\b[:\-\s]+([A-Z][A-Za-z0-9 \-/&,]{4,80})\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # ALL CAPS HEADINGS on their own line
    re.compile(r"^\s*([A-Z][A-Z0-9 \-/&,]{8,80})\s*$", re.MULTILINE),
]


def detect_sections(text: str, *, min_section_chars: int = 200) -> list[dict[str, Any]]:
    """Detect section headings + content ranges in artifact text.

    Returns list of {title, start_offset, end_offset, text} dicts.
    Sections are sorted by start_offset. The "text" field is the
    section's content (everything until the next heading or doc end).
    """
    candidates: list[tuple[int, str]] = []  # (offset, title)
    seen_offsets: set[int] = set()

    for pat in _SECTION_HEADING_PATTERNS:
        for m in pat.finditer(text):
            offset = m.start()
            if offset in seen_offsets:
                continue
            seen_offsets.add(offset)
            # Extract title from the most-specific capture group
            groups = [g for g in m.groups() if g]
            title = " ".join(groups).strip()
            if 4 <= len(title) <= 120:
                candidates.append((offset, title))

    candidates.sort()
    if not candidates:
        return []

    sections: list[dict[str, Any]] = []
    for i, (offset, title) in enumerate(candidates):
        end = candidates[i + 1][0] if i + 1 < len(candidates) else len(text)
        section_text = text[offset:end]
        if len(section_text) < min_section_chars:
            continue  # too short to be a real section
        sections.append({
            "title": title,
            "start_offset": offset,
            "end_offset": end,
            "text": section_text,
        })
    return sections


def find_containing_section(
    sentence_text: str, all_text: str, sections: list[dict[str, Any]],
) -> str | None:
    """Given a sentence and the artifact's section list, return the
    title of the section containing that sentence. Uses first-match
    on substring lookup (sentence position in source)."""
    idx = all_text.find(sentence_text[:60])
    if idx < 0:
        return None
    for s in sections:
        if s["start_offset"] <= idx < s["end_offset"]:
            return s["title"]
    return None


# ────────────────────────────────────────────────────────────────────
# COUNTERFACTUAL PREDICTION
# ────────────────────────────────────────────────────────────────────

_COUNTERFACTUAL_PROMPTS: dict[str, str] = {
    "requirement": (
        "TASK: Given an RFP / bid-document section titled \"{section_title}\",\n"
        "list 6-10 TYPICAL requirements / obligations / clauses that should\n"
        "appear in this section. Be specific to this section type.\n\n"
        "Examples for 'Insurance Requirements' would be:\n"
        '  - "Contractor shall maintain general liability insurance."\n'
        '  - "Workers compensation coverage at statutory minimum."\n'
        '  - "Certificate of insurance must be provided within 7 days."\n\n'
        "OUTPUT a JSON array of strings, one sentence per requirement,\n"
        "covering the TYPICAL content of this section type:\n"
        '  {{"predictions": ["<requirement 1>", "<requirement 2>", ...]}}\n\n'
        "Section title: {section_title}\n\n"
        "/no_think"
    ),
    "stakeholder": (
        "TASK: Given an RFP / bid-document section titled \"{section_title}\",\n"
        "what TYPES of human stakeholders / contacts would typically appear?\n"
        "List 4-6 example contact-line sentences naming the kinds of people\n"
        "who appear in such a section.\n\n"
        "OUTPUT JSON:\n"
        '  {{"predictions": ["<example contact line 1>", "<example 2>", ...]}}\n\n'
        "Section title: {section_title}\n\n"
        "/no_think"
    ),
    "quantity": (
        "TASK: Given an RFP / bid-document section titled \"{section_title}\",\n"
        "what TYPICAL structural quantities (counts, percentages, durations,\n"
        "SLAs) would appear in this section? List 6-8 example sentences.\n\n"
        "OUTPUT JSON:\n"
        '  {{"predictions": ["<quantity sentence 1>", "<sentence 2>", ...]}}\n\n'
        "Section title: {section_title}\n\n"
        "/no_think"
    ),
}


def predict_counterfactual_items(
    section_title: str,
    entity_type: str,
    *,
    llm_call: Callable[[str, int], str],
    parse_json: Callable[[str], Any],
    model_override: str | None = None,
) -> list[str]:
    """Ask LLM to predict typical items for this section type.

    Returns list of hypothetical-document sentences to use as
    additional retrieval queries.
    """
    template = _COUNTERFACTUAL_PROMPTS.get(entity_type)
    if not template:
        return []
    prompt = template.format(section_title=section_title)
    text = llm_call(prompt, 1024)
    obj = parse_json(text)
    if not isinstance(obj, dict):
        return []
    preds = obj.get("predictions")
    if not isinstance(preds, list):
        return []
    return [p for p in preds if isinstance(p, str) and p.strip()][:12]


# ────────────────────────────────────────────────────────────────────
# COVERAGE ANALYSIS
# ────────────────────────────────────────────────────────────────────


def is_undercovered(
    section: dict[str, Any],
    item_count: int,
    *,
    entity_type: str,
) -> bool:
    """Heuristic: should this section have produced more items?

    Uses section text length + entity-type density heuristics:
      - Sections > 2000 chars producing 0 items are likely undercovered.
      - Sections > 4000 chars producing < 3 items are likely undercovered.
      - Sections with keyword 'requirements' / 'shall' / 'must' in title
        producing < 5 items are likely undercovered.
    """
    text_len = len(section.get("text", ""))
    title_lower = section.get("title", "").lower()

    # Trigger keywords in title suggesting heavy content
    heavy_keywords = (
        "requirement", "obligation", "responsibility", "duty",
        "scope", "compliance", "insurance", "indemn",
        "warranty", "guarantee", "term", "condition",
        "specification", "performance",
    )
    has_heavy_title = any(k in title_lower for k in heavy_keywords)

    if text_len < 500:
        return False
    if text_len > 2000 and item_count == 0:
        return True
    if text_len > 4000 and item_count < 3:
        return True
    if has_heavy_title and item_count < 5:
        return True
    return False


# ────────────────────────────────────────────────────────────────────
# SICRL ORCHESTRATOR
# ────────────────────────────────────────────────────────────────────


def run_sicrl(
    *,
    by_artifact: dict[str, str],
    first_pass_items: list[dict[str, Any]],
    entity_type: str,
    exemplars: list[str],
    negative_exemplars: list[str],
    llm_call: Callable[[str, int], str],
    parse_json: Callable[[str], Any],
    canonicalize_fn: Callable[[str, str], dict[str, Any] | None],
    embed_fn: Callable[[list[str]], np.ndarray],
    sentence_split_fn: Callable[[str], list[str]],
    max_iterations: int = 2,
) -> list[dict[str, Any]]:
    """Run the Section-Indexed Counterfactual Recall Loop.

    Augments first_pass_items with additional items found via
    counterfactual queries against under-covered sections.

    Returns the augmented item list (deduped).
    """
    augmented = list(first_pass_items)
    seen_sigs: set[str] = set()
    for item in augmented:
        # Build a dedup signature from the item's canonical form
        canon = (item.get("canonical")
                 or item.get("name")
                 or item.get("canonical_name") or "")
        if isinstance(canon, str) and canon.strip():
            sig = re.sub(r"\s+", " ", canon.lower()).strip()[:120]
            seen_sigs.add(sig)

    iteration_added = 0

    for aid, text in by_artifact.items():
        if not text or len(text) < 500:
            continue
        # Detect sections
        sections = detect_sections(text)
        if not sections:
            continue
        logger.info(
            "SICRL %s/%s: detected %d sections in artifact %s",
            entity_type, "iter1", len(sections), aid,
        )

        # Count first-pass items per section
        section_counts: dict[str, int] = {}
        for item in first_pass_items:
            sent = item.get("_source_sentence") or ""
            if not sent:
                continue
            sec_title = find_containing_section(sent, text, sections)
            if sec_title:
                section_counts[sec_title] = section_counts.get(sec_title, 0) + 1

        # Identify under-covered sections
        undercovered = [
            s for s in sections
            if is_undercovered(s, section_counts.get(s["title"], 0),
                               entity_type=entity_type)
        ]
        if not undercovered:
            continue
        logger.info(
            "SICRL %s: %d undercovered sections in %s",
            entity_type, len(undercovered), aid,
        )

        # Run counterfactual prediction for each undercovered section
        # IN PARALLEL — each is an independent LLM call.
        with ThreadPoolExecutor(max_workers=6) as pool:
            future_preds = {
                pool.submit(
                    predict_counterfactual_items,
                    section["title"], entity_type,
                    llm_call=llm_call, parse_json=parse_json,
                ): section
                for section in undercovered
            }
            for fut in as_completed(future_preds):
                section = future_preds[fut]
                try:
                    predictions = fut.result()
                except Exception:
                    predictions = []
                if not predictions:
                    continue
                # Embed predictions + section sentences
                section_sentences = sentence_split_fn(section["text"])
                if not section_sentences:
                    continue
                section_vecs = embed_fn(section_sentences)
                pred_vecs = embed_fn(predictions)
                if section_vecs.size == 0 or pred_vecs.size == 0:
                    continue
                # Find best section sentence per prediction
                # (cosine similarity, since vectors are L2-normalized)
                sims = pred_vecs @ section_vecs.T  # (P, S)
                # For each prediction, take top-2 sentences above 0.55
                threshold = 0.55
                hits: list[str] = []
                for p_idx in range(len(predictions)):
                    pred_sims = sims[p_idx]
                    top_idx = np.argsort(-pred_sims)[:3]
                    for s_idx in top_idx:
                        if pred_sims[s_idx] >= threshold:
                            hits.append(section_sentences[s_idx])
                # Dedup hits
                unique_hits = list(dict.fromkeys(hits))
                logger.info(
                    "SICRL %s section '%s': %d preds -> %d hits",
                    entity_type, section["title"][:40],
                    len(predictions), len(unique_hits),
                )
                # Canonicalize each hit
                for hit in unique_hits:
                    outcome = canonicalize_fn(hit, entity_type)
                    if not outcome:
                        continue
                    canon = (outcome.get("canonical")
                             or outcome.get("name")
                             or outcome.get("canonical_name") or "")
                    sig = re.sub(r"\s+", " ", str(canon).lower()).strip()[:120]
                    if not sig or sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                    outcome["_source_sentence"] = hit
                    outcome["_source_artifact_id"] = aid
                    outcome["_via"] = "sicrl"
                    outcome["_section"] = section["title"]
                    augmented.append(outcome)
                    iteration_added += 1

    logger.info(
        "SICRL %s: added %d new items (started %d, ended %d)",
        entity_type, iteration_added,
        len(first_pass_items), len(augmented),
    )
    return augmented


__all__ = [
    "detect_sections",
    "find_containing_section",
    "predict_counterfactual_items",
    "is_undercovered",
    "run_sicrl",
]
