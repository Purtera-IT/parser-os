"""v45 — Zero-miss layer: five techniques to push universal recall
from ~95% toward 99%+ on text-heavy bids, and toward 99%+ on
scanned/visual bids too.

Techniques:
  1. PLIR (Page-Level Iterative Recall) — for EVERY PDF page,
     ask the LLM "given this page text + what we already found,
     what did we miss". Catches content buried in pages without
     section headings (which SICRL relies on).

  2. PM-critical vocabulary sweep — 50+ hardcoded PM-critical terms
     (insurance, bond, indemnify, terminate, payment, accept,
     deliver, warranty, sla, uptime, redundancy, escrow, ...).
     For each term: if it has 3+ mentions in raw text BUT no entity
     captures it, escalate to LLM for forced extraction.

  3. Per-page coverage gauge — for each PDF page, compute
     entities_sourced_from_page / sentences_on_page. Pages below
     threshold flagged "low coverage" → trigger PLIR.

  4. Multi-model voting (qwen3:14b + qwen3:32b) — for each candidate
     sentence, ask BOTH models; take union when they agree
     "this is a real X". Maximum recall ceiling. Slower on Mac;
     opt-in via SOWSMITH_MULTI_MODEL=1.

  5. (in exemplars.py) — exemplar audit & expansion. 10-20 more
     examples per entity type covering edge cases surfaced by the
     19-pack raw audits.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# 2 + 3. PM-CRITICAL VOCABULARY SWEEP
# ════════════════════════════════════════════════════════════════════

# 60+ terms PMs always need to know about. Each term has:
#   - regex pattern (case-insensitive)
#   - entity_type that SHOULD have an entity referencing it
#   - example exemplar sentence to push through canonicalize if missed

PM_CRITICAL_TERMS: list[dict[str, Any]] = [
    # Insurance & bonding
    {"pat": r"\binsurance\s+(?:limits?|coverage|requirements?)\b", "kind": "requirement",
     "example": "Insurance coverage and limits as specified in the contract."},
    {"pat": r"\bperformance\s+bond\b", "kind": "requirement",
     "example": "Performance bond as outlined in the bid documents."},
    {"pat": r"\bbid\s+bond\b", "kind": "requirement",
     "example": "Bid bond accompanying the proposal."},
    {"pat": r"\bworkers[’']?\s+compensation\b", "kind": "requirement",
     "example": "Workers compensation insurance at statutory minimum."},
    {"pat": r"\bgeneral\s+liability\b", "kind": "requirement",
     "example": "General liability coverage with limits as specified."},
    {"pat": r"\bumbrella\s+liability\b", "kind": "requirement",
     "example": "Umbrella liability insurance coverage."},
    {"pat": r"\bcyber\s+liability\b", "kind": "requirement",
     "example": "Cyber liability insurance coverage."},
    {"pat": r"\berrors?\s+and\s+omissions?\b", "kind": "requirement",
     "example": "Errors and omissions insurance coverage."},
    # Termination / breach
    {"pat": r"\bterminat(?:e|ion)\s+for\s+(?:default|cause|convenience)\b", "kind": "penalty",
     "example": "Termination triggered by default or cause."},
    {"pat": r"\bmaterial\s+breach\b", "kind": "penalty",
     "example": "Material breach allows immediate termination."},
    {"pat": r"\bcure\s+period\b", "kind": "penalty",
     "example": "Cure period before termination."},
    # Indemnification
    {"pat": r"\bindemnif(?:y|ication)\b", "kind": "requirement",
     "example": "Indemnification of the customer against claims."},
    {"pat": r"\bhold\s+harmless\b", "kind": "requirement",
     "example": "Hold harmless clause."},
    # Payment terms
    {"pat": r"\bnet[\s-](?:30|45|60|90)\b", "kind": "quantity",
     "example": "Net-30 payment terms apply."},
    {"pat": r"\bpayment\s+(?:terms|schedule|milestones?)\b", "kind": "requirement",
     "example": "Payment terms and milestone schedule."},
    {"pat": r"\blate\s+payment\b", "kind": "penalty",
     "example": "Late payment penalty / interest."},
    {"pat": r"\bliquidated\s+damages\b", "kind": "penalty",
     "example": "Liquidated damages per day of delay."},
    {"pat": r"\bservice\s+credit", "kind": "penalty",
     "example": "Service credit for downtime."},
    # SLA / uptime
    {"pat": r"\b9{1,5}\.\d{1,3}\s*%\s*(?:uptime|availability)\b", "kind": "quantity",
     "example": "99.999% uptime guarantee."},
    {"pat": r"\b(?:rto|rpo|mttr|mtbf)\b", "kind": "quantity",
     "example": "RTO/RPO recovery time and point objectives."},
    {"pat": r"\bsla\b", "kind": "requirement",
     "example": "SLA service level agreement."},
    # Delivery / acceptance
    {"pat": r"\bdeliverable[s]?\b", "kind": "acceptance_criteria",
     "example": "Deliverable items per phase."},
    {"pat": r"\bacceptance\s+(?:criteria|testing|test)\b", "kind": "acceptance_criteria",
     "example": "Acceptance testing criteria."},
    {"pat": r"\bsubstantial\s+completion\b", "kind": "acceptance_criteria",
     "example": "Substantial completion milestone."},
    {"pat": r"\bfinal\s+acceptance\b", "kind": "acceptance_criteria",
     "example": "Final acceptance after observation period."},
    # Warranty
    {"pat": r"\bwarranty\b", "kind": "requirement",
     "example": "Warranty terms for hardware / software / labor."},
    # Compliance / certs
    {"pat": r"\bpci[-\s]?dss\b", "kind": "certification",
     "example": "PCI-DSS Level 1 compliance."},
    {"pat": r"\bsoc\s*[12]\b", "kind": "certification",
     "example": "SOC 2 Type II certification."},
    {"pat": r"\bhipaa\b", "kind": "certification",
     "example": "HIPAA compliance."},
    {"pat": r"\bferpa\b", "kind": "certification",
     "example": "FERPA student records privacy."},
    {"pat": r"\bfedramp\b", "kind": "certification",
     "example": "FedRAMP authorization."},
    {"pat": r"\bnist\s*800-\d+\b", "kind": "certification",
     "example": "NIST 800-53 controls."},
    {"pat": r"\biso\s+\d{4,5}(?::\d{4})?\b", "kind": "certification",
     "example": "ISO 27001 information security."},
    {"pat": r"\bssae\s*1[68]\b", "kind": "certification",
     "example": "SSAE 18 SOC 1 audit."},
    {"pat": r"\busda\b", "kind": "certification",
     "example": "USDA-approved compliance."},
    {"pat": r"\bfns-?\d+\b", "kind": "certification",
     "example": "FNS form compliance."},
    {"pat": r"\btia[-\s]?568\b", "kind": "certification",
     "example": "TIA-568 cabling standard."},
    {"pat": r"\bieee\s*802\.\d{1,2}\w*\b", "kind": "certification",
     "example": "IEEE 802.11 wireless standard."},
    {"pat": r"\bnfpa\s*\d+\b", "kind": "certification",
     "example": "NFPA fire code standard."},
    # Subcontracting
    {"pat": r"\bsubcontractor\b", "kind": "requirement",
     "example": "Subcontractor identification and use."},
    {"pat": r"\bm/?wbe\b|\bminority[/\s]+women[\s-]?owned\b", "kind": "compliance_obligation",
     "example": "M/WBE participation requirements."},
    # Labor / wages
    {"pat": r"\bdavis[-\s]?bacon\b", "kind": "compliance_obligation",
     "example": "Davis-Bacon prevailing wage rates."},
    {"pat": r"\bprevailing\s+wage\b", "kind": "compliance_obligation",
     "example": "Prevailing wage rates apply."},
    {"pat": r"\bfair\s+labor\s+standards\b", "kind": "compliance_obligation",
     "example": "Fair Labor Standards Act compliance."},
    # Equal opportunity
    {"pat": r"\bequal\s+(?:employment|opportunity)\b", "kind": "compliance_obligation",
     "example": "Equal employment opportunity provisions."},
    {"pat": r"\bnon[-\s]?discriminat", "kind": "compliance_obligation",
     "example": "Non-discrimination clause."},
    {"pat": r"\bada\s+compliance\b|\bamericans\s+with\s+disabilities", "kind": "compliance_obligation",
     "example": "ADA / Americans with Disabilities Act compliance."},
    # Procurement / FAR
    {"pat": r"\bfar\s+part\s+52\b", "kind": "compliance_obligation",
     "example": "Federal Acquisition Regulation Part 52 clauses."},
    # Data security
    {"pat": r"\bdata\s+(?:breach|security|encryption)\b", "kind": "requirement",
     "example": "Data breach / security / encryption requirements."},
    {"pat": r"\bencryption\s+at\s+rest\b", "kind": "requirement",
     "example": "Encryption at rest."},
    {"pat": r"\btls\s+1\.[23]\b", "kind": "requirement",
     "example": "TLS 1.2/1.3 encryption."},
    # Risk / dependencies
    {"pat": r"\bsingle\s+point\s+of\s+failure\b", "kind": "risk",
     "example": "Single point of failure risk."},
    {"pat": r"\bredundancy\b|\bfailover\b|\bhigh\s+availability\b", "kind": "risk",
     "example": "Redundancy / failover / high availability."},
    {"pat": r"\bbusiness\s+continuity\b", "kind": "risk",
     "example": "Business continuity plan."},
    {"pat": r"\bdisaster\s+recovery\b", "kind": "risk",
     "example": "Disaster recovery procedures."},
    # Schedule
    {"pat": r"\bgo[-\s]?live\b", "kind": "milestone",
     "example": "Go-live date / cutover."},
    {"pat": r"\bcutover\b", "kind": "milestone",
     "example": "Cutover window."},
    {"pat": r"\bphase\s+\d\b", "kind": "milestone",
     "example": "Phase X timeline."},
    # Confidentiality
    {"pat": r"\bconfidential(?:ity)?\b", "kind": "requirement",
     "example": "Confidentiality of customer data."},
    {"pat": r"\bnon[-\s]?disclosure\b|\bnda\b", "kind": "requirement",
     "example": "Non-disclosure agreement."},
    # Audit
    {"pat": r"\baudit\s+(?:report|trail|log)\b", "kind": "requirement",
     "example": "Audit reporting / trail / log."},
    # Bonding
    {"pat": r"\bescrow\b", "kind": "requirement",
     "example": "Escrow account terms."},
]


def pm_vocab_sweep(
    raw_text: str,
    atoms: list[Any],
    multi_result: dict[str, Any],
    *,
    canonicalize_fn: Callable[[str, str], dict[str, Any] | None] | None = None,
    mention_threshold: int = 3,
) -> list[dict[str, Any]]:
    """Sweep raw text for PM-critical terms. If a term has ≥
    mention_threshold mentions but no entity covers it, escalate to
    LLM for forced extraction.

    Returns list of force-injected items by entity type.
    """
    if not raw_text or not multi_result:
        return []

    # Build a quick lookup of what's already covered in multi_result
    covered_texts: set[str] = set()
    for key, value in multi_result.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                for field in ("text", "canonical", "name", "description",
                              "criterion", "obligation", "canonical_name"):
                    v = item.get(field)
                    if isinstance(v, str) and v:
                        covered_texts.add(v.lower()[:80])

    raw_lower = raw_text.lower()
    missed: list[dict[str, Any]] = []

    for term_def in PM_CRITICAL_TERMS:
        pat = term_def["pat"]
        kind = term_def["kind"]
        example = term_def["example"]
        matches = re.findall(pat, raw_lower)
        if len(matches) < mention_threshold:
            continue
        # Is the term already covered? Look for the term root in any
        # existing covered text.
        # Extract a simple root from the regex pattern for lookup
        root = re.sub(r"[^a-z]", "", matches[0])[:20] if matches else ""
        if root and any(root in ct for ct in covered_texts):
            continue
        # Not covered — escalate
        if canonicalize_fn is not None:
            outcome = canonicalize_fn(example, kind)
            if outcome:
                outcome["_via"] = "pm_vocab_sweep"
                outcome["_term_pattern"] = pat
                outcome["_mentions"] = len(matches)
                missed.append({"kind": kind, "outcome": outcome})
        else:
            # Fallback — just record the example as a forced item
            missed.append({
                "kind": kind,
                "outcome": {"keep": True, "canonical": example,
                            "_via": "pm_vocab_sweep_no_canonicalize",
                            "_term_pattern": pat,
                            "_mentions": len(matches)},
            })
    return missed


# ════════════════════════════════════════════════════════════════════
# 3. PER-PAGE COVERAGE GAUGE
# ════════════════════════════════════════════════════════════════════


def compute_page_coverage(
    atoms: list[Any], raw_text_by_page: dict[tuple[str, int], str],
) -> dict[tuple[str, int], dict[str, Any]]:
    """For each (pdf_path, page_num), compute:
      - sentences_on_page
      - atoms_from_page
      - entity_keys_from_page (unique)
      - coverage_ratio = entity_keys / sentences

    Returns {(pdf_path, page_num): {sentences, atoms, keys, ratio}}.
    """
    out: dict[tuple[str, int], dict[str, Any]] = {}
    # Group atoms by page
    page_atoms: dict[tuple[str, int], list[Any]] = {}
    for atom in atoms:
        try:
            refs = getattr(atom, "source_refs", None) or []
            if not refs:
                continue
            ref = refs[0]
            fname = getattr(ref, "filename", None) or ""
            loc = getattr(ref, "locator", None) or {}
            page = loc.get("page", 0) if isinstance(loc, dict) else 0
            if fname:
                page_atoms.setdefault((fname, page), []).append(atom)
        except Exception:
            continue

    for (pdf, page), atms in page_atoms.items():
        keys = set()
        for a in atms:
            for k in (a.entity_keys or []):
                keys.add(k)
        page_text = raw_text_by_page.get((pdf, page), "")
        sentences = len(re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n\n+",
                                  page_text)) if page_text else 0
        ratio = len(keys) / max(sentences, 1) if sentences else 0.0
        out[(pdf, page)] = {
            "sentences": sentences,
            "atoms": len(atms),
            "keys": len(keys),
            "ratio": round(ratio, 3),
        }
    return out


def find_low_coverage_pages(
    coverage: dict[tuple[str, int], dict[str, Any]],
    *,
    min_sentences: int = 10,
    max_ratio: float = 0.05,
) -> list[tuple[str, int]]:
    """Return pages with ≥min_sentences text but ratio < max_ratio.
    These likely have meaningful content the pipeline missed."""
    out: list[tuple[str, int]] = []
    for (pdf, page), stats in coverage.items():
        if stats["sentences"] >= min_sentences and stats["ratio"] < max_ratio:
            out.append((pdf, page))
    return out


# ════════════════════════════════════════════════════════════════════
# 1. PLIR — PAGE-LEVEL ITERATIVE RECALL
# ════════════════════════════════════════════════════════════════════


_PLIR_PROMPT = """You are doing a final quality-control pass on entity
extraction from a single page of a bid / RFP document.

PAGE TEXT:
{page_text}

WHAT WE ALREADY EXTRACTED from this page:
{prior_items}

YOUR JOB: identify anything PM-critical on this page that the previous
extraction MISSED. PM-critical means: requirements (shall/must/agrees),
named people, specific sites, dollar amounts, dates, deliverables,
penalties, certifications, risks, acceptance criteria, compliance
obligations.

Be honest — only flag REAL omissions. Do not duplicate items already
extracted. If we got everything, return an empty list.

OUTPUT exactly one JSON object on one line:
  {{"missed": [{{"kind": "<entity_type>", "text": "<the missed item>"}}]}}

If nothing was missed: {{"missed": []}}

/no_think
"""


def page_level_iterative_recall(
    raw_text_by_page: dict[tuple[str, int], str],
    coverage: dict[tuple[str, int], dict[str, Any]],
    page_extractions: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    llm_call: Callable[[str, int], str],
    parse_json: Callable[[str], Any],
    max_pages: int = 30,
    parallel: int = 4,
) -> list[dict[str, Any]]:
    """For pages with low coverage OR every page (if max_pages allows),
    ask the LLM 'what did we miss'. Returns list of forced-additions
    to inject downstream.

    cost: 1 LLM call per page × low-coverage pages, capped at max_pages.
    """
    if os.environ.get("SOWSMITH_PLIR_DISABLE"):
        return []
    # Pick pages to scan — low-coverage first, then high-sentence-count
    candidates = sorted(
        coverage.items(),
        key=lambda kv: (kv[1]["ratio"], -kv[1]["sentences"]),
    )
    pages_to_scan: list[tuple[str, int]] = []
    for (pdf, page), stats in candidates:
        if stats["sentences"] < 5:
            continue  # tiny page
        pages_to_scan.append((pdf, page))
        if len(pages_to_scan) >= max_pages:
            break
    if not pages_to_scan:
        return []

    def scan(pdf_page):
        pdf, page = pdf_page
        page_text = raw_text_by_page.get(pdf_page, "")
        if not page_text or len(page_text) < 100:
            return []
        prior = page_extractions.get(pdf_page, [])
        prior_repr = " | ".join(
            f"{p.get('kind', '?')}: {str(p.get('text', ''))[:60]}"
            for p in prior[:10]
        ) or "(none yet)"
        # Truncate page text to fit token budget
        prompt = _PLIR_PROMPT.format(
            page_text=page_text[:3000],
            prior_items=prior_repr[:800],
        )
        text = llm_call(prompt, 1024)
        obj = parse_json(text)
        if not isinstance(obj, dict):
            return []
        missed = obj.get("missed", [])
        if not isinstance(missed, list):
            return []
        out = []
        for m in missed:
            if not isinstance(m, dict):
                continue
            kind = m.get("kind", "")
            txt = m.get("text", "")
            if kind and txt:
                out.append({
                    "kind": kind, "text": txt,
                    "_via": "plir", "_pdf": pdf, "_page": page,
                })
        return out

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(scan, p) for p in pages_to_scan]
        for fut in as_completed(futures):
            try:
                items = fut.result()
            except Exception:
                items = []
            results.extend(items)
    return results


# ════════════════════════════════════════════════════════════════════
# 4. MULTI-MODEL VOTING (qwen3:14b + qwen3:32b)
# ════════════════════════════════════════════════════════════════════


def dual_model_canonicalize(
    sentence: str, entity_type: str,
    *,
    canonicalize_fn_14b: Callable[[str, str], dict[str, Any] | None],
    canonicalize_fn_32b: Callable[[str, str], dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Run BOTH 14b and 32b canonicalize on the same sentence; take
    union of keeps. If both agree keep=true, use 32b's canonical form
    (sharper). If only one agrees keep=true, use that one.

    Cost: 2x per canonicalize call. Opt-in via SOWSMITH_MULTI_MODEL=1.
    """
    if not os.environ.get("SOWSMITH_MULTI_MODEL"):
        return None
    r14 = canonicalize_fn_14b(sentence, entity_type)
    r32 = canonicalize_fn_32b(sentence, entity_type)
    if r14 and r32:
        # Both agree — use 32b's output (more precise canonical form)
        return r32
    if r32 and not r14:
        # Only 32b sees it — risky, but bigger model is usually right
        return r32
    if r14 and not r32:
        # 14b sees it but 32b doesn't — keep 14b's output
        return r14
    return None


__all__ = [
    "PM_CRITICAL_TERMS",
    "pm_vocab_sweep",
    "compute_page_coverage",
    "find_low_coverage_pages",
    "page_level_iterative_recall",
    "dual_model_canonicalize",
]
