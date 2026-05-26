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
        "stakeholders": _build_excerpt_for_stakeholders(by_artifact),
        "milestones": _build_excerpt_for_milestones(by_artifact),
        "requirements": _build_excerpt_for_requirements(by_artifact),
        "site_clusters": _build_excerpt_for_site_clusters(by_artifact),
    }

    # Dispatch 5 calls in parallel. ThreadPoolExecutor is fine here:
    # the LLM HTTP call is I/O-bound, so GIL doesn't bottleneck.
    parallel = int(os.environ.get("SOWSMITH_LLM_PARALLEL", str(DEFAULT_PARALLEL)))
    calls: dict[str, Callable[[], Any]] = {
        "customer": lambda: _extract_customer(excerpts["customer"]),
        "stakeholders": lambda: _extract_stakeholders(excerpts["stakeholders"]),
        "milestones": lambda: _extract_milestones(excerpts["milestones"]),
        "requirements": lambda: _extract_requirements(excerpts["requirements"]),
        "site_clusters": lambda: _extract_site_clusters(excerpts["site_clusters"]),
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
    return results


def _empty_result() -> dict[str, Any]:
    return {
        "customer": None,
        "stakeholders": [],
        "milestones": [],
        "requirements": [],
        "site_clusters": [],
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


def _extract_stakeholders(docs_excerpt: str) -> list[dict[str, Any]]:
    if not docs_excerpt:
        return []
    prompt = f"""Identify PEOPLE (stakeholders) named in this bid package.

Include both customer-side and vendor-side individuals: project
managers, technical leads, approvers, escort owners, signatories,
points of contact, etc.

INCLUDE:
- Real human names with optional role / email / phone
- Customer's project sponsor, PM, technical lead
- Vendor's PM, account exec, technical lead

EXCLUDE (these are NOT people):
- Field labels / column headers ("Access Constraint", "Ending Number",
  "Upload Destination", "Tag Prefix", "Asset Type", "Owner")
- Role-only mentions with no name ("the PM", "the architect", "the bidder")
- Department names ("IT Department", "Procurement Office")
- Job titles alone with no person attached
- Generic terms ("contractor", "vendor", "customer", "the team")

{_OUTPUT_RULES}

DOCUMENTS:

{docs_excerpt}

OUTPUT (array of objects):
{{"stakeholders": [
  {{"name": "<Full Name>", "role": "<role or null>", "email": "<email or null>", "phone": "<phone or null>"}},
  ...
]}}

If no real people are named, return: {{"stakeholders": []}}

/no_think"""
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
    if not docs_excerpt:
        return []
    prompt = f"""Identify REQUIREMENTS (what the customer requires the contractor to do).

INCLUDE:
- "Shall" / "must" / "required" clauses from the SOW
- SLAs and performance targets (uptime %, response times)
- Compliance requirements (NFPA, HIPAA, PCI, IEEE, ISO standards)
- Acceptance criteria (functional, performance, security)
- Deliverables (documentation, test reports, training)
- Security requirements (badge, escort, audit)

EXCLUDE:
- Boilerplate ("contractor will comply with applicable laws")
- Vendor-side aspirational statements ("we strive to provide...")
- Pricing terms (those belong elsewhere)
- Project metadata (deal ID, packet version, etc.)

Paraphrase each requirement to ONE concise sentence (≤ 25 words).

{_OUTPUT_RULES}

DOCUMENTS:

{docs_excerpt}

OUTPUT (array of objects):
{{"requirements": [
  {{"text": "<requirement, ≤25 words>", "category": "<sla|compliance|performance|security|deliverable|acceptance|other>"}},
  ...
]}}

If no real requirements found, return: {{"requirements": []}}

/no_think"""
    text = _call_ollama(prompt, max_tokens=2048)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return []
    return _normalize_objects(
        obj.get("requirements"), ("text", "category")
    )


def _extract_site_clusters(docs_excerpt: str) -> list[dict[str, Any]]:
    if not docs_excerpt:
        return []
    prompt = f"""Identify PHYSICAL SITES grouped into clusters.

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
    text = _call_ollama(prompt, max_tokens=2048)
    obj = _parse_json_object(text)
    if not isinstance(obj, dict):
        return []
    return _normalize_site_clusters(obj.get("site_clusters"))


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


def _normalize_objects(
    items: Any, fields: tuple[str, ...], *, is_stakeholder: bool = False
) -> list[dict[str, Any]]:
    """Coerce list of objects to uniform shape; drop malformed.

    For stakeholders, also drops names that look like field labels.
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
        if is_stakeholder and _is_likely_field_label(str(first_value)):
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
    name, NOT a real person.

    Pipeline:
      1. Strip leading articles ("the ", "a ", "an "), repeated.
      2. Exact phrase match against the denylist.
      3. Single-word matching the tail-word denylist.
      4. Trailing-word matching the tail-word denylist (catches
         "Liability Insurance", "Purchase Order", etc.).
      5. ANY org-keyword token present (catches "Hood County
         Emergency", "County Commissioners", "U.S. Postal", etc.).
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
    # ANY org-keyword present → not a person. Catches "Hood County
    # Emergency", "County Commissioners", "U.S. Postal Service".
    # Accept the rare false positive (a real person named "Sherry
    # Court") to keep org names out of the stakeholder list.
    if any(t in _ORG_TOKENS for t in tokens):
        return True
    return False


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
]
