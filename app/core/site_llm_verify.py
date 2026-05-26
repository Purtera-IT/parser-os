"""LLM-based site catalog verification (opt-in, Ollama-backed).

When ``SOWSMITH_SITE_LLM_VERIFY=1`` is set, the site catalog built by
``find_authoritative_site_phrases`` (in site_detection.py) gets a
final pass through a small Ollama LLM to clean residual false
positives that survived the deterministic cross-doc validation —
e.g. famous landmarks mentioned in spec boilerplate, non-scope city
names, sentence-fragment captures.

Configuration (env vars, all optional):
  OLLAMA_HOST        — http://HOST:PORT (default: http://100.114.102.122:11434)
  OLLAMA_MODEL       — model name (default: qwen3:14b)
  SOWSMITH_LLM_TIMEOUT — seconds per call (default: 90)

The verifier is intentionally fail-safe: any HTTP error, timeout, or
malformed model response causes the catalog to pass through
unchanged. No API key required — Ollama runs on a private tailnet
machine.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

DEFAULT_HOST = "http://100.114.102.122:11434"
DEFAULT_MODEL = "qwen3:14b"
DEFAULT_TIMEOUT = 90


_CHUNK_SIZE = 40  # candidates per LLM call


def verify_sites_with_llm(catalog: set[str], atoms: list[Any]) -> set[str]:
    """Filter the candidate site catalog via Ollama LLM.

    Sends the catalog + representative doc excerpts to an LLM. Asks
    the model to label each candidate as KEEP (real project site) or
    DROP. Catalogs larger than ``_CHUNK_SIZE`` are split into multiple
    LLM calls so the response stays bounded and parses reliably.

    Args:
      catalog: set of normalized site phrases. Output of the
        deterministic catalog builder.
      atoms: full atom list — used to extract project context.

    Returns:
      Subset of ``catalog`` the LLM kept. On any failure for a chunk,
      the chunk passes through unchanged (fail-safe).
    """
    if not catalog:
        return catalog

    docs_excerpt = _build_doc_excerpt(atoms, max_per_doc=600, max_total=3000)
    if not docs_excerpt:
        return catalog

    items = sorted(catalog)
    if len(items) <= _CHUNK_SIZE:
        return _verify_chunk(items, docs_excerpt) or catalog

    # Chunk large catalogs and union the keeps.
    kept_total: set[str] = set()
    any_succeeded = False
    for i in range(0, len(items), _CHUNK_SIZE):
        chunk = items[i:i + _CHUNK_SIZE]
        result = _verify_chunk(chunk, docs_excerpt)
        if result is None:
            # Chunk failed — fail-safe: keep all items in this chunk
            kept_total.update(chunk)
        else:
            any_succeeded = True
            kept_total.update(result)
    # If EVERY chunk failed, fall back to full passthrough so we
    # don't accidentally produce an empty catalog from comms errors.
    if not any_succeeded:
        return catalog
    return kept_total


def _verify_chunk(items: list[str], docs_excerpt: str) -> set[str] | None:
    """Send one chunk of candidates to the LLM. Returns the kept set,
    or ``None`` if the call failed / the response was unparseable."""
    if not items:
        return set()
    prompt = _build_prompt(items, docs_excerpt)
    response_text = _call_ollama(prompt)
    if not response_text:
        return None
    return _parse_keep_drop_response(response_text, items)


# ─────────── prompt construction ───────────


def _build_doc_excerpt(atoms: list[Any], *, max_per_doc: int, max_total: int) -> str:
    """Concatenate first-N-chars of each doc's body text into a context blob."""
    by_artifact: dict[str, list[str]] = {}
    for atom in atoms or []:
        aid = getattr(atom, "artifact_id", None)
        if not aid:
            continue
        raw = getattr(atom, "raw_text", None) or ""
        if not raw:
            continue
        by_artifact.setdefault(aid, []).append(raw)
    if not by_artifact:
        return ""
    chunks: list[str] = []
    running_total = 0
    for aid, texts in sorted(by_artifact.items()):
        # Stitch each artifact's first ~max_per_doc chars together
        stitched = " ".join(texts)
        if len(stitched) > max_per_doc:
            stitched = stitched[:max_per_doc]
        # Get a friendly artifact name from the atom source_refs
        fname = None
        for atom in atoms:
            if getattr(atom, "artifact_id", None) == aid and getattr(atom, "source_refs", None):
                fname = getattr(atom.source_refs[0], "filename", None)
                if fname:
                    break
        chunks.append(f"--- {fname or aid} ---\n{stitched}")
        running_total += len(stitched)
        if running_total >= max_total:
            break
    return "\n\n".join(chunks)


def _build_prompt(items: list[str] | set[str], doc_excerpt: str) -> str:
    """Compose the audit prompt.

    Critical design points:
      - List candidates as a numbered list so the model can refer to
        them precisely
      - Request strict JSON output (no commentary) for deterministic
        parsing
      - Give explicit DROP examples (standards bodies, landmarks,
        sentence fragments) so the model knows what to filter
    """
    if isinstance(items, set):
        items = sorted(items)
    numbered = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(items))

    return f"""You are auditing a candidate site list extracted from a managed-services bid package.

The bid package is excerpted below (multiple source documents):

{doc_excerpt}

Here are the candidate site names that were detected:

{numbered}

For each candidate, label whether it is a real PROJECT SITE (a building, school, hospital, facility, or location in scope for THIS specific bid) or a FALSE POSITIVE.

DROP these (NOT project sites):
  - Standards bodies / industry associations (ANSI, ASHRAE, NFPA, "Building Officials Council", "American Wood Preservers")
  - Vendor / product names ("Cisco Systems", "Genetec Security Center")
  - Famous landmarks mentioned only as reference examples ("Chrysler Building" in a Muskegon spec, "Yeon Building")
  - Cities or places mentioned as spec references, NOT in project scope
  - Sentence fragments captured by regex ("consumption annual energy costs neptune municipal building", "performance bonds los medanos college")
  - Generic nouns without a specific name ("medical center", "elementary school", "academy")
  - Spec-doc section headings ("pre bid meeting location", "purchasing office")
  - Street addresses misclassified as sites ("Heck Ave", "Corlies Avenue")

KEEP these (real project sites):
  - Named schools / hospitals / buildings / offices / branches that this bid actually serves
  - Real facility names with proper nouns + facility tail ("Wesley School", "Brookdale Community College")
  - Customer's headquarters / main location

Respond with ONLY a JSON object on a single line. No commentary. No thinking. No markdown.

Format: {{"keep": [<numbers>], "drop": [<numbers>]}}

Example: {{"keep": [1, 3, 7], "drop": [2, 4, 5, 6, 8]}}

/no_think"""


# ─────────── Ollama HTTP call ───────────


def _call_ollama(prompt: str) -> str:
    """POST to /api/generate. Returns the raw response.text or ''."""
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    timeout = int(os.environ.get("SOWSMITH_LLM_TIMEOUT", str(DEFAULT_TIMEOUT)))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 2048,
        },
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


# ─────────── response parsing ───────────


def _parse_keep_drop_response(response_text: str, items: list[str] | set[str]) -> set[str]:
    """Extract the keep-numbers from the model's JSON response.

    The model is asked for ``{"keep": [1, 3, 7], "drop": [...]}``.
    Some models append commentary; we look for the first JSON object
    and parse that.
    """
    if isinstance(items, set):
        items = sorted(items)
    if not items:
        return set()
    # Find the first {...} block in the response
    match = re.search(r"\{[^{}]*\"keep\"\s*:\s*\[[^\]]*\][^{}]*\}", response_text, re.DOTALL)
    if not match:
        return set()
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return set()
    keep_ids = parsed.get("keep")
    if not isinstance(keep_ids, list):
        return set()
    kept: set[str] = set()
    for raw_id in keep_ids:
        try:
            idx = int(raw_id) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(items):
            kept.add(items[idx])
    return kept


__all__ = ["verify_sites_with_llm"]
