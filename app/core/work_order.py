"""Document-grain reassembly — state the job before anything classifies a sentence.

The per-atom pipeline shreds a deal into thousands of spans and then asks, of each
span in isolation, "is this a task?". That question has no good answer at the span
level: "we'll need to get into the closets after 6" is not a task, and "11 APs" is
not a task, but together they are most of a work order. Barton Malow came back as
1,173 atoms and zero usable work lines.

This stage asks the question one grain coarser — per DOCUMENT, then per DEAL:

1. **Relevance.** Every document is judged ``about_this_job`` or
   ``relationship_or_other`` through :func:`app.core.decide.decide`, so a PM can
   correct the judgment and the correction applies to the next deal that looks
   like it. The judged text is the DOCUMENT (title + body); the deal name rides in
   ``context``. That split matters: a lesson keyed on the deal name matches every
   document in the deal and mutes the seam (the document_job_scope regression).
   Abstention keeps the document — losing a relevant document costs a work line,
   keeping an irrelevant one costs the extractor a few tokens.

2. **Reassembly.** What survives is read together, once, and returned as a short
   structured work order: the lines of work, their objects, counts and units, plus
   the deal-level facts that decide how the work is priced (how many sites, whether
   it is after hours, whether anyone is on site, who supplies the equipment).

3. **Minting.** Each work line becomes a ``task`` atom so the existing Deal Kit
   consumers (site anchoring, tier classification, the kit builder) see it without
   learning a second path.

Provenance is inherited, never invented. A minted task is a ``copy.deepcopy`` of the
real atom that best supports it — the precedent set by
:func:`app.core.task_atom_backfill.backfill_quote_task_atoms` — so its
``artifact_id`` and ``source_refs`` name a document that is actually in the compile.
Minting a synthetic ``SourceRef`` instead would name no real artifact and
:func:`app.core.atom_type_sanity.cap_authority_to_source` would cap the atom on its
first pass.

Every minted atom lands at ``machine_extractor`` / ``needs_review``: this stage
summarises, and a summary is a machine's reading of the documents until a PM says
otherwise. It never raises authority and never deletes an atom it did not mint.

Off by default; ``SOWSMITH_WORK_ORDER=1`` turns it on.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

RELATION = "work_order_relevance"
CANDIDATES = ["about_this_job", "relationship_or_other"]
INSTRUCTION = (
    "Is this document about the specific job being quoted — the work to be "
    "performed, its scope, sites, equipment, schedule or price — or about "
    "something else: the business relationship, partnership strategy, the "
    "company's own services, hiring, or a different job for the same people?"
)

WORK_ORDER_PROMPT = """You are reading everything known about one job BEFORE it was quoted.

DEAL: {deal}
WHAT THE DOCUMENTS SAY:
{body}

State the job as a short work order. Return ONLY JSON:
{{"work_lines": [{{"work": "<one short line: the action and the thing>",
                  "object": "<the thing, singular>",
                  "count": <number or null>,
                  "unit": "<what the count counts, singular, in the documents' own words, or null>"}}],
 "site_count": <number or null>,
 "after_hours": <true|false>,
 "no_onsite_hands": <true|false>,
 "customer_supplies_equipment": <true|false>,
 "one_line_summary": "<the whole job in under 15 words>"}}

Only what the documents actually say. Do not invent a count the documents do not
state. If the job is not stated, return empty work_lines."""

# A work line is a summary of a document, so it cannot be a verbatim span. These
# are the shape limits that keep a summary from becoming a paragraph.
_MIN_LINE_CHARS = 8
_MAX_LINE_CHARS = 160


def _env_flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def enabled() -> bool:
    """On by default (PUR-25). Kill switch: SOWSMITH_WORK_ORDER=0 (or false/no/off)."""
    return _env_flag("SOWSMITH_WORK_ORDER", "1")


def _text(atom: Any) -> str:
    raw = getattr(atom, "raw_text", None) or getattr(atom, "text", "") or ""
    return re.sub(r"\s+", " ", str(raw)).strip()


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "")


def _filename(atom: Any) -> str:
    for attr in ("source_filename", "filename"):
        v = getattr(atom, attr, None)
        if v:
            return str(v)
    for ref in (getattr(atom, "source_refs", None) or []):
        fn = getattr(ref, "filename", None)
        if fn is None and isinstance(ref, dict):
            fn = ref.get("filename")
        if fn:
            return str(fn)
    return str(getattr(atom, "artifact_id", "") or "?")


def _locator_key(atom: Any) -> tuple:
    """Where in its document an atom sits, as a totally ordered tuple.

    Upstream extractors run in thread pools, so the order atoms arrive in is not
    guaranteed between compiles. Reading order is recovered from the source
    locator (page/line/row/char offsets) and the atom id breaks every tie, so the
    same set of atoms always yields the same document body.
    """
    parts: list[tuple] = []
    refs = getattr(atom, "source_refs", None) or []
    if refs:
        ref = refs[0]
        loc = getattr(ref, "locator", None)
        if loc is None and isinstance(ref, dict):
            loc = ref.get("locator")
        if isinstance(loc, dict):
            for k in sorted(loc):
                v = loc[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    parts.append((k, 1, 0.0, str(v)))
                else:
                    parts.append((k, 0, float(v), ""))
    return (tuple(parts), str(getattr(atom, "id", "") or ""), _text(atom))


def group_by_document(atoms: list[Any]) -> dict[str, list[Any]]:
    """Atoms bucketed by the document they came from, in a stable order.

    Buckets are keyed by filename and iterate in filename order; atoms inside a
    bucket are sorted by :func:`_locator_key`. Neither depends on input order.
    """
    by: dict[str, list[Any]] = {}
    for a in atoms:
        by.setdefault(_filename(a), []).append(a)
    return {fn: sorted(by[fn], key=_locator_key) for fn in sorted(by)}


def rank_documents(by_doc: dict[str, list[Any]], top_n: int) -> list[tuple[str, list[Any]]]:
    """Largest documents first; the filename breaks ties so truncation is stable."""
    return sorted(by_doc.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top_n]


def _body(doc_atoms: list[Any], limit: int) -> str:
    return " ".join(t for t in (_text(a) for a in doc_atoms) if t)[:limit]


def judge_document(
    title: str,
    body: str,
    *,
    deal_name: str = "",
    project_id: str = "",
) -> tuple[bool, str]:
    """Is this document about the job being quoted? Returns (keep, source).

    The judged TEXT is the document. The deal rides in ``context`` so a learned
    lesson keys on the document's own wording and not on the deal it happened to
    be taught in. Abstention keeps the document.
    """
    if not body or len(body) < _env_int("SOWSMITH_WORK_ORDER_MIN_BODY", 60):
        return False, "too_short"
    try:
        from app.core.decide import DecisionScope, decide

        d = decide(
            RELATION,
            f"DOCUMENT: {title}\n{body}"[:600],
            CANDIDATES,
            instruction=INSTRUCTION,
            context=f"DEAL: {deal_name}"[:1200],
            scope=DecisionScope(deal_id=str(project_id or "")),
            exclude_created_by=("teacher",),
        )
    except Exception as exc:  # pragma: no cover - the seam must never break a compile
        logger.warning("work_order relevance judge failed: %s", exc)
        return True, "error"
    if d is None or d.verdict is None:
        return True, "abstain"
    return d.verdict == "about_this_job", getattr(d, "source", "") or "decided"


def _parse_work_order(raw: str) -> dict:
    """The first JSON object in the reply that looks like a work order.

    A greedy ``{...}`` regex spans from the first brace to the LAST one, so a
    reply with two objects, or prose containing a brace after the JSON, failed to
    parse and silently minted nothing on that run only. Decode object by object.
    """
    text = raw or ""
    dec = json.JSONDecoder()
    first: dict | None = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "work_lines" in obj:
            return obj
        if first is None:
            first = obj
    return first or {}


# ── extraction cache ─────────────────────────────────────────────────
# Mirrors the vision cache in app.core.llm_client: with
# ``SOWSMITH_WORK_ORDER_CACHE_DB`` set, the extraction reply is stored under
# sha256(model + prompt + max_tokens) so identical inputs replay identically.
# Default-off; any cache error falls through to a live call.
_CACHE_CONN = None
_CACHE_PATH = None


def _cache_conn():
    global _CACHE_CONN, _CACHE_PATH
    path = os.environ.get("SOWSMITH_WORK_ORDER_CACHE_DB")
    if not path:
        return None
    if _CACHE_CONN is not None and _CACHE_PATH == path:
        return _CACHE_CONN
    try:
        import sqlite3

        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS work_order_cache "
            "(k TEXT PRIMARY KEY, model TEXT, reply TEXT)"
        )
        conn.commit()
        _CACHE_CONN, _CACHE_PATH = conn, path
        return conn
    except Exception:
        return None


def cache_key(model: str, prompt: str, max_tokens: int) -> str:
    h = hashlib.sha256()
    for part in (model or "", prompt or "", str(int(max_tokens))):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(key: str) -> str | None:
    conn = _cache_conn()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT reply FROM work_order_cache WHERE k=?", (key,)
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _cache_put(key: str, model: str, reply: str) -> None:
    conn = _cache_conn()
    if conn is None or not reply:
        return
    try:
        conn.execute(
            "INSERT OR REPLACE INTO work_order_cache (k, model, reply) VALUES (?,?,?)",
            (key, model, reply),
        )
        conn.commit()
    except Exception:
        pass


def extract_work_order(deal_name: str, kept: list[tuple[str, str]]) -> dict:
    """Read the surviving documents together and state the job once."""
    if not kept:
        return {}
    per_doc = _env_int("SOWSMITH_WORK_ORDER_DOC_CHARS", 2600)
    total = _env_int("SOWSMITH_WORK_ORDER_TOTAL_CHARS", 14000)
    body = "\n\n".join(f"[{fn[:60]}] {tx[:per_doc]}" for fn, tx in kept)[:total]
    prompt = WORK_ORDER_PROMPT.format(deal=deal_name[:70], body=body)
    max_tokens = _env_int("SOWSMITH_WORK_ORDER_MAX_TOKENS", 700)
    try:
        from app.core import llm_client

        model = llm_client.teacher_model()
        key = cache_key(model, prompt, max_tokens)
        raw = _cache_get(key)
        if raw is None:
            raw = llm_client.complete(
                prompt,
                max_tokens=max_tokens,
                seed=_env_int("SOWSMITH_WORK_ORDER_SEED", 0),
            ) or ""
            _cache_put(key, model, raw)
    except Exception as exc:  # pragma: no cover
        logger.warning("work_order extraction failed: %s", exc)
        return {}
    return _parse_work_order(raw)


def _line_label(line: dict) -> str:
    """The work line as one sentence, or '' if the model returned something unusable."""
    work = re.sub(r"\s+", " ", str(line.get("work") or "")).strip(" .;-–—*•·")
    if not (_MIN_LINE_CHARS <= len(work) <= _MAX_LINE_CHARS):
        return ""
    # A line that is only a number, or only punctuation, is not a statement of work.
    if not re.search(r"[A-Za-z]{3}", work):
        return ""
    return work


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


def _authority_rank(atom: Any) -> int:
    """This atom's standing on the shared authority ladder (0 when unknown)."""
    try:
        from app.core.authority import AUTHORITY_RANKS

        return int(AUTHORITY_RANKS.get(getattr(atom, "authority_class", None), 0))
    except Exception:
        return 0


def pick_support(line_text: str, candidates: list[Any]) -> Any | None:
    """The real atom whose wording best supports this work line.

    Overlap decides which document sentence the line was summarised from; authority
    breaks ties. This picks PROVENANCE, not meaning — the meaning came from the
    judged documents. When nothing overlaps, the highest-authority atom in the kept
    set still names a real artifact, which is what the summary is actually sourced
    from.
    """
    if not candidates:
        return None
    want = _tokens(line_text)
    best, best_key = None, (-1.0, -1, 0)
    for a in candidates:
        t = _text(a)
        if not t:
            continue
        have = _tokens(t)
        overlap = len(want & have) / len(want) if want else 0.0
        key = (round(overlap, 3), _authority_rank(a), -len(t))
        # Ties go to the smaller atom id, not to whichever came first.
        if key > best_key or (
            key == best_key
            and best is not None
            and str(getattr(a, "id", "")) < str(getattr(best, "id", ""))
        ):
            best, best_key = a, key
    return best


def mint_work_line_atoms(
    atoms: list[Any],
    work_order: dict,
    support_pool: list[Any],
    *,
    project_id: str,
) -> tuple[list[Any], int]:
    """Turn each work line into a task atom deepcopied from its supporting atom."""
    lines = work_order.get("work_lines")
    if not isinstance(lines, list) or not lines:
        return atoms, 0
    from app.core.ids import stable_id
    from app.core.schemas import AtomType

    existing = {
        re.sub(r"\s+", " ", _text(a).lower())
        for a in atoms
        if _atom_type_str(a) == "task"
    }
    site_count = work_order.get("site_count")
    added: list[Any] = []
    # Output order (and which duplicate wins dedup) must not follow the order the
    # model happened to list lines in: sort by normalized label, then content.
    labelled = sorted(
        (
            (_line_label(line), line)
            for line in lines
            if isinstance(line, dict)
        ),
        key=lambda lp: (lp[0].lower(), json.dumps(lp[1], sort_keys=True, default=str)),
    )
    for label, line in labelled:
        if not label:
            continue
        key = label.lower()
        if key in existing:
            continue
        support = pick_support(label, support_pool)
        if support is None:
            continue
        existing.add(key)

        task = copy.deepcopy(support)
        artifact_id = getattr(support, "artifact_id", "") or ""
        task.id = stable_id("atm", str(artifact_id), "work_order", label)
        task.project_id = project_id
        task.atom_type = AtomType.task
        task.raw_text = label
        task.normalized_text = label.lower()

        val = dict(getattr(task, "value", None) or {})
        # Drop the supporting atom's list polarity: a summary is not an
        # Include bullet, and carrying `list_section` would let
        # demote_email_include_list_microtasks re-type this back to scope_item.
        val.pop("list_section", None)
        val.pop("section_header", None)
        count = line.get("count")
        val.update(
            {
                "kind": "task",
                "text": label,
                "task_tier": "parent",
                "is_quote_line": True,
                "backfilled_from_atom_id": getattr(support, "id", None),
                "backfill_reason": "work_order",
                # The count lives in `value`, never in a `quantity:` entity key:
                # scrub_nondeliverable_quantity_keys strips those off any atom.
                "count": count if isinstance(count, (int, float)) else None,
                "unit": (str(line.get("unit")).strip().lower() or None)
                if line.get("unit")
                else None,
                "object": (str(line.get("object")).strip() or None)
                if line.get("object")
                else None,
                "site_count": site_count if isinstance(site_count, (int, float)) else None,
                "after_hours": bool(work_order.get("after_hours")),
                "no_onsite_hands": bool(work_order.get("no_onsite_hands")),
                "customer_supplies_equipment": bool(
                    work_order.get("customer_supplies_equipment")
                ),
            }
        )
        task.value = val

        # A summary is a machine's reading until a PM confirms it.
        try:
            from app.core.schemas import AuthorityClass, ReviewStatus

            task.authority_class = AuthorityClass.machine_extractor
            task.review_status = ReviewStatus.needs_review
        except Exception:  # pragma: no cover
            pass

        flags = list(getattr(task, "review_flags", None) or [])
        for flag in ("work_order", "task_tier_parent"):
            if flag not in flags:
                flags.append(flag)
        task.review_flags = flags
        added.append(task)

    if not added:
        return atoms, 0
    return atoms + added, len(added)


def apply_work_order(
    atoms: list[Any],
    *,
    project_id: str,
    deal_name: str = "",
) -> tuple[list[Any], int, dict]:
    """Judge, reassemble, mint. Returns (atoms, minted, report).

    ``report`` carries what the stage decided so the compiler can surface it:
    kept/dropped document names, the deal-level facts, and the one-line summary.
    """
    report: dict[str, Any] = {"kept_docs": [], "dropped_docs": [], "summary": {}}
    if not atoms:
        return atoms, 0, report

    by_doc = group_by_document(atoms)
    top_n = _env_int("SOWSMITH_WORK_ORDER_MAX_DOCS", 12)
    body_chars = _env_int("SOWSMITH_WORK_ORDER_JUDGE_CHARS", 1800)
    ranked = rank_documents(by_doc, top_n)

    kept: list[tuple[str, str]] = []
    support_pool: list[Any] = []
    for fn, doc_atoms in ranked:
        body = _body(doc_atoms, body_chars)
        keep, why = judge_document(
            fn, body, deal_name=deal_name, project_id=project_id
        )
        if keep:
            kept.append((fn, _body(doc_atoms, 1_000_000)))
            support_pool.extend(doc_atoms)
            report["kept_docs"].append(fn)
        else:
            report["dropped_docs"].append(f"{fn} ({why})")

    if not kept:
        return atoms, 0, report

    work_order = extract_work_order(deal_name, kept)
    if not work_order:
        return atoms, 0, report

    report["summary"] = {
        k: work_order.get(k)
        for k in (
            "site_count",
            "after_hours",
            "no_onsite_hands",
            "customer_supplies_equipment",
            "one_line_summary",
        )
    }
    atoms, minted = mint_work_line_atoms(
        atoms, work_order, support_pool, project_id=project_id
    )
    return atoms, minted, report


def work_line_labels(atoms: list[Any]) -> list[str]:
    """The minted work lines of a compile, normalized and sorted."""
    return sorted(
        {
            re.sub(r"\s+", " ", _text(a).lower()).strip()
            for a in atoms
            if (getattr(a, "value", None) or {}).get("backfill_reason") == "work_order"
        }
    )


def work_line_agreement(runs: list[list[str]]) -> float:
    """How much N runs of the same deal agree on its work lines, in [0, 1].

    Mean pairwise Jaccard over the normalized label sets. 1.0 means every run
    produced the same lines; fewer than two runs, or runs that all produced
    nothing, count as full agreement.
    """
    sets = [{re.sub(r"\s+", " ", str(x).lower()).strip() for x in r} for r in runs]
    if len(sets) < 2:
        return 1.0
    total, pairs = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            total += (len(sets[i] & sets[j]) / len(union)) if union else 1.0
            pairs += 1
    return total / pairs


def deal_agreement(deal_id: str, runs: list[list[str]]) -> dict[str, Any]:
    """Per-deal scorecard row: agreement plus the lines every run / some run saw."""
    sets = [{re.sub(r"\s+", " ", str(x).lower()).strip() for x in r} for r in runs]
    common = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    return {
        "deal_id": deal_id,
        "runs": len(runs),
        "agreement": round(work_line_agreement(runs), 4),
        "stable_lines": sorted(common),
        "unstable_lines": sorted(union - common),
    }


__all__ = [
    "deal_agreement",
    "rank_documents",
    "work_line_agreement",
    "work_line_labels",
    "RELATION",
    "CANDIDATES",
    "apply_work_order",
    "enabled",
    "extract_work_order",
    "group_by_document",
    "judge_document",
    "mint_work_line_atoms",
    "pick_support",
]
