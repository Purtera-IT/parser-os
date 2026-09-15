"""Does this document describe the job this deal is for?

A deal's attachments are whatever HubSpot associated with it, and on a
programme customer that includes other jobs. Live 010162 ("CDW- Sodexo SD-WAN
Program", 2026-09-15): a kiosk close-down at Delta Admin -- its packing list
("Delta Close Down.pdf") and a smart-hands thread ("RE: CDW Smart Hands SOW
Delta Admin 70598001") -- sat beside the SD-WAN scope. Nothing in either names
a deal number, so document scope (document_lifecycle/scope.py) read them as
this deal's, and the brief came out as a staff-augmentation job about kiosks,
badges and packing boxes, with its Deal Kit tasks drawn from the close-down.

The question is answered per DOCUMENT, through the decide() chokepoint:

    STORE (a PM taught "this thread is a different job")  ->  LLM  ->  this_deal

The judge reads the deal's own name -- the seller's one-line statement of the
work -- against the document's title and its first lines. Only a confident
``other_job`` removes anything, and removal is lossless: the atoms go to the
suppression ledger like every other gate drop, so a PM can see what was set
aside and teach the opposite. No store and no model means nothing moves.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

RELATION = "document_job"
CANDIDATES = ["this_deal", "other_job"]
INSTRUCTION = (
    "A deal is one job for one customer; its name states the work. Decide "
    "whether this document is about THAT job (this_deal) or about a different "
    "engagement for the same customer, such as another site's work order, a "
    "separate service call, or an unrelated project (other_job). Documents that "
    "discuss the deal's work, its pricing, its sites, its schedule or its "
    "contract are this_deal, even when they also mention other topics. Answer "
    "other_job only when the document's subject is clearly a different job; "
    "when in doubt, this_deal."
)
_MIN_CONF = 0.85
_LINES = 12
_LINE_CHARS = 160
_META_KINDS = frozenset({"hubspot_note_meta", "email_header", "email_addressee"})


def enabled() -> bool:
    return os.environ.get("SOWSMITH_DOCUMENT_JOB_SCOPE", "1").strip().lower() not in ("0", "false", "no", "off")


def deal_name_from_manifest(project_dir: Path | str | None) -> str:
    """The CRM deal name from the manifest sidecar, or ''."""
    if not project_dir:
        return ""
    path = Path(project_dir) / ".parser_manifest.json"
    if not path.is_file():
        return ""
    try:
        ctx = json.loads(path.read_text(encoding="utf-8")).get("context") or {}
        crm = ctx.get("crm") if isinstance(ctx, dict) else None
        return str((crm or {}).get("deal_name") or "").strip()
    except (OSError, ValueError, AttributeError):
        return ""


def _atom_type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "")


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _title(atoms: list[Any], filename: str) -> str:
    """What the document calls itself: an email thread's subject, a note's title, else the filename."""
    for a in atoms:
        v = _value(a)
        thread = v.get("email_thread")
        if isinstance(thread, dict) and thread.get("subject"):
            return str(thread["subject"]).strip()
    for a in atoms:
        v = _value(a)
        if v.get("kind") in ("hubspot_note_meta", "hubspot_note_body") and v.get("title"):
            return str(v["title"]).strip()
    return filename


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def common_lines(atoms: list[Any]) -> set[str]:
    """Lines that recur across documents of one deal: banners ("External sender
    Check the sender..."), signatures, disclaimers. They describe no job, so the
    judge should not read them as the document's opening lines. No vocabulary:
    a line is chrome because the deal repeats it, whatever it says."""
    seen: dict[str, set[str]] = {}
    for a in atoms:
        # The opening of the line: a banner repeats its opening even when one
        # copy was joined to itself with a separator.
        key = _norm(getattr(a, "raw_text", ""))[:80]
        if len(key) < 12:
            continue
        doc = str(getattr(a, "source_artifact_id", None) or getattr(a, "artifact_id", None) or "")
        seen.setdefault(key, set()).add(doc)
    return {k for k, docs in seen.items() if len(docs) >= 2}


def _own_words_first(atoms: list[Any]) -> list[Any]:
    """An email's own message before the history it quotes, else document order."""
    own = [a for a in atoms if _value(a).get("quoted") is False]
    if not own:
        return list(atoms)
    rest = [a for a in atoms if _value(a).get("quoted") is not False]
    return own + rest


def document_text(atoms: list[Any], filename: str, common: set[str] | None = None) -> str:
    """The document as the judge reads it: its title and its first lines."""
    lines: list[str] = []
    skip = common or set()
    for a in _own_words_first(atoms):
        v = _value(a)
        if v.get("kind") in _META_KINDS or v.get("field_name") in _META_KINDS or v.get("non_deal") or _atom_type(a) in ("raw_utterance",):
            continue
        t = " ".join(str(getattr(a, "raw_text", "") or "").split())
        if not t or t in lines or _norm(t)[:80] in skip or t.startswith("[Image extracted"):
            continue
        lines.append(t[:_LINE_CHARS])
        if len(lines) >= _LINES:
            break
    if not lines:  # a transcript is nothing but utterances
        for a in atoms:
            t = " ".join(str(getattr(a, "raw_text", "") or "").split())
            if t and len(t) > 40:
                lines.append(t[:_LINE_CHARS])
            if len(lines) >= _LINES:
                break
    return f"DOCUMENT: {_title(atoms, filename)}\n" + "\n".join(f"- {l}" for l in lines)


def judge_documents(
    atoms: list[Any],
    *,
    deal_name: str,
    project_id: str = "",
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    """Partition ``atoms`` into (kept, dropped) and describe each verdict.

    Every document is judged once. A confident ``other_job`` drops the
    document's atoms; anything else keeps them. Without a deal name there is
    nothing to compare against, so nothing is judged.
    """
    if not deal_name or not atoms:
        return list(atoms), [], []
    try:
        from app.core.decide import DecisionScope, decide
    except Exception:  # pragma: no cover
        return list(atoms), [], []

    by_doc: "OrderedDict[str, list[Any]]" = OrderedDict()
    for a in atoms:
        key = str(getattr(a, "source_artifact_id", None) or getattr(a, "artifact_id", None) or "")
        by_doc.setdefault(key, []).append(a)

    scope = DecisionScope(deal_id=str(project_id or ""))
    dropped_ids: set[int] = set()
    verdicts: list[dict[str, Any]] = []
    common = common_lines(atoms)
    for key, doc_atoms in by_doc.items():
        filename = str(getattr(doc_atoms[0], "source_filename", "") or key)
        text = f"DEAL: {deal_name.strip()}\n{document_text(doc_atoms, filename, common)}"
        try:
            d = decide(RELATION, text[:4000], CANDIDATES, instruction=INSTRUCTION, scope=scope)
        except Exception:
            d = None
        verdict = getattr(d, "verdict", None) or "this_deal"
        conf = float(getattr(d, "confidence", 0.0) or 0.0)
        source = getattr(d, "source", "fallback")
        other = verdict == "other_job" and (source == "store" or conf >= _MIN_CONF)
        verdicts.append({
            "filename": filename, "verdict": "other_job" if other else "this_deal",
            "confidence": round(conf, 3), "source": source, "atoms": len(doc_atoms),
            "correction_id": getattr(d, "correction_id", None),
        })
        if other:
            dropped_ids.update(id(a) for a in doc_atoms)
            for a in doc_atoms:
                flags = list(getattr(a, "review_flags", None) or [])
                if "other_job" not in flags:
                    flags.append("other_job")
                try:
                    a.review_flags = flags
                except Exception:
                    pass
    kept = [a for a in atoms if id(a) not in dropped_ids]
    dropped = [a for a in atoms if id(a) in dropped_ids]
    return kept, dropped, verdicts


__all__ = ["RELATION", "CANDIDATES", "INSTRUCTION", "enabled", "deal_name_from_manifest", "common_lines", "document_text", "judge_documents"]
