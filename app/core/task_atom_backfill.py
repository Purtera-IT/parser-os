"""Deterministically mint quote-level task atoms from high-signal notes.

HubSpot notes and short email bullets often carry the actual quoting work units
before a SOW exists:

* Badge/access control setup
* UID Enterprise setup
* Okta integration
* Camera configuration
* Do you have resources for a Ubiquiti install...

The LLM type classifier can leave these as ``scope_item`` or ``open_question``
because they are terse or phrased as a request. This backfill preserves the
original atom and adds a task atom for Deal Kit / site anchoring.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import AtomType

_SOURCE_TYPES = frozenset({"scope_item", "open_question", "requirement", "customer_instruction"})

_QUOTE_TASK_RE = re.compile(
    r"\b("
    r"ubiquiti\s+(?:install|configuration|configure)|"
    r"(?:badge|door)\s*/?\s*access(?:\s+control)?\s+setup|"
    r"access[-\s]*control\s+configuration|"
    r"uid\s+enterprise\s+(?:setup|onboarding)|"
    r"okta\s+(?:integration|provisioning|groups?)|"
    r"camera\s+configuration|"
    r"knowledge\s+transfer|white\s+glove|walk(?:ing)?\s+(?:him|customer|them)\s+through|"
    r"configure(?:d|ing|ation)?\s+(?:installed\s+)?(?:ubiquiti|switches|routers|badge|cameras|aps?)"
    r")\b",
    re.I,
)

_NON_QUOTE_RE = re.compile(
    r"\b(network\s+build\s*out\s+(?:is\s+)?(?:excluded|does\s+not\s+need)|"
    r"general\s+firewall/network\s+configuration)\b|"
    r"^\s*from\s*:|"
    r"\|\s*(?:to|subject|date)\s*:",
    re.I,
)

_NARRATIVE_NOT_TASK_RE = re.compile(
    r"\b("
    r"primary\s+focus\s+areas?|"
    r"customer\s+indicated|"
    r"purtera\s+agreed\s+to|"
    r"considered\s+a\s+(?:significant|hard)\s+requirement|"
    r"we\s+would\s+just\s+need\s+to|"
    r"areas\s+within\s+the\s+office"
    r")\b",
    re.I,
)

_DIRECT_TASK_LABEL_RE = re.compile(
    r"^\s*(?:\*\s*)?("
    r"badge\s*/?\s*access(?:\s+control)?\s+setup|"
    r"uid\s+enterprise\s+setup|"
    r"okta\s+integration|"
    r"camera\s+configuration|"
    r"knowledge\s+transfer\s*/\s*walking\s+(?:him|customer|them)\s+through\s+the\s+setup"
    r")\s*$",
    re.I,
)


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _text(atom: Any) -> str:
    raw = getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
    if raw.strip():
        return raw.strip()
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        return str(val.get("text") or val.get("description") or "").strip()
    return ""


def _clean_candidate(text: str) -> str:
    s = re.sub(r"^\s*(?:\*\s*)+", "", text or "").strip()
    s = re.sub(r"^\*\*(?:HubSpot Note|Unknown)\*\*:\s*", "", s, flags=re.I).strip()
    s = re.sub(r"^HubSpot Note:\s*", "", s, flags=re.I).strip()
    return s.strip(" -")


def _source_kind(atom: Any) -> str:
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        return str(val.get("kind") or "").strip().lower()
    return ""


def _list_section(atom: Any) -> str:
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        return str(val.get("list_section") or "").strip().lower()
    return ""


def _is_email_include_list_item(atom: Any) -> bool:
    """An item under an email ``Include:`` header — quote_line_head groups
    these into umbrella parent tasks; they must not become individual tasks."""
    return _list_section(atom) == "include" and _source_kind(atom) == "email_body_line"


def _candidate_lines(text: str) -> list[str]:
    lines = []
    for line in re.split(r"[\r\n]+", text or ""):
        cleaned = _clean_candidate(line)
        if cleaned:
            lines.append(cleaned)
    if lines:
        return lines
    cleaned = _clean_candidate(text)
    return [cleaned] if cleaned else []


def should_backfill_task(text: str) -> bool:
    label = _clean_candidate(text)
    if not label or len(label) > 450:
        return False
    if _NON_QUOTE_RE.search(label):
        return False
    if _NARRATIVE_NOT_TASK_RE.search(label):
        return False
    if _DIRECT_TASK_LABEL_RE.match(label):
        return True
    # Question-shaped resource asks are a valid parent quote task, but most
    # other prose matches ("Okta is important", transcript snippets) are facts.
    if re.match(r"^(do|can|could)\b", label, re.I) and re.search(r"\bubiquiti\s+install\b", label, re.I):
        return True
    if len(label) > 120:
        return False
    return bool(_QUOTE_TASK_RE.search(label))


def _task_label(text: str) -> str:
    """Preserve verbatim wording — never invent umbrella paraphrases.

    Email Include bullets and note lines must stay exactly as written
    ("walking him through the setup", not "guided handoff"). The only
    rewrite allowed is collapsing a question-shaped Ubiquiti install ask
    into a short parent label when the source is itself a resource question
    (not an Include-list micro-item).
    """
    label = _clean_candidate(text)
    # Question-shaped resource asks only — not Include-list micro-labels.
    if re.match(r"^(?:do|can|could)\b", label, re.I) and re.search(
        r"\bubiquiti\s+install\b", label, re.I
    ):
        return "Ubiquiti configuration / install support"
    return label[:240]


def backfill_quote_task_atoms(atoms: list[Any], *, project_id: str) -> tuple[list[Any], int]:
    """Mint quote-level tasks from notes / asks — never from email Include lists.

    Include/Exclude bullets are evidence atoms (verbatim). Inventing umbrella
    parents like ``Ubiquiti configuration / install support`` or
    ``Knowledge transfer / guided handoff`` from those lists hallucinates
    wording that is not in the source email.
    """
    existing = {
        re.sub(r"\s+", " ", _clean_candidate(_text(a)).lower())
        for a in atoms
        if _atom_type_str(a) == "task"
    }
    added: list[Any] = []

    for atom in atoms:
        if _atom_type_str(atom) not in _SOURCE_TYPES:
            continue
        # Include-list items stay as verbatim evidence — never become tasks
        # and never seed invented umbrella parents.
        if _is_email_include_list_item(atom):
            continue
        kind = _source_kind(atom)
        for line in _candidate_lines(_text(atom)):
            # Avoid broad narrative paragraphs; only promote direct bullets,
            # email-body lines, and question-shaped Ubiquiti resource asks.
            cleaned = _clean_candidate(line)
            direct_label = bool(_DIRECT_TASK_LABEL_RE.match(cleaned))
            if (
                not direct_label
                and kind not in {"email_body_line", "bullet"}
                and not re.match(r"^(?:do|can|could)\b", cleaned, re.I)
            ):
                continue
            if not should_backfill_task(line):
                continue
            label = _task_label(line)
            key = re.sub(r"\s+", " ", label.lower())
            if key in existing:
                continue
            existing.add(key)

            task = copy.deepcopy(atom)
            artifact_id = getattr(atom, "artifact_id", "") or ""
            task.id = stable_id("atm", artifact_id, "quote_task_backfill", label)
            task.project_id = project_id
            task.atom_type = AtomType.task
            task.raw_text = label
            task.normalized_text = label.lower()
            val = dict(getattr(task, "value", None) or {})
            # Strip Include/Exclude polarity so minted tasks are not shown as
            # hallucinated Include bullets in Atom Quality audit.
            val.pop("list_section", None)
            val.pop("section_header", None)
            val.update(
                {
                    "kind": "task",
                    "text": label,
                    "task_tier": "parent",
                    "is_quote_line": True,
                    "backfilled_from_atom_id": getattr(atom, "id", None),
                    "backfill_reason": "quote_task_note",
                }
            )
            task.value = val
            # Clear section_path Include/Exclude so envelope grouping is honest.
            for ref in getattr(task, "source_refs", None) or []:
                loc = getattr(ref, "locator", None)
                if isinstance(loc, dict) and loc.get("section_path") in (["Include"], ["Exclude"]):
                    loc = dict(loc)
                    loc.pop("section_path", None)
                    loc.pop("lead_in", None)
                    ref.locator = loc
            flags = list(getattr(task, "review_flags", None) or [])
            for flag in ("task_backfill", "task_tier_parent"):
                if flag not in flags:
                    flags.append(flag)
            task.review_flags = flags
            added.append(task)

    if not added:
        return atoms, 0
    return atoms + added, len(added)


__all__ = ["backfill_quote_task_atoms", "should_backfill_task"]
