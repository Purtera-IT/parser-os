"""Classify task atoms as quote-level parents vs runbook child steps.

Deal Kit quotes parent work units ("Install AP", "Kiosk install") while
runbook bullets ("verify LED", "connect cable") stay in atoms as children.
Deterministic heuristics — no LLM.
"""

from __future__ import annotations

import re
from typing import Any

_STEP_HEADER_RE = re.compile(r"^\s*step\s+\d+\s*:\s*", re.I)
_PARENT_DELIVERABLE_RE = re.compile(
    r"\b("
    r"install(?:ation)?|deployment|site survey|acceptance test(?:ing)?|"
    r"cable drop|kiosk install|structured cabling|cutover|"
    r"develop schedule|validate deliverables|complete billing|"
    r"wireless ap|access point|conduit drop|hang\s+\d|"
    r"unbox and verify kiosk parts|power on the kiosk"
    r")\b",
    re.I,
)
_CHILD_IMPERATIVE_RE = re.compile(
    r"^\s*("
    r"confirm|verify|locate|identify|determine whether|determine if|"
    r"attach|connect|route|place|keep|match|leave|pull|start|lift|"
    r"install the|open network|put sign|put the|hang|email assigned|"
    r"check that|ensure the|make sure|record |note "
    r")\b",
    re.I,
)
_CHILD_PROCEDURAL_RE = re.compile(
    r"^\s*(if a problem|when |during setup|before powering|after tightening)\b",
    re.I,
)
_SCOPE_VENDOR_RE = re.compile(r"^\s*PurTera will\b", re.I)

_TIER_TYPES = frozenset({"task"})


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_text(atom: Any) -> str:
    raw = getattr(atom, "raw_text", None) or ""
    if raw.strip():
        return raw.strip()
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        for key in ("name", "text", "description", "action"):
            v = val.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _bullet_depth(atom: Any, val: dict[str, Any]) -> int | None:
    depth = val.get("depth")
    if isinstance(depth, int):
        return depth
    if isinstance(depth, str) and depth.isdigit():
        return int(depth)
    refs = getattr(atom, "source_refs", None) or []
    if refs:
        loc = getattr(refs[0], "locator", None) or {}
        if isinstance(loc, dict):
            bd = loc.get("bullet_depth")
            if isinstance(bd, int):
                return bd
    return None


def infer_task_tier(*, text: str, structured: dict[str, Any] | None = None) -> tuple[str, bool]:
    """Return ``(task_tier, is_quote_line)`` for a task-shaped label."""
    structured = structured or {}
    label = (text or "").strip()
    if not label:
        return "child", False

    explicit = structured.get("task_tier")
    if explicit in ("parent", "child"):
        is_quote = structured.get("is_quote_line")
        if is_quote is None:
            is_quote = explicit == "parent"
        return explicit, bool(is_quote)

    kind = str(structured.get("kind") or "")
    depth = structured.get("depth")
    if depth is None:
        depth = _bullet_depth_from_structured(structured)
    if isinstance(depth, str) and depth.isdigit():
        depth = int(depth)

    if _STEP_HEADER_RE.match(label):
        return "parent", True

    if structured.get("task_id"):
        return "parent", True

    if _SCOPE_VENDOR_RE.match(label) and len(label) >= 55:
        return "parent", True

    if _PARENT_DELIVERABLE_RE.search(label) and not _CHILD_IMPERATIVE_RE.match(label):
        return "parent", True

    if structured.get("phase") and not _CHILD_IMPERATIVE_RE.match(label):
        return "parent", True

    if _CHILD_IMPERATIVE_RE.match(label) or _CHILD_PROCEDURAL_RE.match(label):
        return "child", False

    if kind == "bullet" and isinstance(depth, int) and depth >= 1:
        return "child", False

    low = label.lower()
    if low.startswith(("confirm ", "verify ", "locate ", "identify ", "determine ")):
        return "child", False

    if kind != "bullet" and (depth is None or depth == 0):
        if len(label) >= 35:
            return "parent", True

    return "child", False


def _bullet_depth_from_structured(structured: dict[str, Any]) -> int | None:
    depth = structured.get("depth")
    if isinstance(depth, int):
        return depth
    return None


def infer_task_tier_for_atom(atom: Any) -> tuple[str, bool]:
    val = dict(getattr(atom, "value", None) or {})
    return infer_task_tier(text=_atom_text(atom), structured=val)


def _step_parent_label(text: str) -> str | None:
    m = _STEP_HEADER_RE.match(text)
    if not m:
        return None
    body = text[m.end() :].strip()
    return body or text.strip()


def classify_task_tiers(atoms: list[Any]) -> tuple[list[Any], int]:
    """Stamp ``task_tier`` / ``is_quote_line`` on task atoms; link child → parent hints."""
    changed = 0
    last_parent_id: str | None = None
    last_parent_label: str | None = None

    for atom in atoms:
        if _atom_type_str(atom) not in _TIER_TYPES:
            continue

        text = _atom_text(atom)
        val = dict(getattr(atom, "value", None) or {})
        tier, is_quote = infer_task_tier(text=text, structured=val)

        if tier == "parent":
            last_parent_id = str(getattr(atom, "id", "") or "")
            last_parent_label = _step_parent_label(text) or text
        elif is_quote is False and last_parent_id:
            val.setdefault("parent_task_id", last_parent_id)
            if last_parent_label:
                val.setdefault("parent_task_hint", last_parent_label)

        prev_tier = val.get("task_tier")
        prev_quote = val.get("is_quote_line")
        val["task_tier"] = tier
        val["is_quote_line"] = is_quote
        atom.value = val

        flags = list(getattr(atom, "review_flags", None) or [])
        flag = "task_tier_parent" if tier == "parent" else "task_tier_child"
        if flag not in flags:
            flags.append(flag)
            atom.review_flags = flags

        if prev_tier != tier or prev_quote != is_quote:
            changed += 1

    return atoms, changed


def is_quote_line_task_atom(atom: Any) -> bool:
    """Whether a task atom should surface as a Deal Kit quote line."""
    if _atom_type_str(atom) != "task":
        return False
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict) and "is_quote_line" in val:
        return bool(val.get("is_quote_line"))
    tier, is_quote = infer_task_tier_for_atom(atom)
    return is_quote


__all__ = [
    "classify_task_tiers",
    "infer_task_tier",
    "infer_task_tier_for_atom",
    "is_quote_line_task_atom",
]
