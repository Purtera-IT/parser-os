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


TASK_TIER_RELATION = "task_tier"


def _taught_tier(label: str) -> str | None:
    """``parent`` / ``child`` from the feedback store, or None (abstain / no store)."""
    try:
        from app.core.decide import decide, get_store

        if get_store() is None:
            return None
        d = decide(
            TASK_TIER_RELATION, label[:600], ["parent", "child"],
            instruction="Is this line a unit of work a quote prices (parent) or a step inside one (child)?",
            llm=False,
        )
    except Exception:
        return None
    if d is None or d.source != "store" or d.verdict not in ("parent", "child"):
        return None
    return d.verdict


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

    # Taught first. Whether a line is a quote line or a step inside one is a
    # judgment finished Deal Kits already made: 000020 Binghamton priced
    # "Confirm QS1 connectivity between workstations and host" as its own
    # task, and the word list below (a leading "confirm") called it a child
    # step, so Deal Kit never proposed it. A confident taught answer wins; the
    # heuristics are the cold start for everything nobody taught yet.
    taught = _taught_tier(label)
    if taught is not None:
        return taught, taught == "parent"

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


_FUNCTION_WORDS = frozenset({"a", "an", "the", "of", "for", "to", "and", "in", "at", "on", "with", "is", "are", "be"})


def _identity_tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in _FUNCTION_WORDS)


def fold_task_mentions(atoms: list[Any]) -> int:
    """One unit of work stated more than once is one quote line.

    Live 010043 (compile fc2db7e, 2026-09-15): "3 Verkada cameras" in the
    email, "3 cameras-Verkada" in a note, "3 Verkada cameras intsall." in the
    note's title -- three parent tasks, two of them priced from the learned
    5.33 h per camera, so the Deal Kit was offered 32 hours for 16 hours of
    work. The three name the same thing: the words of one are the words of
    another, with at most a word added. That is the whole test -- token sets
    in a subset relation after grammar words are dropped -- so "Update QS1 Host
    PC static IP for the 1517 subnet" and "Update any hardcoded printer IPs
    from 1518 to the 1517 subnet" stay two tasks, and a survey mentioned five
    different ways stays five (that is the store's to teach).

    The fullest mention stays the parent; the others become its children,
    linked by ``parent_task_id`` and marked ``folded_into``, and are no longer
    quote lines, so hours are learned once. Returns how many were folded.
    """
    parents = [
        a for a in atoms
        if _atom_type_str(a) == "task"
        and (getattr(a, "value", None) or {}).get("task_tier") == "parent"
    ]
    toks = {id(a): _identity_tokens(_atom_text(a)) for a in parents}
    # Fullest first, so a shorter mention folds into the richest statement.
    ordered = sorted(parents, key=lambda a: (-len(toks[id(a)]), str(getattr(a, "id", ""))))
    folded = 0
    kept: list[Any] = []
    for a in ordered:
        t = toks[id(a)]
        if len(t) < 2:
            kept.append(a)
            continue
        into = next((k for k in kept if len(toks[id(k)]) >= 2 and (t <= toks[id(k)] or toks[id(k)] <= t)), None)
        if into is None:
            kept.append(a)
            continue
        val = dict(getattr(a, "value", None) or {})
        val["task_tier"] = "child"
        val["is_quote_line"] = False
        val["parent_task_id"] = str(getattr(into, "id", "") or "")
        val["parent_task_hint"] = _atom_text(into)
        val["folded_into"] = str(getattr(into, "id", "") or "")
        a.value = val
        flags = [f for f in (getattr(a, "review_flags", None) or []) if f != "task_tier_parent"]
        if "task_tier_child" not in flags:
            flags.append("task_tier_child")
        if "task_mention_folded" not in flags:
            flags.append("task_mention_folded")
        try:
            a.review_flags = flags
        except Exception:
            pass
        folded += 1
    return folded


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
    "fold_task_mentions",
    "infer_task_tier",
    "infer_task_tier_for_atom",
    "is_quote_line_task_atom",
]
