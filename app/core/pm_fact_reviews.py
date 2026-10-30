"""A PM's fixes to the deal's lines, applied over the heads and every rule.

The deal page's Scope site Evidence (and the Sources reader) saves a PM's verdict on a line: "Looks right", "Not a
site" with its reason, "Fix address" with the right address (Platform-infra ``deal-fact-reviews.js``). Each save
rewrites ``deals/<deal>/orbitbrief/latest/fact_reviews.json``; this applies it on every compile, last of the
decisions, so the quote, the brief and the SOW read the PM's version (owner 2026-10-29: "it saves mine and
overrides the atoms to match that so it fixes it and then when I do the later things it has that context").

A fix is about what a line IS, not something learned later in the deal, so a pre-quote (as-of) run applies it
too. Each touched line says so (``value.pm_fix`` and a ``c3_label`` with ``act: "pm"``) and the rule ledger
records it (``pm_fix.*``): a PM fix is an approved label for the heads' next refit.

Lines are matched by atom id first, then by the review's own words and file (ids change across compiles).
Never raises; no blob, no reviews, or no SOWSMITH_FEEDBACK_BLOB is a normal no-op.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.core.pm_answer_blob import _container_client

NOT_A_SITE = re.compile(r"^\s*not a site\b", re.I)


def _fold(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def load_reviews(deal_id: str) -> list[dict[str, Any]]:
    """The deal's saved reviews, newest first. ``[]`` when unavailable."""
    if not deal_id:
        return []
    cc = _container_client()
    if cc is None:
        return []
    try:
        raw = cc.download_blob(f"deals/{deal_id}/orbitbrief/latest/fact_reviews.json").readall()
        doc = json.loads(raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw)
    except Exception:
        return []
    rows = doc.get("reviews") if isinstance(doc, dict) else None
    return [r for r in rows or [] if isinstance(r, dict) and r.get("status")]


def _filename(atom: Any) -> str:
    refs = getattr(atom, "source_refs", None) or []
    return str(getattr(refs[0], "filename", "") or "") if refs else ""


def _type(atom: Any) -> str:
    t = getattr(atom, "atom_type", None)
    return str(getattr(t, "value", t) or "")


def apply_reviews(atoms: list[Any], reviews: list[dict[str, Any]], types: dict[str, Any] | None = None) -> list[str]:
    """Apply each review to its line(s). Returns the ids changed. ``types`` (the heads' read, atom id -> [type, p])
    follows a retype so no later pass re-keys the line as a site."""
    if not reviews:
        return []
    from app.core import rule_ledger

    by_id = {str(getattr(a, "id", "") or ""): a for a in atoms}
    by_text: dict[str, list[Any]] = {}
    for a in atoms:
        by_text.setdefault(_fold(getattr(a, "raw_text", None) or getattr(a, "text", None)), []).append(a)
    changed: list[str] = []
    for r in reviews:
        hits = [by_id[r["atomId"]]] if r.get("atomId") in by_id else [
            a for a in by_text.get(_fold(r.get("factText")), [])
            if not r.get("sourceFilename") or _filename(a) == r.get("sourceFilename")]
        for a in hits:
            v = getattr(a, "value", None)
            if not isinstance(v, dict):
                v = {}
                try:
                    a.value = v
                except Exception:
                    continue
            aid = str(getattr(a, "id", "") or "")
            was = _type(a)
            why = " · ".join(x for x in (r.get("reason"), r.get("note")) if x) or None
            fix = {"status": r.get("status"), "value": r.get("fixedValue"), "why": why, "by": r.get("reviewer"),
                   "at": r.get("reviewedAt")}
            v["pm_fix"] = fix
            status, fixed = r.get("status"), str(r.get("fixedValue") or "")
            if status == "fixed" and NOT_A_SITE.match(fixed) and was == "physical_site":
                try:
                    from app.core.schemas import AtomType
                    a.atom_type = AtomType.deal_metadata
                except Exception:
                    a.atom_type = "deal_metadata"
                v["kind"] = "pm_not_a_site"
                v["c3_label"] = {"type": "deal_metadata", "p": 1.0, "act": "pm", "was": was, "why": why}
                try:
                    a.entity_keys = [k for k in (getattr(a, "entity_keys", None) or []) if not str(k).startswith("site:")]
                except Exception:
                    pass
                if types is not None and aid in types:
                    types[aid] = ["deal_metadata", 1.0]
                rule_ledger.fire("pm_fix.not_a_site", a, was, "deal_metadata", why or "")
            elif status == "fixed" and fixed and was == "physical_site":
                v["street_address"] = fixed
                v["address"] = fixed
                v["c3_label"] = {**(v.get("c3_label") or {}), "type": "physical_site", "act": "pm", "why": why}
                rule_ledger.fire("pm_fix.address", a, was, "physical_site", why or "")
            elif status == "set_aside":
                v["c3_label"] = {**(v.get("c3_label") or {}), "act": "pm", "set_aside": True, "why": why}
                rule_ledger.fire("pm_fix.set_aside", a, was, "set_aside", why or "")
            elif status == "looks_right":
                v["c3_label"] = {**(v.get("c3_label") or {}), "pm_confirmed": True}
                rule_ledger.fire("pm_fix.looks_right", a, was, was, "")
            else:
                continue
            changed.append(aid)
    return changed


__all__ = ["apply_reviews", "load_reviews"]
