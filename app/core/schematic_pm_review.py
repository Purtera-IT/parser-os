"""Bridge: existing schematic atoms -> PM verification review for PM_HANDOFF.

Surfaces the 100% mechanism in the PM payload the UI reads: every symbol type is
CONFIDENT (count-verified) / FLAGGED (mismatch/low-conf/unknown) / MISSING (legend
declares it, none found). The PM sees exactly what to eyeball — no silent wrongs.
Returns {"present": False} for deals with no schematic atoms (e.g. doc-only deals),
so it adds zero noise where there are no drawings.
"""
from __future__ import annotations
from typing import Any

from app.core.schematic_verification import (
    Detection, LegendEntry, verify_page, FLAGGED, MISSING,
)

_LEGEND = "schematic_legend"
_DET = "schematic_symbol_detection"
_WARN = "schematic_warning"


def _atype(a) -> str:
    t = getattr(a, "atom_type", None)
    return str(getattr(t, "value", t) or "")


def _val(a) -> dict:
    v = getattr(a, "value", None)
    return v if isinstance(v, dict) else {}


def schematic_pm_review(atoms: list[Any]) -> dict:
    legend: list[LegendEntry] = []
    dets: list[Detection] = []
    n_warn = 0
    for a in atoms:
        t = _atype(a)
        v = _val(a)
        if t == _LEGEND:
            entries = v.get("entries")
            entries = entries if isinstance(entries, list) and entries else [v]
            for e in entries:
                if not isinstance(e, dict):
                    continue
                eid = str(e.get("entry_id") or e.get("legend_entry_id")
                          or e.get("normalized_label") or e.get("label_text") or id(e))
                label = (e.get("label_text") or e.get("normalized_label")
                         or e.get("description") or e.get("type") or eid)
                cnt = e.get("count_column", e.get("declared_count", e.get("count")))
                try:
                    cnt = int(cnt) if cnt is not None else None
                except Exception:
                    cnt = None
                legend.append(LegendEntry(entry_id=eid, type_label=str(label)[:60], declared_count=cnt))
        elif t == _DET:
            le = v.get("legend_entry_id")
            dets.append(Detection(
                det_id=str(v.get("detection_id") or id(a)),
                legend_entry_id=str(le) if le else None,
                confidence=float(v.get("confidence") or 1.0),
                verified=bool(v.get("verified", True)),
            ))
        elif t == _WARN:
            n_warn += 1
    if not legend and not dets:
        return {"present": False}
    rep = verify_page(legend, dets)
    queue = [{"type": i.detail.get("type", ""), "status": i.status, "reason": i.reason}
             for i in rep.review_queue][:200]
    return {
        "present": True,
        "coverage_pct": round(rep.coverage * 100),
        "confident_count": len(rep.confident),
        "needs_review_count": len(rep.review_queue),
        "review_queue": queue,           # the PM hand-check list (flag-don't-guess)
        "warnings": n_warn,
        "summary": rep.emitted_accuracy_note,
    }
