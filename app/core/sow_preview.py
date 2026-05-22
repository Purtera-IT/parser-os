"""SOW preview — substrate-side receipt-grounded markdown view.

THIS IS NOT SOWSMITH. The real SOWSmith lives in the ``purpulse.app``
repo with its own policy pack (``SOWSmith_policy_pack_v2.yaml``, ~12k
lines) and field catalog (``SOW_field_catalog_v2.yaml``, ~10.6k lines
= the 707-field SOW Requirements Library). HTML SOW rendering for the
PM handoff path lives in ``Orbitbrief-Core/src/orbitbrief_core/
pm_handoff/render_html.py``.

What this module IS: a deterministic, receipt-grounded preview of the
substrate's view of the project. Useful for:

  * Verifying the parser-os substrate captured every PM-relevant fact
    before handing off to the real SOWSmith downstream
  * Debugging which atoms / packets / contradictions exist in the
    envelope without opening the raw JSON
  * Giving a reviewer a single markdown document they can scan to
    confirm scope_truth / change_order_timeline / risk register /
    open questions look right BEFORE the real SOW is assembled

Consumes the OrbitBrief envelope (built by ``build_orbitbrief_envelope``)
and emits a 17-section markdown brief. Every claim in the output traces
back to a specific atom ID. Missing-fields placeholders surface as
``[NEEDS DATA: field_id]`` so the real SOWSmith can decide whether the
gap is a blocker.

Design principles:
  * Every section pulls FROM the envelope's pre-computed cockpit
    surfaces (pm_dashboard, sow_readiness_scorecard, srl_missing_
    checklist, scope_truth, change_order_timeline, site_readiness,
    stakeholder_load, project_vitals). No re-deriving — Core is truth.
  * Every claim carries its source atom IDs as inline footnotes.
  * MISSING fields surface explicitly so a reviewer sees gaps.
  * Markdown formatting is intentionally rich — headings, tables,
    audit-trail callouts — so the doc renders well anywhere.
"""
from __future__ import annotations

from typing import Any

SOW_VERSION = "parser_os_sow_preview_v1"


def build_sow_markdown(envelope: dict[str, Any]) -> str:
    """Produce a contract-grade SOW markdown document from the envelope.

    The envelope must carry the OrbitBrief-Core cockpit surfaces
    (pm_dashboard, sow_readiness_scorecard, srl_missing_checklist,
    scope_truth, change_order_timeline, site_readiness,
    stakeholder_load, project_vitals). If a surface is missing the
    relevant section is skipped — the SOW is best-effort, not
    best-guessing.
    """
    sections: list[str] = []
    sections.append(_section_header(envelope))
    sections.append(_section_executive_summary(envelope))
    sections.append(_section_project_vitals(envelope))
    sections.append(_section_stakeholders(envelope))
    sections.append(_section_sites(envelope))
    sections.append(_section_scope_of_work(envelope))
    sections.append(_section_change_orders(envelope))
    sections.append(_section_exclusions(envelope))
    sections.append(_section_schedule(envelope))
    sections.append(_section_commercial(envelope))
    sections.append(_section_sla_and_support(envelope))
    sections.append(_section_constraints(envelope))
    sections.append(_section_acceptance(envelope))
    sections.append(_section_risk_register(envelope))
    sections.append(_section_open_questions(envelope))
    sections.append(_section_readiness_audit(envelope))
    sections.append(_section_evidence_trail(envelope))
    sections.append(_section_footer(envelope))
    return "\n\n".join(s for s in sections if s).strip() + "\n"


# ─────────────────────────── sections ───────────────────────────


def _section_header(envelope: dict[str, Any]) -> str:
    project_id = envelope.get("project_id", "Unknown Project")
    compile_id = envelope.get("compile_id", "")
    generated_at = envelope.get("generated_at", "")
    vitals = envelope.get("project_vitals") or {}
    score = vitals.get("score_100", "—")
    band = vitals.get("band", "—")
    band_marker = {
        "green": "READY",
        "yellow": "MINOR GAPS",
        "orange": "NEEDS ATTENTION",
        "red": "SUBSTANTIAL WORK REMAINING",
    }.get(band, band.upper() if isinstance(band, str) else "—")
    return (
        f"# SOW Preview — {_humanize(project_id)}\n\n"
        f"**Project ID:** `{project_id}`  \n"
        f"**Compile ID:** `{compile_id}`  \n"
        f"**Generated:** {generated_at}  \n"
        f"**Readiness:** **{score} / 100** &nbsp;·&nbsp; **{band_marker}**\n\n"
        f"> *Substrate-side preview auto-rendered from the OrbitBrief envelope. "
        f"This is **not** the final SOW — SOWSmith (in `purpulse.app`, with the "
        f"707-field SRL + policy pack) assembles that. This preview is "
        f"receipt-grounded: every claim traces back to a specific atom in the "
        f"source artifacts. Sections marked `[NEEDS DATA]` flag fields the "
        f"substrate could not satisfy from current evidence.*"
    )


def _section_executive_summary(envelope: dict[str, Any]) -> str:
    summary = envelope.get("summary") or {}
    site_count = (envelope.get("site_readiness") or {}).get("site_count", 0)
    scope = envelope.get("scope_truth") or {}
    device_count = scope.get("device_count", 0)
    contested = scope.get("contested_count", 0)
    stakeholders = (envelope.get("pm_dashboard") or {}).get("stakeholders", [])
    money_total = (envelope.get("pm_dashboard") or {}).get("money_summary", {}).get("total", 0)
    artifact_count = summary.get("artifact_count", 0)
    atom_count = summary.get("atom_count", 0)
    out = ["## 1. Executive Summary\n"]
    out.append(
        f"This engagement covers **{device_count}** device categor"
        f"{'y' if device_count == 1 else 'ies'} across **{site_count}** "
        f"site{'s' if site_count != 1 else ''}, scoped from "
        f"**{artifact_count}** source artifact{'s' if artifact_count != 1 else ''} "
        f"comprising **{atom_count}** atomic evidence units."
    )
    if stakeholders:
        out.append(
            f"**{len(stakeholders)}** named stakeholders are on record. "
            "Roles and contact information are summarized in §3."
        )
    if money_total:
        out.append(
            f"Total commercial figures captured across the discovery set: "
            f"**${money_total:,.2f}** (see §10 for the breakdown)."
        )
    if contested:
        out.append(
            f"> ⚠ **{contested}** scope claim{'s' if contested != 1 else ''} "
            f"{'are' if contested != 1 else 'is'} **contested** across source "
            f"artifacts. See §6 (Scope of Work) for the governing values and "
            f"§17 (Evidence Trail) for the full audit."
        )
    return "\n\n".join(out)


def _section_project_vitals(envelope: dict[str, Any]) -> str:
    v = envelope.get("project_vitals") or {}
    if not v:
        return ""
    out = ["## 2. Project Vitals\n"]
    out.append(
        f"| Score | Band | Top Drivers | Top Detractors |\n"
        f"|---|---|---|---|\n"
        f"| **{v.get('score_100', '—')} / 100** | **{v.get('band', '—').upper()}** | "
        f"{', '.join(v.get('top_drivers') or []) or '—'} | "
        f"{', '.join(v.get('top_detractors') or []) or '—'} |"
    )
    out.append("\n**Component breakdown:**\n")
    out.append("| Component | Weight | Score | Contribution |")
    out.append("|---|---|---|---|")
    for c in v.get("components") or []:
        out.append(
            f"| {c.get('name', '—')} | {c.get('weight', 0):.2f} | "
            f"{c.get('raw_score', 0):.2f} | {c.get('contribution', 0):.3f} |"
        )
    return "\n".join(out)


def _section_stakeholders(envelope: dict[str, Any]) -> str:
    dash = envelope.get("pm_dashboard") or {}
    stakeholders = dash.get("stakeholders") or []
    out = ["## 3. Stakeholders\n"]
    if not stakeholders:
        out.append("_[NEEDS DATA: no stakeholders identified in discovery artifacts.]_")
        return "\n".join(out)
    out.append("| Name | Role | Email | Phone |")
    out.append("|---|---|---|---|")
    for s in stakeholders:
        out.append(
            f"| {s.get('name') or '_(unnamed)_'} "
            f"| {s.get('role') or '—'} "
            f"| {s.get('email') or '—'} "
            f"| {s.get('phone') or '—'} |"
        )
    # Load matrix from stakeholder_load for the same names.
    load = envelope.get("stakeholder_load") or {}
    if load.get("stakeholders"):
        out.append("\n**Workload allocation:**\n")
        out.append("| Stakeholder | Risks owned | Critical | High | Action items | Severity load |")
        out.append("|---|---|---|---|---|---|")
        for s in load["stakeholders"]:
            out.append(
                f"| {s.get('slug', '—')} | {s.get('risk_count', 0)} "
                f"| {s.get('critical_risk_count', 0)} | {s.get('high_risk_count', 0)} "
                f"| {s.get('action_item_count', 0)} | {s.get('risk_severity_load', 0)} |"
            )
        if load.get("bottlenecks"):
            out.append(
                f"\n> ⚠ **Workload bottlenecks**: {', '.join(load['bottlenecks'])} — "
                f"carrying ≥2 critical or ≥4 high-severity risks. "
                f"Consider rebalancing before kickoff."
            )
    return "\n".join(out)


def _section_sites(envelope: dict[str, Any]) -> str:
    sr = envelope.get("site_readiness") or {}
    sites = sr.get("sites") or []
    out = ["## 4. Sites & Locations\n"]
    if not sites:
        out.append("_[NEEDS DATA: no sites identified in discovery artifacts.]_")
        return "\n".join(out)
    out.append(
        f"**{sr.get('site_count', 0)}** sites · "
        f"average readiness **{sr.get('avg_readiness', 0):.2f}** / 1.00"
    )
    if sr.get("least_ready_sites"):
        out.append(
            f"\n> ⚠ Sites needing attention: {', '.join(sr['least_ready_sites'])}"
        )
    out.append("\n| Site | Readiness | Devices | Stakeholders | Constraints | Contradictions |")
    out.append("|---|---|---|---|---|---|")
    for s in sites:
        out.append(
            f"| `{s.get('site')}` | {s.get('readiness', 0):.2f} "
            f"| {s.get('device_count', 0)} | {s.get('stakeholder_count', 0)} "
            f"| {s.get('constraint_count', 0)} | {s.get('contradiction_count', 0)} |"
        )
    return "\n".join(out)


def _section_scope_of_work(envelope: dict[str, Any]) -> str:
    scope = envelope.get("scope_truth") or {}
    devices = scope.get("devices") or []
    out = ["## 5. Scope of Work\n"]
    if not devices:
        out.append("_[NEEDS DATA: no device-level scope claims in discovery artifacts.]_")
        return "\n".join(out)
    out.append(
        "The following quantities are the **authoritative scope** after reconciling "
        "every source artifact. When multiple artifacts make competing claims, the "
        "**governing claim** is the one with the highest authority class (signed SOW > "
        "vendor quote > meeting note). Contested entries are flagged and detailed in §17."
    )
    out.append("\n| Device | Site | Quantity | Governing Source | Status |")
    out.append("|---|---|---|---|---|")
    for d in devices:
        contested_marker = "⚠ CONTESTED" if d.get("is_contested") else "✓ uncontested"
        out.append(
            f"| {_humanize(d.get('device', '').replace('device:', ''))} "
            f"| {_humanize(d.get('site', '').replace('site:', ''))} "
            f"| **{d.get('canonical_quantity', '—')}** "
            f"| `{d.get('governing_authority', '—')}` "
            f"| {contested_marker} |"
        )
    if scope.get("contested_count"):
        out.append(
            f"\n> ⚠ **{scope['contested_count']}** scope item"
            f"{'s' if scope['contested_count'] != 1 else ''} contested. "
            f"Resolve before kickoff."
        )
    return "\n".join(out)


def _section_change_orders(envelope: dict[str, Any]) -> str:
    ct = envelope.get("change_order_timeline") or {}
    entries = ct.get("entries") or []
    out = ["## 6. Change Order History\n"]
    if not entries:
        out.append("_No change orders recorded in discovery artifacts._")
        return "\n".join(out)
    out.append(
        f"**{ct.get('entry_count', 0)}** total change events  ·  "
        f"**{ct.get('with_structured_delta', 0)}** carry quantitative deltas  ·  "
        f"**{ct.get('with_approval_signal', 0)}** carry explicit approval"
    )
    out.append("\n| # | Kind | Delta | Approval | Source | Text |")
    out.append("|---|---|---|---|---|---|")
    for i, e in enumerate(entries, start=1):
        delta = e.get("change_delta") or {}
        if delta:
            delta_str = f"{delta.get('from')} → {delta.get('to')} ({delta.get('delta'):+d})"
        else:
            delta_str = "—"
        approval = "✓" if e.get("approval_signal") else "—"
        text = (e.get("text") or "").replace("|", "\\|").replace("\n", " ")[:160]
        out.append(
            f"| {i} | {e.get('kind', '—')} | {delta_str} | {approval} "
            f"| `{(e.get('driven_by') or '—')}` | {text} |"
        )
    return "\n".join(out)


def _section_exclusions(envelope: dict[str, Any]) -> str:
    dash = envelope.get("pm_dashboard") or {}
    excl = dash.get("exclusions") or []
    out = ["## 7. Out of Scope\n"]
    if not excl:
        out.append("_[NEEDS DATA: no explicit out-of-scope statements found.]_")
        return "\n".join(out)
    out.append("The following items are **explicitly excluded** from this engagement:\n")
    for e in excl:
        text = (e.get("text") or "").strip()
        out.append(f"- {text}  \n  _Source: atom `{e.get('atom_id', '—')}`_")
    return "\n".join(out)


def _section_schedule(envelope: dict[str, Any]) -> str:
    dash = envelope.get("pm_dashboard") or {}
    timeline = dash.get("milestones_timeline") or []
    out = ["## 8. Schedule & Milestones\n"]
    if not timeline:
        out.append("_[NEEDS DATA: no project milestones captured.]_")
        return "\n".join(out)
    out.append("| Date (ISO) | Milestone | Source Atom |")
    out.append("|---|---|---|")
    for m in timeline:
        text = (m.get("text") or "").replace("|", "\\|")[:120]
        out.append(f"| **{m.get('iso', '—')}** | {text} | `{m.get('atom_id', '—')}` |")
    return "\n".join(out)


def _section_commercial(envelope: dict[str, Any]) -> str:
    dash = envelope.get("pm_dashboard") or {}
    money = dash.get("money_summary") or {}
    atoms = money.get("atoms") or []
    out = ["## 9. Commercial Terms\n"]
    if not atoms:
        out.append("_[NEEDS DATA: no commercial figures captured in discovery.]_")
        return "\n".join(out)
    total = money.get("total", 0)
    out.append(f"**Total commercial figures across discovery:** ${total:,.2f}\n")
    out.append("| Amount | Context | Source Atom |")
    out.append("|---|---|---|")
    for m in atoms[:25]:
        amt = m.get("amount", 0)
        text = (m.get("text") or "").replace("|", "\\|")[:120]
        out.append(f"| ${amt:,.2f} | {text} | `{m.get('atom_id', '—')}` |")
    if len(atoms) > 25:
        out.append(f"\n_(+{len(atoms) - 25} additional commercial atoms not shown)_")
    return "\n".join(out)


def _section_sla_and_support(envelope: dict[str, Any]) -> str:
    dash = envelope.get("pm_dashboard") or {}
    sla = dash.get("sla_summary") or []
    out = ["## 10. SLA & Support Terms\n"]
    if not sla:
        out.append("_[NEEDS DATA: no SLA targets captured.]_")
        return "\n".join(out)
    out.append("Aggregated SLA targets across all source artifacts:\n")
    out.append("| Targets | Source Excerpt | Source Atom |")
    out.append("|---|---|---|")
    for s in sla:
        targets = s.get("sla") or {}
        target_str = ", ".join(f"`{k}={v}`" for k, v in targets.items()) or "—"
        text = (s.get("text") or "").replace("|", "\\|")[:120]
        out.append(f"| {target_str} | {text} | `{s.get('atom_id', '—')}` |")
    return "\n".join(out)


def _section_constraints(envelope: dict[str, Any]) -> str:
    indexes = envelope.get("indexes") or {}
    by_type = indexes.get("atoms_by_atom_type") or {}
    constraint_ids = by_type.get("constraint", [])
    atoms = envelope.get("atoms") or []
    atom_by_id = {a.get("id"): a for a in atoms}
    out = ["## 11. Site Access, Safety & Constraints\n"]
    if not constraint_ids:
        out.append("_No site-access or safety constraints captured._")
        return "\n".join(out)
    rendered = 0
    for aid in constraint_ids[:30]:
        atom = atom_by_id.get(aid)
        if not atom:
            continue
        text = (atom.get("raw_text") or "").strip().replace("\n", " ")
        if not text:
            continue
        rendered += 1
        out.append(f"- {text[:240]}  \n  _Source: atom `{aid}`_")
    if rendered == 0:
        out.append("_No site-access or safety constraints captured._")
    elif len(constraint_ids) > 30:
        out.append(f"\n_(+{len(constraint_ids) - 30} additional constraint atoms not shown)_")
    return "\n".join(out)


def _section_acceptance(envelope: dict[str, Any]) -> str:
    indexes = envelope.get("indexes") or {}
    by_type = indexes.get("atoms_by_atom_type") or {}
    decision_ids = by_type.get("decision", []) + by_type.get("meeting_commitment", [])
    atoms = envelope.get("atoms") or []
    atom_by_id = {a.get("id"): a for a in atoms}
    out = ["## 12. Acceptance Criteria\n"]
    if not decision_ids:
        # Fallback: look for text-matched acceptance phrases.
        candidates = [
            a for a in atoms
            if any(t in (a.get("raw_text") or "").lower()
                   for t in ("acceptance", "sign-off", "signoff", "completion criteria"))
        ]
        if not candidates:
            out.append("_[NEEDS DATA: no explicit acceptance criteria captured.]_")
            return "\n".join(out)
        for a in candidates[:10]:
            text = (a.get("raw_text") or "").strip()
            out.append(f"- {text[:240]}  \n  _Source: atom `{a.get('id', '—')}`_")
        return "\n".join(out)
    rendered = 0
    for aid in decision_ids[:30]:
        atom = atom_by_id.get(aid)
        if not atom:
            continue
        text = (atom.get("raw_text") or "").strip()
        if not text:
            continue
        rendered += 1
        out.append(f"- {text[:240]}  \n  _Source: atom `{aid}`_")
        if rendered >= 15:
            break
    if rendered == 0:
        out.append("_[NEEDS DATA: no explicit acceptance criteria captured.]_")
    return "\n".join(out)


def _section_risk_register(envelope: dict[str, Any]) -> str:
    dash = envelope.get("pm_dashboard") or {}
    risks_by_owner = dash.get("risks_by_owner") or {}
    unowned = dash.get("risks_unowned") or []
    out = ["## 13. Risk Register\n"]
    total = sum(len(v) for v in risks_by_owner.values()) + len(unowned)
    if total == 0:
        out.append("_No risks captured in discovery artifacts._")
        return "\n".join(out)
    out.append(f"**{total}** risks identified.\n")
    out.append("| ID | Severity | Owner | Summary | Mitigation |")
    out.append("|---|---|---|---|---|")
    rendered = []
    for owner_slug, risks in risks_by_owner.items():
        for r in risks:
            rendered.append((owner_slug, r))
    for r in unowned:
        rendered.append((None, r))
    # Filter out the table header row that leaks in as an atom when
    # the markdown parser classifies the header (| ID | Risk | ... |)
    # as a risk_row. Header rows have risk_id="ID" or "Severity" /
    # severity is a non-severity token.
    def _is_header_row(risk: dict[str, Any]) -> bool:
        rid = (risk.get("risk_id") or "").strip().lower()
        sev = (risk.get("severity") or "").strip().lower()
        return rid in {"id", ""} and sev in {"severity", ""}
    rendered = [pair for pair in rendered if not _is_header_row(pair[1])]
    # Sort by severity weight: critical → high → medium → low
    sev_order = {"critical": 0, "high": 1, "medium": 2, "med": 2, "low": 3}
    rendered.sort(key=lambda pair: sev_order.get(((pair[1].get("severity") or "")).lower(), 9))
    for owner, r in rendered:
        sev = r.get("severity") or "—"
        owner_str = owner or "_(unowned)_"
        summary = (r.get("summary") or "—").replace("|", "\\|")[:120]
        mitigation = (r.get("mitigation") or "—").replace("|", "\\|")[:120]
        out.append(
            f"| {r.get('risk_id') or '—'} | **{sev}** | {owner_str} "
            f"| {summary} | {mitigation} |"
        )
    return "\n".join(out)


def _section_open_questions(envelope: dict[str, Any]) -> str:
    dash = envelope.get("pm_dashboard") or {}
    qs = dash.get("open_questions") or []
    out = ["## 14. Open Questions Before Kickoff\n"]
    if not qs:
        out.append("_No open questions captured._")
        return "\n".join(out)
    out.append(
        f"**{len(qs)}** open question{'s' if len(qs) != 1 else ''} require resolution "
        f"before this SOW can be signed:\n"
    )
    for i, q in enumerate(qs, start=1):
        text = (q.get("text") or "").strip()
        out.append(f"{i}. {text}  \n   _Source: atom `{q.get('atom_id', '—')}`_")
    return "\n".join(out)


def _section_readiness_audit(envelope: dict[str, Any]) -> str:
    sc = envelope.get("sow_readiness_scorecard") or {}
    ck = envelope.get("srl_missing_checklist") or {}
    out = ["## 15. Readiness Audit\n"]
    out.append(
        f"**SOW readiness score:** **{sc.get('readiness_score', 0):.2f}** / 1.00  ·  "
        f"**Grade:** `{sc.get('grade', '—')}`"
    )
    out.append(
        f"**SRL field coverage:** **{ck.get('present_count', 0)} / {ck.get('field_count', 0)}** "
        f"({ck.get('coverage', 0) * 100:.0f}%)"
    )

    # Dimension breakdown
    out.append("\n**Dimension scores:**\n")
    out.append("| Dimension | Score |")
    out.append("|---|---|")
    for dim, d in (sc.get("dimensions") or {}).items():
        out.append(f"| {dim} | {d.get('score', 0):.2f} |")

    # Per-category SRL coverage
    by_cat = ck.get("by_category") or {}
    if by_cat:
        out.append("\n**SRL coverage by category:**\n")
        out.append("| Category | Present / Total | Coverage |")
        out.append("|---|---|---|")
        for cat, stats in sorted(by_cat.items()):
            out.append(
                f"| {cat} | {stats.get('present', 0)} / {stats.get('total', 0)} | "
                f"{stats.get('coverage', 0) * 100:.0f}% |"
            )

    # Missing fields
    missing = ck.get("missing") or []
    if missing:
        out.append("\n**Required fields still missing from discovery:**\n")
        for m in missing:
            out.append(
                f"- ⚠ **[NEEDS DATA: `{m.get('field_id')}`]** "
                f"_{m.get('label')}_ &nbsp;·&nbsp; "
                f"category: `{m.get('category', 'general')}`"
            )
    return "\n".join(out)


def _section_evidence_trail(envelope: dict[str, Any]) -> str:
    scope = envelope.get("scope_truth") or {}
    contested = scope.get("contested") or []
    out = ["## 16. Evidence Trail — Contested Claims\n"]
    if not contested:
        out.append(
            "All scope claims in this SOW are uncontested across the source artifacts. "
            "No reconciliation required."
        )
        return "\n".join(out)
    out.append(
        f"The following **{len(contested)}** scope items had competing claims in the "
        f"discovery artifacts. The governing value (used in §6) is shown alongside the "
        f"competing values and the source atoms that asserted each."
    )
    for c in contested:
        device = _humanize(c.get("device", "").replace("device:", ""))
        site = _humanize(c.get("site", "").replace("site:", ""))
        out.append(
            f"\n### {device} @ {site}\n\n"
            f"**Governing:** {c.get('canonical_quantity')}  ·  "
            f"**Competing:** {', '.join(str(v) for v in c.get('competing_values', []))}\n"
        )
        out.append("| Quantity | Authority | Source Atom | Excerpt |")
        out.append("|---|---|---|---|")
        for audit_row in c.get("audit", []):
            qty = audit_row.get("quantity")
            for claim in audit_row.get("claims", []):
                text = (claim.get("text") or "").replace("|", "\\|").replace("\n", " ")[:120]
                out.append(
                    f"| **{qty}** | `{claim.get('authority_class', '—')}` "
                    f"| `{claim.get('atom_id', '—')}` | {text} |"
                )
    return "\n".join(out)


def _section_footer(envelope: dict[str, Any]) -> str:
    summary = envelope.get("summary") or {}
    return (
        "---\n\n"
        f"_Substrate-side SOW preview ({SOW_VERSION}) auto-rendered from the "
        f"OrbitBrief envelope `{envelope.get('compile_id', '—')}` across "
        f"{summary.get('artifact_count', 0)} source artifacts. This is **not** the "
        f"final SOW — SOWSmith (in `purpulse.app`) assembles the contract-grade "
        f"document from the SRL field catalog and policy pack. This preview is "
        f"receipt-grounded: every claim above traces to an atom ID in the source "
        f"bundle. Audit the substrate at `orbitbrief.input.json`._"
    )


# ─────────────────────────── helpers ───────────────────────────


def _humanize(slug: str) -> str:
    if not slug:
        return "—"
    return slug.replace("_", " ").title()


__all__ = ["build_sow_markdown", "SOW_VERSION"]
