from __future__ import annotations

import re
from collections import defaultdict

from app.core.authority import (
    compare_atoms,
    choose_governing_atoms,
    is_governing_candidate,
    is_scope_impacting_meeting_atom,
)
from app.core.anchors import _best_site_key, make_anchor_signature
from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.packet_certificates import build_packet_certificate
from app.core.risk import packet_pm_sort_key, score_packet_risk
from app.core.schemas import (
    AtomType,
    AuthorityClass,
    EntityRecord,
    EvidenceAtom,
    EvidenceEdge,
    EvidencePacket,
    PacketFamily,
    PacketStatus,
    ReviewStatus,
)

# Any token that can place an atom in a site_access *candidate* group (includes catwalk as location).
_ACCESS_CANDIDATE_TOKENS_RE = re.compile(
    r"\b("
    r"catwalk|"
    r"boom\s+lift|scissor\s+lift|\blifts?\b|"
    r"after[-\s]?hours|after\s+hours|nights?|weekends?|weekdays|"
    r"ceiling\s+access|overhead\s+access|"
    r"escort\s+required|"
    r"badge\s+access|mdf\s+access|idf\s+access|"
    r"restricted\s+access|work\s+window|access\s+window|"
    r"\bescort\b|\bbadge\b|\bmdf\b|\bidf\b|"
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*[-–]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)"
    r")\b",
    re.I,
)

# Strong site-access signal without requiring catwalk pairing (lifts, hours, escort policy, etc.).
_SITE_ACCESS_STRONG_RE = re.compile(
    r"\b("
    r"after[-\s]?hours|after\s+hours|nights?|weekends?|weekdays|"
    r"boom\s+lift|scissor\s+lift|\blifts?\b|"
    r"ceiling\s+access|overhead\s+access|"
    r"escort\s+required|"
    r"badge\s+access|mdf\s+access|idf\s+access|"
    r"restricted\s+access|work\s+window|access\s+window|"
    r"\bescort\b"
    r")\b",
    re.I,
)

_CATWALK_RE = re.compile(r"\bcatwalk\b", re.I)

# Catwalk as a work surface: only elevates to site_access when paired with lift/hours/ceiling/escort/badge/MDF/IDF/etc.
_CATWALK_COMPANION_RE = re.compile(
    r"\b("
    r"lift|boom|scissor|"
    r"after[-\s]?hours|after\s+hours|nights?|weekends?|weekdays|"
    r"ceiling|overhead|"
    r"escort|badge|mdf|idf|restricted|hoist|aerial|rental|"
    r"work\s+window|access\s+window"
    r")\b",
    re.I,
)

_TIME_RANGE_ACCESS_RE = re.compile(
    r"\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*[-–]\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)",
    re.I,
)

ACCESS_TEXT_RE = _ACCESS_CANDIDATE_TOKENS_RE

_SITE_ACCESS_EVIDENCE_AUTHS: frozenset[AuthorityClass] = frozenset(
    {
        AuthorityClass.approved_site_roster,
        AuthorityClass.customer_current_authored,
        AuthorityClass.contractual_scope,
        AuthorityClass.meeting_note,
        AuthorityClass.vendor_quote,
    }
)

_SITE_ACCESS_ATOM_TYPES: frozenset[AtomType] = frozenset(
    {
        AtomType.constraint,
        AtomType.customer_instruction,
        AtomType.action_item,
        AtomType.open_question,
        AtomType.scope_item,
    }
)


def _site_access_combined_blob(group: list[EvidenceAtom]) -> str:
    parts: list[str] = []
    for a in group:
        parts.append(normalize_text(a.raw_text))
        parts.append(normalize_text(str(a.value)))
    return " ".join(parts)


def _site_access_strong_signal(blob: str) -> bool:
    if _SITE_ACCESS_STRONG_RE.search(blob):
        return True
    if _TIME_RANGE_ACCESS_RE.search(blob):
        return True
    if re.search(r"\b(badge|mdf|idf)\s+access\b", blob, re.I):
        return True
    return False


def _site_access_catwalk_paired(blob: str) -> bool:
    return bool(_CATWALK_RE.search(blob) and _CATWALK_COMPANION_RE.search(blob))


def _group_triggers_site_access(group: list[EvidenceAtom]) -> bool:
    blob = _site_access_combined_blob(group)
    if _site_access_strong_signal(blob):
        return True
    return _site_access_catwalk_paired(blob)


def _open_question_access_gate_unknown(atom: EvidenceAtom) -> bool:
    """Badge / MDF / IDF access unclear — route to missing_info (access gate), not generic open_question."""
    if atom.atom_type != AtomType.open_question:
        return False
    blob = normalize_text(atom.raw_text) + " " + normalize_text(str(atom.value))
    if not re.search(r"\b(badge|mdf|idf)\b", blob, re.I):
        return False
    return bool(
        re.search(r"\b(unknown|tbd|unclear|unsure|not\s+sure)\b", blob, re.I) or "?" in (atom.raw_text or "")
    )


def _atom_eligible_site_access_evidence(atom: EvidenceAtom) -> bool:
    if atom.authority_class not in _SITE_ACCESS_EVIDENCE_AUTHS:
        return False
    if atom.atom_type not in _SITE_ACCESS_ATOM_TYPES:
        return False
    if atom.atom_type == AtomType.open_question and _open_question_access_gate_unknown(atom):
        return False
    blob = normalize_text(atom.raw_text) + " " + normalize_text(str(atom.value))
    return bool(_ACCESS_CANDIDATE_TOKENS_RE.search(blob))


def _site_access_group_unresolved(group: list[EvidenceAtom]) -> bool:
    for atom in group:
        if atom.confidence < 0.75:
            return True
        if atom.atom_type in {AtomType.open_question, AtomType.action_item}:
            return True
        blob = normalize_text(atom.raw_text) + " " + normalize_text(str(atom.value))
        if "?" in (atom.raw_text or ""):
            return True
        if re.search(r"\b(unknown|tbd|clarif|confirm|verify|unclear)\b", blob, re.I):
            return True
    return False


def _site_access_packet_reason(group: list[EvidenceAtom]) -> str:
    blob = _site_access_combined_blob(group)
    mentions: list[str] = []
    if _CATWALK_RE.search(blob):
        mentions.append("catwalk")
    if re.search(r"\b(lift|boom|scissor)\b", blob, re.I):
        mentions.append("lift")
    if re.search(r"\b(after[-\s]?hours|after\s+hours|nights?|weekends?)\b", blob, re.I) or _TIME_RANGE_ACCESS_RE.search(
        blob
    ):
        mentions.append("after-hours or work-window timing")
    if re.search(r"\b(ceiling|overhead)\s+access\b", blob, re.I):
        mentions.append("ceiling access")
    if re.search(r"\b(escort|badge|mdf|idf)\b", blob, re.I):
        mentions.append("escort or credential/telecom-room access")
    tail = ", ".join(mentions) if mentions else "physical or time-bounded site access"
    return (
        f"Site access constraints ({tail}) affect mobilization, lifts, or after-hours work; "
        "confirm requirements with the customer or GC before scheduling critical path work."
    )


def _site_access_review_flags(group: list[EvidenceAtom]) -> list[str]:
    blob = _site_access_combined_blob(group)
    flags = ["site_access_physical_constraints"]
    if _CATWALK_RE.search(blob):
        flags.append("site_access_catwalk_context")
    if re.search(r"\b(lift|boom|scissor)\b", blob, re.I):
        flags.append("site_access_lift_equipment")
    if re.search(r"\b(after[-\s]?hours|after\s+hours|nights?|weekends?)\b", blob, re.I) or _TIME_RANGE_ACCESS_RE.search(
        blob
    ):
        flags.append("site_access_after_hours_window")
    return sorted(set(flags))


_SCOPE_INCLUSION_PRIMARY_AUTHORITIES: frozenset[AuthorityClass] = frozenset(
    {
        AuthorityClass.approved_site_roster,
        AuthorityClass.customer_current_authored,
        AuthorityClass.contractual_scope,
    }
)


def _atom_value_dict(atom: EvidenceAtom) -> dict:
    v = atom.value
    return v if isinstance(v, dict) else {}


def _vendor_quote_like(atom: EvidenceAtom) -> bool:
    return atom.authority_class == AuthorityClass.vendor_quote


def _can_act_as_scope_inclusion_governor(atom: EvidenceAtom) -> bool:
    """Written scope / roster / customer current, or explicitly approved meeting notes only."""
    if not is_governing_candidate(atom):
        return False
    if atom.authority_class in _SCOPE_INCLUSION_PRIMARY_AUTHORITIES:
        return True
    if atom.authority_class == AuthorityClass.meeting_note:
        return atom.review_status == ReviewStatus.approved
    return False


def _group_has_scope_inclusion_governor(group: list[EvidenceAtom]) -> bool:
    return any(_can_act_as_scope_inclusion_governor(a) for a in group)


def _inclusion_excluded_signal(value: dict) -> bool:
    if value.get("included") is False:
        return True
    inc = str(value.get("inclusion_status") or "").strip().lower()
    if inc in {"excluded", "by_others", "not_included", "nic", "not included", "by others"}:
        return True
    return False


def _vendor_quantity_zero_or_unknown(atom: EvidenceAtom) -> bool:
    v = _atom_value_dict(atom)
    q = v.get("quantity")
    if q is None:
        raw = str(v.get("quantity_raw") or "").strip().lower()
        return raw in {"", "0", "n/a", "na", "tbd", "unknown"}
    try:
        return float(q) == 0.0
    except (TypeError, ValueError):
        return False


_CERTIFICATION_TOPIC = re.compile(
    r"\b(certification|certificates|certificate|certify|cable\s+certification|tester\s+export|automated\s+tester|"
    r"fluke|test\s+results?|test\s+reports?|tia[-\s]?568|permanent\s+link\s+test|channel\s+test|"
    r"wiremap|pass[-/\s]?fail\s+report|link\s+test)\b",
    re.I,
)

_CERTIFICATION_REQUIREMENT = re.compile(
    r"\b(required|shall|must|provide|submit|include|deliver|fluke|certification\s+report|"
    r"test\s+export|automated\s+tester|test\s+results?)\b",
    re.I,
)

_CERTIFICATION_PRIMARY_AUTHS: frozenset[AuthorityClass] = frozenset(
    {
        AuthorityClass.approved_site_roster,
        AuthorityClass.customer_current_authored,
        AuthorityClass.contractual_scope,
        AuthorityClass.meeting_note,
    }
)


def _blob_is_certification_testing_topic(atom: EvidenceAtom) -> bool:
    blob = normalize_text(atom.raw_text) + " " + normalize_text(str(atom.value))
    if _CERTIFICATION_TOPIC.search(blob):
        return True
    v = _atom_value_dict(atom)
    ik = str(v.get("item_kind") or "").lower()
    ni = str(v.get("normalized_item") or "").lower()
    ck = str(v.get("comparison_key") or "").lower()
    if "cert" in ik or ik in {"certification", "testing", "test"}:
        return True
    if any(t in ni for t in ("cert", "fluke", "test export", "wiremap", "link test", "tia")):
        return True
    if "cert" in ck or "testing" in ck or "test_export" in ck:
        return True
    return False


def _atom_documents_certification_requirement(atom: EvidenceAtom) -> bool:
    """Approved / current / addendum / meeting evidence that cable testing or certification is required or in scope."""
    if atom.authority_class not in _CERTIFICATION_PRIMARY_AUTHS:
        return False
    if not _blob_is_certification_testing_topic(atom):
        return False
    if atom.atom_type in {AtomType.open_question, AtomType.action_item}:
        return True
    if atom.atom_type == AtomType.quantity:
        v = _atom_value_dict(atom)
        ik = str(v.get("item_kind") or "").lower()
        if ik in {"certification", "testing", "test"}:
            return True
        blob = normalize_text(atom.raw_text) + " " + normalize_text(str(v))
        if not _CERTIFICATION_REQUIREMENT.search(blob):
            return False
        try:
            q = float(v.get("quantity") or 0)
        except (TypeError, ValueError):
            q = 0.0
        return q > 0.0 or v.get("included") is True
    if atom.atom_type in {AtomType.scope_item, AtomType.customer_instruction, AtomType.exclusion}:
        blob = normalize_text(atom.raw_text) + " " + normalize_text(str(atom.value))
        if _CERTIFICATION_REQUIREMENT.search(blob):
            return True
        return "?" in (atom.raw_text or "")
    return False


def _primary_atom_triggers_certification_requirement_packet(atom: EvidenceAtom) -> bool:
    return _atom_documents_certification_requirement(atom)


def _gov_sort_key_cert(atom: EvidenceAtom) -> tuple[int, str]:
    prio = {
        AuthorityClass.approved_site_roster: 0,
        AuthorityClass.customer_current_authored: 1,
        AuthorityClass.contractual_scope: 2,
        AuthorityClass.meeting_note: 3,
    }.get(atom.authority_class, 9)
    return (prio, atom.id)


def _vendor_certification_testing_excluded_supporting_atom(atom: EvidenceAtom) -> bool:
    """Vendor quote line excluded / zero-qty on certification or tester export topic."""
    if not _vendor_quote_like(atom):
        return False
    if atom.atom_type not in {AtomType.quantity, AtomType.scope_item, AtomType.vendor_line_item}:
        return False
    if not _blob_is_certification_testing_topic(atom):
        return False
    v = _atom_value_dict(atom)
    if _inclusion_excluded_signal(v):
        return True
    if atom.atom_type == AtomType.quantity and _vendor_quantity_zero_or_unknown(atom):
        return True
    return False


def _action_item_deferred_to_certification_missing_info(atom: EvidenceAtom) -> bool:
    if atom.atom_type != AtomType.action_item:
        return False
    if not _blob_is_certification_testing_topic(atom):
        return False
    return bool(re.search(r"\b(confirm|verify|format)\b", normalize_text(atom.raw_text), re.I))


_PATHWAY_RACEWAY_CONDUIT = re.compile(
    r"\b(raceway|conduit|wire\s*mold|wiremold|pathway|cable\s+pathway|sleeve|j[-\s]?hook|"
    r"surface\s+mount\s+raceway|emt|pvc\s+conduit)\b",
    re.I,
)

_PATHWAY_UNCERTAINTY = re.compile(
    r"\b(unknown|confirm|tbd|verify|may\s+be\s+unusable|unusable|allowance|unit\s+allowance|"
    r"existing\s+raceway|existing\s+conduit|conduit\s+condition|clarif|need\s+to\s+price|"
    r"price\s+new|per\s+affected|which\s+existing)\b",
    re.I,
)

_RACEWAY_PATHWAY_PRIMARY_AUTHS: frozenset[AuthorityClass] = frozenset(
    {
        AuthorityClass.customer_current_authored,
        AuthorityClass.contractual_scope,
        AuthorityClass.meeting_note,
    }
)


def _blob_is_pathway_raceway_conduit(atom: EvidenceAtom) -> bool:
    blob = normalize_text(atom.raw_text) + " " + normalize_text(str(atom.value))
    if _PATHWAY_RACEWAY_CONDUIT.search(blob):
        return True
    v = _atom_value_dict(atom)
    ik = str(v.get("item_kind") or "").lower()
    ni = str(v.get("normalized_item") or "").lower()
    ck = str(v.get("comparison_key") or "").lower()
    if ik in {"raceway", "conduit"}:
        return True
    if "raceway" in ni or "conduit" in ni:
        return True
    if "raceway" in ck or "conduit" in ck or "raceway_conduit" in ck:
        return True
    return False


def _blob_has_pathway_uncertainty(atom: EvidenceAtom) -> bool:
    blob = normalize_text(atom.raw_text) + " " + normalize_text(str(atom.value))
    if "?" in (atom.raw_text or ""):
        return True
    return bool(_PATHWAY_UNCERTAINTY.search(blob))


def _vendor_raceway_conduit_supporting_atom(atom: EvidenceAtom) -> bool:
    """Vendor quote line that supports pathway missing_info (excluded, not included, or zero qty)."""
    if not _vendor_quote_like(atom):
        return False
    if atom.atom_type not in {AtomType.quantity, AtomType.scope_item, AtomType.vendor_line_item}:
        return False
    if not _blob_is_pathway_raceway_conduit(atom):
        return False
    v = _atom_value_dict(atom)
    if _inclusion_excluded_signal(v):
        return True
    if atom.atom_type == AtomType.quantity and _vendor_quantity_zero_or_unknown(atom):
        return True
    return False


def _primary_atom_triggers_raceway_conduit_missing_info(atom: EvidenceAtom) -> bool:
    """Customer / addendum / transcript evidence that raceway or conduit scope, allowance, or condition is unclear."""
    if _primary_atom_triggers_certification_requirement_packet(atom):
        return False
    if atom.authority_class not in _RACEWAY_PATHWAY_PRIMARY_AUTHS:
        return False
    if atom.atom_type not in {
        AtomType.open_question,
        AtomType.customer_instruction,
        AtomType.scope_item,
        AtomType.exclusion,
        AtomType.action_item,
    }:
        return False
    if not _blob_is_pathway_raceway_conduit(atom):
        return False
    if atom.atom_type == AtomType.open_question:
        return True
    if atom.atom_type == AtomType.action_item:
        return _blob_has_pathway_uncertainty(atom) or bool(
            re.search(r"\bverify\b", normalize_text(atom.raw_text), re.I)
        )
    return _blob_has_pathway_uncertainty(atom)


def _action_item_deferred_to_raceway_missing_info(atom: EvidenceAtom) -> bool:
    return atom.atom_type == AtomType.action_item and _primary_atom_triggers_raceway_conduit_missing_info(atom)


def _vendor_scope_pollution_diversion(atom: EvidenceAtom) -> bool:
    if not _vendor_quote_like(atom):
        return False
    v = _atom_value_dict(atom)
    if str(v.get("item_kind") or "") != "power":
        return False
    if v.get("is_scope_pollution_candidate") is True:
        return True
    if str(v.get("scope_relevance") or "") == "scope_pollution_candidate":
        return True
    return False


def _vendor_excluded_line_diversion(atom: EvidenceAtom) -> bool:
    """Vendor line marked excluded / not included must not anchor active scope_inclusion."""
    if not _vendor_quote_like(atom):
        return False
    return _inclusion_excluded_signal(_atom_value_dict(atom))


def _vendor_scope_diversion_kind(atom: EvidenceAtom) -> str | None:
    """Return packet diversion: 'pollution' (scope_exclusion) or 'excluded' (missing_info), else None."""
    if _vendor_scope_pollution_diversion(atom):
        return "pollution"
    if _vendor_excluded_line_diversion(atom):
        return "excluded"
    if _vendor_electrical_power_line(atom):
        return "pollution"
    return None


def _site_keys(atom: EvidenceAtom) -> set[str]:
    return {k for k in atom.entity_keys if k.startswith("site:")}


def _blob_suggests_poe_only_not_branch_circuit(blob: str) -> bool:
    """True when text is PoE / 802.3 style without clear branch-circuit / receptacle electrical context."""
    if not re.search(r"\b(poe|802\.3af|802\.3at|802\.3bt|power\s+over\s+ethernet)\b", blob, re.I):
        return False
    return not bool(
        re.search(r"\b(20\s*amp|120v|240v|receptacle|electrical\s+outlet|branch\s+circuit|utility\s+power)\b", blob, re.I)
    )


def _is_vendor_poe_network_line(atom: EvidenceAtom) -> bool:
    """Vendor line is PoE / network power, not 120V branch-circuit / receptacle scope pollution."""
    if not _vendor_quote_like(atom):
        return False
    v = _atom_value_dict(atom)
    if str(v.get("item_kind") or "") == "poe":
        return True
    ni = str(v.get("normalized_item") or "").lower()
    if ni == "poe" or ni.startswith("poe_") or ni.startswith("poe:"):
        return True
    blob = normalize_text(atom.raw_text) + " " + normalize_text(str(v))
    return _blob_suggests_poe_only_not_branch_circuit(blob)


def _vendor_electrical_power_line(atom: EvidenceAtom) -> bool:
    """Vendor quote line for branch-circuit / receptacle power (not PoE network scope)."""
    if not _vendor_quote_like(atom) or atom.atom_type not in {AtomType.quantity, AtomType.scope_item}:
        return False
    if _is_vendor_poe_network_line(atom):
        return False
    if _vendor_scope_pollution_diversion(atom):
        return True
    v = _atom_value_dict(atom)
    if str(v.get("item_kind") or "") == "power":
        return True
    blob = normalize_text(atom.raw_text) + " " + normalize_text(str(v))
    if re.search(r"\b(20\s*amp|120\s*v|receptacle|electrical\s+outlet|utility\s+power|branch\s+circuit)\b", blob, re.I):
        return True
    return False


_POWER_SCOPE_NEGATION = re.compile(
    r"(not\s+in\s+scope|out\s+of\s+scope|excluded|not\s+included|by\s+others|ignored|"
    r"no\s+electrical|electrical\s+not|power\s+not|does\s+not\s+include|"
    r"outside\s+contractor|by\s+the\s+owner|owner[-\s]furnished)",
    re.I,
)

_GOVERNANCE_POWER_EXCLUSION_AUTHS: frozenset[AuthorityClass] = frozenset(
    {
        AuthorityClass.customer_current_authored,
        AuthorityClass.contractual_scope,
        AuthorityClass.approved_site_roster,
        AuthorityClass.meeting_note,
    }
)


def _power_governance_covers_vendor(gov: EvidenceAtom, vendor: EvidenceAtom) -> bool:
    """True when written / transcript power exclusion applies to this vendor line (site or project-global)."""
    sg, sv = _site_keys(gov), _site_keys(vendor)
    if sg and sv:
        return bool(sg & sv)
    # Missing site on one side is common (transcript exclusions); pair within the same project bundle.
    return True


def _atom_documents_power_out_of_scope(atom: EvidenceAtom) -> bool:
    """Customer / SOW / roster evidence that electrical or utility power work is out of scope."""
    if atom.authority_class not in _GOVERNANCE_POWER_EXCLUSION_AUTHS:
        return False
    if atom.atom_type not in {AtomType.exclusion, AtomType.customer_instruction, AtomType.scope_item}:
        return False
    blob = normalize_text(atom.raw_text) + " " + normalize_text(str(atom.value))
    if _blob_suggests_poe_only_not_branch_circuit(blob):
        return False
    if not re.search(r"\b(power|electrical|utility|20\s*amp|120v|outlet|receptacle|contractor\s+power)\b", blob, re.I):
        return False
    return bool(_POWER_SCOPE_NEGATION.search(blob))


def _gov_sort_key_power(atom: EvidenceAtom) -> tuple[int, str]:
    prio = {
        AuthorityClass.customer_current_authored: 0,
        AuthorityClass.contractual_scope: 1,
        AuthorityClass.approved_site_roster: 2,
        AuthorityClass.meeting_note: 3,
    }.get(atom.authority_class, 9)
    return (prio, atom.id)


def _anchor_for_atoms(atoms: list[EvidenceAtom]) -> tuple[str, str]:
    all_keys = [key for atom in atoms for key in atom.entity_keys]
    site_keys = sorted(k for k in set(all_keys) if k.startswith("site:"))
    device_keys = sorted(k for k in set(all_keys) if k.startswith("device:"))
    if site_keys:
        return "site", site_keys[0]
    if device_keys:
        return "device", device_keys[0]
    return "entity", "unknown"


def _packet_confidence(governing_atoms: list[EvidenceAtom], has_contradiction: bool) -> float:
    if not governing_atoms:
        return 0.0
    value = max(atom.confidence for atom in governing_atoms)
    if has_contradiction:
        value -= 0.15
    return max(0.0, min(1.0, value))


def _select_governing_atoms(
    atoms: list[EvidenceAtom],
    *,
    family: PacketFamily | None = None,
    prefer_customer_exclusion: bool = False,
    prefer_customer_raceway_instruction: bool = False,
    prefer_certification_requirement: bool = False,
) -> list[EvidenceAtom]:
    candidates = [a for a in atoms if a.authority_class != AuthorityClass.deleted_text]
    if not candidates:
        return []
    if family == PacketFamily.scope_exclusion:
        non_vendor = [a for a in candidates if a.authority_class != AuthorityClass.vendor_quote]
        if non_vendor:
            candidates = non_vendor
    if family == PacketFamily.missing_info and prefer_certification_requirement:
        non_vendor = [a for a in candidates if a.authority_class != AuthorityClass.vendor_quote]
        if non_vendor:
            candidates = non_vendor
        cert_govs = [a for a in candidates if _atom_documents_certification_requirement(a)]
        if cert_govs:
            return [min(cert_govs, key=_gov_sort_key_cert)]
    if family == PacketFamily.missing_info and prefer_customer_raceway_instruction:
        non_vendor = [a for a in candidates if a.authority_class != AuthorityClass.vendor_quote]
        if non_vendor:
            candidates = non_vendor
    if prefer_customer_raceway_instruction:
        cust = [
            a
            for a in candidates
            if a.authority_class == AuthorityClass.customer_current_authored
            and a.atom_type in {AtomType.customer_instruction, AtomType.open_question}
        ]
        if cust:
            return sorted(cust, key=lambda a: a.id)[:1]
    if prefer_customer_exclusion:
        customer_exclusions = [
            a
            for a in candidates
            if a.atom_type == AtomType.exclusion and a.authority_class == AuthorityClass.customer_current_authored
        ]
        if customer_exclusions:
            return sorted(customer_exclusions, key=lambda a: a.id)[:1]
    if any(a.authority_class == AuthorityClass.customer_current_authored for a in candidates):
        candidates = [
            a
            for a in candidates
            if a.authority_class != AuthorityClass.quoted_old_email
        ]
    context = {"packet_family": family.value} if family is not None else None
    winners = choose_governing_atoms(candidates, context=context)
    if not winners:
        return []
    best = winners[0]
    for atom in winners[1:]:
        decision = compare_atoms(best, atom, context=context)
        best = best if decision.governing_atom_id == best.id else atom
    return [best]


def _build_packet(
    project_id: str,
    family: PacketFamily,
    atoms: list[EvidenceAtom],
    related_edges: list[EvidenceEdge],
    status: PacketStatus,
    reason: str,
    contradicting_atom_ids: list[str] | None = None,
    review_flags: list[str] | None = None,
    prefer_customer_exclusion: bool = False,
    prefer_customer_raceway_instruction: bool = False,
    prefer_certification_requirement: bool = False,
    owner: str | None = None,
    material_identity: str | None = None,
) -> EvidencePacket:
    governing_atoms = _select_governing_atoms(
        atoms,
        family=family,
        prefer_customer_exclusion=prefer_customer_exclusion,
        prefer_customer_raceway_instruction=prefer_customer_raceway_instruction,
        prefer_certification_requirement=prefer_certification_requirement,
    )
    governing_ids = [a.id for a in governing_atoms]
    support_ids = sorted({a.id for a in atoms if a.id not in set(contradicting_atom_ids or [])})
    contradicting_ids = sorted(set(contradicting_atom_ids or []))
    edge_ids = sorted({e.id for e in related_edges})
    anchor_signature = make_anchor_signature(family, atoms, owner=owner, material_identity=material_identity)
    anchor_type = anchor_signature.anchor_type
    anchor_key = anchor_signature.canonical_key
    atom_ids_for_stable = "|".join(sorted(a.id for a in atoms))

    flags = set(review_flags or [])
    if contradicting_ids:
        flags.add("contradiction_present")
    if any(a.confidence < 0.75 for a in atoms):
        flags.add("low_confidence_atom")
    if any(a.authority_class == AuthorityClass.deleted_text for a in atoms):
        flags.add("deleted_text_present")
    if any("semantic_candidate_linker" in edge.reason.lower() for edge in related_edges):
        flags.add("semantic_candidate_linker")

    effective_status = status
    if not governing_ids and status in {PacketStatus.active, PacketStatus.needs_review}:
        if family == PacketFamily.scope_exclusion and flags & {
            "vendor_scope_pollution_candidate",
            "power_vendor_scope_mismatch",
        }:
            effective_status = PacketStatus.needs_review
        else:
            effective_status = PacketStatus.rejected
    elif governing_atoms and any(atom.review_status.value == "needs_review" for atom in governing_atoms):
        if effective_status == PacketStatus.active:
            effective_status = PacketStatus.needs_review

    packet = EvidencePacket(
        id=stable_id("pkt", project_id, family.value, anchor_signature.hash, atom_ids_for_stable),
        project_id=project_id,
        family=family,
        anchor_type=anchor_type,
        anchor_key=anchor_key,
        anchor_signature=anchor_signature,
        governing_atom_ids=governing_ids,
        supporting_atom_ids=support_ids,
        contradicting_atom_ids=contradicting_ids,
        related_edge_ids=edge_ids,
        confidence=_packet_confidence(governing_atoms, bool(contradicting_ids)),
        status=effective_status,
        reason=reason,
        review_flags=sorted(flags),
    )
    return packet


def _is_risky_action_item(atom: EvidenceAtom) -> bool:
    text = normalize_text(atom.raw_text)
    return any(token in text for token in ("scope", "add", "remove", "price", "cost", "commercial", "change"))


def _is_material_roster_vendor_aggregate_edge(edge: EvidenceEdge) -> bool:
    return (edge.metadata or {}).get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote"


def _material_aggregate_review_flags(family: PacketFamily) -> list[str]:
    flags = ["roster_vendor_aggregate_mismatch", "contradiction_present"]
    if family == PacketFamily.vendor_mismatch:
        flags.append("vendor_scope_quantity_mismatch")
    return flags


def build_packets(
    project_id: str,
    atoms: list[EvidenceAtom],
    entities: list[EntityRecord],
    edges: list[EvidenceEdge],
    attach_metadata: bool = True,
) -> list[EvidencePacket]:
    del entities  # reserved for future packet anchoring refinements
    atom_by_id = {a.id: a for a in atoms}
    packets: list[EvidencePacket] = []
    consumed_by_conflict_or_exclusion: set[str] = set()
    material_edge_ids = {e.id for e in edges if _is_material_roster_vendor_aggregate_edge(e)}

    # 0) Material roster aggregate vs summed vendor primary lines (one packet per edge; metadata from graph).
    for edge in edges:
        if edge.id not in material_edge_ids:
            continue
        md = edge.metadata or {}
        roster_id = md.get("roster_atom_id")
        vendor_ids = md.get("vendor_atom_ids") or []
        identity = md.get("identity")
        pf = md.get("preferred_packet_family")
        if not roster_id or not vendor_ids or not identity or not pf:
            continue
        roster = atom_by_id.get(roster_id)
        vendor_atoms = [atom_by_id[i] for i in vendor_ids if i in atom_by_id]
        if roster is None or not vendor_atoms:
            continue
        try:
            family = PacketFamily(pf)
        except ValueError:
            continue
        atoms_for_packet = [roster] + vendor_atoms
        packet = _build_packet(
            project_id=project_id,
            family=family,
            atoms=atoms_for_packet,
            related_edges=[edge],
            status=PacketStatus.needs_review,
            reason=edge.reason or f"Roster aggregate vs vendor primary lines for {identity}.",
            contradicting_atom_ids=sorted(vendor_ids),
            review_flags=_material_aggregate_review_flags(family),
            material_identity=str(identity),
        )
        packets.append(packet)
        consumed_by_conflict_or_exclusion.update([roster_id, *vendor_ids])

    # 0c) missing_info — certification / tester exports required vs vendor excluded or zero qty
    cert_primaries = [a for a in atoms if _primary_atom_triggers_certification_requirement_packet(a)]
    cert_primaries_sorted = sorted(cert_primaries, key=_gov_sort_key_cert)
    for primary in cert_primaries_sorted:
        if primary.id in consumed_by_conflict_or_exclusion:
            continue
        vendors = [
            v
            for v in atoms
            if v.id != primary.id
            and _vendor_certification_testing_excluded_supporting_atom(v)
            and v.id not in consumed_by_conflict_or_exclusion
        ]
        if not vendors:
            continue
        meeting_extras = [
            a
            for a in cert_primaries_sorted
            if a.id not in consumed_by_conflict_or_exclusion
            and a.id != primary.id
            and a.authority_class == AuthorityClass.meeting_note
            and a.atom_type == AtomType.action_item
            and _action_item_deferred_to_certification_missing_info(a)
        ]
        pack_atoms = [primary] + vendors + meeting_extras
        related_ci = [e for e in edges if any(x.id in {e.from_atom_id, e.to_atom_id} for x in pack_atoms)]
        vendor_ids_sorted = sorted({v.id for v in vendors})
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.missing_info,
            atoms=pack_atoms,
            related_edges=related_ci,
            status=PacketStatus.needs_review,
            reason=(
                "Cable certification / automated tester exports or pass-fail test reports are required by approved or "
                "current scope, but the vendor quote excludes them, marks them not included, or carries zero quantity; "
                "confirm deliverables and report format."
            ),
            contradicting_atom_ids=vendor_ids_sorted,
            review_flags=[
                "certification_testing_export_missing_info",
                "vendor_excluded_line",
                "vendor_quote_not_scope_governor",
            ],
            prefer_certification_requirement=True,
            material_identity="certification",
        )
        packets.append(packet)
        consumed_by_conflict_or_exclusion.update(a.id for a in pack_atoms)

    # 1) quantity_conflict
    for edge in edges:
        if edge.id in material_edge_ids:
            continue
        if edge.edge_type.value != "contradicts":
            continue
        a = atom_by_id.get(edge.from_atom_id)
        b = atom_by_id.get(edge.to_atom_id)
        if (
            not a
            or not b
            or a.id in consumed_by_conflict_or_exclusion
            or b.id in consumed_by_conflict_or_exclusion
            or a.atom_type != AtomType.quantity
            or b.atom_type != AtomType.quantity
        ):
            continue
        qty_a = a.value.get("quantity")
        qty_b = b.value.get("quantity")
        reason = edge.reason if edge.reason else f"Quantity conflict between {qty_a} and {qty_b}."
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.quantity_conflict,
            atoms=[a, b],
            related_edges=[edge],
            status=PacketStatus.needs_review,
            reason=reason,
            contradicting_atom_ids=[a.id, b.id],
        )
        packets.append(packet)
        consumed_by_conflict_or_exclusion.update([a.id, b.id])

    # 2) vendor_mismatch
    for edge in edges:
        if edge.id in material_edge_ids:
            continue
        if edge.edge_type.value != "contradicts":
            continue
        a = atom_by_id.get(edge.from_atom_id)
        b = atom_by_id.get(edge.to_atom_id)
        if not a or not b:
            continue
        authorities = {a.authority_class, b.authority_class}
        if (
            a.atom_type == AtomType.quantity
            and b.atom_type == AtomType.quantity
            and authorities == {AuthorityClass.approved_site_roster, AuthorityClass.vendor_quote}
        ):
            packet = _build_packet(
                project_id=project_id,
                family=PacketFamily.vendor_mismatch,
                atoms=[a, b],
                related_edges=[edge],
                status=PacketStatus.needs_review,
                reason=edge.reason if edge.reason else "Vendor quote quantity does not match scoped quantity.",
                contradicting_atom_ids=[a.id, b.id],
                review_flags=["vendor_scope_quantity_mismatch"],
            )
            packets.append(packet)
            consumed_by_conflict_or_exclusion.update([a.id, b.id])

    # 2b) Written scope excludes electrical power; vendor quote still lists power work.
    for vendor_atom in atoms:
        if vendor_atom.id in consumed_by_conflict_or_exclusion:
            continue
        if not _vendor_electrical_power_line(vendor_atom):
            continue
        candidates_gov = [
            a
            for a in atoms
            if a.id != vendor_atom.id
            and _atom_documents_power_out_of_scope(a)
            and _power_governance_covers_vendor(a, vendor_atom)
        ]
        if not candidates_gov:
            continue
        gov = min(candidates_gov, key=_gov_sort_key_power)
        related_edges = [
            e
            for e in edges
            if vendor_atom.id in {e.from_atom_id, e.to_atom_id} or gov.id in {e.from_atom_id, e.to_atom_id}
        ]
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.scope_exclusion,
            atoms=[gov, vendor_atom],
            related_edges=related_edges,
            status=PacketStatus.needs_review,
            reason=(
                "Vendor quote includes electrical or utility power line items despite written scope / customer "
                "evidence excluding that power work; do not treat vendor power as structured low-voltage scope."
            ),
            contradicting_atom_ids=[vendor_atom.id],
            review_flags=["power_vendor_scope_mismatch", "scope_pollution_vendor_vs_written_exclusion"],
            prefer_customer_exclusion=True,
            material_identity="scope:power",
        )
        packets.append(packet)
        consumed_by_conflict_or_exclusion.add(vendor_atom.id)
        consumed_by_conflict_or_exclusion.add(gov.id)

    # 3) scope_exclusion
    exclusion_atoms = [a for a in atoms if a.atom_type == AtomType.exclusion]
    excludes_edges = [e for e in edges if e.edge_type.value == "excludes"]
    grouped_exclusions: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in exclusion_atoms:
        site_keys = sorted(k for k in atom.entity_keys if k.startswith("site:"))
        bucket = _best_site_key([atom], prioritize_exclusion_text=True)
        if bucket == "site:unknown" and site_keys:
            bucket = site_keys[0]
        grouped_exclusions[bucket].append(atom)
    for bucket_key, ex_atoms in grouped_exclusions.items():
        uniq = {a.id: a for a in ex_atoms}
        grouped_exclusions[bucket_key] = sorted(uniq.values(), key=lambda a: a.id)
    for anchor_key, ex_atoms in grouped_exclusions.items():
        ex_atoms = [a for a in ex_atoms if a.id not in consumed_by_conflict_or_exclusion]
        if not ex_atoms:
            continue
        related = [
            e
            for e in excludes_edges
            if any(
                atom_by_id.get(aid) and anchor_key in atom_by_id[aid].entity_keys
                for aid in (e.from_atom_id, e.to_atom_id)
            )
        ]
        conflict_targets = [
            atom_by_id[e.to_atom_id]
            for e in related
            if atom_by_id.get(e.to_atom_id) is not None
            and atom_by_id[e.to_atom_id].atom_type in {AtomType.scope_item, AtomType.quantity}
            and e.to_atom_id not in consumed_by_conflict_or_exclusion
        ]
        all_atoms = ex_atoms + conflict_targets
        has_transcript_exclusion = any(
            atom.authority_class == AuthorityClass.meeting_note for atom in ex_atoms
        )
        status = (
            PacketStatus.needs_review
            if conflict_targets or has_transcript_exclusion
            else PacketStatus.active
        )
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.scope_exclusion,
            atoms=all_atoms,
            related_edges=related,
            status=status,
            reason="Exclusion directive identified for scoped work.",
            contradicting_atom_ids=[a.id for a in conflict_targets],
            review_flags=["exclusion_present"],
            prefer_customer_exclusion=True,
        )
        packets.append(packet)
        consumed_by_conflict_or_exclusion.update(a.id for a in all_atoms)

    # 4) site_access — lifts, catwalk (when paired), after-hours, escort/credential, ceiling access, work windows
    access_candidates = [
        a
        for a in atoms
        if a.id not in consumed_by_conflict_or_exclusion and _atom_eligible_site_access_evidence(a)
    ]
    by_site_access: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in access_candidates:
        site_key = _best_site_key([atom])
        by_site_access[site_key].append(atom)
    for site_key in sorted(by_site_access.keys()):
        group = sorted(by_site_access[site_key], key=lambda a: a.id)
        if not _group_triggers_site_access(group):
            continue
        related_sa = [
            e for e in edges if any(a.id in {e.from_atom_id, e.to_atom_id} for a in group)
        ]
        status = PacketStatus.needs_review if _site_access_group_unresolved(group) else PacketStatus.active
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.site_access,
            atoms=group,
            related_edges=related_sa,
            status=status,
            reason=_site_access_packet_reason(group),
            review_flags=_site_access_review_flags(group),
        )
        packets.append(packet)

    # 4b) missing_info — badge / MDF / IDF access unknown (open questions only)
    for atom in atoms:
        if atom.id in consumed_by_conflict_or_exclusion:
            continue
        if not _open_question_access_gate_unknown(atom):
            continue
        related_g = [e for e in edges if atom.id in {e.from_atom_id, e.to_atom_id}]
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.missing_info,
            atoms=[atom],
            related_edges=related_g,
            status=PacketStatus.needs_review,
            reason=(
                "Badge, MDF/IDF room, or credential-based access is unknown or unresolved; "
                "confirm escort, badging, and telecom room entry before mobilizing."
            ),
            review_flags=["missing_info_access_gate", "site_access_gate_unknown"],
            material_identity="site_access_gate",
        )
        packets.append(packet)
        consumed_by_conflict_or_exclusion.add(atom.id)

    # 5) meeting_decision
    meeting_decision_atoms = [a for a in atoms if a.atom_type in {AtomType.decision, AtomType.meeting_commitment}]
    for atom in meeting_decision_atoms:
        same_anchor_atoms = [
            other
            for other in atoms
            if other.id != atom.id
            and set(other.entity_keys).intersection(set(atom.entity_keys))
            and other.atom_type in {AtomType.scope_item, AtomType.exclusion, AtomType.quantity, AtomType.customer_instruction}
        ]
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.meeting_decision,
            atoms=[atom] + same_anchor_atoms,
            related_edges=[e for e in edges if atom.id in {e.from_atom_id, e.to_atom_id}],
            status=PacketStatus.needs_review if is_scope_impacting_meeting_atom(atom) else PacketStatus.active,
            reason="Meeting decision captured from transcript evidence.",
            review_flags=["verbal_commitment_requires_confirmation"],
        )
        packets.append(packet)

    # 6) action_item
    action_items = [a for a in atoms if a.atom_type == AtomType.action_item]
    for atom in action_items:
        if atom.id in consumed_by_conflict_or_exclusion:
            continue
        if _action_item_deferred_to_raceway_missing_info(atom):
            continue
        owner = str(atom.value.get("owner", "")).strip().lower()
        risky = _is_risky_action_item(atom)
        status = PacketStatus.active
        if not owner or owner == "unknown" or risky:
            status = PacketStatus.needs_review
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.action_item,
            atoms=[atom],
            related_edges=[e for e in edges if atom.id in {e.from_atom_id, e.to_atom_id}],
            status=status,
            reason="Action item extracted from transcript.",
            owner=owner or "unknown",
        )
        packets.append(packet)

    # 7) missing_info — raceway/conduit pathway, allowance, and condition (before generic open questions)
    raceway_primaries = [a for a in atoms if _primary_atom_triggers_raceway_conduit_missing_info(a)]
    for primary in sorted(raceway_primaries, key=lambda a: a.id):
        if primary.id in consumed_by_conflict_or_exclusion:
            continue
        vendors = [
            v
            for v in atoms
            if v.id != primary.id
            and _vendor_raceway_conduit_supporting_atom(v)
            and v.id not in consumed_by_conflict_or_exclusion
        ]
        pack_atoms = [primary] + vendors
        related_mi = [
            e for e in edges if any(x.id in {e.from_atom_id, e.to_atom_id} for x in pack_atoms)
        ]
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.missing_info,
            atoms=pack_atoms,
            related_edges=related_mi,
            status=PacketStatus.needs_review,
            reason=(
                "Raceway/conduit pathway, allowance, or existing infrastructure condition is unclear "
                "(unknown, verify, allowance, or usability); confirm scope and pricing before treating as firm in-scope."
            ),
            review_flags=[
                "raceway_conduit_pathway_missing_info",
                "pathway_allowance_or_condition_uncertainty",
            ],
            prefer_customer_raceway_instruction=True,
            material_identity="raceway_conduit",
        )
        packets.append(packet)
        consumed_by_conflict_or_exclusion.update(a.id for a in pack_atoms)

    open_questions = [a for a in atoms if a.atom_type == AtomType.open_question]
    for atom in open_questions:
        if atom.id in consumed_by_conflict_or_exclusion:
            continue
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.missing_info,
            atoms=[atom],
            related_edges=[e for e in edges if atom.id in {e.from_atom_id, e.to_atom_id}],
            status=PacketStatus.needs_review,
            reason="Open question from transcript requires clarification.",
        )
        packets.append(packet)

    # 8) customer_override
    customer_instructions = [
        a
        for a in atoms
        if a.atom_type == AtomType.customer_instruction and a.authority_class == AuthorityClass.customer_current_authored
    ]
    for atom in customer_instructions:
        conflicts = [
            other
            for other in atoms
            if other.id != atom.id
            and set(other.entity_keys).intersection(set(atom.entity_keys))
            and other.atom_type in {AtomType.scope_item, AtomType.exclusion, AtomType.quantity}
        ]
        status = PacketStatus.needs_review if conflicts else PacketStatus.active
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.customer_override,
            atoms=[atom] + conflicts,
            related_edges=[e for e in edges if atom.id in {e.from_atom_id, e.to_atom_id}],
            status=status,
            reason="Customer current instruction overrides prior context.",
            contradicting_atom_ids=[c.id for c in conflicts],
            review_flags=["customer_current_override"],
        )
        packets.append(packet)

    # 9) scope_inclusion (requires governing written scope / roster / customer / approved meeting;
    #    vendor quote never defines scope alone; vendor pollution / excluded lines diverted.)
    inclusion_candidates = [
        a
        for a in atoms
        if a.atom_type in {AtomType.scope_item, AtomType.quantity}
        and a.id not in consumed_by_conflict_or_exclusion
    ]
    for atom in inclusion_candidates:
        diversion = _vendor_scope_diversion_kind(atom)
        if diversion is None:
            continue
        related_edges = [e for e in edges if atom.id in {e.from_atom_id, e.to_atom_id}]
        paired_gov: EvidenceAtom | None = None
        if diversion == "pollution":
            power_anchor = "scope:power" if _vendor_electrical_power_line(atom) else None
            gov_cands = [
                a
                for a in atoms
                if a.id != atom.id
                and _atom_documents_power_out_of_scope(a)
                and _power_governance_covers_vendor(a, atom)
            ]
            if gov_cands:
                gov_atom = min(gov_cands, key=_gov_sort_key_power)
                paired_gov = gov_atom
                pollution_atoms = [gov_atom, atom]
                related_edges = [
                    e
                    for e in edges
                    if atom.id in {e.from_atom_id, e.to_atom_id} or gov_atom.id in {e.from_atom_id, e.to_atom_id}
                ]
                packet = _build_packet(
                    project_id=project_id,
                    family=PacketFamily.scope_exclusion,
                    atoms=pollution_atoms,
                    related_edges=related_edges,
                    status=PacketStatus.needs_review,
                    reason=(
                        "Vendor quote includes electrical or utility power line items despite written scope / customer "
                        "evidence excluding that power work; do not treat vendor power as structured low-voltage scope."
                    ),
                    contradicting_atom_ids=[atom.id],
                    review_flags=[
                        "power_vendor_scope_mismatch",
                        "scope_pollution_vendor_vs_written_exclusion",
                        "vendor_scope_pollution_candidate",
                        "vendor_quote_not_scope_governor",
                    ],
                    prefer_customer_exclusion=True,
                    material_identity=power_anchor,
                )
            else:
                packet = _build_packet(
                    project_id=project_id,
                    family=PacketFamily.scope_exclusion,
                    atoms=[atom],
                    related_edges=related_edges,
                    status=PacketStatus.needs_review,
                    reason=(
                        "Vendor electrical/power line is a scope pollution candidate; "
                        "do not treat as structured scope inclusion."
                    ),
                    review_flags=["vendor_scope_pollution_candidate", "vendor_quote_not_scope_governor"],
                    material_identity=power_anchor,
                )
        else:
            cert_mat = "certification" if _vendor_certification_testing_excluded_supporting_atom(atom) else None
            cert_reason = (
                "Cable certification / automated tester exports or pass-fail test reports are required by scope, "
                "but this vendor line is excluded, not included, or zero quantity; confirm deliverables."
            )
            raceway_mat = (
                None
                if cert_mat
                else ("raceway_conduit" if _vendor_raceway_conduit_supporting_atom(atom) else None)
            )
            raceway_reason = (
                "Vendor raceway/conduit line is excluded, not included, or zero quantity; pathway allowance and "
                "existing infrastructure condition remain unclear — confirm against customer scope and allowances."
            )
            packet = _build_packet(
                project_id=project_id,
                family=PacketFamily.missing_info,
                atoms=[atom],
                related_edges=related_edges,
                status=PacketStatus.needs_review,
                reason=cert_reason
                if cert_mat
                else (
                    raceway_reason
                    if raceway_mat
                    else (
                        "Vendor line marked excluded or not included does not establish scope; "
                        "align with written scope, testing requirements, or procurement coverage."
                    )
                ),
                review_flags=["vendor_excluded_line", "vendor_quote_not_scope_governor"]
                + (["certification_testing_export_missing_info"] if cert_mat else [])
                + (["raceway_conduit_pathway_missing_info"] if raceway_mat else []),
                material_identity=cert_mat or raceway_mat,
            )
        packets.append(packet)
        consumed_by_conflict_or_exclusion.add(atom.id)
        if paired_gov is not None:
            consumed_by_conflict_or_exclusion.add(paired_gov.id)

    inclusion_candidates = [
        a
        for a in atoms
        if a.atom_type in {AtomType.scope_item, AtomType.quantity}
        and a.id not in consumed_by_conflict_or_exclusion
    ]
    grouped_inclusions: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in inclusion_candidates:
        _, anchor_key = _anchor_for_atoms([atom])
        grouped_inclusions[anchor_key].append(atom)
    for anchor_key in sorted(grouped_inclusions):
        group_atoms = grouped_inclusions[anchor_key]
        if not _group_has_scope_inclusion_governor(group_atoms):
            continue
        related_edges = [e for e in edges if any(a.id in {e.from_atom_id, e.to_atom_id} for a in group_atoms)]
        packet = _build_packet(
            project_id=project_id,
            family=PacketFamily.scope_inclusion,
            atoms=group_atoms,
            related_edges=related_edges,
            status=PacketStatus.needs_review
            if any(is_scope_impacting_meeting_atom(atom) for atom in group_atoms)
            else PacketStatus.active,
            reason="Scoped inclusion evidence is consistent.",
        )
        packets.append(packet)

    # Deduplicate packets by family + canonical anchor signature + atom set (scope_exclusion can share a site).
    dedup: dict[tuple[str, str, str], EvidencePacket] = {}
    for packet in packets:
        signature_hash = packet.anchor_signature.hash if packet.anchor_signature is not None else packet.anchor_key
        atom_sig = "|".join(
            sorted(set(packet.governing_atom_ids + packet.supporting_atom_ids + packet.contradicting_atom_ids))
        )
        key = (packet.family.value, signature_hash, atom_sig)
        if key not in dedup:
            dedup[key] = packet
        else:
            existing = dedup[key]
            if (
                packet.family == PacketFamily.quantity_conflict
                and "Aggregate scoped quantity" in packet.reason
                and "Aggregate scoped quantity" not in existing.reason
            ):
                dedup[key] = packet
            if (
                packet.family == PacketFamily.vendor_mismatch
                and "Aggregate scoped quantity" in packet.reason
                and "Aggregate scoped quantity" not in existing.reason
            ):
                dedup[key] = packet

    merged = list(dedup.values())
    # Multiple roster↔vendor contradict edges can yield separate vendor_mismatch packets with the same anchor.
    vm_by_sig_hash: dict[str, EvidencePacket] = {}
    deduped_packets: list[EvidencePacket] = []
    for p in merged:
        if p.family != PacketFamily.vendor_mismatch or p.anchor_signature is None:
            deduped_packets.append(p)
            continue
        h = p.anchor_signature.hash
        cur = vm_by_sig_hash.get(h)
        if cur is None:
            vm_by_sig_hash[h] = p
            continue
        if "Aggregate scoped quantity" in p.reason and "Aggregate scoped quantity" not in cur.reason:
            vm_by_sig_hash[h] = p
        elif "Aggregate scoped quantity" in cur.reason and "Aggregate scoped quantity" not in p.reason:
            pass
        elif len(p.reason or "") > len(cur.reason or ""):
            vm_by_sig_hash[h] = p
    deduped_packets.extend(vm_by_sig_hash.values())
    result = deduped_packets
    if attach_metadata:
        atom_by_id = {atom.id: atom for atom in atoms}
        edge_by_id = {e.id: e for e in edges}
        for packet in result:
            packet.certificate = build_packet_certificate(packet, atom_by_id, edge_by_id=edge_by_id)
            packet_atoms = [
                atom_by_id[atom_id]
                for atom_id in (packet.supporting_atom_ids + packet.contradicting_atom_ids)
                if atom_id in atom_by_id
            ]
            packet.risk = score_packet_risk(packet, packet_atoms, edges)
    result = sorted(result, key=lambda p: packet_pm_sort_key(p) if p.risk is not None else (50, 50, 0.0, p.anchor_key, p.id))
    return result
