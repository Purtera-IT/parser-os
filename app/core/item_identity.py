from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.normalizers import normalize_text

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("rj-45", "rj45").replace("cat 6a", "cat6a").replace("cat-6a", "cat6a")
    text = text.replace("cat 6", "cat6").replace("cat-6", "cat6")
    text = text.replace("category 6a", "cat6a").replace("category 6", "cat6")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> set[str]:
    return set(_WORD_RE.findall(_norm(value)))


@dataclass(frozen=True)
class IdentitySpec:
    canonical_key: str
    item_kind: str
    material_family: str
    comparison_group: str
    synonyms: tuple[str, ...]
    required_any_tokens: tuple[tuple[str, ...], ...] = ()
    negative_tokens: tuple[str, ...] = ()
    scope_pollution_candidate: bool = False


@dataclass(frozen=True)
class IdentityResult:
    canonical_key: str
    item_kind: str
    material_family: str
    comparison_group: str
    confidence: float
    matched_by: str
    matched_text: str
    scope_pollution_candidate: bool = False
    inferred: bool = False
    review_flags: tuple[str, ...] = field(default_factory=tuple)


# TODO(strict-schema): Wide copper reference lives in app/domain/copper_cabling.yaml; runtime DomainPack is
# produced via app.domain.loader._adapt_reference_pack_to_domain_pack (subset + default_pack merge).
IDENTITY_SPECS: tuple[IdentitySpec, ...] = (
    IdentitySpec("cat6a_utp", "cable_drop", "copper_cabling", "copper_cable_category_shielding", ("cat6a utp", "cat6a unshielded", "unshielded cat6a", "category 6a utp"), (("cat6a", "utp"), ("cat6a", "unshielded")), ("stp", "shielded")),
    IdentitySpec("cat6a_stp", "cable_drop", "copper_cabling", "copper_cable_category_shielding", ("cat6a stp", "cat6a shielded", "shielded cat6a", "category 6a stp"), (("cat6a", "stp"), ("cat6a", "shielded")), ("utp", "unshielded")),
    IdentitySpec("cat6_utp", "cable_drop", "copper_cabling", "copper_cable_category_shielding", ("cat6 utp", "cat6 unshielded", "unshielded cat6", "category 6 utp", "utp cat6"), (("cat6", "utp"), ("cat6", "unshielded")), ("cat6a", "stp", "shielded")),
    IdentitySpec("cat6_stp", "cable_drop", "copper_cabling", "copper_cable_category_shielding", ("cat6 stp", "cat6 shielded", "shielded cat6", "category 6 stp", "stp cat6"), (("cat6", "stp"), ("cat6", "shielded")), ("cat6a", "utp", "unshielded")),
    IdentitySpec("cat6a", "cable_drop", "copper_cabling", "copper_cable_category", ("cat6a", "category 6a", "augmented category 6"), (("cat6a",),), ("cat6", "utp", "stp")),
    IdentitySpec("cat6", "cable_drop", "copper_cabling", "copper_cable_category", ("cat6", "category 6"), (("cat6",),), ("cat6a",)),
    IdentitySpec("cat5e", "cable_drop", "copper_cabling", "copper_cable_category", ("cat5e", "cat 5e", "category 5e"), (("cat5e",), ("cat", "5e")), ("cat6", "cat6a")),
    IdentitySpec("rj45", "termination", "connector", "copper_endpoint", ("rj45", "data jack", "jack", "modular jack", "keystone jack", "data outlet", "comm outlet", "work area outlet", "ethernet jack", "network port", "termination", "terminations"), (("rj45",), ("data", "jack"), ("comm", "outlet"), ("work", "area", "outlet"), ("termination",)), ("power", "fiber", "speaker", "audio")),
    IdentitySpec("data_drop", "cable_drop", "copper_cabling", "copper_drop", ("data drop", "drop", "cable run", "copper run", "horizontal cabling", "network drop", "ethernet drop", "low voltage drop"), (("drop",), ("cable", "run"), ("horizontal", "cabling")), ("power", "audio", "speaker", "fiber", "rg59")),
    IdentitySpec("patch_panel", "patch_panel", "copper_accessory", "copper_accessory", ("patch panel", "keystone patch panel", "modular patch panel"), (("patch", "panel"),), ("electrical", "power")),
    IdentitySpec("patch_cord", "patch_cord", "copper_accessory", "copper_accessory", ("patch cord", "patch cable", "ethernet patch cable", "jumper"), (("patch", "cord"), ("patch", "cable"), ("jumper",)), ("fiber",)),
    IdentitySpec("faceplate", "faceplate", "copper_accessory", "copper_accessory", ("faceplate", "face plate", "wall plate", "wallplate"), (("faceplate",), ("wall", "plate")), ()),
    IdentitySpec("raceway", "raceway", "pathway", "pathway", ("raceway", "surface raceway", "wiremold", "wire mold", "pathway"), (("raceway",), ("wiremold",), ("pathway",)), ()),
    IdentitySpec("conduit", "conduit", "pathway", "pathway", ("conduit", "emt", "sleeve", "pvc conduit"), (("conduit",), ("emt",), ("sleeve",)), ()),
    IdentitySpec("certification_testing", "certification", "testing", "testing_requirement", ("certification", "certify", "tester export", "test report", "fluke report", "wiremap", "wire map", "tia 568"), (("certification",), ("tester", "export"), ("test", "report"), ("wiremap",), ("wire", "map")), ()),
    IdentitySpec("labeling", "labeling", "documentation", "documentation_requirement", ("label", "labeling", "label standard", "cable id"), (("label",), ("labeling",), ("cable", "id")), ()),
    IdentitySpec("as_built", "documentation", "documentation", "documentation_requirement", ("as built", "as-built", "redline", "record drawing"), (("as", "built"), ("redline",), ("record", "drawing")), ()),
    IdentitySpec("power", "power", "electrical", "scope_pollution", ("power", "electrical", "20 amp", "20a", "120v", "receptacle", "electrical outlet", "circuit"), (("power",), ("20", "amp"), ("electrical",), ("receptacle",), ("circuit",)), ("poe", "ethernet"), True),
    IdentitySpec("poe", "network_power", "network_dependency", "network_dependency", ("poe", "poe+", "poe++", "power over ethernet", "802.3af", "802.3at", "802.3bt"), (("poe",), ("power", "over", "ethernet")), ()),
    IdentitySpec("mdf", "telecom_space", "site_entity", "site_readiness", ("mdf", "main distribution frame", "main telecom room", "core room"), (("mdf",), ("main", "distribution", "frame")), ()),
    IdentitySpec("idf", "telecom_space", "site_entity", "site_readiness", ("idf", "intermediate distribution frame", "telecom closet", "network closet", "wiring closet"), (("idf",), ("telecom", "closet"), ("network", "closet")), ()),
    IdentitySpec("lift_access", "access_constraint", "site_access", "site_access", ("lift", "scissor lift", "boom lift", "catwalk", "aerial lift"), (("lift",), ("catwalk",)), ("elevator",)),
    IdentitySpec("after_hours", "access_constraint", "site_access", "site_access", ("after hours", "after-hours", "night work", "weekend", "maintenance window"), (("after", "hours"), ("night", "work"), ("weekend",), ("maintenance", "window")), ()),
    IdentitySpec("badge_access", "access_constraint", "site_access", "site_access", ("badge", "badge access", "escort", "security escort", "background check", "site access"), (("badge",), ("escort",), ("background", "check"), ("site", "access")), ()),
    IdentitySpec("fiber", "fiber_cable", "fiber", "fiber", ("fiber", "fibre", "fiber optic", "strand", "om3", "om4", "os2", "single mode", "multimode"), (("fiber",), ("fibre",), ("strand",), ("om3",), ("om4",), ("os2",)), ()),
    IdentitySpec("rg59", "coax_cable", "coax", "low_voltage_av", ("rg59", "rg-59", "coax", "coaxial", "bnc"), (("rg59",), ("rg", "59"), ("coax",)), ()),
    IdentitySpec("speaker_cable", "speaker_cable", "av_cabling", "low_voltage_av", ("speaker cable", "speaker wire", "12 awg speaker"), (("speaker", "cable"), ("speaker", "wire")), ()),
    IdentitySpec("audio_cable", "audio_cable", "av_cabling", "low_voltage_av", ("audio cable", "audio signal cable", "microphone cable", "balanced audio"), (("audio", "cable"), ("audio", "signal"), ("microphone", "cable")), ()),
)


def _atom_value(atom_or_value: Any) -> dict[str, Any]:
    if isinstance(atom_or_value, dict):
        if "value" in atom_or_value and isinstance(atom_or_value["value"], dict):
            return atom_or_value["value"]
        return atom_or_value
    value = getattr(atom_or_value, "value", None)
    return value if isinstance(value, dict) else {}


def _atom_raw_text(atom_or_value: Any, raw_text: str | None = None) -> str:
    chunks: list[str] = []
    if raw_text:
        chunks.append(raw_text)
    if isinstance(atom_or_value, dict):
        chunks.append(str(atom_or_value.get("raw_text", "")))
        chunks.append(str(atom_or_value.get("normalized_text", "")))
    else:
        chunks.append(str(getattr(atom_or_value, "raw_text", "")))
        chunks.append(str(getattr(atom_or_value, "normalized_text", "")))
    value = _atom_value(atom_or_value)
    for key in ("normalized_item", "item", "description", "material_spec", "device", "name", "text", "scope", "constraint_type"):
        if value.get(key):
            chunks.append(str(value[key]))
    if isinstance(atom_or_value, dict):
        for key in atom_or_value.get("entity_keys", []) or []:
            chunks.append(str(key).replace(":", " "))
    else:
        for key in getattr(atom_or_value, "entity_keys", []) or []:
            chunks.append(str(key).replace(":", " "))
    return " ".join(c for c in chunks if c)


def _blocked_by_negative_tokens(spec: IdentitySpec, text: str) -> bool:
    toks = _tokens(text)
    norm = _norm(text)
    for neg in spec.negative_tokens:
        neg_norm = _norm(neg)
        if not neg_norm:
            continue
        if " " in neg_norm:
            if neg_norm in norm:
                return True
        elif neg_norm in toks:
            return True
    return False


def _score_spec(spec: IdentitySpec, text: str) -> tuple[float, str]:
    norm = _norm(text)
    toks = _tokens(text)
    if _blocked_by_negative_tokens(spec, text):
        return 0.0, "negative_token_block"

    # Exact/synonym phrase match.
    for syn in spec.synonyms:
        syn_norm = _norm(syn)
        if syn_norm and (syn_norm == norm or re.search(rf"(^|\s){re.escape(syn_norm)}($|\s)", norm)):
            return 0.98 if syn_norm == norm else 0.94, f"synonym:{syn}"

    # Required token groups.
    for group in spec.required_any_tokens:
        group_norm = [_norm(g) for g in group]
        if all(g in toks or g in norm for g in group_norm if g):
            return 0.90, "required_token_group:" + "+".join(group)

    # Token overlap fallback using synonyms.
    best = 0.0
    best_syn = ""
    for syn in spec.synonyms:
        syn_toks = _tokens(syn)
        if not syn_toks:
            continue
        overlap = len(toks & syn_toks) / max(len(syn_toks), 1)
        if overlap > best:
            best = overlap
            best_syn = syn
    if best >= 0.72:
        return 0.72 + min(best - 0.72, 0.18), f"token_overlap:{best_syn}"
    return 0.0, "no_match"


def canonical_item_identity(atom_or_value: Any, raw_text: str | None = None, *, allow_multi: bool = False) -> IdentityResult | list[IdentityResult] | None:
    """Return canonical identity for an atom/value/raw text.

    Deterministic and dependency-free. Designed for use by parsers, graph builder,
    packetizer, gold comparison, and diagnostics.
    """
    text = _atom_raw_text(atom_or_value, raw_text)
    if not text.strip():
        return [] if allow_multi else None

    results: list[IdentityResult] = []
    for spec in IDENTITY_SPECS:
        score, by = _score_spec(spec, text)
        if score <= 0:
            continue
        flags: list[str] = []
        if score < 0.86:
            flags.append("identity_low_confidence")
        results.append(
            IdentityResult(
                canonical_key=spec.canonical_key,
                item_kind=spec.item_kind,
                material_family=spec.material_family,
                comparison_group=spec.comparison_group,
                confidence=round(score, 3),
                matched_by=by,
                matched_text=_norm(text)[:240],
                scope_pollution_candidate=spec.scope_pollution_candidate,
                inferred=False,
                review_flags=tuple(flags),
            )
        )

    # Multi-interpretation: a plain "data drop" often implies a copper run and an RJ45 endpoint,
    # but keep both lower-confidence so packetizer can require review if needed.
    norm = _norm(text)
    if allow_multi and ("data drop" in norm or "network drop" in norm or "comm outlet" in norm or "work area outlet" in norm):
        keys = {r.canonical_key for r in results}
        if "rj45" not in keys:
            results.append(IdentityResult("rj45", "termination", "connector", "copper_endpoint", 0.78, "inferred_from_drop_language", norm[:240], inferred=True, review_flags=("inferred_identity",)))
        if "data_drop" not in keys:
            results.append(IdentityResult("data_drop", "cable_drop", "copper_cabling", "copper_drop", 0.78, "inferred_from_drop_language", norm[:240], inferred=True, review_flags=("inferred_identity",)))

    # Prefer more specific cable identities over generic ones.
    specificity_order = {
        "cat6a_utp": 100, "cat6a_stp": 100, "cat6_utp": 95, "cat6_stp": 95,
        "cat6a": 80, "cat6": 70, "data_drop": 60, "rj45": 60,
    }
    results.sort(key=lambda r: (r.confidence, specificity_order.get(r.canonical_key, 50)), reverse=True)

    if allow_multi:
        deduped: list[IdentityResult] = []
        seen: set[str] = set()
        for r in results:
            if r.canonical_key not in seen:
                deduped.append(r)
                seen.add(r.canonical_key)
        return deduped
    return results[0] if results else None


def canonical_material_key(atom_or_value: Any, raw_text: str | None = None) -> str | None:
    result = canonical_item_identity(atom_or_value, raw_text)
    if isinstance(result, IdentityResult):
        return result.canonical_key
    return None


def normalize_inclusion_status(value: Any) -> str:
    text = _norm(value)
    if not text:
        return "unknown"
    if text in {"yes", "y", "true", "1", "included", "include", "base bid", "base"}:
        return "included"
    if any(k in text for k in ("not included", "excluded", "exclude", "by others", "nic", "n i c", "out of scope")):
        return "excluded"
    if any(k in text for k in ("optional", "option", "alternate", "alt 1", "alt 2")):
        return "optional"
    if "allowance" in text:
        return "allowance"
    if any(k in text for k in ("tbd", "to be determined", "pending", "confirm")):
        return "tbd"
    if text in {"no", "n", "false", "0"}:
        return "excluded"
    return "unknown"


def _numeric_quantity_present(value: dict[str, Any]) -> bool:
    q = value.get("quantity")
    if q is None:
        return False
    try:
        float(q)
        return True
    except (TypeError, ValueError):
        return False


def is_primary_vendor_quantity(value: dict[str, Any], *, raw_text: str = "") -> bool:
    """True when a vendor quote quantity line should count toward primary (non-optional) totals."""
    if not isinstance(value, dict) or not _numeric_quantity_present(value):
        return False
    inc = str(value.get("inclusion_status") or "").strip().lower()
    if not inc or inc == "unknown":
        inc = normalize_inclusion_status(
            value.get("included") if value.get("included") is not None else value.get("notes", "")
        )
    if inc in {"excluded", "optional", "allowance", "tbd"}:
        return False
    if value.get("included") is False:
        return False
    qs = str(value.get("quantity_status") or "").lower()
    if qs in {"allowance", "tbd", "not_applicable", "included_no_qty"}:
        return False
    blob = normalize_text(
        f"{raw_text} {value.get('notes', '')} {value.get('item', '')} {value.get('description', '')}"
    ).lower()
    if re.search(r"\b(not included|by others|\bnics?\b|optional|alternate|allowance only)\b", blob):
        if inc != "included" and value.get("included") is not True:
            return False
    return True


# Explicit roster/vendor identities from parsers must not be replaced by inference.
_PARSER_IDENTITY_PROTECTED: frozenset[str] = frozenset(
    {"rj45", "cat6_utp", "cat6_stp", "cat6a_utp", "cat6a_stp", "total"}
)


def merge_parser_value_identity(value: dict[str, Any], raw_text: str | None = None) -> dict[str, Any]:
    """Apply enrich_value_with_identity when fields are missing or compatible with inference.

    Does not overwrite protected normalized_item values (rj45, cat6_utp, cat6_stp, …).
    If an existing normalized_item disagrees with inferred identity, keep the original key
    but still fill other missing identity fields from the preview.
    """
    if not isinstance(value, dict):
        return value
    if value.get("aggregate") is True and str(value.get("normalized_item", "")).strip().lower() == "total":
        out = dict(value)
        if "inclusion_status" not in out:
            src = out.get("included") if out.get("included") is not None else out.get("notes", "")
            out["inclusion_status"] = normalize_inclusion_status(src)
        return out

    ni = str(value.get("normalized_item") or "").strip().lower()
    if ni in _PARSER_IDENTITY_PROTECTED:
        out = dict(value)
        if "inclusion_status" not in out:
            src = out.get("included") if out.get("included") is not None else out.get("notes", "")
            out["inclusion_status"] = normalize_inclusion_status(src)
        preview = enrich_value_with_identity(dict(value), raw_text=raw_text)
        for k, v in preview.items():
            if k in {"normalized_item", "inclusion_status"}:
                continue
            if k not in out or out[k] in (None, "", [], {}):
                out[k] = v
        return out

    preview = enrich_value_with_identity(dict(value), raw_text=raw_text)
    new_ni = str(preview.get("normalized_item") or "").strip().lower()
    if ni and new_ni and new_ni != ni:
        out = dict(value)
        for k, v in preview.items():
            if k == "normalized_item":
                continue
            if k not in out or out[k] in (None, "", [], {}):
                out[k] = v
        if "inclusion_status" not in out and "inclusion_status" in preview:
            out["inclusion_status"] = preview["inclusion_status"]
        return out
    return preview


def enrich_value_with_identity(value: dict[str, Any], raw_text: str | None = None) -> dict[str, Any]:
    enriched = dict(value)
    result = canonical_item_identity(value, raw_text)
    if isinstance(result, IdentityResult):
        enriched.setdefault("normalized_item", result.canonical_key)
        enriched.setdefault("item_kind", result.item_kind)
        enriched.setdefault("material_family", result.material_family)
        enriched.setdefault("comparison_group", result.comparison_group)
        enriched.setdefault("identity_confidence", result.confidence)
        enriched.setdefault("identity_matched_by", result.matched_by)
        if result.scope_pollution_candidate:
            enriched["is_scope_pollution_candidate"] = True
        if result.review_flags:
            enriched.setdefault("identity_review_flags", list(result.review_flags))
    if "inclusion_status" not in enriched:
        source = enriched.get("included") if enriched.get("included") is not None else enriched.get("notes", "")
        enriched["inclusion_status"] = normalize_inclusion_status(source)
    return enriched
