from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.domain.schemas import DomainPack

DOMAIN_DIR = Path(__file__).resolve().parent
DEFAULT_PACK_ID = "default_pack"


def _candidate_pack_path(pack_id_or_path: str | Path) -> Path:
    candidate = Path(pack_id_or_path)
    if candidate.exists():
        return candidate
    pack_id = str(pack_id_or_path).strip()
    if not pack_id:
        return DOMAIN_DIR / "default_pack.yaml"
    return DOMAIN_DIR / f"{pack_id}.yaml"


def _parse_domain_file(target: Path) -> dict[str, Any]:
    text = target.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        return json.loads(text)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Domain pack must be a mapping object: {target}")
    return loaded


def _looks_like_wide_reference_pack(payload: dict[str, Any]) -> bool:
    """Heuristic: copper_cabling-style JSON reference vs slim DomainPack YAML."""
    if "domain_pack_id" in payload and "pack_id" not in payload:
        return True
    arp = payload.get("artifact_role_patterns")
    if isinstance(arp, dict) and arp:
        sample = next(iter(arp.values()), None)
        if isinstance(sample, dict):
            return True
    return False


def _adapt_reference_pack_to_domain_pack(payload: dict[str, Any], *, source_path: Path) -> DomainPack:
    """
    Map a wide reference pack (e.g. copper_cabling.yaml) onto DomainPack using default_pack as base.

    TODO(strict-schema): Map nested artifact_role_patterns, comparison_rules, and ontology-linked fields
    into DomainPack instead of inheriting defaults; keep ontology/copper_low_voltage_ontology.yaml as
    the semantic source of truth until that adapter exists.
    """
    default_path = DOMAIN_DIR / "default_pack.yaml"
    base = DomainPack.model_validate(yaml.safe_load(default_path.read_text(encoding="utf-8")))
    pack_id = str(payload.get("pack_id") or payload.get("domain_pack_id") or source_path.stem).strip()
    name = str(payload.get("name") or pack_id)
    version = str(payload.get("version") or "0.0.0")
    merged_risk = dict(base.risk_defaults)
    rd = payload.get("risk_defaults")
    if isinstance(rd, dict):
        for k, v in rd.items():
            try:
                merged_risk[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    ref_onto: str | None = None
    onto = DOMAIN_DIR / "ontology" / "copper_low_voltage_ontology.yaml"
    if onto.is_file():
        ref_onto = "ontology/copper_low_voltage_ontology.yaml"
    sl = payload.get("service_lines")
    service_lines = [str(x) for x in sl] if isinstance(sl, list) else base.service_lines
    return base.model_copy(
        update={
            "pack_id": pack_id,
            "name": name,
            "version": version,
            "risk_defaults": merged_risk,
            "service_lines": service_lines or [pack_id],
            "reference_ontology_path": ref_onto,
        }
    )


def load_domain_pack(pack_id_or_path: str | Path | None = None) -> DomainPack:
    if pack_id_or_path is None:
        target = DOMAIN_DIR / "default_pack.yaml"
    else:
        target = _candidate_pack_path(pack_id_or_path)
        if not target.exists():
            target = DOMAIN_DIR / "default_pack.yaml"
    try:
        payload = _parse_domain_file(target)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid domain pack JSON at '{target}': {exc}") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid domain pack YAML at '{target}': {exc}") from None
    except FileNotFoundError:
        raise ValueError(f"Domain pack file not found: {target}") from None
    try:
        return DomainPack.model_validate(payload)
    except ValidationError as exc:
        if _looks_like_wide_reference_pack(payload):
            return _adapt_reference_pack_to_domain_pack(payload, source_path=target)
        raise ValueError(f"Invalid domain pack schema in '{target}': {exc}") from exc

