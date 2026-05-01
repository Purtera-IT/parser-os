from __future__ import annotations

from app.domain.loader import DEFAULT_PACK_ID, load_domain_pack
from app.domain.schemas import DomainEntityType, DomainPack
from app.domain.suggestions import RuleSuggestion, RuleSuggestionFile

_ACTIVE_DOMAIN_PACK: DomainPack | None = None


def get_active_domain_pack() -> DomainPack:
    global _ACTIVE_DOMAIN_PACK
    if _ACTIVE_DOMAIN_PACK is None:
        _ACTIVE_DOMAIN_PACK = load_domain_pack(DEFAULT_PACK_ID)
    return _ACTIVE_DOMAIN_PACK


def set_active_domain_pack(pack: DomainPack) -> None:
    global _ACTIVE_DOMAIN_PACK
    _ACTIVE_DOMAIN_PACK = pack


__all__ = [
    "DomainEntityType",
    "DomainPack",
    "DEFAULT_PACK_ID",
    "load_domain_pack",
    "get_active_domain_pack",
    "set_active_domain_pack",
    "RuleSuggestion",
    "RuleSuggestionFile",
]
