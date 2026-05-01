from __future__ import annotations

from pathlib import Path

from app.parsers.parser_router import parser_capabilities
from app.parsers.registry import choose_parser, get_registered_parsers


def test_parser_capabilities_are_listed() -> None:
    caps = parser_capabilities()
    names = {cap.parser_name for cap in caps}
    assert {"xlsx", "quote", "email", "transcript", "docx"}.issubset(names)


def test_routing_is_deterministic_for_same_artifact(demo_project: Path) -> None:
    target = demo_project / "vendor_quote.xlsx"
    first_parser, first_match, _ = choose_parser(target, domain_pack=None)
    second_parser, second_match, _ = choose_parser(target, domain_pack=None)
    assert first_parser is not None and second_parser is not None
    assert first_match.parser_name == second_match.parser_name
    assert first_match.confidence == second_match.confidence
    assert first_match.reasons == second_match.reasons


def test_registry_returns_registered_parsers() -> None:
    parsers = get_registered_parsers()
    assert parsers
    assert all(hasattr(parser, "capability") for parser in parsers)
