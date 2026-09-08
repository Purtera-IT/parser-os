"""An alias has to be in the documents.

The LLM that clusters site surface forms invents names. Measured on a
140-envelope sample (2026-09-07), 84 of 242 alias names attached to site atoms
appear NOWHERE in their own deal — "rpdu 230 sites", "backup modem 284 sites".
The existing guard is a boilerplate word list, and a word list only knows the
words somebody already wrote down. Provenance is the universal test.
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.core.entity_resolution import (
    _deal_evidence_text,
    _name_appears_in_evidence,
    _normalize_for_evidence,
)


class _Atom:
    def __init__(self, text: str) -> None:
        self.text = text


EVIDENCE = _deal_evidence_text([
    _Atom("625 W. Adams St |, Chicago, IL 60661"),
    _Atom("Merrill Gardens at Tacoma — walkthrough scheduled"),
])


def test_a_name_the_documents_contain_is_kept() -> None:
    assert _name_appears_in_evidence("Merrill Gardens at Tacoma", EVIDENCE) is True
    # Casing and punctuation must not decide it.
    assert _name_appears_in_evidence("merrill gardens at tacoma", EVIDENCE) is True
    assert _name_appears_in_evidence("625 W Adams St", EVIDENCE) is True


def test_a_name_the_documents_never_mention_is_refused() -> None:
    for invented in ("rpdu 230 sites", "backup modem 284 sites", "Jacksonville, FL"):
        assert _name_appears_in_evidence(invented, EVIDENCE) is False, invented


def test_it_degrades_open_when_there_is_no_evidence() -> None:
    """Absence of proof is not proof of absence: with nothing collected we
    cannot show a name was invented, and refusing every alias would be worse
    than the problem."""
    assert _name_appears_in_evidence("Anything At All", "") is True


def test_an_empty_name_is_never_evidence() -> None:
    assert _name_appears_in_evidence("   ", EVIDENCE) is False


def test_evidence_text_reads_objects_and_dicts() -> None:
    assert "chicago" in _deal_evidence_text([_Atom("Chicago, IL")])
    assert "chicago" in _deal_evidence_text([{"text": "Chicago, IL"}])
    assert _deal_evidence_text([]) == ""
    assert _normalize_for_evidence("St. Louis, MO!") == "st louis mo"


def test_the_fusion_block_still_lives_in_its_own_function() -> None:
    """A regression test for a mistake I made writing this change.

    Inserting these module-level helpers into the middle of
    `collect_site_alias_groups` truncated it and moved the cluster-fusion
    block into an unrelated function. It still compiled and the import still
    worked — only the behaviour was gone.
    """
    source = Path("app/core/entity_resolution.py").read_text()
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "collect_site_alias_groups"
    )
    body = ast.get_source_segment(source, fn) or ""
    assert "LLM SITE-CLUSTER FUSION" in body
    assert "_deal_evidence_text" in body
