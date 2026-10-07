"""duplicate_atom_collapse folds a repeat only inside ONE section.

Labeling rule: identical text under a different parent (day, site row,
section) is its own fact, never a repeat. A sentence stated under a scope
heading and again as a field under a service heading of the same document
was folded into the first copy, and the second statement left the compile.

The collapse already never folds across documents (intra-doc only); these
tests pin that it keeps doing so, and that a true repeat inside one section
still folds.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.core.entity_resolution import collapse_duplicate_atoms

SENTENCE = "The provider will deliver remote support during normal business hours."
LONG = (
    "The contractor shall coordinate all access windows with the facility "
    "manager before any work begins on site."
)
LONG_NEAR = LONG.replace("before any", "before all")


class _Atom:
    def __init__(self, text, *, section=None, artifact_id="art_doc1",
                 atom_type="scope_item", confidence=0.8, quoted=False):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.normalized_text = text
        self.value = {"quoted": True} if quoted else {}
        self.confidence = confidence
        self.artifact_id = artifact_id
        self.entity_keys = []
        loc = {}
        if section is not None:
            loc["section_path"] = list(section)
        self.source_refs = [SimpleNamespace(locator=loc)]


def test_same_text_two_sections_of_one_document_stays_two_atoms():
    a = _Atom(SENTENCE, section=["Agreement", "Project Scope"], confidence=0.9)
    b = _Atom(SENTENCE, section=["Agreement", "Service Model"], confidence=0.7)
    out = collapse_duplicate_atoms([a, b])
    assert len(out) == 2


def test_same_text_same_section_still_folds():
    a = _Atom(SENTENCE, section=["Agreement", "Project Scope"], confidence=0.9)
    b = _Atom(SENTENCE, section=["Agreement", "Project Scope"], confidence=0.7)
    out = collapse_duplicate_atoms([a, b])
    assert out == [a]


def test_section_compare_ignores_case_and_spacing():
    a = _Atom(SENTENCE, section=["Project  Scope"], confidence=0.9)
    b = _Atom(SENTENCE, section=["project scope"], confidence=0.7)
    assert len(collapse_duplicate_atoms([a, b])) == 1


def test_a_copy_with_no_section_still_folds():
    # A recall pass re-reading the page carries no heading; it is the same
    # span, not a different parent.
    a = _Atom(SENTENCE, section=["Project Scope"], confidence=0.9)
    b = _Atom(SENTENCE, section=None, atom_type="entity", confidence=0.7)
    assert collapse_duplicate_atoms([a, b]) == [a]


def test_quoted_copy_still_folds_across_sections():
    a = _Atom(SENTENCE, section=["Scope"], confidence=0.9)
    b = _Atom(SENTENCE, section=["Original message"], confidence=0.7, quoted=True)
    assert collapse_duplicate_atoms([a, b]) == [a]


def test_same_text_two_documents_unchanged():
    # Cross-document repetition was never folded here (corroboration); it
    # stays that way whether or not the sections match.
    a = _Atom(SENTENCE, section=["Project Scope"], artifact_id="art_doc1")
    b = _Atom(SENTENCE, section=["Project Scope"], artifact_id="art_doc2")
    c = _Atom(SENTENCE, section=["Service Model"], artifact_id="art_doc3")
    assert len(collapse_duplicate_atoms([a, b, c])) == 3


def test_near_duplicate_two_sections_stays_two_atoms():
    a = _Atom(LONG, section=["Site Access"], confidence=0.9)
    b = _Atom(LONG_NEAR, section=["Schedule"], confidence=0.7)
    assert len(collapse_duplicate_atoms([a, b])) == 2


def test_near_duplicate_same_section_still_folds():
    a = _Atom(LONG, section=["Site Access"], confidence=0.9)
    b = _Atom(LONG_NEAR, section=["Site Access"], confidence=0.7)
    assert collapse_duplicate_atoms([a, b]) == [a]
