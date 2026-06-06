"""v58: the fuzzy prose-dedup pass in ``collapse_duplicate_atoms`` must stay
correct AND bounded.

Background: the pass compared every long prose atom against every kept rep in
its ``(type, first-8-tokens)`` bucket using pure-Python ``difflib``. When
boilerplate makes thousands of *distinct* prose atoms share one bucket key
(a 39k-atom spreadsheet deal where every row starts the same way), that's an
O(n²) wall of ~1ms comparisons — it ran for hours.

The fix swaps difflib for the mathematically-identical-but-C-accelerated
``rapidfuzz`` and caps the number of representatives kept per bucket
(``SOWSMITH_FUZZY_DEDUP_MAX_REPS``). These tests pin the quality contract
(near-dups still collapse, distinct prose survives) and the bound's documented
trade-off (beyond the cap a bucket stops deduping rather than blowing up).
"""

from __future__ import annotations

from app.core.entity_resolution import collapse_duplicate_atoms

# 8-token shared prefix so the atoms below land in ONE fuzzy bucket.
_PREFIX = "Contractor shall ensure that the installed system meets "


class _Atom:
    def __init__(self, atom_type, text, confidence=0.8, artifact_id="art1"):
        self.atom_type = atom_type
        self.artifact_id = artifact_id
        self.raw_text = text
        self.normalized_text = text
        self.confidence = confidence


def test_near_duplicate_prose_collapses() -> None:
    a = _Atom(
        "constraint",
        _PREFIX + "the ANSI grounding standard for all rooftop antenna mounts in zone alpha.",
        confidence=0.9,
    )
    # One trailing-char edit → ratio well above 0.92 → a duplicate.
    b = _Atom(
        "constraint",
        _PREFIX + "the ANSI grounding standard for all rooftop antenna mounts in zone alphaa.",
        confidence=0.5,
    )
    out = collapse_duplicate_atoms([a, b])
    assert len(out) == 1
    assert out[0].confidence == 0.9  # highest-confidence survivor kept


def test_distinct_prose_sharing_prefix_survives() -> None:
    a = _Atom(
        "constraint",
        _PREFIX + "the ANSI grounding standard for all rooftop antenna mounts in zone alpha.",
    )
    b = _Atom(
        "constraint",
        _PREFIX + "the seismic bracing code for underground conduit runs by the parking structure.",
    )
    out = collapse_duplicate_atoms([a, b])
    assert len(out) == 2, "different requirements must not collapse on shared prefix"


def test_rep_cap_bounds_work_and_stops_deduping_past_cap(monkeypatch) -> None:
    # cap=1: only the first distinct rep is retained for comparison.
    monkeypatch.setenv("SOWSMITH_FUZZY_DEDUP_MAX_REPS", "1")
    a = _Atom(
        "constraint",
        _PREFIX + "the ANSI grounding standard for all rooftop antenna mounts in zone alpha.",
    )
    b = _Atom(  # distinct from A — kept, but NOT stored as a rep (cap reached at 1)
        "constraint",
        _PREFIX + "the seismic bracing code for underground conduit runs by the parking structure.",
    )
    a_dup = _Atom(  # near-dup of A; A is the one stored rep → collapses
        "constraint",
        _PREFIX + "the ANSI grounding standard for all rooftop antenna mounts in zone alphaa.",
    )
    b_dup = _Atom(  # near-dup of B; B was never stored → kept (the bounded trade-off)
        "constraint",
        _PREFIX + "the seismic bracing code for underground conduit runs by the parking structuree.",
    )
    out = collapse_duplicate_atoms([a, b, a_dup, b_dup])
    # a_dup folds into a; b_dup escapes because b was past the cap.
    assert len(out) == 3


def test_short_prose_below_threshold_is_untouched() -> None:
    # < 50 chars never enters the fuzzy pass — both survive even if near-identical.
    a = _Atom("constraint", "Use copper cabling only.")
    b = _Atom("constraint", "Use copper cabling only!")
    out = collapse_duplicate_atoms([a, b])
    assert len(out) == 2
