"""Every magic number matches its receipt, or a test names the drift.

The registry (app/core/calibration.py) documents where each load-bearing
constant came from; these tests pin each entry to its live usage site. Change
the code without the registry -- or the registry without the code -- and the
mismatch is named instead of silently accumulating.

Two pinning styles, both deliberate:

* importable module constants are compared directly;
* function-local literals (the PDF thresholds, the density comparison, the
  fold minimum) are pinned by asserting the literal in the SOURCE. Grepping
  source in a test is ugly and is chosen anyway: the alternative is hoisting
  working constants out of 5,000-line modules purely to make them importable,
  which is churn in load-bearing files for the benefit of a test.
"""

from __future__ import annotations

import inspect

from app.core.calibration import REGISTRY


def test_every_entry_carries_a_full_receipt() -> None:
    """Provenance is the point. An entry without it is just another literal."""
    for key, entry in REGISTRY.items():
        assert entry.derivation.strip(), f"{key}: no derivation"
        assert entry.stale_when.strip(), f"{key}: no staleness condition"
        assert entry.corpus_n > 0, f"{key}: corpus size hidden"
        assert entry.derived, f"{key}: no derivation date"


def test_small_corpora_are_stated_not_hidden() -> None:
    """A constant from one deal pack is allowed; pretending otherwise is not."""
    tiny = [k for k, e in REGISTRY.items() if e.corpus_n < 5]
    for key in tiny:
        entry = REGISTRY[key]
        assert "1" in entry.notes or "see" in entry.derivation.lower() or entry.corpus_n == 1, (
            f"{key}: corpus_n={entry.corpus_n} should be acknowledged"
        )


# ── import pins ─────────────────────────────────────────────────────────


def test_prose_floor_pins() -> None:
    from app.parsers import markdown_parser

    assert markdown_parser._MIN_TEXT_LINES == REGISTRY["prose_floor.min_lines"].value
    assert markdown_parser._MIN_TEXT_CHARS == REGISTRY["prose_floor.min_chars"].value


def test_match_threshold_pin() -> None:
    from app.parsers import registry as parser_registry

    assert parser_registry.MATCH_THRESHOLD == REGISTRY["router.match_threshold"].value


# ── source pins, for function-local literals ────────────────────────────


def test_pdf_page_band_pins() -> None:
    from app.parsers import orbitbrief_pdf

    source = inspect.getsource(orbitbrief_pdf)
    low = int(REGISTRY["pdf.low_text_page"].value)
    rich = int(REGISTRY["pdf.text_rich_page"].value)
    assert f"LOW_TEXT_PAGE_THRESHOLD = {low}" in source, (
        f"pdf low-text threshold moved off {low}; update the registry receipt too"
    )
    assert f"TEXT_RICH_PAGE_THRESHOLD = {rich}" in source, (
        f"pdf text-rich threshold moved off {rich}; update the registry receipt too"
    )


def test_speaker_density_pin() -> None:
    from app.parsers import transcript_parser

    source = inspect.getsource(transcript_parser)
    density = REGISTRY["transcript.speaker_density"].value
    assert f">= {density}" in source, (
        f"speaker density moved off {density}; the registry receipt describes "
        "a 19-file measurement that no longer matches the code"
    )


def test_speaker_fold_minimum_pin() -> None:
    from app.core import normalizers

    source = inspect.getsource(normalizers.fold_standalone_speaker_lines)
    minimum = int(REGISTRY["speaker_fold.min_folds"].value)
    assert f"len(folds) >= {minimum}" in source, (
        f"fold minimum moved off {minimum}; update the registry receipt too"
    )


# ── the self-deriving half: constants that recompute their own bounds ───


def test_derivable_constants_sit_inside_their_corpus_bounds() -> None:
    """drift_report() names any constant the eval corpus has outgrown.

    Run in CI so a constant cannot quietly expire: when new corpus cases move
    the supportable range past the shipped value, this test starts failing
    with the constant's name and the evidence, which is the entire point.

    The mechanism earned its keep on its very first run -- it flagged
    impossible bounds [0.037, 0.0] and the cause was a bug in its own
    author's derivation filter (the Otter case, whose 0.0 colon-density is by
    design, had leaked into a bound about the colon dialect).
    """
    from app.core.calibration import drift_report

    problems = drift_report()
    assert problems == [], "\n".join(problems)


def test_derived_bounds_are_sane_ranges() -> None:
    from app.core.calibration import REGISTRY

    for key, entry in REGISTRY.items():
        if entry.re_derive is None:
            continue
        bounds = entry.re_derive()
        assert bounds["lower"] <= bounds["upper"], (
            f"{key}: impossible bounds {bounds} -- the derivation itself is wrong"
        )
        assert bounds.get("evidence"), f"{key}: bounds without evidence"
