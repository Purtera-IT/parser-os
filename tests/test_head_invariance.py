"""Head invariance — the same content must decide the same in every dress.

The parser audit held a document's content constant and varied an incidental
attribute (filename, container, casing); this file applies the same discipline
to the post-parse heads. Ten bleeds were found and fixed on 2026-08-24; each
family below pins either a fixed bleed or an invariant that already held and
must keep holding.
"""

import pytest

from app.core.atom_type_sanity import (
    _classify_quantity,
    _classify_quantity_key,
    _canonical_quantity_noun,
)
from app.core.entity_extraction import _emit_quantity_keys, _slugify
from app.core.site_detection import _normalize, phrase_is_in_catalog
from app.core.vision_extraction import _dedup_rows, _text_verify_row
from app.core.zero_miss import PM_CRITICAL_TERMS, pm_vocab_sweep
import re


# ── atom_type_sanity: dress-blind quantity verdicts ─────────────────────────

class TestSanityQuantityInvariance:
    @pytest.mark.parametrize("dress", [
        "replace approximately 110 existing TVs",
        "REPLACE APPROXIMATELY 110 EXISTING TVS",
        "110 TVs",          # NBSP
        "110  TVs",              # doubled space
        "110 TVs.",              # trailing punctuation
        "1,000 cables",          # thousands separator
        "110 TVs at $450 each",  # deliverable near money stays deliverable
        "$49,500 for 110 TVs",
    ])
    def test_deliverable_in_every_dress(self, dress):
        assert _classify_quantity(dress) == "deliverable"

    @pytest.mark.parametrize("dress", [
        "28.57% margin on hardware",
        "28.57% MARGIN ON HARDWARE",
    ])
    def test_financial_in_every_dress(self, dress):
        assert _classify_quantity(dress) == "financial"

    @pytest.mark.parametrize("key,verdict", [
        ("quantity:110", "ok"),                    # bare count carries no vocab
        ("quantity:110_tvs", "deliverable"),
        ("quantity:110-tvs", "deliverable"),       # hyphen slug == underscore slug
        ("quantity:28_57_margin", "financial"),
        ("quantity:260_pmo_cost", "financial"),
        ("quantity:pmo_cost_260", "financial"),    # noun-leading slug still classified
        ("quantity:118_pricing_lines", "meta"),
    ])
    def test_key_classification_mirrors_text(self, key, verdict):
        assert _classify_quantity_key(key) == verdict

    @pytest.mark.parametrize("raw,canon", [
        ("APs", "access points"), ("aps", "access points"), ("WAPs", "access points"),
        ("Badge Readers", "badge readers"), ("NVRS", "NVRs"),
    ])
    def test_noun_canonicalization_is_case_blind(self, raw, canon):
        assert _canonical_quantity_noun(raw) == canon


# ── site_detection: one site name, every dress, one catalog entry ───────────

class TestSiteNormalizeInvariance:
    def test_id_dressings_collapse(self):
        forms = ["ATL-HQ-01", "atl_hq_01", "ATL HQ 01", "Atl/HQ/01", "atl hq 01"]
        assert len({_normalize(f) for f in forms}) == 1

    def test_period_is_dress_not_content(self):
        # FIXED BLEED: _normalize kept periods, so "St. Louis Office" never
        # equalled its own entity-key slug and catalog lookups missed.
        assert _normalize("St. Louis Office") == _normalize("st_louis_office")
        assert _normalize("Bldg. 5") == _normalize("bldg_5")

    def test_decimals_stay_self_equal(self):
        # periods->spaces on BOTH sides of any comparison: a decimal cannot
        # collide with the digit-run it is not ("4.5" != "45").
        assert _normalize("Suite 4.5") == _normalize("suite 4.5")
        assert _normalize("Suite 4.5") != _normalize("Suite 45")

    @pytest.mark.parametrize("probe", [
        "CHIPOTLE STORE #1442", "chipotle_store_1442",
        "St. Louis Office", "st louis office", "ST_LOUIS_OFFICE",
    ])
    def test_catalog_membership_is_dress_invariant(self, probe):
        catalog = {_normalize("Chipotle Store #1442"), _normalize("St. Louis Office")}
        assert phrase_is_in_catalog(probe, catalog)


# ── zero_miss.pm_vocab_sweep: counting and coverage must see through layout ─

class TestVocabSweepInvariance:
    def test_no_pattern_uses_capturing_groups(self):
        # findall() returns group contents when a pattern captures; the sweep
        # derives its dedup root from matches[0], so a capturing group would
        # silently corrupt the covered-check.
        offenders = [t["pat"] for t in PM_CRITICAL_TERMS if re.search(r"\((?!\?)", t["pat"])]
        assert offenders == []

    def test_covered_multiword_term_does_not_reescalate(self):
        # FIXED BLEED: the de-spaced root ("performancebond") was compared
        # against spaced covered text, so every covered multi-word term
        # re-escalated to the LLM on every deal.
        raw = ("The performance bond requirement is stated. A performance bond "
               "shall be provided. Performance bond value equals 100% of contract.")
        covered = {"requirement": [{"text": "Performance bond as outlined in the bid documents."}]}
        fired = [m for m in pm_vocab_sweep(raw, [], covered)
                 if "performance" in m["outcome"].get("_term_pattern", "")]
        assert fired == []

    def test_pdf_hyphenation_does_not_hide_mentions(self):
        # FIXED BLEED: "perfor-\nmance bond" counted as zero mentions.
        raw = ("The performance bond requirement is stated. A performance bond "
               "shall be provided. Performance bond value equals 100% of contract.")
        hyph = raw.replace("performance bond", "perfor-\nmance bond")
        n_plain = len(pm_vocab_sweep(raw, [], {"requirement": []}))
        n_hyph = len(pm_vocab_sweep(hyph, [], {"requirement": []}))
        assert n_hyph == n_plain

    def test_apostrophe_style_is_invisible(self):
        curly = ("Workers’ compensation insurance. Workers’ compensation "
                 "limits. Workers’ compensation statutory.")
        straight = curly.replace("’", "'")
        assert (len(pm_vocab_sweep(curly, [], {"requirement": []}))
                == len(pm_vocab_sweep(straight, [], {"requirement": []})))


# ── entity_extraction: one quantity, every path, one key ────────────────────

class TestQuantityKeyInvariance:
    def test_structured_and_text_paths_agree(self):
        assert (_emit_quantity_keys({"quantity": 50}, "")
                == _emit_quantity_keys(None, "Qty: 50")
                == _emit_quantity_keys(None, "QTY: 50")
                == _emit_quantity_keys(None, "qty=50")
                == {"quantity:50"})

    def test_thousands_separator_is_dress(self):
        assert _emit_quantity_keys(None, "Qty: 1,000") == {"quantity:1000"}

    @pytest.mark.parametrize("value,text", [
        (None, "Qty: 0"),          # FIXED BLEED: zero emitted a key
        (None, "Qty: 2000000"),    # FIXED BLEED: absurd count emitted a key
        ({"quantity": 0}, ""),     # FIXED BLEED: structured zero emitted a key
        ({"quantity": 2_000_000}, ""),
    ])
    def test_plausibility_gate_covers_every_path(self, value, text):
        # the noun-anchored path always dropped 0 / >100k; the structured and
        # "Qty:" paths did not, so junk keys polluted rollups and conflicts.
        assert _emit_quantity_keys(value, text) == set()

    def test_slug_is_dress_blind(self):
        assert _slugify("Bâtiment 12") == _slugify("Batiment 12")
        assert _slugify("GEBÄUDE 5") == _slugify("gebaude 5")
        assert _slugify("Suite 400") == _slugify("Suite 400")


# ── vision_extraction: tile overlap dedup + hallucination verify ────────────

class TestVisionInvariance:
    def test_tile_whitespace_dress_dedups(self):
        # FIXED BLEED: overlapping tile crops render the SAME row with
        # different internal spacing; the dedup key treated dress as content.
        rows = [
            {"kind": "money", "text": "50 × Access Point = $12,000", "category": None},
            {"kind": "money", "text": "50  ×  Access Point  =  $12,000", "category": None},
            {"kind": "money", "text": "50 × access point = $12,000", "category": None},
        ]
        assert len(_dedup_rows(rows)) == 1

    def test_verify_tolerates_ligatures_and_hyphenation(self):
        # measured tolerance, pinned: the 0.30 overlap floor absorbs PDF
        # ligatures (fi -> ﬁ) and a couple of hyphenated line breaks.
        row = "final configuration of the office network"
        page = ("The final configuration of the office network is documented "
                "here in full detail.")
        assert _text_verify_row(row, page)
        assert _text_verify_row(row, page.replace("fi", "ﬁ"))
        assert _text_verify_row(
            row, page.replace("configuration", "confi-\nguration"))
