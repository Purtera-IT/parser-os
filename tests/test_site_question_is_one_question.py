"""The question a PM answers and the question the pipeline asks are one question.

Found by tracing what a "One site / Two sites" answer would actually do. It
would have done nothing: the answer was stored under relation `same_site` with
verdicts "same"/"different", while `semantic_site_fusion_groups` asks decide()
for `same_physical_site` between "same_site"/"distinct_site", using a different
exemplar built from slugs. Three mismatches in one path — a lesson banked where
nothing looks for it.
"""

from __future__ import annotations

from app.core.pm_feedback import HEAD_REGISTRY
from app.core.site_duplicate_candidates import (
    SITE_PAIR_CANDIDATES,
    SITE_PAIR_RELATION,
    pair_exemplar,
    site_duplicate_candidates,
)

PALO_ALTO = {
    "site": "site:palo_alto_ca_94304", "facility_name": "Palo Alto Office",
    "address": "3300 Hillview Ave", "city": "Palo Alto", "state": "CA", "anchored": True,
}
HILLVIEW = {
    "site": "site:symphonyai_hillview_office",
    "facility_name": "Symphony Ai Hillview Office", "anchored": False,
}


def test_the_head_stores_answers_where_the_pipeline_looks() -> None:
    spec = HEAD_REGISTRY["site"]
    assert spec.relation == SITE_PAIR_RELATION
    assert tuple(spec.candidates) == SITE_PAIR_CANDIDATES


def test_the_declared_verdicts_are_the_ones_that_decide() -> None:
    """A closed set is what makes an out-of-vocabulary answer refused at the
    write boundary instead of silently shaping the boundary and being dropped."""
    assert set(SITE_PAIR_CANDIDATES) == {"same_site", "distinct_site"}
    assert HEAD_REGISTRY["site"].mode == "classify"


def test_the_fusion_pass_asks_with_the_exemplar_the_answer_is_taught_on() -> None:
    from app.core.entity_resolution import semantic_site_fusion_groups

    rows = {r["site"]: r for r in (PALO_ALTO, HILLVIEW)}
    asked: list[str] = []

    # Capture what decide() is handed, without deciding anything.
    import app.core.decide as decide_mod

    real = decide_mod.decide

    class _Undecided:
        verdict = None
        source = "fallback"

    def _spy(**kwargs):
        asked.append(kwargs.get("text", ""))
        return _Undecided()

    import os

    os.environ["SOWSMITH_NEURAL_SITE_FUSION"] = "1"
    decide_mod.decide = _spy
    try:
        semantic_site_fusion_groups(set(rows), rows)
    finally:
        decide_mod.decide = real
        os.environ.pop("SOWSMITH_NEURAL_SITE_FUSION", None)

    taught = pair_exemplar(PALO_ALTO, HILLVIEW)
    assert asked, "the fusion pass asked nothing"
    assert taught in asked, f"asked {asked!r}, taught {taught!r}"
    # And the address is in it, because that is what the judgement turns on.
    assert "3300 hillview ave" in taught


def test_the_shortlist_and_the_fusion_pass_agree_on_a_real_pair() -> None:
    candidate = site_duplicate_candidates([PALO_ALTO, HILLVIEW])[0]
    assert candidate["exemplar"] == pair_exemplar(PALO_ALTO, HILLVIEW)


def test_the_exemplar_survives_two_descriptions_of_one_site() -> None:
    """The fifth instance of one bug, ended by normalisation rather than by
    aligning one more pair of call sites.

    The panel described a site from the readiness row ("Symphony AI Hillview
    office"); the fusion pass described the same site from its slug, because
    no atom carried that key ("symphony ai hillview office"). One character of
    case, and a taught answer could not be retrieved.
    """
    anchor = {
        "site": "site:palo_alto_ca_94304", "facility_name": "Palo Alto",
        "street_address": "3300 Hillview Ave", "city": "Palo Alto", "state": "CA",
    }
    from_readiness = {
        "site": "site:symphony_ai_hillview_office",
        "facility_name": "Symphony AI Hillview office",
    }
    from_slug = {
        "site": "site:symphony_ai_hillview_office",
        "facility_name": "symphony ai hillview office",
    }
    assert pair_exemplar(anchor, from_readiness) == pair_exemplar(anchor, from_slug)


def test_whitespace_does_not_change_the_key() -> None:
    a = {"site": "site:a", "facility_name": "Palo  Alto", "street_address": "3300 Hillview Ave"}
    b = {"site": "site:a", "facility_name": "Palo Alto", "street_address": "3300 Hillview Ave"}
    other = {"site": "site:b", "facility_name": "Hillview"}
    assert pair_exemplar(a, other) == pair_exemplar(b, other)


def test_the_address_is_still_distinguishing() -> None:
    """Punctuation is kept on purpose — two addresses on one street must not
    collapse into the same key."""
    other = {"site": "site:z", "facility_name": "Hillview"}
    a = {"site": "site:a", "facility_name": "Office", "street_address": "3300 Hillview Ave"}
    b = {"site": "site:a", "facility_name": "Office", "street_address": "3400 Hillview Ave"}
    assert pair_exemplar(a, other) != pair_exemplar(b, other)
