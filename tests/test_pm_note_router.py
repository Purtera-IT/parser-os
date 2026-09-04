"""A PM writes what they mean; the router works out what that corrects.

The chip loop made a PM pick a head before they could teach anything, and the
question loop only ever spoke to one head. These pin the general form: one
note, several lessons, each aimed at the head that got it wrong, carrying the
circumstance it applies under and the reason the PM gave.
"""

from __future__ import annotations

from app.core.feedback_store import FeedbackStore, condition_holds
from app.core.pm_feedback import HEAD_REGISTRY, apply_pm_correction
from app.core.pm_note_router import (
    apply_note,
    extract_condition,
    extract_rationale,
    route_note,
    split_clauses,
)
from app.core.terminology import apply_preferred_terms, preferred_terms

NO_LLM = lambda note: {}  # noqa: E731 - the pattern layer must stand alone


def _heads(routing):
    return {l.head for l in routing.lessons}


def test_a_wording_preference_becomes_a_terminology_lesson() -> None:
    r = route_note(
        "I prefer SLO instead of SLA because we guarantee objectives, not agreements.",
        synthesize=NO_LLM,
    )
    assert _heads(r) == {"terminology"}
    lesson = r.lessons[0]
    assert lesson.new_value == "SLO"
    assert lesson.old_value == "SLA"
    assert lesson.rationale == "we guarantee objectives, not agreements"


def test_a_type_correction_routes_to_the_type_head() -> None:
    r = route_note(
        "This should be a constraint not a scope_item because it says the tech must be escorted.",
        synthesize=NO_LLM,
    )
    assert "type" in _heads(r)
    assert next(l for l in r.lessons if l.head == "type").new_value == "constraint"


def test_stop_asking_routes_to_the_gap_head_and_always_ask_is_its_opposite() -> None:
    stop = route_note("Stop asking about the PM fee on staff aug deals.", synthesize=NO_LLM)
    assert next(l for l in stop.lessons if l.head == "gap").new_value == "invalid"
    always = route_note("Always ask who signs acceptance.", synthesize=NO_LLM)
    assert next(l for l in always.lessons if l.head == "gap").new_value == "valid"


def test_our_own_company_is_never_a_site_routes_to_admission() -> None:
    r = route_note("PurTera is our own company, it is never a site.", synthesize=NO_LLM)
    assert "admission" in _heads(r)
    assert next(l for l in r.lessons if l.head == "admission").new_value == "drop"


def test_one_note_can_carry_several_lessons() -> None:
    r = route_note(
        "Prefer SLO instead of SLA. Also stop asking about the PM fee. "
        "And that line should be a constraint.",
        synthesize=NO_LLM,
    )
    assert {"terminology", "gap", "type"} <= _heads(r)


def test_the_circumstance_is_read_from_the_note_and_grounded_in_the_deal() -> None:
    facts = {"owner": ["Chase Whitfield", "Patrick Kelly"]}
    cond = extract_condition("when Chase is assigned we bill the blended rate", facts)
    assert cond == {"field": "owner", "equals": "chase_whitfield"}
    # Stated the other way round is the same fact.
    assert extract_condition("Chase quoted this one", facts)["equals"] == "chase_whitfield"
    # No circumstance at all is not a condition.
    assert extract_condition("Prefer SLO instead of SLA", facts) == {}


def test_every_lesson_in_a_conditional_note_carries_the_condition() -> None:
    r = route_note(
        "When Chase is assigned, stop asking about the PM fee and prefer SLO instead of SLA.",
        facts={"owner": ["Chase Whitfield"]},
        synthesize=NO_LLM,
    )
    assert r.lessons
    assert all(l.condition == {"field": "owner", "equals": "chase_whitfield"} for l in r.lessons)


def test_a_conditional_lesson_only_fires_where_the_condition_holds() -> None:
    relations = {"when": {"field": "owner", "equals": "chase"}}
    assert condition_holds(relations, {"owner": "Chase Whitfield"}) is True
    assert condition_holds(relations, {"owner": "Patrick Kelly"}) is False
    # No facts at all: silence, never a guess.
    assert condition_holds(relations, None) is False
    assert condition_holds({}, None) is True


def test_the_reason_is_kept_verbatim_never_interpreted() -> None:
    why = extract_rationale("Drop that because the customer already answered it in the SOW")
    assert why == "the customer already answered it in the SOW"
    assert extract_rationale("Drop that.") == ""


def test_clauses_split_but_short_fragments_are_not_lessons() -> None:
    assert split_clauses("One thing here. Another thing there.") == [
        "One thing here.",
        "Another thing there.",
    ]
    assert split_clauses("ok.") == []


def test_the_llm_only_proposes_inside_the_registry() -> None:
    # The exemplars quote the note, so only the head and verdict checks decide.
    note = "Nobody needs to be asked who signs acceptance on these dental sites."

    def rogue(_note):
        return {
            "lessons": [
                {"head": "not_a_head", "exemplar": note, "new_value": "y"},
                {"head": "type", "exemplar": note, "new_value": "not_an_atom_type"},
                {"head": "gap", "exemplar": "who signs acceptance on these dental sites", "new_value": "invalid"},
            ]
        }

    r = route_note(note, synthesize=rogue)
    assert _heads(r) == {"gap"}, "invented heads and verdicts are refused"


def test_a_note_it_cannot_read_is_reported_not_guessed() -> None:
    r = route_note("zzz qqq wibble", synthesize=NO_LLM)
    assert r.lessons == []
    assert r.unrouted


def test_apply_note_commits_every_lesson_to_the_store() -> None:
    store = FeedbackStore(":memory:")
    out = apply_note(
        "Prefer SLO instead of SLA because we guarantee objectives. Also stop asking about the PM fee.",
        store=store,
        deal_id="deal-1",
        pm="griffin@purtera-it.com",
        facts={"owner": ["Chase Whitfield"]},
    )
    assert len(out["committed"]) == 2
    heads = {c["head"] for c in out["committed"]}
    assert heads == {"terminology", "gap"}
    assert all(c["correction_id"] for c in out["committed"])


def test_a_preference_rewrites_what_we_author_and_says_why() -> None:
    store = FeedbackStore(":memory:")
    apply_pm_correction(
        store,
        {
            "head": "terminology",
            "dealId": "deal-1",
            "targetId": "",
            "text": "SLA",
            "oldValue": "SLA",
            "newValue": "SLO",
            "scope": "deal",
            "context": "we guarantee objectives, not agreements",
        },
    )
    terms = preferred_terms(store, deal_id="deal-1")
    assert terms and terms[0]["prefer"] == "SLO" and terms[0]["instead_of"] == "SLA"
    assert "objectives" in terms[0]["why"]

    text, applied = apply_preferred_terms("Confirm the SLA for each site.", terms)
    assert text == "Confirm the SLO for each site."
    assert applied == [terms[0]["correction_id"]]

    # Whole words only: a preference for SLA never touches SLAB.
    untouched, _ = apply_preferred_terms("Mount the SLAB before the racks.", terms)
    assert untouched == "Mount the SLAB before the racks."


def test_a_preference_scoped_to_a_deal_stays_on_that_deal() -> None:
    store = FeedbackStore(":memory:")
    apply_pm_correction(
        store,
        {
            "head": "terminology",
            "dealId": "deal-1",
            "targetId": "",
            "text": "SLA",
            "oldValue": "SLA",
            "newValue": "SLO",
            "scope": "deal",
            "context": "",
        },
    )
    assert preferred_terms(store, deal_id="deal-1")
    assert preferred_terms(store, deal_id="deal-2") == []


def test_a_conditional_preference_waits_for_its_circumstance() -> None:
    store = FeedbackStore(":memory:")
    apply_pm_correction(
        store,
        {
            "head": "terminology",
            "dealId": "deal-1",
            "targetId": "",
            "text": "SLA",
            "oldValue": "SLA",
            "newValue": "SLO",
            "scope": "deal",
            "context": "Chase's deals only",
            "relations": {"when": {"field": "owner", "equals": "chase"}},
        },
    )
    assert preferred_terms(store, deal_id="deal-1", facts={"owner": "Chase Whitfield"})
    assert preferred_terms(store, deal_id="deal-1", facts={"owner": "Patrick Kelly"}) == []
    assert preferred_terms(store, deal_id="deal-1") == []


def test_terminology_is_a_registered_head_like_any_other() -> None:
    assert HEAD_REGISTRY["terminology"].relation == "preferred_term"
    assert HEAD_REGISTRY["terminology"].mode == "extract"


def test_a_standalone_actor_sentence_sets_the_note_context() -> None:
    r = route_note(
        "Chase quoted this one. Stop asking who pays for the PM effort. Prefer SLO instead of SLA.",
        facts={"owner": ["Chase Whitfield"]},
        synthesize=NO_LLM,
    )
    assert r.unrouted == [], "the actor sentence is context, not an unread clause"
    assert {l.head for l in r.lessons} == {"gap", "terminology"}
    assert all(l.condition == {"field": "owner", "equals": "chase_whitfield"} for l in r.lessons)


def test_a_reason_belongs_to_its_own_sentence() -> None:
    r = route_note(
        "Prefer SLO instead of SLA because we commit to objectives. "
        "Stop asking who provisions access since CDW always does it.",
        synthesize=NO_LLM,
    )
    term = next(l for l in r.lessons if l.head == "terminology")
    gap = next(l for l in r.lessons if l.head == "gap")
    assert term.rationale == "we commit to objectives"
    assert gap.rationale == "CDW always does it"
    assert "CDW" not in term.rationale


def test_an_inline_circumstance_conditions_only_its_own_sentence() -> None:
    r = route_note(
        "When Chase is assigned stop asking about the PM fee. Prefer SLO instead of SLA.",
        facts={"owner": ["Chase Whitfield"]},
        synthesize=NO_LLM,
    )
    gap = next(l for l in r.lessons if l.head == "gap")
    term = next(l for l in r.lessons if l.head == "terminology")
    assert gap.condition == {"field": "owner", "equals": "chase_whitfield"}
    assert term.condition == {}, "the wording preference was not about Chase"


def test_a_note_is_learned_from_the_card_the_pm_was_looking_at() -> None:
    live = [
        "Who provisions the techs' access, and is that access included in the per-site fee?",
        "Confirm AP count/model for this quote.",
    ]
    r = route_note(
        "Stop asking who provisions the techs' access because CDW always does it.",
        questions=live,
        synthesize=NO_LLM,
    )
    lesson = next(l for l in r.lessons if l.head == "gap")
    assert lesson.exemplar == live[0], "the card beats the paraphrase as a prototype"


def test_grounding_refuses_a_card_that_is_about_something_else() -> None:
    from app.core.pm_note_router import ground_in_questions

    live = ["Confirm AP count/model for this quote.", "What are the approved work hours?"]
    assert ground_in_questions("Who provides the customer bridge?", live) == ""


def test_the_gap_head_keeps_a_bar_that_clears_its_own_noise() -> None:
    """Measured on the live corpus: a reworded same-ask scores 0.754-0.768 and
    a DIFFERENT ask about the same site scores 0.746. Eight thousandths apart,
    so the default bars cannot separate them and the gap head carries its own.
    A note is grounded in the card the PM was looking at, which scores ~1.0,
    so the higher bar costs nothing that matters."""
    from app.core.pm_feedback import _HEAD_THRESHOLDS, _THRESHOLD_DEAL, _threshold_for

    assert _threshold_for("gap", "deal") == 0.82
    assert _threshold_for("gap", "global") == 0.88
    assert _threshold_for("type", "deal") == _THRESHOLD_DEAL
    assert _HEAD_THRESHOLDS["gap"][0] > 0.746, "must clear the measured false positive"


def test_repeated_judgments_relax_the_bar_but_never_below_the_heads_own_floor() -> None:
    from app.core.pm_feedback import _merge_with_existing, pm_correction_to_correction

    class Store:
        def __init__(self):
            self.rows = {}

        def get(self, cid):
            return self.rows.get(cid)

    store = Store()
    seen = []
    for deal in ("a", "b", "c", "d", "e", "f"):
        corr = pm_correction_to_correction(
            {
                "head": "gap",
                "dealId": deal,
                "targetId": "r1",
                "text": "Who signs acceptance?",
                "oldValue": "valid",
                "newValue": "invalid",
                "scope": "global",
            }
        )
        corr = _merge_with_existing(store, corr)
        store.rows[corr.id] = corr
        seen.append(corr.threshold)
    assert seen[0] == 0.88
    assert seen[1] < seen[0], "evidence buys reach"
    assert min(seen) >= 0.82, "never below the bar one in-deal correction clears"


def test_an_unreadable_note_and_an_offline_model_are_not_the_same_thing(monkeypatch) -> None:
    """Live: the service's model endpoint pointed at an address it cannot
    route to, so every note the shapes could not read cost 39 seconds and came
    back empty. Empty because the note said nothing teachable, and empty
    because nothing was listening, looked identical to the PM."""
    from app.core import pm_note_router as router

    monkeypatch.setattr(router, "_default_synthesize", lambda note: {})
    offline = router.route_note("The facilities team owns every cable pathway on hospital work.")
    assert offline.lessons == []
    assert offline.unrouted
    assert offline.model_unavailable is True
    assert offline.as_dict()["model_unavailable"] is True

    # A model that answers and simply finds nothing teachable is not an outage.
    monkeypatch.setattr(router, "_default_synthesize", lambda note: {"lessons": []})
    quiet = router.route_note("The facilities team owns every cable pathway on hospital work.")
    assert quiet.model_unavailable is False


def test_a_readable_note_never_waits_on_the_model() -> None:
    """The shapes answer first, so the common notes cost nothing at all."""
    from app.core import pm_note_router as router

    def explode(note):
        raise AssertionError("the model must not be called for a note the shapes can read")

    r = router.route_note("Prefer SLO instead of SLA because we commit to objectives.", synthesize=explode)
    assert [l.head for l in r.lessons] == ["terminology"]


def test_a_lesson_may_only_quote_the_person_who_wrote_the_note() -> None:
    """Asked to produce examples, a model hands the prompt's examples back.
    Live: one note about cable pathway returned eight lessons — gap invalid AND
    gap valid, admission drop AND keep, "norm 100", "router project_metadata" —
    every one lifted from the instructions, none of them said by anybody."""
    from app.core.pm_note_router import route_note

    def echoes_the_prompt(note):
        return {
            "lessons": [
                {"head": "gap", "exemplar": "an ask should not be put to the customer", "new_value": "invalid"},
                {"head": "norm", "exemplar": "correcting a figure, a rate or a quantity", "new_value": "100"},
                {"head": "router", "exemplar": "saying what kind of work the deal is", "new_value": "project_metadata"},
            ]
        }

    r = route_note(
        "The facilities team owns every cable pathway on hospital work.",
        synthesize=echoes_the_prompt,
    )
    assert r.lessons == [], "nothing the PM did not say gets banked"


def test_a_verdict_the_head_cannot_return_is_refused() -> None:
    """The model returned "gap_valid" — the relation, not a verdict — and, for
    admission, the PM's whole sentence. Neither can ever fire."""
    from app.core.pm_feedback import HEAD_REGISTRY
    from app.core.pm_note_router import route_note

    assert HEAD_REGISTRY["gap"].candidates == ("valid", "invalid")
    assert HEAD_REGISTRY["admission"].candidates == ("keep", "drop")

    note = "The facilities team owns every cable pathway on hospital work."

    def wrong_verdict(_note):
        return {"lessons": [{"head": "gap", "exemplar": note, "new_value": "gap_valid"}]}

    assert route_note(note, synthesize=wrong_verdict).lessons == []

    def right_verdict(_note):
        return {"lessons": [{"head": "gap", "exemplar": note, "new_value": "invalid"}]}

    kept = route_note(note, synthesize=right_verdict).lessons
    assert [(l.head, l.new_value) for l in kept] == [("gap", "invalid")]


def test_one_note_cannot_mean_both() -> None:
    """A head that comes back with two verdicts for one note is the model
    hedging, and banking either would be a coin toss on the PM's behalf."""
    from app.core.pm_note_router import route_note

    note = "The facilities team owns every cable pathway on hospital work."

    def hedges(_note):
        return {
            "lessons": [
                {"head": "gap", "exemplar": note, "new_value": "invalid"},
                {"head": "gap", "exemplar": note, "new_value": "valid"},
            ]
        }

    assert route_note(note, synthesize=hedges).lessons == []
