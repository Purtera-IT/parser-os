"""The router's evaluation gate, and the per-atom head it is meant to gate.

Both exist because of one measurement: across every training database and the
whole correction store, this repo holds

    service_routing training rows       0
    service_routing corrections         0 of 136
    human-authored corrections          0 of 136  (all universal-audit/mined-audit/seed)
    deal-scoped corrections             0 of 136  (all global)

so the router has no training data here and no human label anywhere in the
system. A gate that cannot refuse would have printed a number anyway. These
tests pin the refusals first, because that is the behaviour that matters while
the gold set is empty.
"""

from __future__ import annotations

import re

import pytest

from app.core.atom_route_head import PerAtomRouter
from app.eval.router_eval import (
    RouterCase,
    compare_routers,
    evaluate_router,
    readiness,
    split_by_deal,
)

_LABELS = ["wireless", "audio_visual", "low_voltage_cabling", "staff_augmentation"]


def _cases(n_deals: int = 30, *, author: str = "cliff@purtera.com", labels=None) -> list[RouterCase]:
    labels = labels or _LABELS
    return [
        RouterCase(f"deal_{d:02d}", labels[d % len(labels)], f"scope text {d}", author)
        for d in range(n_deals)
        for _ in range(2)
    ]


def _perfect(case: RouterCase):
    return (case.gold_label, 0.9)


def _always_wireless(_case: RouterCase):
    return ("wireless", 0.9)


# ── the split ───────────────────────────────────────────────────────────


def test_the_split_is_by_deal_and_never_leaks() -> None:
    """Row-level splitting leaks, and quietly.

    Corrections inside one deal share a customer vocabulary, the same
    documents and the same PM in the same hour, so a row-level split puts
    near-duplicates on both sides and reports a number as circular as the one
    it replaced.
    """
    cases = _cases()
    trainable, holdout = split_by_deal(cases)
    assert not ({c.deal_id for c in trainable} & {c.deal_id for c in holdout})
    assert trainable and holdout


def test_the_split_is_stable_across_runs() -> None:
    """A deal must land on the same side forever, or two runs a month apart
    are not comparable and the hold-out drifts as corrections accumulate."""
    cases = _cases()
    first = {c.deal_id for c in split_by_deal(cases)[1]}
    second = {c.deal_id for c in split_by_deal(cases)[1]}
    assert first == second


# ── the four refusals ───────────────────────────────────────────────────


def test_refuses_when_the_holdout_is_too_small() -> None:
    report = evaluate_router(_perfect, _cases(5))
    assert not report.reportable
    assert any("held-out deals" in r for r in report.refusals)
    assert "NO NUMBER REPORTED" in report.summary()


def test_refuses_on_a_single_label() -> None:
    report = evaluate_router(_perfect, _cases(20, labels=["wireless"]))
    assert not report.reportable
    assert any("single label" in r for r in report.refusals)


def test_refuses_machine_authored_gold() -> None:
    """The failure this whole module exists to prevent.

    All 136 corrections in the store today are machine-written
    (universal-audit / mined-audit / seed). Scoring against those would
    reproduce the circular 91% exactly.
    """
    report = evaluate_router(_perfect, _cases(20, author="mined-audit"))
    assert not report.reportable
    assert any("human-authored" in r for r in report.refusals)


def test_refuses_a_leaked_split() -> None:
    cases = _cases(20)
    report = evaluate_router(_perfect, cases, trainable=cases[:4])
    assert not report.reportable
    assert any("BOTH" in r for r in report.refusals)


# ── abstention must not be scored as success ────────────────────────────


def test_abstaining_does_not_inflate_accuracy() -> None:
    """A router that answers only its easiest deals scores 1.0 on them.

    Coverage is reported beside accuracy so that is visible rather than
    flattering: silence is neither right nor wrong, and is never a hit.
    """
    cases = _cases(30)

    def shy(case: RouterCase):
        return (case.gold_label, 0.9) if case.deal_id.endswith(("0", "1")) else None

    report = evaluate_router(shy, cases)
    assert report.reportable
    assert report.accuracy == pytest.approx(1.0)
    assert report.coverage < 0.4, "the point is that coverage exposes the abstaining"


def test_a_router_that_guesses_the_mode_does_not_beat_the_baseline() -> None:
    report = evaluate_router(_always_wireless, _cases(30))
    assert report.reportable
    assert report.accuracy <= report.majority_baseline
    assert "NOT BEATEN" in report.summary()


# ── promotion is judged against what ships, not against nothing ─────────


def test_promotion_requires_beating_the_shipping_router() -> None:
    """"Better than guessing" is not the question when an LLM base is already
    answering correctly. The bar is the incumbent."""
    cases = _cases(30)
    verdict = compare_routers(_always_wireless, _perfect, cases)
    assert not verdict.promote
    assert any("does not beat the shipping router" in b for b in verdict.blockers)


def test_promotion_rejects_accuracy_bought_by_abstaining() -> None:
    """A candidate that is perfect on a third of the deals has not improved
    anything -- it has handed the hard ones back and taken credit for the rest."""
    cases = _cases(30)

    def shy(case: RouterCase):
        return (case.gold_label, 0.9) if case.deal_id.endswith(("0", "1")) else None

    def decent(case: RouterCase):
        return (case.gold_label, 0.8) if not case.deal_id.endswith("9") else ("wireless", 0.4)

    verdict = compare_routers(shy, decent, cases)
    assert not verdict.promote
    assert any("abstaining is not an improvement" in b for b in verdict.blockers)


def test_a_genuinely_better_head_is_promoted() -> None:
    cases = _cases(30)

    def sloppy(case: RouterCase):
        return ("wireless", 0.5) if case.deal_id.endswith(("3", "7")) else (case.gold_label, 0.8)

    verdict = compare_routers(_perfect, sloppy, cases)
    assert verdict.promote, verdict.summary()


def test_readiness_reports_a_distance_not_a_score() -> None:
    """With an empty gold set the honest answer is how far away it is."""
    text = readiness([])
    assert "NOT enough to gate anything" in text
    assert "more held-out deal(s) needed" in text


# ── the per-atom head ───────────────────────────────────────────────────


_PATTERNS = {
    "wireless": r"access point|wireless|wi-fi|wlan|heatmap|meraki mr",
    "audio_visual": r"display|projector|crestron|microphone|video wall|teams room",
    "low_voltage_cabling": r"cat6|fiber|drop|patch panel|idf|cable tray|fluke",
}


def _keyword_classifier(texts):
    """Stand-in for the trained atom head, with a real abstain path.

    The head is injected exactly as ``ContrastiveTypeKNN`` injects ``embed_fn``,
    so the aggregator and its decision rule are testable without a GPU artifact
    present -- which matters, because none of the promoted stores
    (_contrastive_router, _contrastive_type, _facet_store) exist on this
    machine.
    """
    out = []
    for text in texts:
        hits = [(k, len(re.findall(p, text, re.I))) for k, p in _PATTERNS.items()]
        hits = [(k, n) for k, n in hits if n]
        if len(hits) == 1:
            out.append((hits[0][0], min(0.6 + 0.2 * hits[0][1], 1.0)))
        else:
            out.append(None)  # ambiguous or silent -> contributes nothing
    return out


_WLAN = [
    "Install twenty Meraki MR46 access points across five floors.",
    "Provide a wireless heatmap survey before installation.",
    "Mount access points to the ceiling grid.",
    "Configure WLAN SSIDs per site.",
    "Validate wi-fi coverage after install.",
    "Stage the wireless controller before the site visit.",
    "Escort access is required at the dock before 2pm.",
]
_CABLING = [
    "Run Cat6 to each outlet.",
    "Terminate every drop at the patch panel.",
    "Certify each drop with a Fluke tester.",
    "Install cable tray above the ceiling.",
]


def test_a_clear_deal_routes_and_carries_its_evidence() -> None:
    """The route arrives with the atoms that caused it.

    That is the point of classifying atoms rather than one blob: a PM
    correcting "this is not wireless" can be shown the atoms that said it was.
    """
    decision = PerAtomRouter(classify=_keyword_classifier).route_texts(_WLAN)
    assert decision.label == "wireless"
    assert decision.evidence, "a route with no evidence is the blob problem again"
    assert all(v.label == "wireless" for v in decision.evidence)
    assert "wireless" in decision.explain()


def test_a_genuinely_mixed_deal_abstains_rather_than_picking() -> None:
    router = PerAtomRouter(classify=_keyword_classifier)
    decision = router.route_texts(_WLAN[:4] + _CABLING)
    assert decision.abstained
    assert "mixed" in decision.abstain_reason


def test_thin_evidence_abstains() -> None:
    router = PerAtomRouter(classify=_keyword_classifier)
    decision = router.route_texts(["Escort access is required.", "Work occurs after hours."])
    assert decision.abstained
    assert "atoms voted" in decision.abstain_reason


def test_a_real_second_workstream_is_reported_not_discarded() -> None:
    """A deal that is mostly wireless with real cabling has both. The blob
    representation has to pick one and lose the other."""
    router = PerAtomRouter(classify=_keyword_classifier)
    decision = router.route_texts(_WLAN[:6] + ["Wireless AP labelling."] + _CABLING[:3])
    assert decision.label == "wireless"
    assert decision.secondary == "low_voltage_cabling"


def test_bom_rows_cannot_outvote_the_scope() -> None:
    """A TV install reads as 313 cabling material rows against 52 scope atoms.

    Counting those routes the deal by its bill of materials instead of its
    work, which is why the noise types are filtered before any vote.
    """

    class _Atom:
        def __init__(self, text: str, atom_type: str = "scope_item") -> None:
            self.raw_text = text
            self.atom_type = atom_type

    atoms = [_Atom(t) for t in _WLAN[:6]]
    atoms += [_Atom("Cat6 patch panel 48-port, qty 12", "pricing_assumption") for _ in range(40)]
    decision = PerAtomRouter(classify=_keyword_classifier).route_deal(atoms)
    assert decision.label == "wireless"
    assert decision.considered_atoms == 6


def test_deal_size_does_not_move_a_clear_route() -> None:
    """Every atom votes and the decision is a SHARE, so there is no sampling
    step left to have a cliff in -- which was the defect in the blob path."""
    router = PerAtomRouter(classify=_keyword_classifier)
    small = router.route_texts(_WLAN)
    padded = router.route_texts(_WLAN + ["Work occurs after hours."] * 200)
    assert small.label == padded.label == "wireless"


# ── promotion must survive a small hold-out ─────────────────────────────


def test_a_coin_flip_advantage_is_not_promoted() -> None:
    """The trap in a weekly promote-if-better rule.

    At the hold-out sizes this will have for a long time -- tens, not hundreds
    -- the standard error on a raw accuracy is around ten points, so a
    candidate identical to the incumbent wins roughly half of any weekly
    comparison by chance. Promoting on that reads as an unstable head when it
    is an unstable measurement.

    Both routers answer the SAME deals, so only their disagreements carry
    information, and McNemar asks whether those favour the candidate more than
    a coin would.
    """
    cases = _cases(30)
    losing = {"deal_03", "deal_07", "deal_11"}
    winning = {"deal_02", "deal_05", "deal_09", "deal_13"}

    def candidate(case: RouterCase):
        return ("wireless", 0.6) if case.deal_id in losing else (case.gold_label, 0.8)

    def incumbent(case: RouterCase):
        return ("wireless", 0.6) if case.deal_id in winning else (case.gold_label, 0.8)

    verdict = compare_routers(candidate, incumbent, cases)
    assert not verdict.promote
    assert any("consistent with chance" in b for b in verdict.blockers), verdict.summary()


def test_identical_routers_have_nothing_to_promote_on() -> None:
    cases = _cases(30)
    verdict = compare_routers(_perfect, _perfect, cases)
    assert not verdict.promote
    assert any("never disagree" in b for b in verdict.blockers)


def test_mcnemar_needs_real_evidence_before_it_clears() -> None:
    """Written out rather than imported so the eval runs anywhere. These are
    the exact thresholds a weekly promotion decision will be held to."""
    from app.eval.router_eval import _mcnemar_p

    assert _mcnemar_p(0, 0) == 1.0          # no disagreements -> no evidence
    assert _mcnemar_p(4, 3) == pytest.approx(1.0)   # split -> nothing
    assert _mcnemar_p(3, 0) > 0.05          # three straight wins is not enough
    assert _mcnemar_p(6, 0) <= 0.05         # six is


# ── shadow, and the queue that spends PM attention where it buys most ───


def test_disagreement_is_the_top_of_the_queue() -> None:
    """One tap settles an argument between two routers.

    A deal both routers agree on teaches almost nothing -- they already have
    that case. A deal they split on is where the decision is hard, so the
    label lands in the hold-out as a hard case. That also corrects a bias the
    correction set would otherwise carry: a set built from errors a PM
    happened to notice skews toward the obvious, one built from disagreements
    skews toward the difficult.
    """
    from app.core.router_shadow import decide_ask

    ask, why = decide_ask("wireless", "low_voltage_cabling", base_confidence=0.9)
    assert ask and "disagree" in why


def test_agreement_does_not_cost_a_pm_a_tap() -> None:
    """A queue that cries wolf is a queue nobody clears."""
    from app.core.router_shadow import decide_ask

    assert decide_ask("wireless", "wireless", base_confidence=0.9) == (False, "")
    assert decide_ask("wireless", None, base_confidence=0.88)[0] is False


@pytest.mark.parametrize(
    "base, head, expect",
    [
        (None, None, True),               # nothing placed it -- any label is the first evidence
        ("wireless", None, True),         # base alone and unsure
    ],
)
def test_thin_signal_is_also_worth_asking(base, head, expect) -> None:
    from app.core.router_shadow import decide_ask

    assert decide_ask(base, head, base_confidence=0.2)[0] is expect


def test_a_shadow_row_carries_both_answers_and_the_input_version(tmp_path, monkeypatch) -> None:
    """The head's answer must ride in provenance, never as the label.

    A head's own output becoming its training target is the circularity this
    whole effort exists to break. The label is the BASE's answer, because the
    base is what shipped.

    The row also carries ``scope_summary_version``: without it, a retrospective
    eval cannot tell a head that got worse from a representation that changed
    underneath it.
    """
    import json
    import sqlite3

    db = tmp_path / "training.db"
    monkeypatch.setenv("SOWSMITH_TRAINING_LOG_DB", str(db))
    import app.core.training_log as tl

    tl._LOG = None  # the module caches one log per process
    from app.core import router_shadow

    record = router_shadow.record(
        deal_id="deal_42",
        project_id="p",
        base_label="wireless",
        base_confidence=0.82,
        head_label="low_voltage_cabling",
        head_confidence=0.55,
        provenance={"scope_summary_version": 2},
        scope_summary="install twenty access points",
    )
    assert record.logged and record.ask_pm
    rows = list(sqlite3.connect(db).execute(
        "select relation, label, deal_id, split, provenance from training_rows"
    ))
    assert len(rows) == 1
    relation, label, deal_id, split, prov = rows[0]
    assert relation == "service_routing"
    assert label == "wireless", "the label is what SHIPPED, not what the head guessed"
    assert deal_id == "deal_42"
    assert split in {"train", "holdout"}, "deal-based split assigned on write"
    payload = json.loads(prov)
    assert payload["head_label"] == "low_voltage_cabling"
    assert payload["scope_summary_version"] == 2
    tl._LOG = None


def test_an_unrouted_deal_is_not_logged_as_training_data(tmp_path, monkeypatch) -> None:
    """A row whose label is None teaches nothing.

    Logging it would inflate n_train with silence -- which is arguably how the
    blob registry came to report one training row and call the head trained.
    """
    import sqlite3

    db = tmp_path / "training.db"
    monkeypatch.setenv("SOWSMITH_TRAINING_LOG_DB", str(db))
    import app.core.training_log as tl

    tl._LOG = None
    from app.core import router_shadow

    record = router_shadow.record(
        deal_id="deal_43", base_label=None, head_label=None, scope_summary="x"
    )
    assert record.ask_pm, "nothing placed it, so a label here is the first evidence"
    assert not record.logged
    if db.exists():
        n = sqlite3.connect(db).execute("select count(*) from training_rows").fetchone()[0]
        assert n == 0
    tl._LOG = None


def test_the_router_offers_candidates_before_a_head_exists() -> None:
    """A correction chip with no candidates cannot be tapped.

    Zero of the 136 corrections in the store are routing corrections, and the
    chip has to work now -- with the head off and no promoted store on disk,
    which is the state of every environment today.
    """
    from app.core.service_router import known_packs

    packs = known_packs()
    assert "wireless" in packs and "audio_visual" in packs
    assert "ambiguous" not in [p.lower() for p in packs]


# ── the compile path ────────────────────────────────────────────────────


def test_routing_reaches_the_envelope_even_with_no_head() -> None:
    """The disabled path is the ONLY path taken today, so it has to carry.

    ``service_routing`` used to be added to the envelope only when a head was
    loaded. No promoted store exists in any environment, so the key was absent
    on every compile -- and with it the ``candidates`` a correction chip needs
    to offer a choice, and the record of what the router saw.

    Widening it is safe: ``compute_output_signature`` hashes the
    CompileResult -- atoms, entities, edges, packets -- not the envelope, so
    no signature moves.
    """
    from app.core.service_router import build_service_routing

    out = build_service_routing(
        [], [], deal_id="deal_1", project_id="deal_1", base=None, base_observed=False
    )
    assert out["enabled"] is False
    assert out["candidates"], "the chip needs something to offer"
    assert "shadow" in out, "the observation must travel with the deal"


def test_no_base_in_this_process_is_not_a_missing_route() -> None:
    """parser-os has no service-pack base; the keyword pack_prior is in brief-gen.

    Treating its silence as "nothing placed this deal" would raise a hand on
    every single compile, and a queue that flags everything is a queue nobody
    reads. Whoever can see BOTH answers is who raises the hand.
    """
    from app.core.router_shadow import decide_ask

    assert decide_ask(None, None, base_observed=False) == (False, "")
    assert decide_ask(None, None, base_observed=True)[0] is True


def test_the_head_s_own_answer_never_becomes_a_training_label(tmp_path, monkeypatch) -> None:
    """The circularity this whole effort exists to break.

    With no base observed there is no trustworthy label, so nothing is written
    to the training log -- however confident the head was. The observation
    still returns to the caller and rides in the envelope, so the input is
    preserved for the day a PM does supply a label.
    """
    import sqlite3

    db = tmp_path / "training.db"
    monkeypatch.setenv("SOWSMITH_TRAINING_LOG_DB", str(db))
    import app.core.training_log as tl

    tl._LOG = None
    from app.core import router_shadow

    record = router_shadow.record(
        deal_id="deal_9",
        base_label=None,
        base_observed=False,
        head_label="wireless",
        head_confidence=0.97,
        scope_summary="install access points",
    )
    assert record.head_label == "wireless"
    assert not record.logged, "a head's own output is not a label"
    if db.exists():
        n = sqlite3.connect(db).execute("select count(*) from training_rows").fetchone()[0]
        assert n == 0
    tl._LOG = None
