"""Which of the training blob's rows reach ml/c3 (``DealExample.from_training_blob``),
on invented rows.

* a link whose key this parse no longer has is never matched by its text
  across the deal: it heals only to the one line of its own file with that
  text, and is otherwise dropped and counted; a highlighted span (no key) is
  still found by its text; a rule card's link starts from the line the rule
  fired on;
* the person's row wins over the assistant's twin of it for labels,
  judgments and links; a twin label stands in only where no person labeled
  the line; a model's draft verdict or link never trains;
* judgment rows no head reads (rule, norm, ...) and the deal-level answers
  are counted, not silently lost; deal-size heads train over every line.
"""
from __future__ import annotations

import pytest

from ml.c3.data import DealExample

PM = "pm@example.com"
TWIN_BOT = "claude-code (assistant)"


def _atom(k, text, t, filename, doc):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": doc, "filename": filename,
            "doc_kind": "email", "entered_at": f"2026-05-01T10:{t:02d}:00Z",
            "speaker_role": "customer", "speaker_side": "customer"}


SAME = "Bring the lift on day one."
ATOMS = [
    _atom("k_a1", SAME, 0, "kickoff.eml", "art-kick"),           # the line, in the kickoff email
    _atom("k_b1", SAME, 1, "sow.docx", "art-sow"),               # the same words in another file
    _atom("k_a2", "Lift is a 19 ft scissor lift.", 2, "kickoff.eml", "art-kick"),
    _atom("k_b2", "Customer signs off on the lift plan.", 3, "sow.docx", "art-sow"),
    _atom("k_c1", "Two techs on site.", 4, "notes.txt", "art-notes"),
    _atom("k_c2", "Two techs on site.", 5, "notes.txt", "art-notes"),   # twice in one file
]


def _link(from_key, to_key, rel="supports", head="type", from_text="", labeler=PM, **kw):
    return {"from_head": head, "from_key": from_key, "from_text": from_text, "to_label_key": to_key,
            "relation": rel, "note": "", "labeler": labeler, "created_at": "2026-05-02T10:00:00Z", **kw}


def _deal(blob):
    return DealExample.from_training_blob({"labels": [], "links": [], "judgments": [], **blob},
                                          ATOMS, deal_id="synthetic-loader-rows")


# ------------------------------------------------------------ 1. stale keys

def test_a_stale_key_is_not_matched_to_the_same_words_in_another_file():
    # The row this link was drawn on is gone; its words survive only in the SOW.
    deal = _deal({"links": [_link("lbl_gone", "k_b2", from_text=SAME)]})
    assert deal.edges == []
    assert deal.link_stats["stale_key_dropped"] == 1
    assert deal.link_stats["unresolved"] == 1


def test_a_stale_key_heals_to_the_one_line_of_its_own_file():
    # The old key's label row names its file; one line of that file has the words.
    labels = [{"label_key": "lbl_old_a1", "text": SAME, "filename": "kickoff.eml", "labeler": PM}]
    deal = _deal({"labels": labels, "links": [_link("lbl_old_a1", "k_b2", from_text=SAME)]})
    assert deal.edges == [("k_a1", "k_b2", "supports")]          # not k_b1, the SOW's copy
    assert deal.link_stats["healed_in_file"] == 1
    assert deal.link_stats["stale_key_dropped"] == 0


def test_a_stale_key_whose_file_has_the_words_twice_is_dropped():
    labels = [{"label_key": "lbl_old_c", "text": "Two techs on site.", "filename": "notes.txt", "labeler": PM}]
    deal = _deal({"labels": labels, "links": [_link("lbl_old_c", "k_a2", from_text="Two techs on site.")]})
    assert deal.edges == []
    assert deal.link_stats["stale_key_dropped"] == 1


def test_a_stale_to_key_heals_inside_the_file_the_link_names_or_drops():
    deal = _deal({"links": [
        _link("k_a2", "lbl_old_to", to_text=SAME, to_filename="kickoff.eml"),
        _link("k_b2", "lbl_old_to2", to_text="Lift is a 19 ft scissor lift.", to_filename="sow.docx"),
    ]})
    assert deal.edges == [("k_a2", "k_a1", "supports")]          # healed in kickoff.eml
    assert deal.link_stats["healed_in_file"] == 1
    assert deal.link_stats["stale_key_dropped"] == 1             # the words are not in sow.docx


def test_a_highlighted_span_with_no_key_is_still_found_by_its_text():
    deal = _deal({"links": [_link("k_b2", None, rel="context", to_kind="text",
                                  to_text="Lift is a 19 ft scissor lift.")]})
    assert deal.edges == [("k_b2", "k_a2", "context")]
    assert deal.link_stats["text_span"] == 1


def test_a_rule_card_link_starts_from_the_line_the_rule_fired_on():
    rule = {"head": "rule", "target_key": "rule:lift_terms", "verdict": "should_fire", "labeler": PM,
            "target": {"source": {"atomId": "at-k_b1", "text": SAME, "filename": "sow.docx"}}}
    deal = _deal({"judgments": [rule],
                  "links": [_link("rule:lift_terms", "k_b2", head="rule", from_text=SAME)]})
    assert deal.edges == [("k_b1", "k_b2", "supports")]          # not k_a1, the first line with the words
    assert deal.link_stats["card_anchored"] == 1


# ------------------------------------------------------------ 2. person rows

def _label(key, typ, labeler, at, **kw):
    return {"label_key": key, "label_type": typ, "labeler": labeler, "labeled_at": at,
            "note": f"{typ} by {labeler}", **kw}


def test_the_person_label_wins_over_a_later_assistant_twin():
    deal = _deal({"labels": [
        _label("k_a1", "requirement", PM, "2026-05-02T10:00:00Z"),
        _label("k_a1", "schedule", TWIN_BOT, "2026-05-03T10:00:00Z"),       # saved later
        _label("k_a2", "equipment", "claude-code", "2026-05-03T10:00:00Z"),  # only the assistant
        _label("k_b2", "deliverable", "claude-code", "2026-05-03T10:00:00Z"),
        _label("k_b2", "requirement", PM, "2026-05-01T10:00:00Z"),
    ]})
    got = {a.key: a.label for a in deal.atoms if a.label}
    assert got["k_a1"]["label_type"] == "requirement"
    assert got["k_b2"]["label_type"] == "requirement"               # "claude-code" is not a person
    assert got["k_a2"]["label_type"] == "equipment"                 # no person row: the twin stands in
    assert deal.label_stats == {"person": 2, "twin_superseded": 2, "twin_used": 1,
                                "twin_only_skipped": 0, "duplicate": 0}


def test_a_missed_line_with_a_twin_is_one_line_with_the_person_label():
    missed = dict(origin="labeler", text="Door frames are steel.", doc_id="art-kick")
    deal = _deal({"labels": [
        _label("lbl_missed", "requirement", PM, "2026-05-02T10:00:00Z", **missed),
        _label("lbl_missed", "_keep", TWIN_BOT, "2026-05-03T10:00:00Z", **missed),
    ]})
    rows = [a for a in deal.atoms if a.key == "lbl_missed"]
    assert len(rows) == 1 and rows[0].label["label_type"] == "requirement"


def _j(head, verdict, key, labeler=PM, at="2026-05-02T10:00:00Z", **kw):
    return {"head": head, "verdict": verdict, "target_key": key, "labeler": labeler, "judged_at": at,
            "target": kw.pop("target", {}), **kw}


def test_one_judgment_per_card_the_latest_person_row():
    pair = {"a": {"atomId": "at-k_a1"}, "b": {"atomId": "at-k_b2"}}
    deal = _deal({"judgments": [
        _j("conflict", "contradicts", "edge_1", target=pair, at="2026-05-02T10:00:00Z"),
        _j("conflict", "supports", "edge_1", target=pair, at="2026-05-04T10:00:00Z"),       # changed mind
        _j("conflict", "unrelated", "edge_1", labeler=TWIN_BOT, target=pair, at="2026-05-05T10:00:00Z"),
        _j("conflict", "contradicts", "edge_2", labeler=TWIN_BOT, target=pair),            # draft only
    ]})
    assert [(j["target_key"], j["verdict"]) for j in deal.judgments] == [("edge_1", "supports")]
    assert deal.judgment_stats["duplicate"] == 1
    assert deal.judgment_stats["twin_superseded"] == 1
    assert deal.judgment_stats["twin_only_skipped"] == 1


def test_one_link_per_key_and_drafts_never_train():
    deal = _deal({"links": [
        _link("k_a2", "k_b2"), _link("k_a2", "k_b2"),                   # saved twice
        _link("k_a2", "k_b2", labeler=TWIN_BOT),
        _link("k_a1", "k_b2", labeler="claude-code"),                    # a draft nobody accepted
    ]})
    assert deal.edges == [("k_a2", "k_b2", "supports")]
    assert deal.link_stats["duplicate"] == 1
    assert deal.link_stats["twin_superseded"] == 1
    assert deal.link_stats["twin_only_skipped"] == 1


# ------------------------------------------------------------ 3. what no head reads

@pytest.fixture(scope="module")
def schema():
    pytest.importorskip("torch")
    from ml.c3.schema import load_schema
    return load_schema()


def test_rows_no_head_reads_are_counted_and_deal_heads_train(schema):
    from ml.c3.data import featurize
    deal = _deal({
        "judgments": [
            _j("billing_type", "t_and_m", "deal:billing_type"),
            _j("rule", "should_not_fire", "rule:lift_terms",
               target={"source": {"atomId": "at-k_b1"}}),
            _j("norm", "24", "norm:k_a2"),
        ],
        "deal_answers": [{"labeler": PM, "primary_service": "security_camera", "declared_site_count": 3},
                         {"labeler": TWIN_BOT, "primary_service": "av", "declared_site_count": 1}],
    })
    assert deal.judgment_stats["deal_answers_untrained"] == 1
    b = featurize(deal, schema)
    assert [(j.key, j.size, len(j.lines)) for j in b.judged] == [("jdg:billing_type", "deal", len(ATOMS))]
    st = b.judgment_stats
    assert st["trained:deal"] == 1
    assert st["no_head:rule"] == 1 and st["no_head:norm"] == 1 and st["no_head"] == 2
    assert st["deal_answers_untrained"] == 1
