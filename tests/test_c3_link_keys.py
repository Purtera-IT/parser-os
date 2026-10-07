"""Links drawn on cards whose key is not a line's label_key still start from
the right line (ml/c3/data.py ``_blob_links``), on invented rows.

* a Dropped card's ``sup:<label_key>@line_start=...`` key names that exact
  copy, not the first line with the same words;
* a Places card's ``site:<atom_id>`` key names that atom;
* a conflict or site-pair card starts from its a side (its b side when the
  link points at the a side); a question card from its source line;
* a question card with no line stays unresolved and is counted;
* ordinary keys and text fallbacks resolve as before; no self-links.
"""
from __future__ import annotations

from ml.c3.data import DealExample

PM = "pm@example.com"


def _atom(k, text, t, atom_id=None):
    return {"label_key": k, "atom_id": atom_id or f"at-{k}", "text": text, "doc_id": "email-1",
            "doc_kind": "email", "entered_at": f"2026-05-01T10:{t:02d}:00Z",
            "speaker_role": "customer", "speaker_side": "customer"}


TWIN = "Bring two ladders to the north dock."
ATOMS = [
    _atom("k_orig", TWIN, 0),                                  # the kept line
    _atom("k_drop", TWIN, 1),                                  # its dropped copy, same words
    _atom("k_other", "Ladders are stored in the blue van.", 2),
    _atom("k_site1", "Work is at the Cedar Lane depot.", 3, atom_id="atm_copy_s1"),
    _atom("k_site2", "Cedar Lane depot, gate B.", 4),
    _atom("k_win1", "Access window is mornings only.", 5),
    _atom("k_win2", "Access window is afternoons only.", 6),
    _atom("k_q", "How many techs are needed?", 7),
    _atom("k_ans", "Plan for three techs.", 8),
]


def _link(from_key, to_key, rel="same_as", head="type", from_text="", **kw):
    return {"from_head": head, "from_key": from_key, "from_text": from_text, "to_label_key": to_key,
            "relation": rel, "note": "", "labeler": PM, **kw}


JUDGMENTS = [
    {"head": "conflict", "target_key": "edge_c1", "verdict": "contradicts", "labeler": PM,
     "target": {"a": {"atomId": "at-k_win1", "text": ATOMS[5]["text"]},
                "b": {"atomId": "at-k_win2", "text": ATOMS[6]["text"]}}},
    # A site pair after a re-parse: the ids changed, the a side's text still matches.
    {"head": "site", "target_key": "pair:p1", "verdict": "same_site", "labeler": PM,
     "target": {"a": {"evidence": {"atomId": "old-id", "text": ATOMS[3]["text"]}},
                "b": {"evidence": {"atomId": "at-k_site2"}}}},
    {"head": "gap", "target_key": "gap:g1", "verdict": "valid", "labeler": PM,
     "target": {"source": {"atomId": "at-k_q"}}},
    # A generated question: no line at all.
    {"head": "gap", "target_key": "gap:g2", "verdict": "valid", "labeler": PM, "target": {}},
]


def _deal(links, judgments=()):
    return DealExample.from_training_blob({"labels": [], "links": links, "judgments": list(judgments)},
                                          ATOMS, deal_id="synthetic-link-keys")


def test_dropped_card_key_names_its_copy_not_the_text_twin():
    deal = _deal([
        _link("sup:k_drop@line_start=12,sentence_index=3", "k_orig", head="suppression", from_text=TWIN),
        _link("sup:k_drop", "k_other", rel="context", head="suppression", from_text=TWIN),
    ])
    assert ("k_drop", "k_orig", "same_as") in deal.edges        # was a self-link on k_orig
    assert ("k_drop", "k_other", "context") in deal.edges       # was drawn from k_orig
    assert deal.link_stats["self_links"] == 0


def test_places_card_key_names_its_atom():
    deal = _deal([_link("site:atm_copy_s1", "k_site2", head="facility", from_text="Cedar Lane")])
    assert deal.edges == [("k_site1", "k_site2", "same_as")]


def test_conflict_and_site_pair_cards_start_from_their_a_side():
    deal = _deal([
        _link("edge_c1", "k_other", rel="context", head="conflict"),
        _link("pair:p1", "k_site2", rel="supports", head="site"),
        _link("gap:g1", "k_ans", rel="answers", head="gap"),
    ], JUDGMENTS)
    assert ("k_win1", "k_other", "context") in deal.edges
    assert ("k_site1", "k_site2", "supports") in deal.edges
    assert ("k_ans", "k_q", "answers") in deal.edges            # question card: answer -> question
    assert deal.link_stats["card_anchored"] == 3


def test_question_card_with_no_line_is_counted_unresolved():
    deal = _deal([_link("gap:g2", "k_ans", rel="answers", head="gap",
                        from_text="Generated question nobody wrote down?")], JUDGMENTS)
    assert deal.edges == []
    assert deal.link_stats["card_no_line"] == 1
    assert deal.link_stats["unresolved"] == 1


def test_ordinary_keys_and_text_fallback_unchanged():
    deal = _deal([
        _link("k_other", "k_orig", rel="supports"),
        # An older parse's key: its text still finds the line.
        _link("lbl_stale", "k_other", rel="context", from_text="Plan for three techs."),
        # A text key whose text is the target's own: a self-link, dropped and counted.
        _link("lbl_stale2", "k_orig", from_text=TWIN),
    ], JUDGMENTS)
    assert sorted(deal.edges) == [("k_ans", "k_other", "context"), ("k_other", "k_orig", "supports")]
    assert deal.link_stats["self_links"] == 1
    assert deal.link_stats["edges"] == 2
    assert deal.link_stats["card_anchored"] == 0


def test_pair_card_link_to_its_own_a_side_starts_from_b_not_a_self_link():
    deal = _deal([_link("edge_c1", "k_win1", rel="supports", head="conflict")], JUDGMENTS)
    assert deal.edges == [("k_win2", "k_win1", "supports")]
    assert deal.link_stats["self_links"] == 0
