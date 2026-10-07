"""Rows a labeler marks "do not learn from this" never reach training.

Old hand-built Deal Kit lines were marked four different ways across deals
(a note prefix, weight_tier='exclude', consumer='ignore',
reads_set.exclude_from_training). Rejecting them would teach the heads to drop
real Deal Kit facts; full-weight rows would teach them the old manual lines.
"""
from app.learning.human_labels import IngestReport, rows_for_deal

PERSON = "developer@purtera-it.com"


def _label(key, text, **kw):
    return {"label_key": key, "atom_id": f"a_{key}", "labeler": PERSON,
            "label_type": "commercial_term", "decide_text": text, "text": text, **kw}


def _doc(*labels, links=(), judgments=()):
    return {"deal_id": "999001", "labels": list(labels), "links": list(links),
            "judgments": list(judgments)}


KEEP = _label("lbl_keep", "Margin % on Total Deal: 30.43%")


def _keys(rows):
    return {(r.get("provenance") or {}).get("label_key") or r.get("label_key") for r in rows}


def _texts(rows):
    return " ".join(str(r.get("raw_text") or "") + " " + str(r.get("masked_text") or "") for r in rows)


def test_every_marker_excludes_the_label():
    markers = [
        {"note": "EXCLUDE_FROM_TRAINING: old manual Deal Kit row"},
        {"note": "  exclude_from_training: lower case"},
        {"weight_tier": "exclude"},
        {"consumer": "ignore"},
        {"weight_tier": "slight", "reads_set": {"exclude_from_training": True}},
    ]
    for i, m in enumerate(markers):
        gone = _label(f"lbl_x{i}", f"Old manual Deal Kit line number {i}", **m)
        report = IngestReport()
        rows = rows_for_deal(_doc(KEEP, gone), report)
        assert f"Old manual Deal Kit line number {i}" not in _texts(rows), m
        assert "30.43%" in _texts(rows)
        assert report.skipped.get("excluded from training") == 1


def test_links_and_judgments_touching_an_excluded_atom_are_dropped():
    gone = _label("lbl_gone", "Old manual Deal Kit total line", note="EXCLUDE_FROM_TRAINING: manual")
    link_from = {"labeler": PERSON, "relation": "contradicts", "from_key": "lbl_gone",
                 "from_text": "Old manual Deal Kit total line", "to_text": "Margin % on Total Deal: 30.43%"}
    link_to = {"labeler": PERSON, "relation": "contradicts", "from_key": "lbl_keep",
               "from_text": "Margin % on Total Deal: 30.43%", "to_atom_id": "a_lbl_gone",
               "to_text": "Old manual Deal Kit total line"}
    rows = rows_for_deal(_doc(KEEP, gone, links=[link_from, link_to]))
    assert "Old manual Deal Kit total line" not in _texts(rows)


def test_unmarked_slight_tier_still_trains():
    lb = _label("lbl_s", "A slight but real line of scope", weight_tier="slight")
    rows = rows_for_deal(_doc(lb))
    assert "A slight but real line of scope" in _texts(rows)


# --- train_for: the quote parser and the delivery parser (Atlas, runbook) ---

from app.learning.human_labels import DELIVERY_PARSER, QUOTE_PARSER, train_for  # noqa: E402


def test_train_for_reads_list_string_or_column():
    assert train_for({"reads_set": {"train_for": ["quote_parser", "delivery_parser"]}}) == {
        QUOTE_PARSER, DELIVERY_PARSER}
    assert train_for({"reads_set": {"train_for": "delivery_parser"}}) == {DELIVERY_PARSER}
    assert train_for({"train_for": "quote_parser, delivery_parser"}) == {QUOTE_PARSER, DELIVERY_PARSER}
    assert train_for({"reads_set": {}}) is None


def test_old_deal_kit_tagged_for_delivery_reaches_only_the_delivery_export():
    kit = _label("lbl_kit", "Deal Kit: 4 techs for 3 days at 2 sites",
                 note="EXCLUDE_FROM_TRAINING: old manual Deal Kit",
                 reads_set={"train_for": ["delivery_parser"]})
    quote = _texts(rows_for_deal(_doc(KEEP, kit)))
    assert "4 techs" not in quote and "30.43%" in quote
    delivery = _texts(rows_for_deal(_doc(KEEP, kit), parser=DELIVERY_PARSER))
    assert "4 techs" in delivery
    assert "30.43%" not in delivery  # untagged rows stay quote-only


def test_train_for_without_quote_parser_leaves_quote_training():
    only = _label("lbl_only", "SOW clause: crew of two on site", reads_set={"train_for": "delivery_parser"})
    both = _label("lbl_both", "SOW clause: work starts June 9",
                  reads_set={"train_for": ["quote_parser", "delivery_parser"]})
    quote = _texts(rows_for_deal(_doc(only, both)))
    assert "crew of two" not in quote and "June 9" in quote
    delivery = _texts(rows_for_deal(_doc(only, both), parser=DELIVERY_PARSER))
    assert "crew of two" in delivery and "June 9" in delivery


def test_delivery_export_is_facts_only():
    tag = {"train_for": ["delivery_parser"]}
    fact = _label("lbl_fact", "Install 12 displays in the lobby", reads_set=tag)
    reject = {**_label("lbl_rej", "Thanks so much, talk soon", reads_set=tag), "label_type": "_keep"}
    chat = {**_label("lbl_chat", "Hope you had a good weekend", reads_set=tag), "label_type": "small_talk"}
    answered = {**_label("lbl_ans", "Yes, the lift is available", reads_set=tag), "label_type": "answered_question"}
    trigger = _label("lbl_trig", "Once the PO lands we schedule the crew",
                     reads_set={**tag, "trigger_event": "po_received"})
    link = {"labeler": PERSON, "relation": "context", "from_key": "lbl_fact", "to_key": "lbl_chat",
            "from_text": "Install 12 displays in the lobby", "to_text": "Hope you had a good weekend"}
    rows = rows_for_deal(_doc(fact, reject, chat, answered, trigger, links=[link]), parser=DELIVERY_PARSER)
    text = _texts(rows)
    assert "12 displays" in text
    for gone in ("Thanks so much", "good weekend", "lift is available", "PO lands"):
        assert gone not in text, gone


def test_judgments_keyed_by_kind_and_id_on_an_excluded_atom_are_dropped():
    """Judgment keys are `<kind>:<id>`; the excluded label carries the bare id.

    Before, only a judgment whose own note started with the marker was dropped,
    so every verdict on an old manual Deal Kit row trained."""
    gone = _label("lbl_gone", "Old manual Deal Kit total line", note="EXCLUDE_FROM_TRAINING: manual")
    other = _label("lbl_other", "Install 12 displays in the lobby")

    def jd(head, verdict, key, text, **kw):
        return {"labeler": PERSON, "head": head, "verdict": verdict, "target_key": key, "text": text, **kw}

    judgments = [
        jd("suppression", "should_have_been_kept", "atom:a_lbl_gone", "Old manual Deal Kit judged by atom id"),
        jd("site", "same_site", "site:a_lbl_gone", "Old manual Deal Kit judged as a site"),
        jd("suppression", "correctly_dropped", "sup:lbl_gone", "Old manual Deal Kit judged by label key"),
        jd("conflict", "unrelated", "xdoc:0123abcd", "Old manual Deal Kit judged in a pair",
           target={"a": {"atomId": "a_lbl_keep"}, "b": {"atomId": "a_lbl_gone"}}),
        jd("gap", "valid", "gap:4567cdef", "Old manual Deal Kit judged as a gap source",
           target={"source": {"atomId": "a_lbl_gone"}}),
        # Unrelated atoms and hashed keys still train.
        jd("suppression", "should_have_been_kept", "atom:a_lbl_other", "A kept atom judged by atom id"),
        jd("gap", "valid", "gap:atom:a_lbl_gone", "A gap key that only looks like an atom key"),
    ]
    report = IngestReport()
    rows = rows_for_deal(_doc(KEEP, gone, other, judgments=judgments), report)
    text = _texts(rows)
    assert "Old manual Deal Kit" not in text
    assert "A kept atom judged by atom id" in text
    assert "A gap key that only looks like an atom key" in text
    assert report.skipped["judgment touches an atom excluded from training"] == 5


def test_delivery_export_keeps_judgments_on_its_own_facts_by_kind_and_id():
    tag = {"train_for": ["delivery_parser"]}
    fact = _label("lbl_fact", "Install 12 displays in the lobby", reads_set=tag)
    j = {"labeler": PERSON, "head": "suppression", "verdict": "should_have_been_kept",
         "target_key": "atom:a_lbl_fact", "text": "Install 12 displays, judged"}
    rows = rows_for_deal(_doc(fact, judgments=[j]), parser=DELIVERY_PARSER)
    assert "judged" in _texts(rows)
