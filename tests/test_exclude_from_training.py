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
