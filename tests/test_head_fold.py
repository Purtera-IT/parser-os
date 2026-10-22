"""Which copy of a folded line survives is the keep head's call (head_fold).

A dedup stage finds that two lines are one (extraction); which copy the deal
keeps is read:co_action's decision. Hermetic: readings are stubbed, atoms are
plain objects, all text made up.
"""
from __future__ import annotations

import app.core.c3_type_base as c3
from app.core.head_fold import HEAD_FOLD_KEY, fold_by_heads, fold_losers
from app.core.suppression_ledger import SURVIVOR_KEY


class _A:
    def __init__(self, atom_id, text, *, artifact="art1", stage=None, survivor=None, quoted=False):
        self.id = atom_id
        self.raw_text = text
        self.artifact_id = artifact
        self.project_id = "deal-test"
        self.source_refs = []
        self.review_flags = []
        self.value = {"quoted": quoted}
        if stage:
            self.review_flags = [f"suppressed:{stage}"]
            self.value["_suppression"] = {"stage": stage, "reason": "quoted copy"}
            self.value[SURVIVOR_KEY] = {"atom_id": survivor, "artifact_id": artifact, "stage": stage}


_LINE = "He also confirmed that he will need us to run the cabling too."


def _reads(table):
    return lambda aid, head: table.get(aid)


def test_head_kept_quoted_copy_replaces_a_rejected_survivor():
    authored = _A("a_auth", _LINE)
    other = _A("a_other", "Site walk is Tuesday.")
    q1 = _A("a_q1", _LINE, artifact="art2", stage="quoted_history_dedup", survivor="a_auth", quoted=True)
    q2 = _A("a_q2", _LINE, artifact="art3", stage="quoted_history_dedup", survivor="a_auth", quoted=True)
    ledger = [q1, q2]
    reads = {"a_auth": ("reject", 0.7), "a_q1": ("keep", 0.62), "a_q2": ("keep", 0.89), "a_other": ("keep", 1.0)}
    atoms, swaps = fold_by_heads([other, authored], ledger, reading=_reads(reads))
    assert [a.id for a in atoms] == ["a_other", "a_q2"]  # same position, surest kept copy
    assert swaps == [(q2, authored)]
    assert authored in ledger and q2 not in ledger and q1 in ledger
    assert authored.value[SURVIVOR_KEY]["atom_id"] == "a_q2"
    assert "suppressed:quoted_history_dedup" in authored.review_flags
    assert not any(str(f).startswith("suppressed:") for f in q2.review_flags)
    assert "_suppression" not in q2.value and SURVIVOR_KEY not in q2.value
    assert q2.value[HEAD_FOLD_KEY]["over"] == "a_auth"
    assert q1.value[SURVIVOR_KEY]["atom_id"] == "a_q2"  # the other copy names the new survivor


def test_fold_stands_when_survivor_is_kept_or_unread_or_no_copy_is_kept():
    for survivor_reading, copy_reading in ((("keep", 0.6), ("keep", 0.99)),  # survivor kept
                                           (None, ("keep", 0.99)),           # survivor never read
                                           (("reject", 0.9), ("reject", 0.8)),
                                           (("reject", 0.9), ("keep", 0.3))):  # copy below the bar
        s = _A("s", _LINE)
        c = _A("c", _LINE, stage="quoted_history_dedup", survivor="s", quoted=True)
        reads = {k: v for k, v in (("s", survivor_reading), ("c", copy_reading)) if v is not None}
        atoms, swaps = fold_by_heads([s], [c], reading=_reads(reads))
        assert atoms == [s] and swaps == []


def test_only_named_stages_and_named_survivors_are_losers():
    a = _A("x", _LINE, stage="quoted_history_dedup", survivor="s")
    b = _A("y", _LINE, stage="chrome", survivor="s")
    c = _A("z", _LINE, stage="quoted_history_dedup")
    c.value.pop(SURVIVOR_KEY)
    assert fold_losers([a, b, c]) == [a]


def test_folded_copies_are_read_with_the_deal(monkeypatch):
    """also_read: every read of the deal holds the copies after the compile's lines."""
    monkeypatch.setattr(c3, "_DEAL", {})
    c3.set_deal([])
    kept = [_A("k1", "Install two cameras."), _A("k2", _LINE)]
    copy = _A("c1", _LINE, artifact="art1", stage="quoted_history_dedup", survivor="k2", quoted=True)
    c3.also_read([copy, kept[0]])  # one already in the list is not read twice
    _deal, rows, keys = c3._rows(kept, {})
    assert keys == ["k1", "k2", "c1"]
    assert [r["atom_id"] for r in rows] == ["k1", "k2", "c1"]
    c3.set_deal([])
    _deal, rows, keys = c3._rows(kept, {})
    assert keys == ["k1", "k2"]
