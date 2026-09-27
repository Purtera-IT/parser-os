"""The only stage that can put a fact back.

Every other gate in the compile takes atoms away, and every defect found on
010180 in the week to 2026-09-27 was something being taken away: eight rooms
deleted by a substance gate, a drawing set aside as another job, a paragraph
glued into one atom, Outlook invite fields admitted as scope, wreckage minted
from a floor plan's walls.

`span_admission` already serves the trained admission heads, but it re-types
atoms that EXIST. A line the parser skipped has no atom to re-type, so nothing
could recover it. Measured on 010180: 2,634 unclaimed lines across 38
documents, 910 of them read by nothing at all.

The safety argument is the asymmetry, so it is what these tests pin: this stage
may only ADD.
"""
from __future__ import annotations

import pytest

from app.core import line_admission as LA


def _coverage(*lines):
    return {"text": [{"artifact_id": "art_1", "unclaimed": list(lines)}]}


def _line(text, state="unread", n=1):
    return {"line": n, "text": text, "state": state}


# ---------------------------------------------------------------- selection

def test_the_same_line_is_judged_once_however_often_it_repeats():
    """A quoted thread carries a signature down every reply. On 010180, 910
    unclaimed lines are 34 distinct texts and "Bell Works | 101 Crawfords
    Corner Road" is 174 of them. Judging it 174 times is 174 chances to be
    inconsistent about one decision."""
    sig = "Bell Works | 101 Crawfords Corner Road, Holmdel NJ"
    cov = _coverage(*[_line(sig, n=i) for i in range(1, 175)])
    rows = LA.unclaimed_lines(cov)
    assert len(rows) == 1
    assert rows[0]["repeats"] == 174


def test_a_fragment_is_too_short_to_be_a_fact():
    """The shapes that made 18 of 010180's 40 drawing atoms wreckage: a word
    split by a wall, a table cell, a page number."""
    cov = _coverage(_line("as: cial Leasi"), _line("REE N 1200"), _line("JAN"))
    assert LA.unclaimed_lines(cov) == []


def test_a_real_sentence_is_considered():
    cov = _coverage(_line("The riser room is locked after 6pm and the super has the only key."))
    rows = LA.unclaimed_lines(cov)
    assert len(rows) == 1
    assert rows[0]["state"] == "unread"


def test_chrome_and_images_are_not_candidates():
    """Coverage already recognised these as furniture; a header is not a line
    somebody failed to read."""
    cov = _coverage(
        _line("Page 3 of 11 -- FlexTrade New Office Cabling", state="chrome"),
        _line("floorplan.png", state="image"),
    )
    assert LA.unclaimed_lines(cov) == []


def test_a_dropped_line_is_considered_as_well_as_an_unread_one():
    """"suppressed" is an atom a gate removed -- exactly the eight rooms. Those
    are the lines most worth a second look."""
    cov = _coverage(_line("Electrical connections will be provided by the landlord.",
                          state="suppressed"))
    assert len(LA.unclaimed_lines(cov)) == 1


# ---------------------------------------------------------------- admission

def _run(monkeypatch, verdict, *, enabled=True):
    monkeypatch.setenv("SOWSMITH_LINE_ADMISSION", "1" if enabled else "0")
    monkeypatch.setattr(LA, "_verdict", lambda text, scope: (verdict, "store", 0.9))
    cov = _coverage(_line("The riser room is locked after 6pm and the super has the only key."))
    return LA.admit_missed_lines(cov, project_id="d1", make_atom=lambda line: {"text": line["text"]})


def test_a_keep_mints_an_atom(monkeypatch):
    atoms, verdicts = _run(monkeypatch, "keep")
    assert len(atoms) == 1
    assert verdicts[0]["admitted"] is True


@pytest.mark.parametrize("verdict", ["drop", None, "", "maybe"])
def test_anything_short_of_keep_leaves_the_line_alone(monkeypatch, verdict):
    """Guess-free. An abstain, a drop, or a head that has never been taught all
    leave the line exactly where the parser left it."""
    atoms, verdicts = _run(monkeypatch, verdict)
    assert atoms == []
    assert verdicts[0]["admitted"] is False


def test_it_is_off_until_switched_on(monkeypatch):
    atoms, verdicts = _run(monkeypatch, "keep", enabled=False)
    assert (atoms, verdicts) == ([], [])


def test_every_line_it_looked_at_is_reported(monkeypatch):
    """A stage that reports only what it added cannot be audited. The lines it
    declined are the ones worth arguing about."""
    monkeypatch.setenv("SOWSMITH_LINE_ADMISSION", "1")
    monkeypatch.setattr(LA, "_verdict", lambda text, scope: ("drop", "store", 0.8))
    cov = _coverage(
        _line("The riser room is locked after 6pm and the super has the only key."),
        _line("Suite 1316, 3rd Floor, Holmdel, NJ 07733", n=2),
    )
    atoms, verdicts = LA.admit_missed_lines(cov, project_id="d1", make_atom=lambda l: {"t": l})
    assert atoms == []
    assert len(verdicts) == 2
    assert all(v["verdict"] == "drop" for v in verdicts)


def test_it_cannot_remove_anything():
    """The whole safety argument, stated as a test: the module has no path that
    deletes or re-types an existing atom. It takes coverage and returns new
    atoms; the compiler's atom list is not an argument."""
    import inspect

    src = inspect.getsource(LA)
    for forbidden in ("atoms.remove", "atom_type =", ".pop(", "del atom"):
        assert forbidden not in src, forbidden
    assert "atoms" not in inspect.signature(LA.admit_missed_lines).parameters
