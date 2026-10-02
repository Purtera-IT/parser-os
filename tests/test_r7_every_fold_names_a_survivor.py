"""No atom vanishes without a recorded survivor.

Every stage that removes an atom is either a FOLD (the atom's words live on
in another atom, which its suppressed entry must name) or a DROP (chrome, a
heading, a gate's verdict: marked ``kind: "drop"`` with its reason). Shape of
the live failures (synthetic text): a site line and a contact line folded into
an atom a later stage removed, leaving suppressed entries whose survivor was
null or no longer in the result.
"""
from __future__ import annotations

from app.core.cross_doc_copies import settle_folds
from app.core.email_threading import dedup_quoted_history
from app.core.entity_resolution import collapse_duplicate_atoms
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.suppression_ledger import (
    SURVIVOR_KEY,
    capture_suppressed,
    settle_ledger,
    suppression_kind,
    take_folds,
)

_N = [0]


def _atom(text, *, art="art_a", atype=AtomType.scope_item, value=None, conf=0.9, line=None):
    _N[0] += 1
    n = _N[0]
    ref = SourceRef(id=f"src_{n}", artifact_id=art, artifact_type=ArtifactType.txt, filename=f"{art}.md",
                    locator={"line_start": line or n, "line_end": line or n},
                    extraction_method="test", parser_version="t")
    return EvidenceAtom(
        id=f"atm_{n:04d}", project_id="p", artifact_id=art, atom_type=atype,
        raw_text=text, normalized_text=text.lower(), value=dict(value or {}), entity_keys=[],
        source_refs=[ref], authority_class=AuthorityClass.machine_extractor, confidence=conf,
        review_flags=[], review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


def _suppress(atom, stage, reason="r"):
    capture_suppressed([atom], [], stage=stage, reason=reason)
    return atom


def test_a_drop_stage_is_marked_a_drop_with_its_reason():
    a = _suppress(_atom("Signature: ________"), "execution_boilerplate_drop", "boilerplate")
    assert a.value["_suppression"] == {"stage": "execution_boilerplate_drop", "reason": "boilerplate", "kind": "drop"}
    b = _suppress(_atom("Install two cameras"), "duplicate_atom_collapse")
    assert b.value["_suppression"]["kind"] == "fold"
    assert suppression_kind(a) == "drop" and suppression_kind(b) == "fold"


def test_a_fold_whose_survivor_was_folded_again_is_repointed_along_the_chain():
    end = _atom("100 Main St, Springfield, OH 45501", atype=AtomType.physical_site)
    mid = _suppress(_atom("100 Main St, Springfield, OH", atype=AtomType.physical_site), "semantic_dedup")
    mid.value[SURVIVOR_KEY] = {"atom_id": end.id}
    first = _suppress(_atom("100 Main St Springfield", atype=AtomType.physical_site), "pre_classify_dedup")
    first.value[SURVIVOR_KEY] = {"atom_id": mid.id}
    atoms, supp, counts = settle_ledger([end], [mid, first])
    assert atoms == [end] and len(supp) == 2
    assert first.value[SURVIVOR_KEY]["atom_id"] == end.id and first.value[SURVIVOR_KEY]["via"] == mid.id
    assert counts["repointed"] == 1


def test_a_site_whose_survivor_is_gone_comes_back():
    site = _suppress(_atom("100 Main St, Springfield, OH 45501", atype=AtomType.physical_site), "semantic_dedup")
    site.value[SURVIVOR_KEY] = {"atom_id": "atm_gone"}
    atoms, supp, counts = settle_ledger([], [site])
    assert atoms == [site] and supp == [] and counts["restored"] == 1
    assert not any(f.startswith("suppressed:") for f in site.review_flags)
    assert "_suppression" not in site.value and site.value["_restored"]["stage"] == "semantic_dedup"


def test_a_site_folded_into_a_non_site_that_was_dropped_comes_back():
    gate = _suppress(_atom("VC Links"), "substance_gate")
    site = _suppress(_atom("VC Links, 100 Main St", atype=AtomType.physical_site), "semantic_dedup")
    site.value[SURVIVOR_KEY] = {"atom_id": gate.id}
    atoms, _supp, _ = settle_ledger([], [gate, site])
    assert site in atoms and gate not in atoms


def test_a_fold_into_a_dropped_atom_is_recorded_as_a_drop_naming_it():
    gate = _suppress(_atom("Yeah."), "substance_gate", "backchannel filler")
    echo = _suppress(_atom("Yeah."), "duplicate_atom_collapse")
    echo.value[SURVIVOR_KEY] = {"atom_id": gate.id}
    atoms, supp, counts = settle_ledger([], [gate, echo])
    assert atoms == [] and len(supp) == 2
    assert echo.value["_suppression"]["kind"] == "drop"
    assert gate.id in echo.value["_suppression"]["drop_reason"]
    assert "backchannel filler" in echo.value["_suppression"]["drop_reason"]


def test_a_fold_naming_nothing_names_the_standing_line_with_its_words_or_comes_back():
    kept = _atom("Complete billing tasks")
    twin = _suppress(_atom("Complete billing tasks"), "quote_line_head")
    lone = _suppress(_atom("Develop schedule based on stakeholder requirements"), "quote_line_head")
    atoms, supp, counts = settle_ledger([kept], [twin, lone])
    assert twin.value[SURVIVOR_KEY]["atom_id"] == kept.id
    assert lone in atoms and supp == [twin]


def test_duplicate_collapse_names_the_atom_it_kept():
    take_folds()
    a = _atom("Contractor shall provide and install category 6 cabling to every outlet in rooms 101 to 120.", conf=0.9)
    b = _atom("Contractor shall provide and install category 6 cabling to every outlet in rooms 101 to 120 .", conf=0.5)
    c = _atom("Install two cameras", conf=0.9)
    d = _atom("Install two cameras", conf=0.4)
    before = [a, b, c, d]
    after = collapse_duplicate_atoms(list(before))
    out, _copies, restored = settle_folds(before, after, [], take_folds(), stage="duplicate_atom_collapse",
                                          make_copies=False)
    assert restored == [] and set(map(id, out)) == {id(a), id(c)}
    assert b.value[SURVIVOR_KEY]["atom_id"] == a.id
    assert d.value[SURVIVOR_KEY]["atom_id"] == c.id


def test_a_quoted_echo_names_the_authored_line_in_another_file():
    thread = {"thread_id": "t1", "thread_index": 0}
    authored = _atom("Please confirm the lift is on site Monday morning.", art="art_mail1",
                     value={"email_thread": dict(thread)})
    echo = _atom("Please confirm the lift is on site Monday morning.", art="art_mail2",
                 value={"email_thread": {"thread_id": "t1", "thread_index": 1}, "quoted": True})
    take_folds()
    before = [authored, echo]
    kept, dropped = dedup_quoted_history(list(before))
    assert dropped == [echo]
    out, copies, restored = settle_folds(before, kept, [], take_folds(), stage="quoted_history_dedup",
                                         make_copies=False)
    assert out == [authored] and copies == [] and restored == []
    assert echo.value[SURVIVOR_KEY]["atom_id"] == authored.id
