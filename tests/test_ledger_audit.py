"""Ledger audit: report-only check for atoms hidden by a dedup or fold stage.

Synthetic shapes mirroring five real cases (no customer text):
a restored drop, a table row folded into a name-and-title atom, a clean
sentence folded into a merged one, a PMO list item dropped from its list, and
a quoted email list with only some items folded.
"""
from __future__ import annotations

import copy
from pathlib import Path

from app.core.ids import stable_id
from app.core.ledger_audit import audit_ledger
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.suppression_ledger import (
    SURVIVOR_KEY,
    capture_suppressed,
    mark_dropped_not_folded,
    settle_ledger,
)


def _atom(aid: str, text: str, *, art: str = "art_sow", loc: dict | None = None, value: dict | None = None,
          atom_type: AtomType = AtomType.scope_item, flags: list[str] | None = None) -> EvidenceAtom:
    return EvidenceAtom(
        id=aid,
        project_id="p",
        artifact_id=art,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=dict(value or {}),
        source_refs=[SourceRef(
            id=stable_id("src", aid), artifact_id=art, artifact_type=ArtifactType.docx,
            filename=f"{art}.docx", locator=dict(loc or {}), extraction_method="test", parser_version="t",
        )],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=list(flags or []),
        parser_version="t",
    )


def _fold(atom: EvidenceAtom, stage: str, survivor: EvidenceAtom | str | None) -> EvidenceAtom:
    capture_suppressed([atom], [], stage=stage, reason="test fold")
    if survivor is not None:
        sid = survivor if isinstance(survivor, str) else survivor.id
        atom.value[SURVIVOR_KEY] = {"atom_id": sid, "stage": stage}
    return atom


def _ids(report: dict, rule: str) -> list[str]:
    return [e.get("atom_id") for e in report["examples"][rule]]


# (a) unresolved survivor ------------------------------------------------------

def test_fold_with_null_survivor_is_flagged():
    """A full table row folded with no survivor (the row's email cell is lost)."""
    kept = [_atom("k_row", "Pat Example | Field Lead", loc={"table_index": 1, "row": 1})]
    row = _fold(_atom("s_row", "Pat Example | Field Lead | pat@example.test",
                      loc={"table_index": 1, "row": 1}), "pre_classify_dedup", None)
    report = audit_ledger(kept, [row])
    assert report["counts"]["unresolved_survivor"] == 1
    assert _ids(report, "unresolved_survivor") == ["s_row"]
    assert report["examples"]["unresolved_survivor"][0]["survivor_id"] is None


def test_fold_into_an_id_nothing_answers_to_is_flagged():
    kept = [_atom("k1", "Install two access points")]
    s = _fold(_atom("s1", "Mount the camera"), "semantic_dedup", "atm_gone")
    report = audit_ledger(kept, [s])
    assert _ids(report, "unresolved_survivor") == ["s1"]


def test_survivor_reached_through_a_chain_or_via_alias_is_not_flagged():
    standing = _atom("k_final", "Label every cable at both ends")
    mid = _fold(_atom("s_mid", "Label every cable at both ends"), "semantic_dedup", standing)
    first = _fold(_atom("s_first", "Label every cable at both ends"), "pasted_note_dedup", mid)
    # A re-pointed fold whose original name is only known as `via`.
    other = _fold(_atom("s_other", "Label every cable at both ends"), "duplicate_atom_collapse", "atm_old")
    aliased = _atom("s_alias", "Label every cable at both ends")
    _fold(aliased, "semantic_dedup", None)
    aliased.value[SURVIVOR_KEY] = {"atom_id": "k_final", "via": "atm_old", "stage": "semantic_dedup"}
    report = audit_ledger([standing], [mid, first, other, aliased])
    assert report["counts"]["unresolved_survivor"] == 0


def test_deliberate_drop_needs_no_survivor():
    s = _atom("s_drop", "Page 3 of 9")
    mark_dropped_not_folded(s, "repeated page footer")
    capture_suppressed([s], [], stage="semantic_dedup", reason="footer")
    report = audit_ledger([], [s])
    assert report["counts"]["unresolved_survivor"] == 0
    assert report["drops"] == 1 and report["folds"] == 0


def test_restored_fold_is_flagged():
    """settle_ledger brings back a fold that ends nowhere; the audit names it."""
    kept = [_atom("k1", "Site walk scheduled")]
    s = _fold(_atom("s_name", "Jordan Sample", art="art_note", atom_type=AtomType.stakeholder),
              "stakeholder_dedup", None)
    atoms, ledger, counts = settle_ledger(kept, [s])
    assert counts["restored"] == 1
    report = audit_ledger(atoms, ledger)
    ex = report["examples"]["unresolved_survivor"]
    assert [e["atom_id"] for e in ex] == ["s_name"]
    assert ex[0]["stage"] == "stakeholder_dedup" and ex[0]["why"].startswith("restored")


def test_copy_whose_canonical_is_missing_is_flagged():
    copy_atom = _atom("k_copy", "Provide a lift", art="art_email", flags=["cross_doc_copy"],
                      value={"duplicate_of": {"atom_id": "atm_x", "canonical_missing": True}})
    report = audit_ledger([copy_atom], [])
    assert _ids(report, "unresolved_survivor") == ["k_copy"]


# (b) merged survivor / lossy survivor -----------------------------------------

def test_clean_sentence_folded_into_a_merged_survivor_is_flagged():
    merged = _atom("k_merged", "All work is performed at the customer's listed sites Scheduling: "
                               "subject to technician availability",
                   loc={"paragraph_index": 83, "sentence_index": 1})
    clean = _fold(_atom("s_clean", "All work is performed at the customer's listed sites.",
                        loc={"paragraph_index": 6}), "semantic_dedup", merged)
    report = audit_ledger([merged], [clean])
    assert report["counts"]["merged_survivor"] == 1
    assert report["examples"]["merged_survivor"][0]["survivor_id"] == "k_merged"
    assert report["counts"]["unresolved_survivor"] == 0


def test_same_text_fold_is_neither_merged_nor_lossy():
    k = _atom("k1", "- Replace the core switch")
    s = _fold(_atom("s1", "Replace the core switch."), "semantic_dedup", k)
    report = audit_ledger([k], [s])
    assert report["findings"] == 0


def test_row_folded_into_a_shorter_record_is_lossy():
    kept = _atom("k_row", "Pat Example | Field Lead", loc={"table_index": 1, "row": 1})
    row = _fold(_atom("s_row", "Pat Example | Field Lead | pat@example.test",
                      loc={"table_index": 1, "row": 1}), "pre_classify_dedup", kept)
    report = audit_ledger([kept], [row])
    assert _ids(report, "lossy_survivor") == ["s_row"]
    assert report["counts"]["partial_list"] == 0  # the row itself still stands


# (c) partial list -------------------------------------------------------------

_PMO = ["Proposal", "Scope", "Provider responsibilities", "Admin Responsibilities"]


def test_one_item_dropped_from_a_docx_list_is_flagged():
    items = ["Draft the schedule", "Assign the crew", "Confirm deliverables", "Close out paperwork"]
    atoms = [_atom(f"k{i}", t, loc={"paragraph_index": 91 + i, "section_path": _PMO}) for i, t in enumerate(items)]
    last = atoms.pop()
    mark_dropped_not_folded(last, "admin task, not a quote line")
    capture_suppressed([last], [], stage="quote_line_head", reason="admin task")
    report = audit_ledger(atoms, [last])
    ex = report["examples"]["partial_list"]
    assert len(ex) == 1
    assert ex[0]["items"] == 4 and ex[0]["suppressed_items"] == 1
    assert ex[0]["suppressed"][0]["atom_id"] == "k3"
    assert ex[0]["suppressed"][0]["kind"] == "drop"
    assert ex[0]["list"]["by"] == "section"
    assert report["counts"]["unresolved_survivor"] == 0


def _quoted_items(prefix: str = "e") -> list[EvidenceAtom]:
    words = ["Network support", "Server support", "Virtualization support", "Patch management",
             "End-user support", "Incident response"]
    val = {"message_index": 10, "quoted": True, "list_item": True, "list_label": "Original request:"}
    return [_atom(f"{prefix}{i}", w, art="art_email", value=dict(val, line=145),
                  loc={"kind": "email_body_line"}) for i, w in enumerate(words)]


def test_quoted_email_list_with_some_items_folded_is_flagged():
    note = [_atom(f"n{i}", w, art="art_note") for i, w in enumerate(["Network support", "Server support",
                                                                    "Incident response"])]
    items = _quoted_items()
    folded = []
    for idx, n in ((0, note[0]), (1, note[1]), (5, note[2])):
        folded.append(_fold(items[idx], "pasted_note_dedup", n))
    # The same line read twice: a second suppressed atom for one item.
    twin = _fold(_atom("e0_twin", "Network support", art="art_email",
                       value={"message_index": 10, "list_item": True, "list_label": "Original request:",
                              "line": 145}), "semantic_dedup", note[0])
    kept = note + [items[i] for i in (2, 3, 4)]
    report = audit_ledger(kept, folded + [twin])
    ex = report["examples"]["partial_list"]
    assert len(ex) == 1
    assert ex[0]["items"] == 6 and ex[0]["suppressed_items"] == 3
    assert sorted(e["atom_id"] for e in ex[0]["suppressed"]) == ["e0", "e1", "e5"]
    assert ex[0]["list"]["by"] in {"list", "mail_line"}  # one source line split into items
    assert report["counts"]["unresolved_survivor"] == 0


def test_list_folded_all_or_none_is_not_flagged():
    note = [_atom(f"n{i}", a.raw_text, art="art_note") for i, a in enumerate(_quoted_items())]
    all_folded = [_fold(a, "pasted_note_dedup", note[i]) for i, a in enumerate(_quoted_items())]
    assert audit_ledger(note, all_folded)["counts"]["partial_list"] == 0
    assert audit_ledger(_quoted_items(), [])["counts"]["partial_list"] == 0


def test_duplicate_item_folded_into_its_own_list_is_not_flagged():
    items = _quoted_items()
    dup = _atom("e_dup", "Network support", art="art_email",
                value={"message_index": 10, "list_item": True, "list_label": "Original request:", "line": 150})
    _fold(dup, "duplicate_atom_collapse", items[0])
    assert audit_ledger(items, [dup])["counts"]["partial_list"] == 0


def test_sentences_of_one_paragraph_are_not_a_list():
    a = _atom("k0", "First sentence here.", loc={"paragraph_index": 4, "sentence_index": 0,
                                                 "lead_in": ["Intro"], "section_path": ["S"]})
    b = _atom("s1", "Second sentence here.", loc={"paragraph_index": 4, "sentence_index": 1,
                                                  "lead_in": ["Intro"], "section_path": ["S"]})
    mark_dropped_not_folded(b, "boilerplate")
    capture_suppressed([b], [], stage="execution_boilerplate_drop", reason="x")
    assert audit_ledger([a], [b])["counts"]["partial_list"] == 0


def test_envelope_rows_are_read_too():
    """The envelope's `suppressed` rows carry stage and a standing survivor or null."""
    kept = [{"id": "k1", "artifact_id": "a", "text": "Name | Title", "structured": {}, "locator": {}}]
    rows = [
        {"id": "s1", "artifact_id": "a", "text": "Name | Title | x@example.test", "stage": "pre_classify_dedup",
         "survivor": {"id": "k1", "text": "Name | Title"}},
        {"id": "s2", "artifact_id": "a", "text": "Lost line", "stage": "semantic_dedup", "survivor": None},
        {"id": "s3", "artifact_id": "a", "text": "Heading", "stage": "section_heading", "survivor": None},
        # The row says it is a deliberate drop: no survivor is owed.
        {"id": "s4", "artifact_id": "a", "text": "Admin line", "stage": "quote_line_head", "survivor": None,
         "kind": "drop", "drop_reason": "admin task"},
    ]
    report = audit_ledger(kept, rows)
    assert _ids(report, "lossy_survivor") == ["s1"]
    assert _ids(report, "unresolved_survivor") == ["s2"]


def test_audit_never_changes_its_inputs_and_caps_examples():
    kept = [_atom("k1", "Keep me")]
    supp = [_fold(_atom(f"s{i}", f"Lost line {i}"), "semantic_dedup", None) for i in range(30)]
    before = (copy.deepcopy([a.model_dump() for a in kept]), copy.deepcopy([a.model_dump() for a in supp]))
    report = audit_ledger(kept, supp, max_examples=5)
    assert report["counts"]["unresolved_survivor"] == 30
    assert len(report["examples"]["unresolved_survivor"]) == 5
    assert ([a.model_dump() for a in kept], [a.model_dump() for a in supp]) == before


# Wiring -----------------------------------------------------------------------

def test_compile_and_envelope_carry_the_audit_and_main_fixtures_have_no_unresolved_survivor(
    demo_project: Path,
) -> None:
    """The synthetic demo deal compiles with zero (a) findings, and the report
    rides on the compile result and the envelope without changing the atoms."""
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope

    result = compile_project(project_dir=demo_project, project_id="demo_project")
    audit = result.ledger_audit
    assert audit["schema"] == "ledger_audit/v1"
    assert audit["kept_atoms"] == len(result.atoms)
    assert audit["suppressed_atoms"] == len(result.suppressed_atoms)
    assert audit["counts"]["unresolved_survivor"] == 0, audit["examples"]["unresolved_survivor"]
    assert audit == audit_ledger(result.atoms, result.suppressed_atoms)
    envelope = build_orbitbrief_envelope(project_dir=demo_project, compile_result=result)
    assert envelope["ledger_audit"] == audit


def test_docx_stress_fixtures_have_no_unresolved_survivor(tmp_path: Path) -> None:
    import scripts.build_docx_stress as stress
    from app.core.compiler import compile_project

    out = tmp_path / "artifacts"
    out.mkdir()
    for build in (stress.da_simple_sow, stress.db_table_in_docx, stress.dc_bullet_list_scope,
                  stress.dd_multi_section_msa, stress.de_signature_block, stress.df_site_roster_docx):
        build(out)
    result = compile_project(project_dir=out, project_id="docx_stress", allow_unverified_receipts=True)
    assert result.ledger_audit["counts"]["unresolved_survivor"] == 0, result.ledger_audit["examples"]


def test_mail_lists_read_as_list_whole_reads_them_and_its_copies_count_as_kept(tmp_path: Path, monkeypatch):
    """A quoted mail list whose items fold onto a note one by one is a
    partial list; once list_whole gives the folded items back as the mail's
    cross-document copies, it is accounted for."""
    import app.core.list_whole as list_whole
    from tests.test_list_folds_whole import ITEMS, LEAD, MAIL_ONLY, _compile

    note = " - ".join([LEAD, *ITEMS])
    mail = " - ".join([LEAD, *ITEMS, MAIL_ONLY])
    (tmp_path / "whole").mkdir()
    (tmp_path / "holes").mkdir()
    r, _mail_id = _compile(tmp_path / "whole", note, mail)
    assert r.ledger_audit["counts"]["partial_list"] == 0
    assert r.ledger_audit["counts"]["unresolved_survivor"] == 0

    monkeypatch.setattr(list_whole, "keep_lists_whole", lambda atoms, supp: (atoms, supp, []))
    r, mail_id = _compile(tmp_path / "holes", note, mail)
    ex = r.ledger_audit["examples"]["partial_list"]
    assert len(ex) == 1 and ex[0]["artifact_id"] == mail_id
    assert ex[0]["list"]["by"] == "mail_line"
    assert ex[0]["suppressed_items"] == len(ITEMS) + 1  # the lead segment folded too
