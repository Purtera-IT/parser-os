"""Declared-scope reconciliation — pins for the Marion County lesson.

Four fixes pinned together:
1. word-numbers are dress ("ten locations" == "10 locations");
2. a quoted-email-only declaration still raises the question, flagged
   unconfirmed;
3. declared > found  ->  exactly one site_count_gap open_question;
4. per-site SOWs referenced with none in the intake -> exactly one
   referenced_sows_missing open_question.

Plus the negatives that keep the pass quiet on healthy deals — the PM
question-quality audit taught us a bad question is worse than none.
"""

import pytest

from app.core.declared_scope import declared_scope_questions
from app.core.entity_extraction import _emit_quantity_keys, parse_quantity_spans
from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(atom_id, atom_type, authority, text, *, entity_keys=(),
          filename="fixture.txt", artifact_id="art_1"):
    return EvidenceAtom(
        id=atom_id, project_id="proj_1", artifact_id=artifact_id,
        atom_type=atom_type, raw_text=text, normalized_text=text.lower(),
        value={"text": text}, entity_keys=list(entity_keys),
        source_refs=[SourceRef(
            id=stable_id("src", atom_id), artifact_id=artifact_id,
            artifact_type=ArtifactType.txt, filename=filename,
            locator={}, extraction_method="test", parser_version="test")],
        authority_class=authority, confidence=0.9,
        review_status=ReviewStatus.auto_accepted, review_flags=[],
        parser_version="test",
    )


def _site(atom_id, slug):
    return _atom(atom_id, AtomType.physical_site,
                 AuthorityClass.approved_site_roster,
                 f"Site {slug}", entity_keys=[f"site:{slug}"])


# ── fix 1: word-numbers are dress ───────────────────────────────────────

class TestWordNumbers:
    def test_ten_equals_10(self):
        word = _emit_quantity_keys({}, "SOW's for each of the ten locations.")
        digit = _emit_quantity_keys({}, "SOW's for each of the 10 locations.")
        assert word == digit == {"quantity:10"}

    @pytest.mark.parametrize("text,want", [
        ("five sites in scope", {"quantity:5"}),
        ("twelve schools total", {"quantity:12"}),
        ("twenty timeclocks to install", {"quantity:20"}),
    ])
    def test_countable_nouns(self, text, want):
        assert _emit_quantity_keys({}, text) == want

    def test_non_countable_context_stays_silent(self):
        assert _emit_quantity_keys({}, "a ten minute call about scope") == set()
        assert _emit_quantity_keys({}, "one of the best options") == set()

    def test_trace_mirror_agrees(self):
        spans = parse_quantity_spans("each of the ten locations")
        assert any(s["quantity"] == 10 for s in spans)


# ── fixes 2+3: declared vs found ────────────────────────────────────────

_QUOTED_DECL = "I have created SOW's for each of the ten locations."


class TestSiteCountGap:
    def test_marion_county_shape_raises_the_question(self):
        # The founding fixture: declaration only in a quoted email,
        # zero physical sites found.
        atoms = [
            _atom("d1", AtomType.deal_metadata, AuthorityClass.quoted_old_email,
                  _QUOTED_DECL),
            _atom("s1", AtomType.scope_item,
                  AuthorityClass.customer_current_authored,
                  "Please send me a list of locations and quantity of timeclocks."),
        ]
        qs = declared_scope_questions(project_id="proj_1", atoms=atoms)
        gap = [q for q in qs if q.value["declared_scope"]["kind"] == "site_count_gap"]
        assert len(gap) == 1
        q = gap[0]
        assert q.atom_type == AtomType.open_question
        assert q.review_status == ReviewStatus.needs_review
        assert "declare 10 locations" in q.raw_text
        assert "0 identified" in q.raw_text
        # fix 2: the quoted-only source is flagged unconfirmed
        assert "quoted email" in q.raw_text
        assert q.value["declared_scope"]["declaration_confirmed"] is False
        assert "quantity:10" in q.entity_keys

    def test_confirmed_authority_drops_the_qualifier(self):
        atoms = [_atom("d1", AtomType.scope_item,
                       AuthorityClass.customer_current_authored,
                       "The rollout covers 10 locations in Marion County.")]
        qs = declared_scope_questions(project_id="proj_1", atoms=atoms)
        gap = [q for q in qs if q.value["declared_scope"]["kind"] == "site_count_gap"]
        assert len(gap) == 1
        assert "quoted email" not in gap[0].raw_text
        assert gap[0].value["declared_scope"]["declaration_confirmed"] is True

    def test_found_meeting_declared_stays_silent(self):
        atoms = [_atom("d1", AtomType.scope_item,
                       AuthorityClass.customer_current_authored,
                       "The rollout covers three locations.")]
        atoms += [_site(f"p{i}", f"school_{i}") for i in range(3)]
        qs = declared_scope_questions(project_id="proj_1", atoms=atoms)
        assert [q for q in qs
                if q.value["declared_scope"]["kind"] == "site_count_gap"] == []

    def test_no_declaration_no_question(self):
        atoms = [_atom("s1", AtomType.scope_item,
                       AuthorityClass.customer_current_authored,
                       "Install the equipment per the attached runbook.")]
        assert declared_scope_questions(project_id="proj_1", atoms=atoms) == []

    def test_single_site_phrase_is_not_a_declaration(self):
        # "one location" (n < 2) is prose, not a countable scope claim.
        atoms = [_atom("s1", AtomType.scope_item,
                       AuthorityClass.customer_current_authored,
                       "Work happens at one location downtown.")]
        assert declared_scope_questions(project_id="proj_1", atoms=atoms) == []

    def test_word_and_digit_declarations_agree(self):
        base = _atom("d1", AtomType.deal_metadata,
                     AuthorityClass.quoted_old_email, _QUOTED_DECL)
        digit = _atom("d1", AtomType.deal_metadata,
                      AuthorityClass.quoted_old_email,
                      "I have created SOW's for each of the 10 locations.")
        qa = declared_scope_questions(project_id="proj_1", atoms=[base])
        qb = declared_scope_questions(project_id="proj_1", atoms=[digit])
        assert (qa[0].value["declared_scope"]["declared_count"]
                == qb[0].value["declared_scope"]["declared_count"] == 10)

    def test_deterministic_ids(self):
        atoms = [_atom("d1", AtomType.deal_metadata,
                       AuthorityClass.quoted_old_email, _QUOTED_DECL)]
        a = declared_scope_questions(project_id="proj_1", atoms=atoms)
        b = declared_scope_questions(project_id="proj_1", atoms=list(atoms))
        assert [q.id for q in a] == [q.id for q in b]


# ── fix 4: referenced per-site documents missing ────────────────────────

class TestReferencedSows:
    def test_referenced_but_absent_raises_one_question(self):
        atoms = [
            _atom("d1", AtomType.deal_metadata, AuthorityClass.quoted_old_email,
                  _QUOTED_DECL),
            _atom("d2", AtomType.deal_metadata, AuthorityClass.quoted_old_email,
                  "A statement of work per site was prepared."),
        ]
        qs = declared_scope_questions(project_id="proj_1", atoms=atoms)
        missing = [q for q in qs
                   if q.value["declared_scope"]["kind"] == "referenced_sows_missing"]
        assert len(missing) == 1          # one question, not one per mention
        assert "no SOW file" in missing[0].raw_text

    def test_present_sow_file_stays_silent(self):
        atoms = [
            _atom("d1", AtomType.deal_metadata, AuthorityClass.quoted_old_email,
                  _QUOTED_DECL, filename="Charles Street SOW v2.pdf"),
        ]
        qs = declared_scope_questions(project_id="proj_1", atoms=atoms)
        assert [q for q in qs
                if q.value["declared_scope"]["kind"] == "referenced_sows_missing"] == []
