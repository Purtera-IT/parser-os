"""Graph-builder + packetizer invariance — same content, any dress, same graph.

Continuation of the seam audit (test_head_invariance.py): 7 bleeds found on
2026-08-25, all fixed here. The passing invariants worth keeping honest are
pinned too — above all, that the edge set does not depend on atom input order.
"""

import random

import pytest

from app.core.graph_builder import (
    _canonical_material_key,
    _collapse_duplicate_row_nodes,
    _looks_like_email_header,
    _noun_anchored_quantity,
    _quantity_value,
    build_edges,
)
from app.core.ids import stable_id
from app.core.packetizer import (
    _EXPLICIT_EXCLUSION_RE,
    _PROCUREMENT_STATUS_RE,
    _SITE_ACCESS_STRONG_RE,
    _TIME_RANGE_ACCESS_RE,
    _gov_sort_key_cert,
)
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(atom_id, atom_type, authority, entity_keys, *, quantity=None,
          text="text", value_extra=None, artifact_id="art_1"):
    value = {"text": text}
    if quantity is not None:
        value["quantity"] = quantity
    if value_extra:
        value.update(value_extra)
    return EvidenceAtom(
        id=atom_id, project_id="proj_1", artifact_id=artifact_id,
        atom_type=atom_type, raw_text=text, normalized_text=text.lower(),
        value=value, entity_keys=entity_keys,
        source_refs=[SourceRef(
            id=stable_id("src", atom_id), artifact_id=artifact_id,
            artifact_type=ArtifactType.txt, filename="fixture.txt",
            locator={}, extraction_method="test", parser_version="test")],
        authority_class=authority, confidence=0.9,
        review_status=ReviewStatus.auto_accepted, review_flags=[],
        parser_version="test",
    )


# ── graph: number formatting is dress ───────────────────────────────────────

class TestQuantityFormatting:
    def test_thousands_separator_binds_like_plain(self):
        # FIXED BLEED: \b\d{1,5}\b split "1,000" into 1 and 000, binding nothing.
        plain = _noun_anchored_quantity("Install 1000 access points.", "access_point")
        comma = _noun_anchored_quantity("Install 1,000 access points.", "access_point")
        assert comma == plain == 1000

    def test_casing_is_dress(self):
        assert (_noun_anchored_quantity("install 50 access points.", "access_point")
                == _noun_anchored_quantity("INSTALL 50 ACCESS POINTS.", "access_point"))

    def test_string_quantity_with_comma_reconciles(self):
        # FIXED BLEED: value.quantity "1,000" parsed to None -> silently
        # excluded from reconciliation while 1000 participated.
        a = _atom("a1", AtomType.quantity, AuthorityClass.approved_site_roster,
                  [], quantity="1,000")
        b = _atom("a2", AtomType.quantity, AuthorityClass.approved_site_roster,
                  [], quantity=1000)
        assert _quantity_value(a) == _quantity_value(b) == 1000.0

    def test_material_key_survives_dressings(self):
        def key(ni):
            return _canonical_material_key(_atom(
                "m", AtomType.quantity, AuthorityClass.approved_site_roster,
                [], quantity=5, value_extra={"normalized_item": ni}))
        assert len({key(d) for d in ["Cat6A", "Cat 6A", "CAT6A patch", "cat-6a"]}) == 1


# ── graph: structural invariants that held and must keep holding ────────────

class TestGraphDeterminism:
    def _fixture(self):
        return [
            _atom("s1", AtomType.scope_item, AuthorityClass.customer_current_authored,
                  ["device:access_point", "site:hq"],
                  text="Install 50 access points at HQ", artifact_id="art_A"),
            _atom("q2", AtomType.quantity, AuthorityClass.vendor_quote,
                  ["device:access_point", "quantity:40"], quantity=40,
                  text="Access points qty 40", artifact_id="art_B"),
            _atom("x1", AtomType.exclusion, AuthorityClass.customer_current_authored,
                  ["device:access_point"],
                  text="Roof-mounted access points are excluded", artifact_id="art_A"),
            _atom("c1", AtomType.constraint, AuthorityClass.meeting_note,
                  ["site:hq"], text="Work at HQ only after hours", artifact_id="art_C"),
            _atom("q3", AtomType.quantity, AuthorityClass.approved_site_roster,
                  ["device:access_point", "quantity:50"], quantity=50,
                  text="50 access points", artifact_id="art_C"),
        ]

    def test_edge_set_is_input_order_invariant(self):
        atoms = self._fixture()

        def sig(edges):
            return sorted((str(e.edge_type), e.from_atom_id, e.to_atom_id)
                          for e in edges)

        base = sig(build_edges("proj_1", list(atoms), []))
        assert base, "fixture must produce edges or this test is vacuous"
        for seed in (1, 7, 42):
            shuffled = list(atoms)
            random.Random(seed).shuffle(shuffled)
            assert sig(build_edges("proj_1", shuffled, [])) == base

    def test_row_collapse_winner_is_order_independent(self):
        li = _atom("v1", AtomType.vendor_line_item, AuthorityClass.vendor_quote, [],
                   quantity=68, value_extra={"source_row_key": "q.xlsx:Quote:row_4"})
        qt = _atom("q1", AtomType.quantity, AuthorityClass.vendor_quote, [],
                   quantity=68, value_extra={"source_row_key": "q.xlsx:Quote:row_4"})
        assert ([a.id for a in _collapse_duplicate_row_nodes([li, qt])]
                == [a.id for a in _collapse_duplicate_row_nodes([qt, li])]
                == ["v1"])

    @pytest.mark.parametrize("dress,want", [
        ("From: bob@x.com", True), ("FROM: bob@x.com", True),
        ("  From : bob@x.com", True), ("from bob@x.com", False),
    ])
    def test_email_header_gate_dress(self, dress, want):
        assert _looks_like_email_header(dress) is want


# ── packetizer: gate regexes see through layout ─────────────────────────────

class TestPacketizerGateDress:
    @pytest.mark.parametrize("dress", [
        "Roof work is not included in this quote",
        "Roof work is NOT INCLUDED in this quote",
        "Roof work is not  included in this quote",   # double space
        "Roof work is not\nincluded in this quote",   # FIXED BLEED: PDF wrap
        "This item is not in\nscope for phase 1",
        "Cabling is out of\nscope",
        "Patching by\nothers",
    ])
    def test_explicit_exclusion_sees_through_whitespace(self, dress):
        assert _EXPLICIT_EXCLUSION_RE.search(dress)

    @pytest.mark.parametrize("dress", [
        "Customer Pending", "CUSTOMER PENDING", "post-closeout",
        "post closeout", "do not order yet", "Awaiting  PO",
    ])
    def test_procurement_status_dress(self, dress):
        assert _PROCUREMENT_STATUS_RE.search(dress)

    @pytest.mark.parametrize("dress", [
        "after-hours work required", "after hours work required",
        "AFTER HOURS work required", "after\nhours work required",
        "SCISSOR  LIFT needed", "escort required at all times",
    ])
    def test_site_access_signal_dress(self, dress):
        assert _SITE_ACCESS_STRONG_RE.search(dress)

    @pytest.mark.parametrize("dress", [
        "7am-4pm", "7am - 4pm", "7am–4pm", "7:00 am – 4:00 pm", "7AM-4PM",
    ])
    def test_time_range_dash_dress(self, dress):
        assert _TIME_RANGE_ACCESS_RE.search(dress)

    def test_governor_sort_tie_broken_by_id(self):
        a = _atom("aaa", AtomType.requirement, AuthorityClass.meeting_note, [])
        b = _atom("bbb", AtomType.requirement, AuthorityClass.meeting_note, [])
        assert ([x.id for x in sorted([a, b], key=_gov_sort_key_cert)]
                == [x.id for x in sorted([b, a], key=_gov_sort_key_cert)])
