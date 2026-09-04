"""A rate is not a place.

Live 010300: the PSOW's pricing table has a unit column reading "Per Item",
"Per Foot", "Per New Cable". Those phrases entered the authoritative site
catalog, keyed 19 atoms to `site:per_item` — including the clause about 169
locations and the weekly-submission contract term — and were fused in as an
alias of the real Atlanta office, so the site roster listed a billing unit
among its names.

Shape, not vocabulary: a place name never opens with a distributive
preposition. The token boundary matters, which is why Perry Street survives.
"""

from __future__ import annotations

from app.core.entity_extraction import (
    _emit_sites,
    is_site_boilerplate_slug,
    slug_is_unit_of_measure,
)
from app.core.site_detection import _looks_like_site_phrase

UNITS = ["per_item", "per_new_cable", "per_foot", "each_site", "every_run"]
PLACES = [
    "2970_brandywine_rd_ste_200",
    "perry_street_parking_deck",
    "atl_hq_01",
    "permian_basin_yard",
    "peachtree_center",
]


def test_a_unit_of_measure_is_not_a_place() -> None:
    for slug in UNITS:
        assert slug_is_unit_of_measure(slug), slug
        assert is_site_boilerplate_slug(slug), slug


def test_a_place_whose_name_merely_starts_with_those_letters_survives() -> None:
    for slug in PLACES:
        assert not slug_is_unit_of_measure(slug), slug
        assert not is_site_boilerplate_slug(slug), slug


def test_the_authoritative_catalog_refuses_a_unit_phrase() -> None:
    assert _looks_like_site_phrase("Per Item") is False
    assert _looks_like_site_phrase("Per New Cable") is False
    assert _looks_like_site_phrase("Perry Street Parking Deck") is True


def test_the_pricing_rows_that_caused_it_emit_no_site() -> None:
    rows = [
        "48 Hour Cancellation or Turnaway Fee – Per $500.00 Item – Per Item | 1 $500.00",
        "Cable Run Cable Management – Per New Cable $17.00 Run – Per Item | 1 $17.00",
        "Cat6 Cable Plenum – Per Foot – Per Item $1.25 | 1 $1.25",
    ]
    for row in rows:
        assert not {k for k in _emit_sites(row) if k.startswith("site:")}, row


def test_a_real_site_sentence_still_emits_its_site() -> None:
    keys = _emit_sites("Services will be performed at the Perry Street Parking Deck.")
    assert "site:perry_street_parking_deck" in keys


def test_the_llm_cluster_path_screens_its_aliases() -> None:
    """The fusion path took the model's aliases unchecked, which is how a
    pricing unit ended up as an alias of a real office."""
    import inspect

    from app.core import entity_resolution

    src = inspect.getsource(entity_resolution)
    cluster = src.split("LLM SITE-CLUSTER FUSION")[1].split("SEMANTIC SITE FUSION")[0]
    assert cluster.count("is_site_boilerplate_slug") >= 2, "aliases and canonical name are both screened"


def test_a_speaker_repeating_themselves_is_one_claim() -> None:
    """Live 010300: "Send me new Bold's quote to you." and "Send me new bold's
    quote." both reached the queue, because the dedup key is a text truncation
    and the two truncations differ. On the page that is two things to do."""
    from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
    from app.core.semantic_dedup import collapse_repeated_speech

    def turn(i, text, artifact="ff", atype=AtomType.action_item, spoken=True):
        refs = (
            [
                SourceRef(
                    id=f"s{i}",
                    artifact_id=artifact,
                    artifact_type="transcript",
                    filename="call.json",
                    extraction_method="t",
                    parser_version="t",
                    locator={"utterance_index": i} if spoken else {"page": 1, "line": i},
                )
            ]
            if True
            else []
        )
        return EvidenceAtom(
            id=f"a{i}", project_id="p", artifact_id=artifact, atom_type=atype,
            raw_text=text, normalized_text=text.lower(), value={},
            authority_class=AuthorityClass.meeting_note, confidence=0.6,
            review_status=ReviewStatus.needs_review, entity_keys=[],
            parser_version="t", source_refs=refs,
        )

    spoken = [
        turn(1, "Send me new Bold's quote to you."),
        turn(2, "Send me new bold's quote."),
        turn(3, "Ship the switches to the Atlanta office."),
    ]
    out = collapse_repeated_speech(spoken)
    assert [a.raw_text for a in out] == [
        "Send me new Bold's quote to you.",
        "Ship the switches to the Atlanta office.",
    ], "the fuller wording survives, the retelling goes"

    # Two similar DOCUMENT lines are two facts and must both survive.
    written = [
        turn(1, "Provider will install phones at Site A.", artifact="psow", spoken=False),
        turn(2, "Provider will install phones at Site B.", artifact="psow", spoken=False),
    ]
    assert len(collapse_repeated_speech(written)) == 2


def test_a_question_from_a_call_is_never_a_constraint() -> None:
    """Live 010300: "Is the children dentistry done after hours?" was emitted
    twice — once as an open_question and once as a CONSTRAINT — so the brief
    carried a question mark as a rule the crew had to work to."""
    import json
    import tempfile
    from pathlib import Path

    from app.core.schemas import AtomType
    from app.parsers.transcript_parser import TranscriptParser

    payload = {
        "schema": "fireflies.transcript.utterances.v1",
        "id": "01M1KWDX5FJCYZ5BAF5JC8W0QC",
        "title": "call",
        "utterances": [
            {"speaker": "S1", "text": "Is the children dentistry done after hours?", "index": 0},
            {"speaker": "S2", "text": "That has got to be done on Saturday, Sunday only after hours.", "index": 1},
        ],
    }
    path = Path(tempfile.mkdtemp()) / "call.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    atoms = TranscriptParser().parse_artifact("p", "art", path)
    by_text = {}
    for a in atoms:
        by_text.setdefault(a.raw_text, set()).add(a.atom_type)
    question = by_text["Is the children dentistry done after hours?"]
    assert AtomType.open_question in question
    assert AtomType.constraint not in question, "asking whether it is so does not make it so"
    assert AtomType.constraint in by_text["That has got to be done on Saturday, Sunday only after hours."]
