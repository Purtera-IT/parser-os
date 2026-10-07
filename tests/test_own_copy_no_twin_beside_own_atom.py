"""A document never shows one line twice: its own atom and a minted copy.

Live 000132: three forwarded emails quote the same six-line city list. One
email's own city atoms were folded away (a pasted-note fold) and the sweep
minted a fresh cross-document copy for each line from the survivor in another
email -- the labelling walk listed the folded atom's row AND the copy, every
city twice. In a third email the site dedup kept each city for its words
(retyped deal_metadata, its refs merged onto the surviving site); the sweep
held that it lacked a physical_site of the line and minted one beside it.

Synthetic shapes only.
"""
from __future__ import annotations

import pytest

from app.core.cross_doc_copies import COPY_FLAG, ensure_own_copies, is_cross_doc_copy
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _ref(artifact, line, msg):
    return SourceRef(id=f"src_{artifact}_{line}", artifact_id=artifact, artifact_type=ArtifactType.email,
                     filename=f"{artifact}.eml", locator={"message_index": msg, "line_start": line,
                                                          "line_end": line, "quoted": True},
                     extraction_method="test", parser_version="t")


def _atom(aid, text, artifact, line, msg, *, atype=AtomType.physical_site, flags=()):
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id=artifact, atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value={"quoted": True, "message_index": msg},
        entity_keys=[], source_refs=[_ref(artifact, line, msg)],
        authority_class=AuthorityClass.machine_extractor, confidence=0.7,
        review_flags=list(flags), review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


CITY = "Springfield, ZZ"


def _lines(_aid):
    return ["Sites", CITY, "Shelbyville, ZZ"]


def test_folded_own_atom_comes_back_instead_of_a_minted_twin():
    survivor = _atom("atm_keep", CITY, "mail_b", 40, 2)
    folded = _atom("atm_mine", CITY, "mail_a", 30, 1, flags=["suppressed:pasted_note_dedup"])
    survivor.source_refs.append(folded.source_refs[0].model_copy(deep=True))

    out = ensure_own_copies([survivor], [], doc_lines=_lines, dropped=[folded])

    assert [a.id for a in out] == ["atm_mine"]          # its own atom, no atm_copy_
    assert is_cross_doc_copy(folded)
    assert not any(f.startswith("suppressed:") for f in folded.review_flags)
    assert folded.value["duplicate_of"]["atom_id"] == "atm_keep"
    assert "mail_a" in survivor.value["also_in_documents"]


def test_own_atom_on_another_line_still_gets_a_minted_copy():
    survivor = _atom("atm_keep", CITY, "mail_b", 40, 2)
    elsewhere = _atom("atm_other", CITY, "mail_a", 99, 1, flags=["suppressed:pasted_note_dedup"])
    survivor.source_refs.append(_ref("mail_a", 30, 1))

    out = ensure_own_copies([survivor], [], doc_lines=_lines, dropped=[elsewhere])

    assert len(out) == 1 and out[0].id.startswith("atm_copy")
    assert out[0].source_refs[0].locator["line_start"] == 30
    assert COPY_FLAG not in elsewhere.review_flags


def test_site_kept_for_its_words_is_not_given_a_site_copy_beside_it():
    survivor = _atom("atm_keep", CITY, "mail_b", 40, 2)
    kept_words = _atom("atm_words", CITY, "mail_c", 50, 3, atype=AtomType.deal_metadata,
                       flags=["city_site_list", "kept_over_site_dedup"])
    survivor.source_refs.append(kept_words.source_refs[0].model_copy(deep=True))

    out = ensure_own_copies([survivor, kept_words], [], doc_lines=_lines, dropped=[])

    assert out == []


def test_other_types_of_the_same_line_still_get_their_own_copy():
    # Without the site-dedup flag a different type is a different atom of the
    # line (a Deal Kit row is a raw_table_row AND a bom_line): unchanged.
    survivor = _atom("atm_keep", CITY, "mail_b", 40, 2)
    other = _atom("atm_meta", CITY, "mail_c", 50, 3, atype=AtomType.deal_metadata)
    survivor.source_refs.append(other.source_refs[0].model_copy(deep=True))

    out = ensure_own_copies([survivor, other], [], doc_lines=_lines, dropped=[])

    assert len(out) == 1 and out[0].id.startswith("atm_copy")


def test_survivor_retyped_later_still_restores_the_own_atom_of_its_line():
    survivor = _atom("atm_keep", "Incident response", "mail_b", 40, 2, atype=AtomType.service_line)
    folded = _atom("atm_mine", "Incident response", "mail_a", 30, 1, atype=AtomType.scope_item,
                   flags=["suppressed:pasted_note_dedup"])
    survivor.source_refs.append(folded.source_refs[0].model_copy(deep=True))

    out = ensure_own_copies([survivor], [], doc_lines=lambda _a: ["Incident response"], dropped=[folded])

    assert [a.id for a in out] == ["atm_mine"]
    assert folded.atom_type == AtomType.scope_item
