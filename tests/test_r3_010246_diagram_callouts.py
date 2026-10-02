"""A callout read off a diagram is a reject-able label, not site_infrastructure.

Deal 010246 (re-run on #268): the install drawing's callouts -- "Solar
Panel", "Cell Modem", "Junction Box" -- came back typed
``site_infrastructure``. A callout names a part of a picture and states
nothing about the site, so after typing it becomes deal_metadata, chatter,
rejected_by ``diagram_label`` (the model's guess kept as an alt type). A
short line that does state an infrastructure fact keeps its type, and prose
or text not from a picture is untouched.
"""
from __future__ import annotations

import pytest

from app.core.diagram_labels import retype_diagram_labels
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _atom(text: str, *, value=None, locator=None, method="pdf_text", art=ArtifactType.pdf,
          typ=AtomType.site_infrastructure) -> EvidenceAtom:
    return EvidenceAtom(
        id=f"x{abs(hash((text, method, str(locator), str(value))))}", project_id="p", artifact_id="a",
        atom_type=typ, raw_text=text, normalized_text=text.lower(), value=dict(value or {}),
        entity_keys=["site:pole_3"],
        source_refs=[SourceRef(id="s", artifact_id="a", artifact_type=art, filename="d.pdf",
                               locator=dict(locator or {"page": 2}), extraction_method=method,
                               parser_version="t")],
        authority_class=AuthorityClass.contractual_scope, confidence=0.6,
        review_status=ReviewStatus.needs_review, review_flags=[], parser_version="t",
    )


def _t(a) -> str:
    return str(getattr(a.atom_type, "value", a.atom_type))


@pytest.mark.parametrize("text", ["Solar Panel", "Cell Modem", "Junction Box", "Pole 3"])
@pytest.mark.parametrize("where", ["vision", "drawing_page"])
def test_picture_callout_becomes_a_diagram_label(text: str, where: str) -> None:
    if where == "vision":
        a = _atom(text, value={"via": "pdf_image_vision", "image_kind": "diagram"}, method="pdf_image_vision")
    else:
        a = _atom(text, locator={"page": 2, "on_drawing": True})
    assert retype_diagram_labels([a]) == 1
    assert _t(a) == "deal_metadata"
    assert "chatter" in a.review_flags and "diagram_label" in a.review_flags
    assert a.value["chatter"] is True and a.value["rejected_by"] == "diagram_label"
    assert a.value["alt_atom_types"] == ["site_infrastructure"]
    assert not [k for k in a.entity_keys if k.startswith("site:")]


@pytest.mark.parametrize("text", ["MDF in room 112", "100 Mbps circuit", "42U rack", "IDF-2"])
def test_infrastructure_fact_on_a_drawing_keeps_its_type(text: str) -> None:
    a = _atom(text, locator={"page": 2, "on_drawing": True})
    assert retype_diagram_labels([a]) == 0
    assert _t(a) == "site_infrastructure"


def test_prose_and_text_layer_labels_are_untouched() -> None:
    prose = _atom("The cell modem is mounted inside the junction box on pole 3.",
                  value={"via": "pdf_image_vision"}, method="pdf_image_vision")
    plain = _atom("Solar Panel")  # a body-text line of a document, not a picture
    other = _atom("Solar Panel", value={"via": "pdf_image_vision"}, typ=AtomType.scope_item)
    assert retype_diagram_labels([prose, plain, other]) == 0
    assert _t(prose) == "site_infrastructure" and _t(plain) == "site_infrastructure"
    assert _t(other) == "scope_item"
