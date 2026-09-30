# -*- coding: utf-8 -*-
"""One answer to "is this model path on?", parsed one way.

`SOWSMITH_DISABLE_LLM` reads like "do not call a text LLM". It turned off every
model path in the compile INCLUDING VISION -- `vision_endpoint_reachable()`
returns False under it, so the parser stops reading pictures. It also switches
off two whole stages (`typed_atom_classification`, and entity extraction via
`multi_entity_llm`) and the vendor-versus-job-site address judgment.

That mattered beyond naming: every audit tool in `_tools/` sets this flag, so a
stage that was SWITCHED OFF looked like a stage that ran and decided nothing.

Two defects fixed here.

**The parse.** Eight of the nine call sites tested the variable with bare
truthiness, where the string "0" is true; `semantic_role` parsed it properly.
So `SOWSMITH_DISABLE_LLM=0` -- which any reader takes to mean "models on" --
disabled eight paths and left the address judgment running. Neither
configuration, silently.

**Vision was three pipelines and only one honoured the switch.**
`pdf_image_vision` and `linked_picture_vision` checked their own flags and
nothing else, so a "no models" run still read images out of PDFs and still
fetched linked drawings. Dev sets both to 1, so dev reads pictures by two
routes while an audit run read none by any route.
"""
from __future__ import annotations

import pytest

from app.core import model_gates as gates


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("SOWSMITH_NO_MODELS", "SOWSMITH_DISABLE_LLM",
                 "SOWSMITH_VISION_DISABLE", "SOWSMITH_PDF_IMAGE_VISION",
                 "SOWSMITH_LINKED_PICTURE_VISION"):
        monkeypatch.delenv(name, raising=False)


def test_the_accurate_name_works(monkeypatch) -> None:
    monkeypatch.setenv("SOWSMITH_NO_MODELS", "1")
    assert gates.models_disabled()


def test_the_old_name_still_works(monkeypatch) -> None:
    """Deployed configs use it; renaming must not break them."""
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    assert gates.models_disabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_zero_means_models_on(monkeypatch, value: str) -> None:
    """The bug: bare truthiness made "0" disable eight paths while the ninth
    read it as enabled."""
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", value)
    assert not gates.models_disabled()


@pytest.mark.parametrize("value", ["1", "true", "YES", "On"])
def test_the_usual_spellings_all_disable(monkeypatch, value: str) -> None:
    monkeypatch.setenv("SOWSMITH_NO_MODELS", value)
    assert gates.models_disabled()


def test_no_models_does_not_stop_every_vision_route(monkeypatch) -> None:
    """The switch reaches the DRAWING pass and nothing else.

    This is the surprise, and it is deliberate rather than tidy: a "no models"
    compile still reads images out of PDFs and still fetches linked drawings.
    Coupling those two to the switch breaks 42 tests in
    `test_pdf_image_vision.py`, which set the switch for the whole suite (see
    tests/conftest.py) and then exercise the pass against a mocked client -- so
    their independence is load-bearing, not an oversight. Changing it is a
    decision for a person.

    The test asserts what IS true, so that whoever makes that decision sees
    this fail and knows they are changing something real."""
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setenv("SOWSMITH_LINKED_PICTURE_VISION", "1")
    monkeypatch.setenv("SOWSMITH_NO_MODELS", "1")

    assert not gates.drawing_vision_enabled(), "the drawing pass DOES honour it"
    assert gates.pdf_image_vision_enabled(), "PDF images do not honour it"
    assert gates.linked_picture_vision_enabled(), "linked pictures do not honour it"
    assert gates.any_vision_enabled(), (
        "a 'no models' compile still reads pictures -- if this ever fails, "
        "somebody coupled the passes to the switch, which is a real change"
    )


def test_each_pipeline_keeps_its_own_polarity(monkeypatch) -> None:
    """Drawings are OFF when their flag is set; the other two are ON when
    theirs are. Inconsistent, deliberately preserved -- dev depends on it."""
    assert gates.drawing_vision_enabled()
    monkeypatch.setenv("SOWSMITH_VISION_DISABLE", "1")
    assert not gates.drawing_vision_enabled()

    assert not gates.pdf_image_vision_enabled()
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    assert gates.pdf_image_vision_enabled()


def test_dev_config_reads_as_a_contradiction(monkeypatch) -> None:
    """Exactly what the dev worker sets today. Someone reading it concludes
    vision is off; two of three pipelines are on."""
    monkeypatch.setenv("SOWSMITH_VISION_DISABLE", "1")
    monkeypatch.setenv("SOWSMITH_PDF_IMAGE_VISION", "1")
    monkeypatch.setenv("SOWSMITH_LINKED_PICTURE_VISION", "1")
    assert not gates.drawing_vision_enabled()
    assert gates.pdf_image_vision_enabled()
    assert gates.linked_picture_vision_enabled()
    assert gates.any_vision_enabled(), "dev DOES read pictures, by two routes"


def test_the_pipelines_route_through_the_gate() -> None:
    """The predicates each module exposes must be the shared ones, or the next
    edit re-introduces a fourth opinion."""
    from app.core import linked_picture_vision, pdf_image_vision, semantic_role
    assert linked_picture_vision.enabled() is gates.linked_picture_vision_enabled()
    assert pdf_image_vision.enabled() is gates.pdf_image_vision_enabled()
    assert semantic_role._llm_disabled() is gates.models_disabled()


def test_describe_answers_the_question_a_person_asks() -> None:
    d = gates.describe()
    assert set(d) == {"models_disabled", "drawing_vision", "pdf_image_vision",
                      "linked_picture_vision", "any_vision"}
