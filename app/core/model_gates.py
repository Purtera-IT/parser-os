"""One place that answers "is this model path on?".

There were sixteen answers to that question across nine modules, and they did
not agree with each other.

The global switch
-----------------
``SOWSMITH_DISABLE_LLM`` reads like "do not call a text LLM". What it actually
turns off:

    vision_extraction       the drawing/CAD vision pass, via
                            `vision_endpoint_reachable()` returning False --
                            the parser stops READING PICTURES
    typed_atom_classifier   the whole `typed_atom_classification` stage
    multi_entity_llm        entity extraction
    semantic_role           `classify_role`, which decides whether an address
                            is the vendor's own or a job site
    site_llm_verify         site verification
    reranker                retrieval reranking
    plain_rule_compiler     falls back to the deterministic compiler
    ollama_host             reachability probes

So it is not "no LLM", it is **no model paths at all, vision included**. Nobody
reading the old name expects "the parser goes blind to drawings", and that
mattered: every audit tool in ``_tools/`` sets this flag, so a stage that was
SWITCHED OFF looked like a stage that ran and decided nothing.

``SOWSMITH_NO_MODELS`` is the accurate name. ``SOWSMITH_DISABLE_LLM`` is
honoured as a deprecated alias so nothing deployed breaks.

The parsing bug this fixes
--------------------------
Eight of the nine call sites tested the variable with bare truthiness::

    if os.environ.get("SOWSMITH_DISABLE_LLM"):     # "0" is truthy!

while :mod:`app.core.semantic_role` parsed it properly. So
``SOWSMITH_DISABLE_LLM=0`` -- which any reader would take to mean "models on"
-- disabled eight paths and left the vendor-address judgment running: a
split-brain compile that is neither configuration. Everything here parses the
same way.

Vision is three pipelines, not one
----------------------------------
There is no single vision switch, and the polarity is not consistent:

    SOWSMITH_VISION_DISABLE          vision_extraction (drawings, tiled)  OFF when set
    SOWSMITH_PDF_IMAGE_VISION        pdf_image_vision (images in PDFs)    ON when set
    SOWSMITH_LINKED_PICTURE_VISION   linked_picture_vision (email links)  ON when set

Dev currently sets all three to ``1``, which reads as a contradiction and
means: drawings off, PDF images on, linked pictures on. That is why this module
exposes one predicate per pipeline rather than a fourth flag -- a fifth name in
this space would make it worse, not better.

The global switch wins over the DRAWING pass only. It does not reach
``pdf_image_vision`` or ``linked_picture_vision``, so a "no models" compile
still reads images out of PDFs and still fetches linked drawings. That is
surprising, it is the behaviour today, and the two predicates below say so --
changing it breaks 42 tests that exercise those passes against mocked clients
with the switch on, so it is a decision for a person rather than a tidy-up.
"""
from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def models_disabled() -> bool:
    """True when every model path must take its deterministic fallback.

    Reads ``SOWSMITH_NO_MODELS``, or the deprecated ``SOWSMITH_DISABLE_LLM``.
    """
    return _flag("SOWSMITH_NO_MODELS") or _flag("SOWSMITH_DISABLE_LLM")


def llm_disabled() -> bool:
    """Deprecated alias for :func:`models_disabled`, kept for call sites that
    read better this way. Same answer."""
    return models_disabled()


# ---- vision, one predicate per pipeline ------------------------------------


def drawing_vision_enabled() -> bool:
    """The tiled drawing/CAD pass in :mod:`app.core.vision_extraction`.

    Negative flag, preserved: ``SOWSMITH_VISION_DISABLE`` turns it OFF.
    """
    return not models_disabled() and not _flag("SOWSMITH_VISION_DISABLE")


def pdf_image_vision_enabled() -> bool:
    """Images embedded in a PDF. Positive flag: ``SOWSMITH_PDF_IMAGE_VISION``.

    Deliberately does NOT consult :func:`models_disabled`, which is surprising
    and is the current behaviour rather than a considered one. It is load-bearing:
    ``tests/conftest.py`` sets the kill-switch for the whole suite and then
    exercises this pass against a mocked client, so 42 tests in
    ``test_pdf_image_vision.py`` depend on the pass running with the switch on.

    So today a "no models" compile still reads images out of PDFs. Coupling it
    to the switch is a decision with test consequences, not a tidy-up, and it is
    left for a person to make.
    """
    return _flag("SOWSMITH_PDF_IMAGE_VISION")


def linked_picture_vision_enabled() -> bool:
    """A drawing linked from an email, fetched from the deal's own blob.

    Positive flag: ``SOWSMITH_LINKED_PICTURE_VISION``. Same caveat as
    :func:`pdf_image_vision_enabled`: it does not consult the global switch, so
    a "no models" compile still fetches and reads linked drawings.
    """
    return _flag("SOWSMITH_LINKED_PICTURE_VISION")


def any_vision_enabled() -> bool:
    """Whether the compile will read a picture at all, by any route.

    The question a person actually means when they ask "is vision on?", and the
    one the deployed config cannot be read to answer.
    """
    return (
        drawing_vision_enabled()
        or pdf_image_vision_enabled()
        or linked_picture_vision_enabled()
    )


def describe() -> dict[str, bool]:
    """Every gate's current answer, for a log line or an audit tool."""
    return {
        "models_disabled": models_disabled(),
        "drawing_vision": drawing_vision_enabled(),
        "pdf_image_vision": pdf_image_vision_enabled(),
        "linked_picture_vision": linked_picture_vision_enabled(),
        "any_vision": any_vision_enabled(),
    }
