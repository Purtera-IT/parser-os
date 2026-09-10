"""An image we described keeps its pixels, so the thing can be shown.

A crop's bytes survived only when the image was DISPUTED -- persisted so a
human grading the queue could see what the gate skipped. An image the pipeline
read SUCCESSFULLY had its description kept and its pixels thrown away, so
nothing downstream could show the thing the description is about.

Deal 51318992 has two described diagrams and 153 blobs, not one of them an
image. A question about a component is far easier to answer next to the
component.
"""

from __future__ import annotations

import app.core.pdf_image_vision as piv


class _Marker:
    id = "atm_marker"
    project_id = "proj"
    artifact_id = "art"
    parser_version = "pdf_image_vision_v1"
    value: dict = {}


def _head(atoms):
    return next(
        (a for a in atoms if (a.value or {}).get("fact_kind") == "image_description"),
        None,
    )


def _facts(atoms):
    return [a for a in atoms if str((a.value or {}).get("fact_kind", "")).startswith("image_fact:")]


def _describe(monkeypatch, *, thumb=("data:image/jpeg;base64,AAAA", None), facts=2):
    """Run _describe with the VLM and page context stubbed out."""
    payload = {
        "description": "A diagram showing the UTM device and its bracket.",
        "facts": [{"kind": "component", "text": f"Part {i}"} for i in range(facts)],
    }
    import json as _json

    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: _json.dumps(payload))
    monkeypatch.setattr(piv, "_context_guard", lambda *a, **k: True)
    monkeypatch.setattr(piv, "_maybe_described_thumb", lambda crop: thumb)
    return piv._describe(
        marker=_Marker(), pdf_name="guide.pdf", page_index=0,
        region_ref="page0/image7", crop=b"\x89PNG-bytes", envelope="page words here",
        image_kind="diagram", caption="", grounding="page words here", guard_min=0.0,
    )


def test_the_description_carries_the_image(monkeypatch):
    atoms = _describe(monkeypatch)
    assert _head(atoms).value["thumb"] == "data:image/jpeg;base64,AAAA"


def test_one_thumbnail_per_image_never_per_fact(monkeypatch):
    """A diagram yielding eight facts must not embed itself eight times."""
    atoms = _describe(monkeypatch, facts=8)
    assert len(_facts(atoms)) == 8
    assert sum(1 for a in atoms if "thumb" in (a.value or {})) == 1


def test_a_budget_refusal_is_recorded_not_swallowed(monkeypatch):
    """Never a silent zero -- the receipt says why there are no pixels."""
    atoms = _describe(monkeypatch, thumb=(None, "budget_exhausted"))
    head = _head(atoms)
    assert "thumb" not in head.value
    assert head.value["thumb_error"] == "budget_exhausted"


def test_no_thumbnail_and_no_error_leaves_the_atom_clean(monkeypatch):
    atoms = _describe(monkeypatch, thumb=(None, None))
    head = _head(atoms)
    assert "thumb" not in head.value and "thumb_error" not in head.value


def test_the_description_itself_is_unchanged(monkeypatch):
    atoms = _describe(monkeypatch)
    assert _head(atoms).raw_text == "A diagram showing the UTM device and its bracket."


def test_descriptions_never_spend_the_skip_receipt_budget(monkeypatch):
    """Separate allowances: a picture-heavy document must not leave a disputed
    image without the thumbnail the person grading it needs."""
    piv._thumb_budget["used"] = 0
    piv._described_thumb_budget["used"] = 0
    monkeypatch.delenv("SOWSMITH_PDF_IMAGE_DESCRIBED_THUMB_MAX", raising=False)
    import json as _json

    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: _json.dumps(
        {"description": "A diagram.", "facts": []}))
    monkeypatch.setattr(piv, "_context_guard", lambda *a, **k: True)
    piv._describe(
        marker=_Marker(), pdf_name="g.pdf", page_index=0, region_ref="page0/image7",
        crop=b"\x89PNG", envelope="ctx", image_kind="diagram", caption="",
        grounding="ctx", guard_min=0.0,
    )
    assert piv._thumb_budget["used"] == 0
