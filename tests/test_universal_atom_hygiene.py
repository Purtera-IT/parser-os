"""Universal atom hygiene P1–P12."""
from __future__ import annotations

from types import SimpleNamespace

from app.core.universal_atom_hygiene import (
    apply_universal_atom_hygiene,
    drop_resolved_vision_stubs,
    is_email_or_marketing_chrome,
    is_shred_atom,
    is_speculative_risk_text,
    is_vision_stub,
    unwrap_vision_text,
)


def test_unwrap_json_and_mismatched_wrappers():
    assert unwrap_vision_text('["The image shows a display."]').startswith("The image")
    assert "Behind TV" in unwrap_vision_text("[\"The image shows 'Behind TV 1'.\"]")
    plain = unwrap_vision_text("Cables run across the floor, approximately 10 feet.")
    assert plain.startswith("Cables run")


def test_stub_and_chrome_and_shred():
    assert is_vision_stub("[Image extracted - awaiting OCR / vision] page8/image64")
    assert is_email_or_marketing_chrome("Quotes in 24–48 hours")
    assert is_email_or_marketing_chrome("Powered by Mimecast")
    assert is_email_or_marketing_chrome("From: Patrick Kelly <patrick@purtera-it.com>")
    assert is_shred_atom("SS")
    assert is_shred_atom("&nbsp;")
    assert not is_email_or_marketing_chrome(
        "Site survey and assessment has already been done by SHI internal team."
    )


def test_speculative_risk_vs_grounded():
    assert is_speculative_risk_text(
        "The floor is carpeted, which may pose a slight trip hazard if cables are not properly managed."
    )
    assert not is_speculative_risk_text(
        "Annotation: replication cable should be moved behind the wall."
    )


def test_drop_stubs_even_when_vision_succeeds_same_region():
    stub = SimpleNamespace(
        raw_text="[Image extracted - awaiting OCR / vision] page9/image67",
        text="",
        atom_type="deal_metadata",
        value={"region_ref": "page9/image67"},
        source_refs=[],
        confidence=0.5,
    )
    fact = SimpleNamespace(
        raw_text="HDMI over Ethernet adapter retained with Yealink system.",
        text="",
        atom_type="scope_item",
        value={
            "via": "pdf_image_vision",
            "fact_kind": "image_fact:cable",
            "region_ref": "page9/image67",
        },
        source_refs=[],
        confidence=0.7,
    )
    kept, dropped = drop_resolved_vision_stubs([stub, fact])
    assert fact in kept
    assert stub in dropped
    assert len(kept) == 1


def test_apply_universal_hygiene_end_to_end():
    atoms = [
        SimpleNamespace(
            raw_text='["The image depicts a conference room setup with a long table."]',
            normalized_text="",
            atom_type="deal_metadata",
            value={
                "via": "pdf_image_vision",
                "fact_kind": "image_description",
                "region_ref": "page5/image56",
            },
            source_refs=[],
            confidence=0.6,
        ),
        SimpleNamespace(
            raw_text="Cables run across the floor, approximately 10 feet to a network receptacle.",
            normalized_text="",
            atom_type="deal_metadata",
            value={
                "via": "pdf_image_vision",
                "fact_kind": "image_fact:cable",
                "region_ref": "page6/image59",
            },
            source_refs=[],
            confidence=0.7,
        ),
        SimpleNamespace(
            raw_text="Quotes in 24–48 hours",
            normalized_text="",
            atom_type="scope_item",
            value={},
            source_refs=[],
            confidence=0.5,
        ),
        SimpleNamespace(
            raw_text="The floor is carpeted, which may pose a slight trip hazard if cables are not properly managed.",
            normalized_text="",
            atom_type="risk",
            value={"via": "pdf_image_vision", "fact_kind": "image_fact:risk"},
            source_refs=[],
            confidence=0.5,
        ),
        SimpleNamespace(
            raw_text="[Image extracted - awaiting OCR / vision] page0/image14",
            normalized_text="",
            atom_type="deal_metadata",
            value={"region_ref": "page0/image14"},
            source_refs=[],
            confidence=0.4,
        ),
        SimpleNamespace(
            raw_text="SS",
            normalized_text="",
            atom_type="deal_metadata",
            value={},
            source_refs=[],
            confidence=0.3,
        ),
    ]
    kept, dropped, stats = apply_universal_atom_hygiene(atoms)
    texts = [a.raw_text for a in kept]
    assert any("10 feet" in t for t in texts)
    assert any("conference room" in t.lower() for t in texts)  # unwrapped blurb kept
    assert not any("Quotes in 24" in t for t in texts)
    assert not any("may pose" in t for t in texts)
    assert not any("awaiting OCR" in t for t in texts)
    assert not any(t == "SS" for t in texts)
    assert stats["dropped_stubs"] >= 1
    assert stats["dropped_chrome"] >= 1
    assert stats["dropped_spec_risk"] >= 1
    # P12: cable fact retagged to scope_item
    cable = next(a for a in kept if "10 feet" in a.raw_text)
    at = getattr(cable.atom_type, "value", cable.atom_type)
    assert str(at) == "scope_item"
