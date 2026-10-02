"""Generic words in the device vocabulary only tag when used as devices.

A construction-camera install (cameras on poles, breakers, power lines,
electrical cabinets) came back with ``device:rack`` on every "cabinet" and a
school Chromebook rollout picked up similar noise. These pin the false-
positive classes and, beside each, the true positive the same word must keep.
"""
from __future__ import annotations

import pytest

from app.core.entity_extraction import enrich_atoms, extract_keys
from app.core.normalizers import extract_meeting_entities
from app.domain import load_domain_pack, set_active_domain_pack


@pytest.fixture(scope="module", params=["default_pack", "networking_pack"])
def pack(request):
    return load_domain_pack(request.param)


def _devices(text: str, pack) -> set[str]:
    return {k for k in extract_keys(text, pack=pack) if k.startswith("device:")}


# ── cabinet / rack ─────────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "Install a 42U rack in the MDF",
    "Install network cabinet in IDF 2",
    "Wall-mount 12U cabinet in the IDF for the switches",
    "The server rack in the MDF needs a new PDU",
    "Provide a 19\" cabinet for the core switch",
    "Install rack in the telecom room",
])
def test_rack_true_positives(text, pack):
    assert "device:rack" in _devices(text, pack)


@pytest.mark.parametrize("text", [
    "Electrical cabinets will be provided by the GC",
    "Kitchen cabinets and countertops are by others",
    "Mount the camera on the cabinet",
    "Install the camera on the electrical cabinet near the gate",
    "Run power from the breaker to the cabinet on the pole",
    "Install a bike rack by the entrance",
    "Electrical cabinet near the MDF door",  # IT cue nearby, but an electrical cabinet
])
def test_rack_false_positives(text, pack):
    assert "device:rack" not in _devices(text, pack)


def test_camera_still_tagged_beside_a_cabinet(pack):
    assert "device:ip_camera" in _devices("Mount the camera on the electrical cabinet", pack)


# ── other generic-word classes ─────────────────────────────────────────
@pytest.mark.parametrize("text,key", [
    # verb uses
    ("OxBlue will monitor the site 24/7 and provide time-lapse", "device:display"),
    ("Please monitor progress and display the feed on the website", "device:display"),
    ("Switch the power off before servicing", "device:switch"),
    ("Need to switch the vendor", "device:switch"),
    # modifier makes it another object
    ("Breaker panel located in the electrical room", "device:controller"),
    ("Install a patch panel", "device:controller"),
    ("The tower crane will block the view", "device:desktop"),
    ("The tower crane will block the view", "device:workstation"),
    ("We may need a strike date for the crane", "device:electric_strike"),
    ("the financial controller approved", "device:controller"),
    # shipping / phrasal UPS
    ("Ship via UPS ground; tracking to follow", "device:ups"),
    ("Two follow-ups on the cam lock", "device:ups"),
    ("Two follow-ups on the cam lock", "device:ip_camera"),
    # units of measure after a number
    ("Ship 50 pcs of bracket", "device:workstation"),
    # transcript speaker labels
    ("Speaker 2: yeah", "device:speaker"),
    # accounts payable
    ("AP department will process the invoice", "device:access_point"),
    ("Send the invoices to AP", "device:access_point"),
])
def test_generic_word_false_positives(text, key):
    pack = load_domain_pack("default_pack")
    assert key not in _devices(text, pack)


@pytest.mark.parametrize("text,key", [
    ("Connect the monitor to the dock", "device:display"),
    ("Mount 2 monitors per desk", "device:display"),
    ("Display wall mounted in lobby", "device:display"),
    ("Connect the cameras to switch 2", "device:switch"),
    ("Provide 24 port PoE switch", "device:switch"),
    ("Install the access control panel by the door", "device:controller"),
    ("Wire the panel to the door strike and card reader", "device:controller"),
    ("Install electric strike on door 4", "device:electric_strike"),
    ("Replace the UPS in the MDF", "device:ups"),
    ("Image 30 PCs for the lab", "device:workstation"),
    ("Install 40 APs on the ceiling", "device:access_point"),
    ("Install ceiling speaker", "device:speaker"),
    ("Reader 2 at Door 4", "device:card_reader"),
    ("We will deploy 450 Chromebooks to students", "device:workstation"),
])
def test_generic_word_true_positives(text, key):
    pack = load_domain_pack("default_pack")
    assert key in _devices(text, pack)


def test_endpoint_pack_tower_and_dock():
    pack = load_domain_pack("endpoint_imac_pack")
    assert "device:desktop" not in _devices("Cell tower behind the school blocks the view", pack)
    assert "device:docking_station" not in _devices("Deliver carts to the loading dock", pack)
    assert "device:desktop" in _devices("Replace the Dell tower and keyboard at each desk", pack)
    assert "device:docking_station" in _devices("Connect the laptop to the USB-C dock", pack)


def test_electrical_pack_amp_is_a_unit():
    pack = load_domain_pack("electrical_pack")
    assert "device:breaker" in _devices("Install a 20 amp breaker", pack)
    assert "device:amplifier" not in _devices("Install a 20 amp breaker", pack)


def test_legal_boilerplate_names_no_devices(pack):
    text = (
        "Contractor shall indemnify and hold harmless the Owner from any liability "
        "for damage to cameras, cabinets or servers hereunder."
    )
    assert _devices(text, pack) == set()


# ── the parser-side taggers use the same gate ──────────────────────────
def test_meeting_entities_skip_speaker_labels_and_verbs():
    set_active_domain_pack(load_domain_pack("default_pack"))
    try:
        assert not any(k.startswith("device:") for k in extract_meeting_entities("Speaker 2: yeah we can monitor the progress"))
        assert "device:rack" not in extract_meeting_entities("Electrical cabinets are by the GC")
        assert "device:rack" in extract_meeting_entities("Install a 42U rack in the MDF")
    finally:
        set_active_domain_pack(load_domain_pack("default_pack"))


def test_email_parser_cabinet():
    from app.parsers.email_parser import EmailParser

    set_active_domain_pack(load_domain_pack("default_pack"))
    p = EmailParser()
    assert "device:rack" not in p._extract_entity_keys("Mount the camera on the electrical cabinet")
    assert "device:rack" in p._extract_entity_keys("Install a 42U rack in the MDF")


# ── chatter-flagged atoms carry no device key ──────────────────────────
def test_chatter_flagged_atom_gets_no_device_keys():
    from app.core.schemas import (
        ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef,
    )

    def _atom(i: str, text: str, flags: list[str], keys: list[str]) -> EvidenceAtom:
        return EvidenceAtom(
            id=i, project_id="p", artifact_id="a", atom_type=AtomType.scope_item,
            raw_text=text, normalized_text=text.lower(), value={}, entity_keys=keys,
            source_refs=[SourceRef(
                id=f"sr_{i}", artifact_id="a", artifact_type=ArtifactType.pdf,
                filename="x.pdf", locator={"page": 1}, extraction_method="test",
                parser_version="test_v1",
            )],
            receipts=[], authority_class=AuthorityClass.contractual_scope,
            confidence=0.9, review_status=ReviewStatus.auto_accepted, review_flags=flags,
            parser_version="test_v1",
        )

    pack = load_domain_pack("default_pack")
    chat = _atom("a1", "Thanks, the cameras look great", ["chatter"], [])
    chat_keyed = _atom("a2", "Thanks for the switch update", ["chatter"], ["device:switch"])
    real = _atom("a3", "Install 4 cameras on the poles", [], [])
    enrich_atoms([chat, chat_keyed, real], pack)
    assert not any(k.startswith("device:") for k in chat.entity_keys)
    assert not any(k.startswith("device:") for k in chat_keyed.entity_keys)
    assert "device:ip_camera" in real.entity_keys
