"""Evidence-anchor gate on the service-router neural head.

The contrastive head can confidently embed UPS/APC battery installs near
wireless exemplars. The gate abstains unless real WLAN/AV/cabling anchors
appear in the corpus — universal, not deal-specific.
"""
from __future__ import annotations

from app.core.service_router import _evidence_anchor_satisfied, build_service_routing


def test_wireless_gate_rejects_ups_battery_scope() -> None:
    corpus = (
        "Install customer provided APCRBC140 battery pack into the UPS. "
        "Remove existing battery pack. Power on equipment. Tampa Florida."
    )
    assert _evidence_anchor_satisfied("wireless", corpus) is False


def test_wireless_gate_accepts_ap_wlan_scope() -> None:
    corpus = (
        "Install 24 Cisco access points and configure SSID / VLAN matrix. "
        "Perform wireless survey and RF heatmap deliverables."
    )
    assert _evidence_anchor_satisfied("wireless", corpus) is True


def test_audio_visual_gate_requires_multiple_anchors() -> None:
    assert _evidence_anchor_satisfied("audio_visual", "Install one display") is False
    assert _evidence_anchor_satisfied(
        "audio_visual",
        "Crestron control processor, Biamp Tesira DSP, and Teams Room codec.",
    ) is True


def test_unknown_pack_has_no_gate() -> None:
    assert _evidence_anchor_satisfied("electrical", "UPS battery only") is True
    assert _evidence_anchor_satisfied("staff_augmentation", "contractors") is True


def test_build_service_routing_abstains_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("SOWSMITH_SERVICE_ROUTING", raising=False)
    out = build_service_routing([], [])
    assert out == {"enabled": False}


def test_build_service_routing_abstains_on_missing_wireless_anchors(monkeypatch) -> None:
    monkeypatch.setenv("SOWSMITH_SERVICE_ROUTING", "1")

    class _FakeHead:
        def classify(self, _text: str):
            return ("wireless", 0.92)

    monkeypatch.setattr("app.core.service_router._load_head", lambda: _FakeHead())
    atoms = [
        {
            "atom_type": "task",
            "text": "Install customer provided APCRBC140 battery pack into the UPS.",
        },
        {
            "atom_type": "scope_item",
            "text": "Customer provides the battery pack at the Tampa office.",
        },
    ]
    docs = [{"filename": "010097 - Stinson battery install.docx"}]
    out = build_service_routing(atoms, docs)
    assert out["enabled"] is True
    assert out.get("abstained") is True
    assert out.get("primary") is None
    assert out.get("abstain_reason") == "missing_evidence_anchors"
    assert out.get("neural_primary") == "wireless"


def test_build_service_routing_emits_wireless_when_anchors_present(monkeypatch) -> None:
    monkeypatch.setenv("SOWSMITH_SERVICE_ROUTING", "1")

    class _FakeHead:
        def classify(self, _text: str):
            return ("wireless", 0.91)

    monkeypatch.setattr("app.core.service_router._load_head", lambda: _FakeHead())
    atoms = [
        {
            "atom_type": "task",
            "text": "Install 40 Meraki MR46 access points campus-wide.",
        },
        {
            "atom_type": "scope_item",
            "text": "Configure SSID and WPA3 auth; deliver RF heatmap.",
        },
    ]
    docs = [{"filename": "wireless-refresh-sow.docx"}]
    out = build_service_routing(atoms, docs)
    assert out["enabled"] is True
    assert out.get("abstained") is not True
    assert out["primary"] == "wireless"
    assert out["confidence"] == 0.91
