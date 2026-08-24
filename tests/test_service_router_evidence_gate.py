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
    """Disabled still means "route nothing" -- but it no longer means "say nothing".

    This asserted an exact ``{"enabled": False}`` and now checks the invariant
    instead, because the disabled path deliberately carries three additions:

    * ``candidates`` -- a correction chip cannot offer a choice without them,
      and the chip has to work BEFORE a head exists, which is the situation
      today and for a while yet.
    * ``shadow`` -- the base's answer is logged as a training row here. The
      blob registry holds one router training row of one class, unchanged for
      ten days, so every compile that routed a deal without recording it was a
      training example thrown away.
    * ``ask_pm`` -- the queue signal.

    The routing contract is unchanged: nothing is routed, no ``primary``.
    """
    monkeypatch.delenv("SOWSMITH_SERVICE_ROUTING", raising=False)
    out = build_service_routing([], [])
    assert out["enabled"] is False
    assert not out.get("primary")
    assert out["candidates"], "a chip with no candidates cannot be tapped"


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
    # Reported confidence is CLAMPED (_conf_ceiling, default 0.8): the head's raw
    # similarity saturates near 1.0 while held-out accuracy is far lower, and
    # OrbitBrief's gap checklist would treat a confidently-wrong domain label as
    # certain. The uncapped score stays available as raw_confidence.
    assert out["confidence"] == 0.8
    assert out["raw_confidence"] == 0.91


# ── the gate must read evidence, not filenames ──────────────────────────


_BATTERY_SCOPE = [
    "Replace forty APC Smart-UPS battery modules at the Atlanta facility.",
    "Dispose of spent lead-acid cells per state regulation.",
    "Escort access is required at the dock before 2pm.",
    "Verify runtime under load after replacement.",
]
_WLAN_SCOPE = [
    "Install twenty Meraki MR46 access points across five floors.",
    "Provide a wireless heatmap survey before installation.",
    "Terminate AP drops to the WLC in IDF 3.",
]


class _Atom:
    def __init__(self, text: str) -> None:
        self.raw_text = text
        self.atom_type = "scope_item"


def _gate(scope: list[str], filenames: list[str]) -> bool:
    from app.core.service_router import _corpus_text, _evidence_anchor_satisfied

    corpus = _corpus_text([_Atom(t) for t in scope], [{"filename": f} for f in filenames])
    return _evidence_anchor_satisfied("wireless", corpus)


def test_renaming_files_cannot_open_the_evidence_gate() -> None:
    """The gate is the last thing standing between a confident vote and a bad route.

    Its docstring records why it exists: Stinson's battery install scored
    wireless@0.92 with zero WLAN anchors, so a neural vote alone is not enough
    and real evidence must be present. ``_corpus_text`` fed it the FILENAMES
    alongside the atom bodies, so the evidence it demanded could be supplied by
    renaming a file:

        battery deal, files "attachment_b.pdf" / "scope_of_work.docx"  -> False
        the same deal, files "Wireless AP Survey.pdf" /
                             "Wi-Fi Heatmap Report.docx"              -> TRUE

    Underscored variants happened not to fire, because a regex word boundary
    does not break on "_" -- so whether the hole opened depended on the
    customer's file-naming habit, which is worse than failing outright.
    """
    assert not _gate(_BATTERY_SCOPE, ["attachment_b.pdf", "scope_of_work.docx"])
    assert not _gate(_BATTERY_SCOPE, ["Wireless AP Survey.pdf", "Wi-Fi Heatmap Report.docx"])
    assert not _gate(_BATTERY_SCOPE, ["wireless_ap_survey.pdf", "wifi_heatmap_report.docx"])


def test_a_real_wlan_deal_still_clears_the_gate() -> None:
    """The other half: removing filenames must not cost real detections."""
    assert _gate(_WLAN_SCOPE, ["attachment_b.pdf"])
    assert _gate(_WLAN_SCOPE, ["Wireless AP Survey.pdf"])


# ── the head's input must not lurch on one extra atom ───────────────────


def test_the_scope_sample_does_not_cliff_when_a_deal_grows() -> None:
    """``bodies[:: len // 40]`` was a step function.

    The divisor held across a band of deal sizes and then incremented, and at
    that moment the sample changed wholesale -- one atom arriving at the wrong
    count could replace almost the entire input to the route. Measured over
    deals of 150-400 atoms, comparing each size against one atom more (overlap
    of selected positions):

        stride         worst 0.053   mean 0.974   7 cliffs below 0.5
        proportional   worst 0.333   mean 0.333   250
        consistent     worst 0.951   mean 0.994   0

    Proportional spacing was tried first and rejected by that measurement: it
    removes the cliffs by making every addition churn the sample. Hashing each
    atom's own text decouples membership from the deal's size entirely.
    """
    from app.core.service_router import _scope_summary

    docs = [{"filename": "attachment_b.pdf"}]
    scope = [
        "Contractor shall install forty IP cameras at the Atlanta facility.",
        "All drops terminate in IDF 3 and are certified to TIA-568.",
        "Escort access is required at the Atlanta dock before 2pm.",
        "Mid-turn jumpers are excluded from this bill of materials.",
    ]
    base_atoms = [_Atom(f"{scope[i % len(scope)]} (row {i})") for i in range(200)]

    def _picks(atoms: list[_Atom]) -> set[str]:
        summary = _scope_summary(atoms, docs)
        return {ln for ln in summary.splitlines() if ln.startswith("- ")}

    previous = _picks(base_atoms)
    worst = 1.0
    for extra in range(1, 61):
        atoms = base_atoms + [_Atom(f"Unrelated pricing note {i}.") for i in range(extra)]
        current = _picks(atoms)
        overlap = len(previous & current) / len(previous | current)
        worst = min(worst, overlap)
        previous = current
    assert worst > 0.5, (
        f"one extra atom replaced most of the routing input (worst overlap {worst:.3f})"
    )


def test_the_representation_carries_a_version() -> None:
    """A head fitted on v1's stride sample is out of distribution against v2.

    Recording the version next to the input is the difference between noticing
    a mismatch and puzzling over a head that quietly got worse.
    """
    from app.core.service_router import SCOPE_SUMMARY_VERSION

    assert SCOPE_SUMMARY_VERSION >= 2
