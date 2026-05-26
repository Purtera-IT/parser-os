"""PR2 — domain-pack detection_targets schema + per-pack coverage."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.loader import DOMAIN_DIR, load_domain_pack
from app.domain.schemas import DetectionTargetSpec, DomainPack

# Packs that must declare at least one load-bearing detection target.
PACKS_WITH_TARGETS = sorted(
    [
        "access_control",
        "av",
        "bms",
        "copper_cabling",
        "das",
        "datacenter_field",
        "edge_iot_security",
        "electrical",
        "endpoint_imac",
        "fire_safety",
        "itad",
        "network_modernization",
        "networking",
        "paging",
        "pos_commerce",
        "security_camera",
        "structured_backbone_fiber",
        "wireless",
    ]
)


@pytest.mark.parametrize("pack_id", PACKS_WITH_TARGETS)
def test_pack_has_at_least_one_load_bearing_target(pack_id: str) -> None:
    pack = load_domain_pack(pack_id)
    assert pack.detection_targets, f"{pack_id} declares no detection_targets"
    load_bearing = [t for t in pack.detection_targets if t.completeness == "load_bearing"]
    assert load_bearing, f"{pack_id} has detection_targets but none are load_bearing"


def test_default_pack_has_empty_detection_targets() -> None:
    pack = load_domain_pack(None)
    assert pack.detection_targets == []


def test_every_pack_loads_without_error() -> None:
    failures: list[tuple[str, str]] = []
    for path in sorted(DOMAIN_DIR.glob("*.yaml")):
        try:
            pack = load_domain_pack(path.stem.replace("_pack", ""))
        except Exception as exc:
            failures.append((path.name, repr(exc)))
            continue
        assert isinstance(pack, DomainPack)
    assert not failures, f"pack load failures: {failures}"


def test_detection_target_modalities_are_validated() -> None:
    with pytest.raises(Exception):
        DetectionTargetSpec.model_validate({
            "key": "x",
            "entity_key": "device:x",
            "completeness": "load_bearing",
            "modalities": [],
        })


def test_detection_target_requires_non_empty_keys() -> None:
    with pytest.raises(Exception):
        DetectionTargetSpec.model_validate({
            "key": "  ",
            "entity_key": "device:x",
            "modalities": ["text_tag"],
        })
    with pytest.raises(Exception):
        DetectionTargetSpec.model_validate({
            "key": "ok",
            "entity_key": "",
            "modalities": ["text_tag"],
        })


def test_resolved_aliases_for_security_camera_ptz_pulls_device_alias_table() -> None:
    pack = load_domain_pack("security_camera")
    ptz = next(t for t in pack.detection_targets if t.key == "ptz_camera")
    resolved = [a.lower() for a in pack.resolved_target_aliases(ptz)]
    # device_aliases.ip_camera plus explicit aliases like 'ptz'
    assert any("ptz" == a or a.startswith("ptz ") for a in resolved)
    # has many camera-shaped aliases pulled from ip_camera
    assert any("camera" in a for a in resolved)


def test_resolved_aliases_dedupes_case_insensitively() -> None:
    pack = load_domain_pack("security_camera")
    bullet = next(t for t in pack.detection_targets if t.key == "bullet_camera")
    resolved = pack.resolved_target_aliases(bullet)
    norm = [a.lower() for a in resolved]
    assert len(norm) == len(set(norm)), "duplicate aliases not deduped"


def test_target_keys_unique_per_pack() -> None:
    for pack_id in PACKS_WITH_TARGETS:
        pack = load_domain_pack(pack_id)
        keys = [t.key for t in pack.detection_targets]
        assert len(keys) == len(set(keys)), f"{pack_id} has duplicate target keys: {keys}"


def test_no_broken_aliases_from_references() -> None:
    """Every aliases_from path must resolve to an existing device_aliases key.

    A broken reference silently degrades target matching because the parser's
    alias resolver returns an empty list, falling back to the explicit aliases
    only. This used to ship with 21 broken references; that mistake doesn't
    get to happen quietly again.
    """
    failures: list[str] = []
    for pack_id in PACKS_WITH_TARGETS:
        pack = load_domain_pack(pack_id)
        da_keys = set(pack.device_aliases.keys())
        for target in pack.detection_targets:
            for ref in target.aliases_from:
                head, _, tail = ref.partition(".")
                if head != "device_aliases" or not tail:
                    failures.append(f"{pack_id}.{target.key}: malformed path {ref!r}")
                    continue
                if tail not in da_keys:
                    failures.append(
                        f"{pack_id}.{target.key}: device_aliases.{tail} missing"
                        f" (available: {sorted(da_keys)})"
                    )
    assert not failures, "broken aliases_from references:\n  " + "\n  ".join(failures)


def test_every_load_bearing_target_has_non_empty_resolved_aliases() -> None:
    """A load-bearing target with an empty resolved alias bag can't match the
    legend, so it effectively becomes a permanent legend_gap warning. Guarantee
    every load-bearing target has *something* to match against.
    """
    failures: list[str] = []
    for pack_id in PACKS_WITH_TARGETS:
        pack = load_domain_pack(pack_id)
        for target in pack.detection_targets:
            if target.completeness != "load_bearing":
                continue
            resolved = pack.resolved_target_aliases(target)
            if not resolved and not target.aliases:
                failures.append(f"{pack_id}.{target.key} has zero aliases")
    assert not failures, "load-bearing targets with no aliases:\n  " + "\n  ".join(failures)


def test_copper_cabling_wide_reference_adapter_carries_targets() -> None:
    pack = load_domain_pack("copper_cabling")
    assert pack.pack_id == "copper_cabling"
    keys = {t.key for t in pack.detection_targets}
    # Must include MDF/IDF/drop/run from the wide reference pack
    assert {"copper_mdf", "copper_idf", "copper_data_drop", "copper_cable_run"}.issubset(keys)


def test_security_camera_declares_canonical_camera_subtypes() -> None:
    pack = load_domain_pack("security_camera")
    keys = {t.key for t in pack.detection_targets}
    must_have = {"fixed_dome_camera", "bullet_camera", "ptz_camera", "panoramic_camera", "lpr_camera", "nvr"}
    missing = must_have - keys
    assert not missing, f"security_camera missing canonical subtypes: {sorted(missing)}"


def test_fire_safety_declares_canonical_device_set() -> None:
    pack = load_domain_pack("fire_safety")
    keys = {t.key for t in pack.detection_targets}
    must_have = {"fire_panel", "smoke_detector", "heat_detector", "pull_station", "horn_strobe"}
    missing = must_have - keys
    assert not missing, f"fire_safety missing canonical devices: {sorted(missing)}"


def test_access_control_declares_canonical_device_set() -> None:
    pack = load_domain_pack("access_control")
    keys = {t.key for t in pack.detection_targets}
    must_have = {"card_reader", "ac_controller", "rex_device", "door_contact", "electric_strike", "mag_lock"}
    missing = must_have - keys
    assert not missing, f"access_control missing canonical devices: {sorted(missing)}"
