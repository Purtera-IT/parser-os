"""Append ``parent_entity_keys`` to subtype detection targets across packs.

For each (pack, target_key) pair listed below, this script appends a
``parent_entity_keys: [...]`` block to the target's YAML stanza so
cross-artifact quantity conflicts can roll a schematic subtype count
up to the broader BOM/RFP entity key.

Idempotent: skips targets that already declare ``parent_entity_keys``.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOMAIN = REPO / "app" / "domain"


# Each entry: (file, target_key, parents). Camera subtypes already done
# in security_camera_pack.yaml in round-3; this file fills in everything
# else the boss-review follow-up flagged.
ROLLUPS: list[tuple[str, str, list[str]]] = [
    # security_camera: NVR rolls up to a generic recorder bucket; the
    # camera subtypes already carry [device:ip_camera, device:camera].
    ("security_camera_pack.yaml", "nvr", ["device:recorder", "device:storage"]),
    # edge_iot_security: the only camera target
    ("edge_iot_security_pack.yaml", "edge_ip_camera", ["device:ip_camera", "device:camera"]),
    ("edge_iot_security_pack.yaml", "edge_nvr", ["device:nvr", "device:recorder"]),
    ("edge_iot_security_pack.yaml", "edge_door_reader", ["device:reader", "device:card_reader"]),
    # fire_safety: detector subtypes roll up to device:detector
    ("fire_safety_pack.yaml", "smoke_detector", ["device:detector", "device:fire_device"]),
    ("fire_safety_pack.yaml", "heat_detector", ["device:detector", "device:fire_device"]),
    ("fire_safety_pack.yaml", "duct_detector", ["device:detector", "device:fire_device"]),
    ("fire_safety_pack.yaml", "horn_strobe", ["device:notification_appliance", "device:fire_device"]),
    ("fire_safety_pack.yaml", "fire_speaker", ["device:notification_appliance", "device:fire_device"]),
    ("fire_safety_pack.yaml", "pull_station", ["device:initiating_device", "device:fire_device"]),
    ("fire_safety_pack.yaml", "fire_panel", ["device:panel", "device:fire_device"]),
    ("fire_safety_pack.yaml", "fire_subpanel", ["device:panel", "device:fire_device"]),
    # access_control
    ("access_control_pack.yaml", "card_reader", ["device:reader", "device:access_device"]),
    ("access_control_pack.yaml", "rex_device", ["device:rex", "device:access_device"]),
    ("access_control_pack.yaml", "door_contact", ["device:contact", "device:access_device"]),
    ("access_control_pack.yaml", "electric_strike", ["device:lock_hardware", "device:access_device"]),
    ("access_control_pack.yaml", "mag_lock", ["device:lock_hardware", "device:access_device"]),
    ("access_control_pack.yaml", "ac_controller", ["device:controller", "device:access_device"]),
    # av — display + projector roll up to device:display, speakers/mics to themselves
    ("av_pack.yaml", "av_display", ["device:display", "device:av_device"]),
    ("av_pack.yaml", "av_projector", ["device:display", "device:projector", "device:av_device"]),
    ("av_pack.yaml", "av_speaker", ["device:speaker", "device:av_device"]),
    ("av_pack.yaml", "av_mic", ["device:microphone", "device:av_device"]),
    ("av_pack.yaml", "av_vc_camera", ["device:camera", "device:av_device"]),
    ("av_pack.yaml", "av_rack", ["device:rack", "device:av_device"]),
    # wireless
    ("wireless_pack.yaml", "wireless_ap", ["device:access_point", "device:wireless_device"]),
    ("wireless_pack.yaml", "wireless_antenna", ["device:antenna", "device:wireless_device"]),
    ("wireless_pack.yaml", "wireless_controller", ["device:wlc", "device:wireless_device"]),
    # das — donor + remote antennas roll up to device:antenna
    ("das_pack.yaml", "donor_antenna", ["device:antenna", "device:das_device"]),
    ("das_pack.yaml", "das_antenna", ["device:antenna", "device:das_device"]),
    ("das_pack.yaml", "bda", ["device:das_amplifier", "device:das_device"]),
    ("das_pack.yaml", "das_head_end", ["device:das_amplifier", "device:das_device"]),
    ("das_pack.yaml", "das_remote_unit", ["device:das_amplifier", "device:das_device"]),
    # paging — speaker variants roll up to device:speaker
    ("paging_pack.yaml", "paging_ceiling_speaker", ["device:speaker", "device:paging_device"]),
    ("paging_pack.yaml", "paging_horn_speaker", ["device:speaker", "device:paging_device"]),
    ("paging_pack.yaml", "paging_amplifier", ["device:amplifier", "device:paging_device"]),
    ("paging_pack.yaml", "paging_mic", ["device:microphone", "device:paging_device"]),
    # bms — controllers + sensors roll up
    ("bms_pack.yaml", "bms_controller", ["device:controller", "device:bms_device"]),
    ("bms_pack.yaml", "thermostat", ["device:controller", "device:bms_device"]),
    ("bms_pack.yaml", "bms_sensor", ["device:sensor", "device:bms_device"]),
    # networking — switch/router/firewall roll up to network device
    ("networking_pack.yaml", "network_switch", ["device:switch", "device:network_device"]),
    ("networking_pack.yaml", "network_router", ["device:router", "device:network_device"]),
    ("networking_pack.yaml", "network_firewall", ["device:firewall", "device:network_device"]),
    ("network_modernization_pack.yaml", "nm_switch", ["device:switch", "device:network_device"]),
    ("network_modernization_pack.yaml", "nm_router", ["device:router", "device:network_device"]),
    ("network_modernization_pack.yaml", "nm_firewall", ["device:firewall", "device:network_device"]),
    # electrical
    ("electrical_pack.yaml", "electrical_panel", ["device:panel", "device:electrical_device"]),
    ("electrical_pack.yaml", "transformer", ["device:transformer", "device:electrical_device"]),
    ("electrical_pack.yaml", "receptacle", ["device:outlet", "device:electrical_device"]),
    # datacenter
    ("datacenter_field_pack.yaml", "dc_rack", ["device:rack", "device:datacenter_device"]),
    ("datacenter_field_pack.yaml", "rack_pdu", ["device:pdu", "device:datacenter_device"]),
    ("datacenter_field_pack.yaml", "tor_switch", ["device:switch", "device:datacenter_device"]),
    # pos
    ("pos_commerce_pack.yaml", "pos_terminal", ["device:pos_register", "device:pos_device"]),
    ("pos_commerce_pack.yaml", "pin_pad", ["device:payment_device", "device:pos_device"]),
    ("pos_commerce_pack.yaml", "cash_drawer", ["device:cash_drawer", "device:pos_device"]),
    # structured cabling
    ("structured_backbone_fiber_pack.yaml", "fiber_trunk", ["device:cable", "device:cabling_device"]),
    ("structured_backbone_fiber_pack.yaml", "fiber_patch_panel", ["device:patch_panel", "device:cabling_device"]),
    # endpoint
    ("endpoint_imac_pack.yaml", "laptop_dock", ["device:dock", "device:endpoint_device"]),
    ("endpoint_imac_pack.yaml", "monitor", ["device:display", "device:endpoint_device"]),
    ("endpoint_imac_pack.yaml", "user_printer", ["device:printer", "device:endpoint_device"]),
]


def _find_target_block(text: str, target_key: str) -> tuple[int, int] | None:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s*-\s*key:\s*{re.escape(target_key)}\s*$", line):
            start = i
            break
    if start is None:
        return None
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if not lines[j].strip():
            continue
        indent = len(lines[j]) - len(lines[j].lstrip())
        if lines[j].lstrip().startswith("- ") and indent == base_indent:
            end = j
            break
    return (start, end)


def apply_rollup(path: Path, target_key: str, parents: list[str]) -> bool:
    text = path.read_text(encoding="utf-8")
    span = _find_target_block(text, target_key)
    if span is None:
        return False
    lines = text.splitlines()
    start, end = span
    block_lines = lines[start:end]
    if any(line.lstrip().startswith("parent_entity_keys:") for line in block_lines):
        return False
    base_indent = len(block_lines[0]) - len(block_lines[0].lstrip())
    field_indent = " " * (base_indent + 2)
    item_indent = " " * (base_indent + 4)
    insert_at = len(block_lines)
    while insert_at > 1 and not block_lines[insert_at - 1].strip():
        insert_at -= 1
    new_lines = [field_indent + "parent_entity_keys:"]
    for p in parents:
        new_lines.append(item_indent + f"- {p}")
    block_lines = block_lines[:insert_at] + new_lines + block_lines[insert_at:]
    new_text = "\n".join(lines[:start] + block_lines + lines[end:])
    if text.endswith("\n"):
        new_text += "\n"
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    changed = 0
    for fname, target_key, parents in ROLLUPS:
        p = DOMAIN / fname
        if not p.exists():
            print(f"  SKIP (missing): {fname}")
            continue
        if apply_rollup(p, target_key, parents):
            changed += 1
            print(f"  WROTE {fname}: {target_key} -> {parents}")
        else:
            print(f"  NOOP  {fname}: {target_key}")
    print(f"\ntotal targets updated: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
