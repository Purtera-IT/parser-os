"""One-shot script that appends ``detection_targets`` blocks to each domain pack.

Run once. Idempotent: skips packs that already have a non-empty
``detection_targets`` block. Sources its target lists from a single
literal table so the per-pack YAML stays human-reviewable.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOMAIN = REPO / "app" / "domain"


# Each entry: (pack-local key, entity_key, ontology_key | None, aliases_from list, completeness, modalities, optional aliases)
TARGETS: dict[str, list[tuple]] = {
    "security_camera_pack.yaml": [
        ("fixed_dome_camera",     "device:fixed_dome_camera",     None, ["device_aliases.ip_camera"], "load_bearing", ["text_tag", "glyph_template"], ["dome", "fixed dome", "mini dome", "indoor dome"]),
        ("bullet_camera",         "device:bullet_camera",         None, ["device_aliases.ip_camera"], "load_bearing", ["text_tag", "glyph_template"], ["bullet", "bullet camera"]),
        ("ptz_camera",            "device:ptz_camera",            None, ["device_aliases.ip_camera"], "load_bearing", ["text_tag", "glyph_template"], ["ptz", "ptz camera", "pan tilt zoom"]),
        ("panoramic_camera",      "device:panoramic_camera",      None, ["device_aliases.ip_camera"], "load_bearing", ["text_tag", "glyph_template"], ["panoramic", "360", "fisheye", "multi-sensor", "multisensor"]),
        ("lpr_camera",            "device:lpr_camera",            None, ["device_aliases.ip_camera"], "load_bearing", ["text_tag", "glyph_template"], ["lpr", "license plate", "lpr camera"]),
        ("nvr",                   "device:nvr",                   None, ["device_aliases.nvr"], "load_bearing", ["text_tag", "glyph_template"]),
        ("camera_switch",         "device:poe_switch",            None, [], "load_bearing", ["text_tag", "glyph_template"], ["poe switch", "camera switch"]),
        ("camera_mount",          "device:camera_mount",          None, ["device_aliases.mount"], "informational", ["text_tag", "glyph_template"]),
        ("weatherproof_housing",  "device:weatherproof_housing",  None, ["device_aliases.housing"], "informational", ["text_tag", "glyph_template"]),
        ("camera_junction_box",   "device:junction_box",          None, [], "informational", ["text_tag", "glyph_template"], ["jb", "junction box", "j-box"]),
        ("camera_conduit",        "device:conduit",               None, [], "informational", ["line_run", "vector_shape"], ["conduit", "emt", "pathway"]),
    ],
    "av_pack.yaml": [
        ("av_display",        "device:av_display",       None, ["device_aliases.display"], "load_bearing", ["text_tag", "glyph_template"]),
        ("av_projector",      "device:projector",        None, ["device_aliases.projector"], "load_bearing", ["text_tag", "glyph_template"]),
        ("av_speaker",        "device:speaker",          None, ["device_aliases.speaker"], "load_bearing", ["text_tag", "glyph_template"]),
        ("av_mic",            "device:microphone",       None, ["device_aliases.microphone"], "load_bearing", ["text_tag", "glyph_template"]),
        ("av_vc_camera",      "device:vc_camera",        None, ["device_aliases.camera"], "load_bearing", ["text_tag", "glyph_template"], ["ptz camera", "vc camera"]),
        ("av_rack",           "device:av_rack",          None, ["device_aliases.rack"], "load_bearing", ["text_tag", "glyph_template"]),
        ("av_floor_box",      "device:floor_box",        None, [], "informational", ["text_tag", "glyph_template"], ["floor box", "poke-thru"]),
        ("av_touch_panel",    "device:touch_panel",      None, [], "informational", ["text_tag", "glyph_template"], ["touch panel", "control panel"]),
        ("av_dsp",            "device:dsp",              None, [], "informational", ["text_tag", "glyph_template"], ["dsp", "amplifier", "switcher"]),
        ("av_conduit",        "device:conduit",          None, [], "informational", ["line_run", "vector_shape"], ["conduit", "emt"]),
    ],
    "wireless_pack.yaml": [
        ("wireless_ap",          "device:wireless_ap",       "wireless_access_point", ["device_aliases.access_point"], "load_bearing", ["text_tag", "glyph_template"]),
        ("wireless_antenna",     "device:antenna",           None, ["device_aliases.antenna"], "load_bearing", ["text_tag", "glyph_template"]),
        ("wireless_controller",  "device:wlc",               None, ["device_aliases.controller"], "load_bearing", ["text_tag", "glyph_template"], ["wlc", "wireless controller"]),
        ("wireless_switch",      "device:poe_switch",        None, ["device_aliases.switch"], "load_bearing", ["text_tag", "glyph_template"]),
        ("wireless_heatmap_zone","zone:wireless_coverage",   None, [], "informational", ["zone"], ["heat map", "heatmap", "coverage zone"]),
        ("wireless_uplink",      "device:uplink_drop",       None, [], "informational", ["line_run", "vector_shape"], ["uplink", "drop"]),
    ],
    "fire_safety_pack.yaml": [
        ("fire_panel",        "device:fire_panel",        None, ["device_aliases.panel"], "load_bearing", ["text_tag", "glyph_template"], ["facp", "fire alarm control panel"]),
        ("fire_subpanel",     "device:fire_subpanel",     None, [], "informational", ["text_tag", "glyph_template"], ["sub panel", "sub-panel"]),
        ("smoke_detector",    "device:smoke_detector",    None, ["device_aliases.smoke_detector"], "load_bearing", ["text_tag", "glyph_template"]),
        ("heat_detector",     "device:heat_detector",     None, ["device_aliases.heat_detector"], "load_bearing", ["text_tag", "glyph_template"]),
        ("pull_station",      "device:pull_station",      None, ["device_aliases.pull_station"], "load_bearing", ["text_tag", "glyph_template"]),
        ("horn_strobe",       "device:horn_strobe",       None, ["device_aliases.horn_strobe"], "load_bearing", ["text_tag", "glyph_template"]),
        ("fire_speaker",      "device:fire_speaker",      None, ["device_aliases.speaker"], "informational", ["text_tag", "glyph_template"]),
        ("duct_detector",     "device:duct_detector",     None, [], "informational", ["text_tag", "glyph_template"], ["duct detector", "beam detector"]),
        ("fire_module",       "device:fire_module",       None, [], "informational", ["text_tag", "glyph_template"], ["control module", "monitor module"]),
        ("nac_slc_run",       "device:nac_slc_run",       None, [], "informational", ["line_run"], ["nac", "slc", "fire conduit"]),
    ],
    "access_control_pack.yaml": [
        ("card_reader",      "device:card_reader",      None, ["device_aliases.reader"], "load_bearing", ["text_tag", "glyph_template"], ["card reader", "prox reader", "keypad reader"]),
        ("ac_controller",    "device:ac_controller",    None, ["device_aliases.controller"], "load_bearing", ["text_tag", "glyph_template"], ["controller", "access panel"]),
        ("rex_device",       "device:rex",              None, ["device_aliases.rex"], "load_bearing", ["text_tag", "glyph_template"]),
        ("door_contact",     "device:door_contact",     None, ["device_aliases.door_contact"], "load_bearing", ["text_tag", "glyph_template"], ["dps", "door position switch"]),
        ("electric_strike",  "device:electric_strike",  None, ["device_aliases.electric_strike"], "load_bearing", ["text_tag", "glyph_template"]),
        ("mag_lock",         "device:mag_lock",         None, ["device_aliases.mag_lock"], "load_bearing", ["text_tag", "glyph_template"]),
        ("ac_power_supply",  "device:power_supply",     None, [], "informational", ["text_tag", "glyph_template"], ["power supply", "psu"]),
        ("ac_junction",      "device:junction_box",     None, [], "informational", ["text_tag", "glyph_template"], ["junction box", "jb"]),
    ],
    "bms_pack.yaml": [
        ("bms_controller",     "device:bms_controller",     None, ["device_aliases.controller"], "load_bearing", ["text_tag", "glyph_template"], ["ddc", "jace", "bms controller"]),
        ("thermostat",         "device:thermostat",         None, ["device_aliases.thermostat"], "load_bearing", ["text_tag", "glyph_template"]),
        ("bms_sensor",         "device:bms_sensor",         None, ["device_aliases.sensor"], "load_bearing", ["text_tag", "glyph_template"], ["temp sensor", "humidity sensor", "co2 sensor"]),
        ("vav_box",            "device:vav_box",            None, [], "load_bearing", ["text_tag", "glyph_template"], ["vav"]),
        ("ahu",                "device:ahu",                None, [], "load_bearing", ["text_tag", "glyph_template"], ["ahu", "air handler"]),
        ("rtu",                "device:rtu",                None, [], "informational", ["text_tag", "glyph_template"], ["rtu", "rooftop unit"]),
        ("vrf_unit",           "device:vrf_unit",           None, [], "informational", ["text_tag", "glyph_template"], ["vrf", "vrv"]),
        ("bms_actuator",       "device:actuator",           None, [], "informational", ["text_tag", "glyph_template"], ["damper", "valve actuator"]),
        ("bms_trunk",          "device:bms_trunk",          None, [], "informational", ["line_run"], ["bacnet", "modbus", "ms/tp"]),
    ],
    "electrical_pack.yaml": [
        ("electrical_panel",   "device:electrical_panel",   None, ["device_aliases.panel"], "load_bearing", ["text_tag", "glyph_template"], ["panelboard", "load center"]),
        ("transformer",        "device:transformer",        None, [], "load_bearing", ["text_tag", "glyph_template"], ["xfmr", "transformer"]),
        ("disconnect_switch",  "device:disconnect",         None, [], "load_bearing", ["text_tag", "glyph_template"], ["disconnect", "fused disconnect"]),
        ("receptacle",         "device:receptacle",         None, ["device_aliases.receptacle"], "load_bearing", ["text_tag", "glyph_template"], ["outlet", "duplex", "quad", "gfci"]),
        ("home_run_circuit",   "device:home_run",           None, [], "informational", ["line_run"], ["home run", "circuit"]),
        ("breaker",            "device:breaker",            None, ["device_aliases.breaker"], "informational", ["text_tag", "glyph_template"]),
        ("electrical_conduit", "device:conduit",            None, [], "informational", ["line_run", "vector_shape"], ["emt", "conduit", "feeder"]),
        ("ground_busbar",      "device:ground_busbar",      None, [], "informational", ["text_tag", "glyph_template"], ["ground bar", "busbar"]),
    ],
    "paging_pack.yaml": [
        ("paging_ceiling_speaker",  "device:speaker",            None, ["device_aliases.speaker"], "load_bearing", ["text_tag", "glyph_template"], ["ceiling speaker", "pendant speaker"]),
        ("paging_horn_speaker",     "device:horn_speaker",       None, [], "load_bearing", ["text_tag", "glyph_template"], ["horn speaker", "outdoor horn"]),
        ("paging_amplifier",        "device:amplifier",          None, ["device_aliases.amplifier"], "load_bearing", ["text_tag", "glyph_template"]),
        ("paging_mic",              "device:paging_mic",         None, ["device_aliases.microphone"], "load_bearing", ["text_tag", "glyph_template"], ["paging mic", "call station"]),
        ("intercom_station",        "device:intercom",           None, ["device_aliases.intercom"], "informational", ["text_tag", "glyph_template"]),
        ("paging_controller",       "device:paging_controller",  None, ["device_aliases.controller"], "informational", ["text_tag", "glyph_template"]),
        ("paging_zone",             "zone:paging_zone",          None, [], "informational", ["zone"], ["paging zone", "zone "]),
    ],
    "das_pack.yaml": [
        ("bda",              "device:bda",              None, ["device_aliases.bda"], "load_bearing", ["text_tag", "glyph_template"], ["bda", "signal booster"]),
        ("donor_antenna",    "device:donor_antenna",    None, [], "load_bearing", ["text_tag", "glyph_template"], ["donor antenna"]),
        ("das_antenna",      "device:das_antenna",      None, ["device_aliases.antenna"], "load_bearing", ["text_tag", "glyph_template"], ["remote antenna", "indoor antenna"]),
        ("das_head_end",     "device:das_head_end",     None, [], "load_bearing", ["text_tag", "glyph_template"], ["master unit", "head-end", "head end"]),
        ("das_remote_unit",  "device:das_remote_unit",  None, [], "load_bearing", ["text_tag", "glyph_template"], ["remote unit"]),
        ("das_splitter",     "device:das_splitter",     None, ["device_aliases.splitter"], "informational", ["text_tag", "glyph_template"], ["splitter", "tap", "combiner"]),
        ("das_cable_run",    "device:das_cable_run",    None, [], "informational", ["line_run"], ["coax", "fiber", "das run"]),
    ],
    "pos_commerce_pack.yaml": [
        ("pos_terminal",     "device:pos_terminal",      None, ["device_aliases.pos_terminal"], "load_bearing", ["text_tag", "glyph_template"], ["register", "pos"]),
        ("pin_pad",          "device:pin_pad",           None, ["device_aliases.pin_pad"], "load_bearing", ["text_tag", "glyph_template"], ["payment device", "pin entry"]),
        ("receipt_printer",  "device:receipt_printer",   None, [], "informational", ["text_tag", "glyph_template"], ["printer"]),
        ("scanner",          "device:scanner",           None, [], "informational", ["text_tag", "glyph_template"], ["barcode scanner"]),
        ("cash_drawer",      "device:cash_drawer",       None, ["device_aliases.cash_drawer"], "load_bearing", ["text_tag", "glyph_template"]),
        ("pos_data_drop",    "device:data_drop",         None, [], "load_bearing", ["text_tag", "glyph_template"], ["data drop", "data jack"]),
        ("pos_power_outlet", "device:receptacle",        None, [], "informational", ["text_tag", "glyph_template"], ["outlet", "receptacle"]),
    ],
    "datacenter_field_pack.yaml": [
        ("dc_rack",          "device:rack",          None, ["device_aliases.rack"], "load_bearing", ["text_tag", "glyph_template"], ["rack", "cabinet"]),
        ("rack_pdu",         "device:rack_pdu",      None, ["device_aliases.pdu"], "load_bearing", ["text_tag", "glyph_template"], ["pdu"]),
        ("tor_switch",       "device:tor_switch",    None, ["device_aliases.tor"], "load_bearing", ["text_tag", "glyph_template"], ["tor", "top of rack"]),
        ("dc_patch_panel",   "device:patch_panel",   None, [], "load_bearing", ["text_tag", "glyph_template"], ["patch panel"]),
        ("cable_tray",       "device:cable_tray",    None, [], "informational", ["line_run", "vector_shape"], ["cable tray", "ladder rack"]),
        ("dc_floor_tile",    "marker:floor_tile",    None, [], "informational", ["text_tag"], ["floor tile", "row marker"]),
        ("power_whip",       "device:power_whip",    None, [], "informational", ["line_run"], ["power whip"]),
    ],
    "networking_pack.yaml": [
        ("network_switch",   "device:switch",           None, ["device_aliases.switch"], "load_bearing", ["text_tag", "glyph_template"]),
        ("network_router",   "device:router",           None, ["device_aliases.router"], "load_bearing", ["text_tag", "glyph_template"]),
        ("network_firewall", "device:firewall",         None, ["device_aliases.firewall"], "load_bearing", ["text_tag", "glyph_template"]),
        ("network_controller","device:controller",      None, [], "informational", ["text_tag", "glyph_template"], ["gateway", "controller"]),
        ("wan_circuit",      "device:wan_circuit",      None, [], "informational", ["text_tag", "line_run"], ["wan", "circuit", "isp drop"]),
        ("network_uplink",   "device:uplink",           None, [], "informational", ["line_run"], ["uplink", "fiber uplink"]),
        ("network_rack",     "device:rack",             None, ["device_aliases.rack"], "load_bearing", ["text_tag", "glyph_template"]),
        ("network_pdu_ups",  "device:pdu_ups",          None, [], "informational", ["text_tag", "glyph_template"], ["pdu", "ups"]),
        ("network_patch_panel","device:patch_panel",    None, [], "load_bearing", ["text_tag", "glyph_template"], ["patch panel"]),
    ],
    "network_modernization_pack.yaml": [
        ("nm_switch",       "device:switch",         None, ["device_aliases.switch"], "load_bearing", ["text_tag", "glyph_template"]),
        ("nm_router",       "device:router",         None, ["device_aliases.router"], "load_bearing", ["text_tag", "glyph_template"]),
        ("nm_firewall",     "device:firewall",       None, [], "load_bearing", ["text_tag", "glyph_template"], ["firewall"]),
        ("nm_rack",         "device:rack",           None, [], "informational", ["text_tag", "glyph_template"], ["rack"]),
        ("nm_uplink",       "device:uplink",         None, [], "informational", ["line_run"], ["uplink"]),
    ],
    "edge_iot_security_pack.yaml": [
        ("edge_ip_camera",   "device:ip_camera",      "edge_ip_security_camera", ["device_aliases.ip_camera"], "load_bearing", ["text_tag", "glyph_template"]),
        ("edge_nvr",         "device:nvr",            None, ["device_aliases.nvr"], "load_bearing", ["text_tag", "glyph_template"]),
        ("edge_door_reader", "device:card_reader",    None, ["device_aliases.reader"], "load_bearing", ["text_tag", "glyph_template"]),
        ("edge_iot_gateway", "device:iot_gateway",    None, [], "load_bearing", ["text_tag", "glyph_template"], ["iot gateway"]),
        ("edge_poe_switch",  "device:poe_switch",     None, [], "load_bearing", ["text_tag", "glyph_template"], ["poe switch"]),
        ("edge_sensor",      "device:sensor",         None, [], "informational", ["text_tag", "glyph_template"], ["sensor"]),
        ("edge_enclosure",   "device:enclosure",      None, [], "informational", ["text_tag", "glyph_template"], ["enclosure", "cabinet"]),
    ],
    "itad_pack.yaml": [
        ("itad_asset_marker",  "marker:asset",          None, [], "load_bearing", ["text_tag"], ["asset", "device"]),
        ("itad_rack",          "device:rack",           None, [], "informational", ["text_tag", "glyph_template"], ["rack"]),
        ("itad_pallet",        "marker:pallet",         None, [], "informational", ["text_tag"], ["pallet", "bin", "cage"]),
        ("itad_wipe_station",  "device:wipe_station",   None, [], "informational", ["text_tag", "glyph_template"], ["wipe", "sanitization"]),
        ("itad_pickup_zone",   "zone:pickup",           None, [], "informational", ["zone"], ["pickup", "staging"]),
    ],
    "endpoint_imac_pack.yaml": [
        ("workstation_marker", "marker:workstation",   None, [], "load_bearing", ["text_tag", "glyph_template"], ["workstation", "desk", "seat"]),
        ("laptop_dock",        "device:dock",          None, ["device_aliases.dock"], "load_bearing", ["text_tag", "glyph_template"], ["dock", "docking station"]),
        ("monitor",            "device:monitor",       None, ["device_aliases.monitor"], "informational", ["text_tag", "glyph_template"]),
        ("user_printer",       "device:printer",       None, ["device_aliases.printer"], "informational", ["text_tag", "glyph_template"]),
        ("user_phone",         "device:phone",         None, [], "informational", ["text_tag", "glyph_template"], ["phone", "handset"]),
        ("user_network_jack",  "device:data_drop",     None, [], "informational", ["text_tag", "glyph_template"], ["jack", "drop"]),
    ],
    "structured_backbone_fiber_pack.yaml": [
        ("mdf",                "room:mdf",              None, [], "load_bearing", ["text_tag", "glyph_template"], ["mdf", "main distribution frame"]),
        ("idf",                "room:idf",              None, [], "load_bearing", ["text_tag", "glyph_template"], ["idf", "intermediate distribution frame"]),
        ("fiber_trunk",        "device:fiber_trunk",    None, ["device_aliases.trunk"], "load_bearing", ["line_run"], ["fiber trunk", "backbone"]),
        ("fiber_patch_panel",  "device:patch_panel",    None, ["device_aliases.patch_panel"], "load_bearing", ["text_tag", "glyph_template"]),
        ("fiber_riser",        "device:fiber_riser",    None, [], "load_bearing", ["line_run"], ["riser", "backbone riser"]),
        ("fiber_strand_count", "device:strand_count",   None, [], "informational", ["text_tag"], ["strands", "sm", "om"]),
    ],
}


def append_targets(path: Path, targets: list[tuple]) -> bool:
    text = path.read_text(encoding="utf-8")
    if re.search(r"^detection_targets:", text, flags=re.M):
        return False
    if not text.endswith("\n"):
        text += "\n"
    lines = ["detection_targets:"]
    for entry in targets:
        key, entity_key, ontology_key, aliases_from, completeness, modalities, *rest = entry
        explicit_aliases = rest[0] if rest else []
        lines.append(f"  - key: {key}")
        lines.append(f"    entity_key: {entity_key}")
        if ontology_key:
            lines.append(f"    ontology_key: {ontology_key}")
        if aliases_from:
            lines.append("    aliases_from:")
            for a in aliases_from:
                lines.append(f"      - {a}")
        if explicit_aliases:
            lines.append("    aliases:")
            for a in explicit_aliases:
                lines.append(f"      - {a!r}")
        lines.append(f"    completeness: {completeness}")
        lines.append("    modalities:")
        for m in modalities:
            lines.append(f"      - {m}")
    text += "\n" + "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    changed = 0
    for fname, targets in TARGETS.items():
        p = DOMAIN / fname
        if not p.exists():
            print(f"  SKIP (missing): {fname}")
            continue
        if append_targets(p, targets):
            changed += 1
            print(f"  WROTE {fname}: {len(targets)} targets")
        else:
            print(f"  SKIP (already has detection_targets): {fname}")
    print(f"\ntotal packs updated: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
