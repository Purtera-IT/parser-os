"""Unit tests for app.takeoff.quote_unitizer — synthetic device inputs."""
from __future__ import annotations

from app.takeoff.quote_unitizer import quote_lines_for_devices
from app.takeoff.schemas import BBox, DeviceInstance


def _wn_device(
    *,
    device_id: str,
    multiplier: int,
    floor_label: str = "Level 5-12 and 15",
    home_run_to: str = "IDF-5",
) -> DeviceInstance:
    return DeviceInstance(
        id=device_id,
        page_index=9,
        sheet_number="T1.06",
        sheet_name="LEVEL 5-12 AND LEVEL 15 FLOOR PLAN",
        raw_symbol="WN",
        normalized_class="wireless_node_outlet",
        system="structured_cabling_wireless",
        bbox=BBox(x0=100, y0=200, x1=116, y1=210),
        floor_label=floor_label,
        levels_represented=["5", "6", "7", "8", "9", "10", "11", "12", "15"],
        multiplier=multiplier,
        home_run_to=home_run_to,
    )


def test_wn_multiplier_9_produces_9_drops() -> None:
    device = _wn_device(device_id="dev_wn_1", multiplier=9)
    lines = quote_lines_for_devices([device])
    by_key = {line.item_key: line for line in lines}

    assert by_key["cat6_wireless_node_drop"].quantity == 9
    assert by_key["wireless_node_work_area_termination"].quantity == 9
    assert by_key["wireless_node_patch_panel_port"].quantity == 9
    assert by_key["wireless_node_copper_certification_test"].quantity == 9
    assert by_key["wireless_node_label_pair"].quantity == 9
    assert by_key["wireless_node_service_loop_allowance_ft"].quantity == 90
    assert by_key["wireless_node_service_loop_allowance_ft"].unit == "ft"


def test_wn_multiplier_1_produces_1_drop_and_10ft_loop() -> None:
    device = _wn_device(device_id="dev_wn_solo", multiplier=1)
    lines = quote_lines_for_devices([device])
    by_key = {line.item_key: line for line in lines}
    assert by_key["cat6_wireless_node_drop"].quantity == 1
    assert by_key["wireless_node_service_loop_allowance_ft"].quantity == 10


def test_two_devices_same_floor_share_a_quote_line() -> None:
    devices = [
        _wn_device(device_id="dev_a", multiplier=1),
        _wn_device(device_id="dev_b", multiplier=1),
    ]
    lines = quote_lines_for_devices(devices)
    drops = [l for l in lines if l.item_key == "cat6_wireless_node_drop"]
    assert len(drops) == 1
    assert drops[0].quantity == 2
    assert sorted(drops[0].source_device_ids) == ["dev_a", "dev_b"]


def test_wn_notes_include_wifi_survey_caveat() -> None:
    device = _wn_device(device_id="dev_wn_1", multiplier=1)
    lines = quote_lines_for_devices([device])
    drop = next(l for l in lines if l.item_key == "cat6_wireless_node_drop")
    note_blob = " ".join(drop.notes).lower()
    assert "wi-fi vendor survey" in note_blob
    assert "poe" in note_blob
