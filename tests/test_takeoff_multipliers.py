"""Pure unit tests for app.takeoff.multipliers — title-only inputs."""
from __future__ import annotations

import pytest

from app.takeoff.multipliers import (
    floor_label_for_title,
    levels_from_title,
    multiplier_for_title,
)


@pytest.mark.parametrize(
    "title, expected_levels, expected_multiplier",
    [
        ("LOWER LOBBY FLOOR PLAN", ["Lower Lobby"], 1),
        ("LOBBY LEVEL FLOOR PLAN", ["Lobby"], 1),
        ("LEVEL 2 BALLROOM FLOOR PLAN", ["2"], 1),
        (
            "LEVEL 5-12 AND LEVEL 15 FLOOR PLAN",
            ["5", "6", "7", "8", "9", "10", "11", "12", "15"],
            9,
        ),
        ("LEVEL 17-18 FLOOR PLAN", ["17", "18"], 2),
        ("LEVEL 19-23 FLOOR PLAN", ["19", "20", "21", "22", "23"], 5),
        ("ROOF PLAN", ["Roof"], 1),
        ("SERVICE LEVEL PLAN", ["Service"], 1),
        ("LEVEL 24 FLOOR PLAN", ["24"], 1),
        ("LEVEL 25 FLOOR PLAN", ["25"], 1),
    ],
)
def test_multiplier_for_title(
    title: str, expected_levels: list[str], expected_multiplier: int
) -> None:
    levels, multiplier = multiplier_for_title(title)
    assert levels == expected_levels
    assert multiplier == expected_multiplier


def test_levels_from_unknown_title_is_empty() -> None:
    assert levels_from_title("KEYED NOTE LEGEND") == []
    assert multiplier_for_title("") == ([], 1)


def test_floor_label_strips_plan_suffix() -> None:
    assert floor_label_for_title("LEVEL 24 FLOOR PLAN") == "LEVEL 24"
    assert floor_label_for_title("LOWER LOBBY FLOOR PLAN") == "Lower Lobby"
