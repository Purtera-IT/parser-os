"""The one list of atom types -- ``app/core/atom_types.json``.

Before this file there were three lists that disagreed: the prod enum
(``schemas.AtomType``), the LLM prompt's ``_TAXONOMY`` and the heads' label
space (``runpod_detector/taxonomy.MICRO_TO_FACET``). ``work_scope_item`` and
``rate_card`` lived in the label space and nowhere else, and adding a type
meant finding every list by hand.

The JSON is the source. ``tests/test_atom_type_registry.py`` fails when any
other list drifts from it, and its message names the file to change. The
human labeler (purpulse ``/pm/quoting/atom-labeling``) serves the same JSON,
so labelers pick from exactly the types the heads train on.

``status``:
  live  -- in the prod enum; the parser may emit it.
  v2    -- label-only: people can label it and heads can learn it, but the
           prod enum does not accept it yet. Promote by adding it to
           ``AtomType`` and flipping the status.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("atom_types.json")
KEEP = "_keep"


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def type_names(*, label_space: str | None = None, status: str | None = None) -> tuple[str, ...]:
    return tuple(
        t["name"] for t in load_registry()["types"]
        if (label_space is None or t["label_space"] == label_space)
        and (status is None or t["status"] == status)
    )


def facet_of(name: str) -> str | None:
    for t in load_registry()["types"]:
        if t["name"] == name:
            return t["facet"]
    return None


def coarse_of(name: str) -> str | None:
    if name == KEEP:
        return KEEP
    for t in load_registry()["types"]:
        if t["name"] == name:
            return t["coarse"]
    return None
