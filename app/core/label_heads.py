"""Which model head each label field trains -- ``app/core/label_heads.json``.

``atom_types.json`` says what a label may hold: the types, the readings, the
relations. This file says what each of them is FOR: the head of the C3 design
(docs/C3_HEADS.md) that a field supervises, the space that head
lives in, and whether it is universal or one company's.

Three consumers read it, so they cannot drift apart:

* the labeling page groups a card into one section per space, in this order
  (Platform-infra serves the JSON from ``/types``);
* the label API checks a save against it;
* the trainer tags every row with its head (``head_of_task``).

``tests/test_label_heads.py`` fails when a reading, relation, judgment head or
training task has no head, or has two.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

HEADS_PATH = Path(__file__).with_name("label_heads.json")

#: Placeholder in a head's ``tasks`` for every company profile.
COMPANY_SLOT = "{company}"


@lru_cache(maxsize=1)
def load_heads() -> dict[str, Any]:
    return json.loads(HEADS_PATH.read_text(encoding="utf-8"))


def heads() -> list[dict[str, Any]]:
    return list(load_heads()["heads"])


def head(key: str) -> dict[str, Any] | None:
    return next((h for h in load_heads()["heads"] if h["key"] == key), None)


def _owner(field: str, value: str) -> str | None:
    for h in load_heads()["heads"]:
        if value in h.get(field, []):
            return h["key"]
    return None


def head_of_read(key: str) -> str | None:
    return _owner("reads", key)


def head_of_relation(relation: str) -> str | None:
    return _owner("relations", relation)


def head_of_judgment(judgment_head: str) -> str | None:
    return _owner("judgments", judgment_head)


def head_of_task(task: str) -> str | None:
    """The head a training task belongs to, or None when nothing claims it.

    ``reads:<key>`` follows the reading; ``<task>_reason`` follows its task;
    a company task (``policy:purtera``) matches its ``{company}`` template.
    """
    if task.startswith("reads:"):
        return head_of_read(task[len("reads:"):])
    for h in load_heads()["heads"]:
        for t in h.get("tasks", []):
            if t == task:
                return h["key"]
            if COMPANY_SLOT in t:
                pre, post = t.split(COMPANY_SLOT, 1)
                mid = task[len(pre):len(task) - len(post)] if post else task[len(pre):]
                if task.startswith(pre) and task.endswith(post) and mid and ":" not in mid:
                    return h["key"]
    if task.endswith("_reason"):
        return head_of_task(task[: -len("_reason")])
    return None


def space_of_task(task: str) -> str | None:
    key = head_of_task(task)
    h = head(key) if key else None
    return h["space"] if h else None


__all__ = [
    "COMPANY_SLOT",
    "HEADS_PATH",
    "head",
    "head_of_judgment",
    "head_of_read",
    "head_of_relation",
    "head_of_task",
    "heads",
    "load_heads",
    "space_of_task",
]
