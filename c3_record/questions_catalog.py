"""The universal question list the base asks of every deal (``questions.json``).

Each entry is a question text; the base's Questions head (``jdg:gap``) reads
the whole deal and says whether THIS deal needs it asked and nothing answers
it yet. Adding a trade or a company's questions is adding text here: the head
reads the wording, so no retrain is needed to ask it (labeling then sharpens
it). ``layer`` keeps base questions (any field-services company) apart from
company ones."""
from __future__ import annotations

import functools
import json
from pathlib import Path


@functools.lru_cache(maxsize=1)
def catalog() -> tuple[dict, ...]:
    try:
        data = json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8"))
    except Exception:
        return ()
    return tuple(q for q in data.get("questions") or [] if q.get("text"))


def texts(layers: tuple[str, ...] = ("base", "purtera")) -> list[str]:
    return [q["text"] for q in catalog() if q.get("layer", "base") in layers]


def by_text() -> dict[str, dict]:
    return {q["text"]: q for q in catalog()}
