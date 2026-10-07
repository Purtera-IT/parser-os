"""The closed list of reason codes per judgment head -- ``judgment_reasons.json``.

``atom_label_judgments.reason`` trains ``<relation>_reason`` as a class. A free
sentence there is a class with one example, which no head can learn, so only a
code listed for the judgment's head counts as a reason. Platform-infra vendors
the same file (``azure-function-api/shared/judgment-reasons.json``) and checks
every save against it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REASONS_PATH = Path(__file__).with_name("judgment_reasons.json")


@lru_cache(maxsize=1)
def load_reasons() -> dict[str, Any]:
    return json.loads(REASONS_PATH.read_text(encoding="utf-8"))


def codes_for(head: str) -> frozenset[str]:
    """The reason codes a judgment on ``head`` may carry (empty for an unlisted head)."""
    return frozenset(c["value"] for c in load_reasons()["heads"].get(str(head or ""), []))


def is_reason_code(head: str, reason: Any) -> bool:
    return str(reason or "").strip() in codes_for(head)
