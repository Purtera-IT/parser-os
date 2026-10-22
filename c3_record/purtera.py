"""The Purtera layer's own facts (purtera_layer.json): who Purtera's partners
are. The universal supplier head answers "third_party" for a reseller such as
CDW; to Purtera that reseller is a partner. Until supplier is retrained in the
company layer, this maps the base's answer for Purtera."""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path


@functools.lru_cache(maxsize=1)
def partners() -> tuple[str, ...]:
    try:
        return tuple(json.loads(Path(__file__).with_name("purtera_layer.json").read_text(encoding="utf-8"))["partners"])
    except Exception:
        return ()


def supplier_for_purtera(answer: str, text: str, context: str = "") -> str:
    """The base's supplier answer as Purtera reads it: a third party named as
    one of Purtera's partners (in the line or its document) is ``partner``."""
    if answer != "third_party":
        return answer
    blob = f"{text} {context}"
    if any(re.search(rf"\b{re.escape(p)}\b", blob, re.I) for p in partners()):
        return "partner"
    return answer
