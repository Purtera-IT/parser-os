from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def _normalize_part(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip().lower())
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(_normalize_part(part) for part in parts)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def canonical_json_hash(obj: Any) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
