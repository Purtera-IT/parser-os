"""Verified Run E split roles; unknown policies fail closed.

The owner approved moving 010215 to training on 2026-10-23. Golden tests,
reserved tests, pending tests, published kits and selection remain protected.
The bundled versioned contract supplies the approved split on hosts without the
customer-local manifest. An explicit missing manifest, or any present malformed,
stale or unknown manifest, is an error; none can fall back to a weaker split.
This contract is split policy, not source/task/INIT admissibility or a data freeze.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable

CONTRACT_PATH = Path(__file__).with_name("runE") / "split_contract.json"
APPROVED_CONTRACT_SHA256 = "721bbbe89232ea751b757713d0f62fc42c6307ed0d1ae4bb167031b4be85cd2f"
ROLE_FIELDS = frozenset({"golden_holdout", "reserved_holdout", "pending_holdout",
                         "selection", "kit_holdout", "released_to_training"})


def load_split_contract(path: str | Path = CONTRACT_PATH) -> dict:
    """Only the exact reviewed semantic policy is supported, never arbitrary JSON."""
    try:
        contract = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("missing or invalid split contract") from exc
    if (not isinstance(contract, dict)
            or set(contract) != ROLE_FIELDS | {"schema_version", "policy_id"}
            or type(contract["schema_version"]) is not int or contract["schema_version"] != 1):
        raise ValueError("unknown split contract schema/version")
    seen = set()
    for field in sorted(ROLE_FIELDS):
        values = contract[field]
        if (not isinstance(values, list) or not values
                or not all(isinstance(v, str) and re.fullmatch(r"[0-9]{6}", v) for v in values)
                or len(values) != len(set(values)) or seen & set(values)):
            raise ValueError("invalid, duplicate or overlapping split roles: " + field)
        seen.update(values)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != APPROVED_CONTRACT_SHA256:
        raise ValueError("unknown/unreviewed split policy; a reviewed contract/code refresh is required")
    return contract


def verify_holdout_manifest(path: str | Path, contract: dict) -> frozenset[str]:
    """The actual manifest must match the approved reserved AND pending roles."""
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != APPROVED_CONTRACT_SHA256:
        raise ValueError("unknown/unreviewed split contract")
    try:
        blob = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("missing or invalid actual holdout manifest") from exc
    if not isinstance(blob, dict):
        raise ValueError("invalid actual holdout manifest")
    for name, role in (("deals", "reserved_holdout"), ("pending", "pending_holdout")):
        values = blob.get(name)
        if (not isinstance(values, list) or not all(isinstance(v, str) for v in values)
                or len(values) != len(set(values)) or set(values) != set(contract[role])):
            raise ValueError("actual holdout manifest differs from verified split: " + name)
    return frozenset(blob["deals"]) | frozenset(blob["pending"])


_contract_override = os.environ.get("C3_SPLIT_CONTRACT")
SPLIT_CONTRACT = load_split_contract(CONTRACT_PATH if _contract_override is None else _contract_override)
LABEL_HOLDOUT = tuple(SPLIT_CONTRACT["golden_holdout"])
KIT_HOLDOUT = tuple(SPLIT_CONTRACT["kit_holdout"])
HOLDOUT_B = tuple(SPLIT_CONTRACT["reserved_holdout"])
PENDING_HOLDOUT = tuple(SPLIT_CONTRACT["pending_holdout"])
SELECTION = frozenset(SPLIT_CONTRACT["selection"])


def _holdout_b_file() -> frozenset[str]:
    explicit = os.environ.get("C3_HOLDOUT2")
    path = Path(explicit) if explicit is not None else Path.home() / "c3-local/runE/holdout2.json"
    if explicit is not None or path.exists():
        return verify_holdout_manifest(path, SPLIT_CONTRACT)
    # Known bundled policy, not a guessed split when an explicit policy failed.
    return frozenset(HOLDOUT_B) | frozenset(PENDING_HOLDOUT)


FIXED_HOLDOUT = (frozenset(LABEL_HOLDOUT) | frozenset(KIT_HOLDOUT)
                 | _holdout_b_file() | SELECTION)


def deal_number(deal_id: str) -> str:
    """The source deal of a part, without a file suffix; callers remove @view."""
    head = str(deal_id or "").partition("#c3part")[0].strip()
    return re.sub(r"\.json$", "", head)


def refuse_fixed_holdout(train_ids: Iterable[str], *, allow_selection: bool = False) -> None:
    protected = FIXED_HOLDOUT - SELECTION if allow_selection else FIXED_HOLDOUT
    hit = sorted({deal_number(i).partition("@")[0] for i in train_ids} & protected)
    if hit:
        raise ValueError("A fixed holdout/selection deal is in the training set: " + ", ".join(hit))
