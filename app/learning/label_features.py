"""Facts a sentence encoder cannot see, handed to the head as numbers.

Measured, on the strings the ingest actually produces (bge-small, 384d):

    "…I will get a conversation going…  [from: cdw.com (reseller, theirs) -> purtera-it.com (internal, ours)]"
    "…I will get a conversation going…  [from: purtera-it.com (internal, ours) -> cdw.com (reseller, theirs)]"
                                                                            cosine 0.9967

Those two strings carry opposite answers for ``blocked_on`` -- us when a
reseller writes it to us, them when we write it to them -- and a mean-pooled
embedding puts them on top of each other. Every shape was tried: short suffix,
short prefix, plain English ("They wrote to us:"), role words, prefix *and*
suffix. The best was 0.9925. All collapsed.

The fault is not the labeling and not the head. A categorical fact laundered
through a sentence encoder arrives as noise: two words inside a hundred move a
mean-pooled vector by nothing. So it stops being text. The head is numpy over
a vector, and these facts are appended to that vector as a small explicit
block, where a linear projection can weight them directly.

(A supervised-contrastive fine-tune of the encoder CAN learn to attend to the
tag, because its loss punishes exactly this collapse -- see
``runpod_detector/train_contrastive_encoder_gpu.py``. That is the better fix
and it needs a GPU and thousands of rows. This one costs nothing and works
today, and the two compose.)

The block is deliberately tiny and categorical. Anything the words already
carry stays in the words.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

#: How much of the augmented vector the feature block is worth.
#:
#: An L2-normalized 384-dim embedding spreads ~1.0 of norm over 384 dims, so a
#: raw 1.0 one-hot would swamp the text entirely. This is the fraction of the
#: final vector's norm given to the whole block, chosen by sweep (see
#: tests/test_label_features.py): it must pull the two `blocked_on` strings
#: apart without collapsing the distance between genuinely similar sentences.
FEATURE_WEIGHT = 0.45

_SIDES = ("ours", "theirs")
_ROLES = ("internal", "reseller", "vendor", "manufacturer", "installer", "customer")

#: Ordered, and the order is the contract: a head fitted on this block reads
#: the same dimensions at query time. Appending is safe, reordering is not.
FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"from_side:{s}" for s in _SIDES)
    + ("from_side:unknown",)
    + tuple(f"to_side:{s}" for s in _SIDES)
    + ("to_side:unknown",)
    + tuple(f"from_role:{r}" for r in _ROLES)
    + ("from_role:unknown",)
    + ("internal_only", "crosses_sides", "has_lead_in", "has_section", "has_table", "has_below")
)
FEATURE_DIM = len(FEATURE_NAMES)


def _side(p: Any) -> str:
    v = str((p or {}).get("side") or "").strip().lower() if isinstance(p, dict) else ""
    return v if v in _SIDES else "unknown"


def _role(p: Any) -> str:
    v = str((p or {}).get("role") or "").strip().lower() if isinstance(p, dict) else ""
    return v if v in _ROLES else "unknown"


def _first(v: Any) -> Any:
    if isinstance(v, list) and v and isinstance(v[0], dict):
        return v[0]
    return None


def features_for(label: dict[str, Any]) -> list[float]:
    """The block for one label row, in ``FEATURE_NAMES`` order."""
    by = label.get("said_by") if isinstance(label.get("said_by"), dict) else None
    to = _first(label.get("said_to"))
    from_side, to_side, from_role = _side(by), _side(to), _role(by)
    on = {
        f"from_side:{from_side}",
        f"to_side:{to_side}",
        f"from_role:{from_role}",
    }
    if label.get("internal_only"):
        on.add("internal_only")
    # The direction itself, as one dimension: a projection can use it without
    # having to compose two one-hots.
    if from_side in _SIDES and to_side in _SIDES and from_side != to_side:
        on.add("crosses_sides")
    if label.get("lead_in"):
        on.add("has_lead_in")
    if label.get("section"):
        on.add("has_section")
    if label.get("table_ref"):
        on.add("has_table")
    if label.get("neighbors_below"):
        on.add("has_below")
    return [1.0 if name in on else 0.0 for name in FEATURE_NAMES]


def augment(X: np.ndarray, F: Sequence[Sequence[float]] | np.ndarray,
            *, weight: float = FEATURE_WEIGHT) -> np.ndarray:
    """``[embedding * (1-w) | features * w]``, L2-normalized.

    The head's whole geometry is cosine, so both parts must be scaled before
    the concatenation or the longer one decides everything.
    """
    X = np.asarray(X, dtype=np.float32)
    F = np.asarray(F, dtype=np.float32)
    if F.ndim == 1:
        F = F.reshape(1, -1)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if F.shape[0] != X.shape[0] or F.shape[1] != FEATURE_DIM:
        raise ValueError(f"feature block must be ({X.shape[0]}, {FEATURE_DIM}), got {F.shape}")
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    Fn = F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-12, None)
    Z = np.concatenate([Xn * (1.0 - weight), Fn * weight], axis=1).astype(np.float32)
    return Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12, None)


__all__ = ["FEATURE_NAMES", "FEATURE_DIM", "FEATURE_WEIGHT", "features_for", "augment"]
