"""Training coefficients, kept outside normalized background row weights.

Caps bound nominal loss coefficient mass over a pass, not gradient norms or
Adam updates (which also depend on the examples, clipping and optimizer state).
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict


def cap_scales(mass: dict[str, float], cap: float, *, name: str = "deal") -> dict[str, float]:
    """Downscale large sources to at most ``cap`` of effective coefficient mass.

    A sparse head with fewer than ceil(1/cap) supported sources cannot satisfy
    the cap. Report that fact and leave its learning intact. Zero-mass sources
    keep 1, never a zero that could later silence supervision.
    """
    if not math.isfinite(cap) or not 0 < cap <= 1:
        raise ValueError("cap must be finite and in (0, 1]")
    if any(not math.isfinite(v) or v < 0 for v in mass.values()):
        raise ValueError("cap mass must be finite and nonnegative")
    live = {k: v for k, v in mass.items() if v > 0}
    scales = dict.fromkeys(mass, 1.0)
    if not live:
        warnings.warn(f"{name} cap {cap:g} infeasible: no supported source deals; unchanged", RuntimeWarning)
        return scales
    if len(live) * cap < 1 - 1e-12:
        warnings.warn(f"{name} cap {cap:g} infeasible: {len(live)} supported source deals; "
                      "learning unchanged", RuntimeWarning)
        return scales
    # Highest clipping threshold satisfying max(mass) / sum(mass) <= cap.
    lo, hi = 0.0, max(live.values())
    for _ in range(100):
        mid = (lo + hi) / 2
        total = sum(min(v, mid) for v in live.values())
        if mid <= cap * total * (1 + 1e-14):
            lo = mid
        else:
            hi = mid
    for key, value in live.items():
        scales[key] = min(1.0, lo / value)
    return scales


def head_cap_scales(train, schema, part_scale, caps, head_weight=None):
    """Per-part head multipliers on the actual selected parts, across all views.

    Account for the heads-average divisor and explicit part scale. The returned
    coefficient applies to direct categorical head CE/BCE; auxiliary losses
    retain their own supervision policy. No data/labels or view masks change.
    """
    from .losses import has_answers
    from .loop import source_deal_id

    weights = head_weight or {}
    result = {b.deal_id: {} for b in train}
    for key, cap in caps.items():
        opp = schema.by_key()[key]
        mass = defaultdict(float)
        supported = []
        for batch in train:
            view = batch.head_scale.get(key, 1.0)
            if not view or not has_answers(batch, key):
                continue
            denom = max(sum(weights.get(o.key, 1.0) for o in schema.select(layer=opp.layer)
                            if batch.head_scale.get(o.key, 1.0) and has_answers(batch, o.key)), 1.0)
            coefficient = part_scale[batch.deal_id] * weights.get(key, 1.0) * view / denom
            if coefficient <= 0:
                continue
            source = source_deal_id(batch.deal_id)
            mass[source] += coefficient
            supported.append((batch.deal_id, source))
        factors = cap_scales(dict(mass), cap, name=key)
        for part, source in supported:
            result[part][key] = factors[source]
    return result
