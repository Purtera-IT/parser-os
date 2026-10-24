"""Training coefficients, kept outside normalized background row weights.

Caps bound nominal loss coefficient mass over a pass, not gradient norms or
Adam updates (which also depend on the examples, clipping and optimizer state).
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict


def validate_balance_metadata(version, config, *, deal_cap=None, head_caps=None, source="input"):
    """Caps require rebuilt inputs declaring the exact requested cap policy.

    No caps means the legacy/direct-call contract stays unchanged. Metadata is
    validated, never inferred from row weights (which also encode background).
    """
    if deal_cap is None and not head_caps:
        return
    expected = {"deal_cap": deal_cap, "head_caps": head_caps or {}}
    def valid_cap(value):
        return type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1

    if ((deal_cap is not None and not valid_cap(deal_cap))
            or any(not valid_cap(v) for v in (head_caps or {}).values())):
        raise ValueError("runtime caps must be finite and in (0, 1]")
    if type(version) is not int or version != 1:
        raise ValueError(f"{source}: runtime caps require rebuilt balance_version=1 inputs; "
                         "legacy baked-in factors must be removed by rebuilding")
    valid = (isinstance(config, dict) and set(config) == {"deal_cap", "head_caps"}
             and (config["deal_cap"] is None or valid_cap(config["deal_cap"]))
             and isinstance(config["head_caps"], dict)
             and all(isinstance(k, str) and valid_cap(v) for k, v in config["head_caps"].items()))
    if not valid or config != expected:
        raise ValueError(f"{source}: balance_config must match runtime caps {expected!r}; rebuild inputs "
                         "with the matching cap policy")


def validate_balance_inputs(train, *, deal_cap=None, head_caps=None):
    """Validate the actual loaded parts before scaling or model initialization."""
    for batch in train:
        validate_balance_metadata(getattr(batch, "balance_version", None),
                                  getattr(batch, "balance_config", None),
                                  deal_cap=deal_cap, head_caps=head_caps, source=batch.deal_id)


def check_balance_files(paths, *, deal_cap=None, head_caps=None):
    """Run E's early preflight on exactly the selected training files."""
    import json
    from pathlib import Path

    if deal_cap is None and not head_caps:
        return
    for path in paths:
        blob = json.loads(Path(path).read_text())
        validate_balance_metadata(blob.get("balance_version"), blob.get("balance_config"),
                                  deal_cap=deal_cap, head_caps=head_caps, source=str(path))


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fail closed on legacy or mismatched capped training inputs")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--deal-cap", type=float, default=None)
    parser.add_argument("--head-cap", action="append", default=[])
    args = parser.parse_args()
    head_caps = {}
    for spec in args.head_cap:
        key, sep, value = spec.rpartition("=")
        if not sep or not key:
            parser.error(f"invalid --head-cap {spec!r}; expected key=fraction")
        head_caps[key] = float(value)
    check_balance_files(args.train, deal_cap=args.deal_cap, head_caps=head_caps)
    print(f"balance input contract checked: {len(args.train)} training files")


if __name__ == "__main__":
    main()
