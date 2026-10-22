"""Average the weights of several checkpoints from one run.

    python -m ml.c3.average --out runC.avg.pt runC.pt.lines@8 runC.pt.lines@12 runC.pt.lines@16

Checks along one run sit in the same basin, so their mean often reads more
steadily than any single one when validation swings from check to check.
Every file must hold the same tensors with the same shapes and the same
schema; float tensors are averaged, anything else is taken from the first.
A slim checkpoint's left-out encoder record is kept when all agree.
"""
from __future__ import annotations

import argparse

import torch


def _load(path: str) -> dict:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def average(paths: list[str]) -> dict:
    if len(paths) < 2:
        raise ValueError("Average at least two checkpoints")
    first = _load(paths[0])
    keys = set(first["state"])
    total = {k: (v.double() if v.is_floating_point() else v) for k, v in first["state"].items()}
    step = int(first["step"])
    for path in paths[1:]:
        p = _load(path)
        if p.get("schema_sha256") != first.get("schema_sha256"):
            raise ValueError(f"{path} was saved against a different schema")
        if set(p["state"]) != keys:
            raise ValueError(f"{path} holds different tensors")
        if p.get("pretrained_body") != first.get("pretrained_body"):
            raise ValueError(f"{path} left out a different encoder")
        step = max(step, int(p["step"]))
        for k, v in p["state"].items():
            if v.shape != total[k].shape:
                raise ValueError(f"{path}: {k} has a different shape")
            if v.is_floating_point():
                total[k] += v.double()
    n = len(paths)
    state = {k: (v / n).to(first["state"][k].dtype) if first["state"][k].is_floating_point() else v
             for k, v in total.items()}
    return dict(first, state=state, step=step)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args()
    torch.save(average(args.checkpoints), args.out)
    print(f"averaged {len(args.checkpoints)} checkpoints into {args.out}", flush=True)


if __name__ == "__main__":
    main()
