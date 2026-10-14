"""Write a checkpoint without the frozen pretrained encoder, for uploads.

    python -m ml.c3.slim --checkpoint runA.heads.pt --out runA.heads.slim.pt

The encoder is downloaded here and every saved encoder tensor is compared
with it; only tensors that match exactly are left out (a trained layer
stays in). The slim file records a sha256 of what it left out, and
``gates.load_checkpoint`` refills it from the reader's own download and
refuses when that download differs. Prints sizes by part, never weights.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import torch

from .gates import slim_state
from .loop import live_student
from .schema import load_schema


def _mb(state: dict[str, torch.Tensor]) -> dict[str, float]:
    seen, out = set(), defaultdict(float)
    for k, v in state.items():
        ptr = (v.untyped_storage().data_ptr(), v.storage_offset())
        if ptr in seen:
            continue
        seen.add(ptr)
        out[k.split(".")[0]] += v.numel() * v.element_size() / 1e6
    return dict(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    args = p.parse_args()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    fresh = live_student(load_schema(), args.encoder)
    print("before MB " + " ".join(f"{k}={v:.0f}" for k, v in sorted(_mb(payload["state"]).items())))
    slim, body = slim_state(payload["state"], fresh)
    if not body:
        raise SystemExit("No saved encoder tensor matches the download; nothing to leave out")
    print("after MB " + " ".join(f"{k}={v:.0f}" for k, v in sorted(_mb(slim).items())))
    payload = dict(payload, state=slim, pretrained_body=body)
    torch.save(payload, args.out)
    print(f"left out {body['tensors']} encoder tensors (sha256 {body['sha256'][:12]}); "
          f"wrote {args.out} {os.path.getsize(args.out) / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
