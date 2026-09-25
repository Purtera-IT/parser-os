"""Generate SYNTHETIC labelled work orders for Division proposal (PUR-29).

No real deal data. Raw label spellings follow the PM value frequencies
(Network 29, Networking 17, EUC 16, IMAC 16, Camera Install 8, AV 6,
Network/LV 6, Cable 6, Staff Aug 4, Cabling 3, palletize 3), scaled by
--scale so every class survives a stratified holdout. The vocabulary below is
only a generator for fake work orders; the proposer never sees it.

~12% of examples mix work from two divisions and carry ``"ambiguous": true``.
"""
from __future__ import annotations

import argparse
import json
import random

FREQ = {"Network": 29, "Networking": 17, "EUC": 16, "IMAC": 16, "Camera Install": 8, "AV": 6,
        "Network/LV": 6, "Cable": 6, "Staff Aug": 4, "Cabling": 3, "palletize": 3}

# division family -> (objects, actions, units)
WORK = {
    "net": (["access point", "network switch", "firewall", "router", "wireless controller", "console server"],
            ["install", "configure", "mount", "replace", "stage"], ["device", "ap", "unit"]),
    "netlv": (["access point", "network switch", "low voltage drop", "patch panel", "data drop"],
              ["install", "terminate", "pull", "configure"], ["drop", "device", "run"]),
    "euc": (["laptop", "desktop", "monitor", "docking station", "user profile"],
            ["deploy", "image", "migrate", "refresh"], ["user", "seat", "device"]),
    "imac": (["workstation", "desk phone", "printer", "cubicle equipment"],
             ["move", "add", "change", "relocate", "disconnect"], ["move", "desk", "user"]),
    "cam": (["security camera", "ip camera", "nvr", "camera mount"],
            ["install", "aim", "mount", "commission"], ["camera", "device"]),
    "av": (["display", "projector", "conference room speaker", "video bar", "touch panel"],
           ["install", "mount", "commission", "calibrate"], ["room", "display"]),
    "cable": (["cat6 cable", "fiber run", "cable tray", "jack", "cable drop"],
              ["pull", "terminate", "certify", "label", "dress"], ["drop", "run", "foot"]),
    "staff": (["field technician", "onsite resource", "help desk analyst"],
              ["provide", "staff", "dispatch"], ["hour", "week", "resource"]),
    "pallet": (["pallet", "shipment", "hardware kit", "carton"],
               ["palletize", "shrink wrap", "stage", "ship", "receive"], ["pallet", "carton"]),
}
FAMILY = {"Network": "net", "Networking": "net", "EUC": "euc", "IMAC": "imac", "Camera Install": "cam",
          "AV": "av", "Network/LV": "netlv", "Cable": "cable", "Staff Aug": "staff", "Cabling": "cable",
          "palletize": "pallet"}


def _line(rng, fam):
    objs, acts, units = WORK[fam]
    obj = rng.choice(objs)
    return {"work": f"{rng.choice(acts)} {obj}s", "object": obj, "count": rng.choice([1, 2, 4, 8, 12, 24, 60, 150]),
            "unit": rng.choice(units)}


def generate(scale: int = 3, seed: int = 29, ambiguous_rate: float = 0.12) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for label, n in FREQ.items():
        for i in range(n * scale):
            fam = FAMILY[label]
            amb = rng.random() < ambiguous_rate
            lines = [_line(rng, fam) for _ in range(rng.randint(1, 3))]
            if amb:
                other = rng.choice([f for f in WORK if f != fam])
                lines = [_line(rng, fam)] + [_line(rng, other)]
                rng.shuffle(lines)
            out.append({
                "id": f"syn-{FAMILY[label]}-{len(out):04d}",
                "division": label,
                "ambiguous": amb,
                "work_order": {"work_lines": lines, "after_hours": rng.random() < 0.3,
                               "no_onsite_hands": fam in ("pallet",) and rng.random() < 0.5},
            })
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/division/synthetic_labelled.json")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seed", type=int, default=29)
    a = ap.parse_args()
    data = {"synthetic": True, "source": "scripts/make_synthetic_division_labels.py",
            "examples": generate(a.scale, a.seed)}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print(f"wrote {len(data['examples'])} synthetic examples to {a.out}")
