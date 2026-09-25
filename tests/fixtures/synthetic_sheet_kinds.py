"""Synthetic worksheets for the structural sheet head (PUR-51/52).

Entirely generated: no real customer, deal or document content. Each kind has
its own structure (header vocabulary, column types, row shape). Header words
are deliberately chosen to MISS the rule classifier's scope vocabulary so every
generated sheet is a rules fallthrough -- the population the head must learn.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

KINDS = ("pricing", "contact_list", "schedule", "junk", "scope")

_HEADERS = {
    "pricing": [["Line", "Service", "Rate", "Amount"], ["Ref", "Offering", "Rate", "Amount", "Discount"]],
    "contact_list": [["Name", "Title", "Email", "Phone"], ["Contact", "Role", "Email", "Mobile"]],
    "schedule": [["Task", "Start", "Finish", "Owner"], ["Phase", "Start", "Finish", "Duration"]],
    "junk": [["Customer Code", "Order Date", "Status", "Region"], ["Code", "Order Date", "Status", "Batch"]],
    "scope": [["Item", "Work", "Notes"], ["Step", "Work", "Notes"]],
}


def _row(kind: str, rng: random.Random, width: int) -> list[str]:
    if kind == "pricing":
        base = [str(rng.randint(1, 99)), "svc " + rng.choice("abcdef"), f"${rng.randint(50, 400)}.00", f"${rng.randint(500, 9000)}.00"]
        extra = [f"{rng.randint(0, 20)}%"]
    elif kind == "contact_list":
        n = rng.choice(["alex", "sam", "jo", "kim"])
        base = [f"{n} person", rng.choice(["pm", "tech", "lead"]), f"{n}@example.invalid", f"555-01{rng.randint(10, 99)}"]
        extra = []
    elif kind == "schedule":
        base = [f"phase {rng.randint(1, 9)}", f"2030-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}", f"2030-0{rng.randint(1, 9)}-2{rng.randint(0, 8)}", rng.choice(["a", "b"])]
        extra = []
    elif kind == "junk":
        base = [f"C{rng.randint(10000, 99999)}", f"2029-1{rng.randint(0, 2)}-0{rng.randint(1, 9)}", rng.choice(["Open", "Closed"]), rng.choice(["N", "S"])]
        extra = []
    else:
        base = [str(rng.randint(1, 50)), "install the " + rng.choice(["rack", "panel", "mount"]) + " in the closet", "per plan " + rng.choice("xyz")]
        extra = []
    return (base + extra)[:width]


def make_rows(kind: str, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    header = rng.choice(_HEADERS[kind])
    n = rng.randint(6, 30) if kind != "junk" else rng.randint(40, 120)
    return [list(header)] + [_row(kind, rng, len(header)) for _ in range(n)]


def write_corpus(root: Path, *, customers: tuple[str, ...] = ("cust_a", "cust_b", "cust_c"), per_kind: int = 4) -> dict[str, str]:
    """Write ``root/<customer>/wb_<kind>_<i>.csv``; return {relative path: kind}."""
    truth: dict[str, str] = {}
    seed = 0
    for cust in customers:
        for kind in KINDS:
            for i in range(per_kind):
                seed += 1
                p = root / cust / f"wb_{kind}_{i}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("w", newline="") as fh:
                    csv.writer(fh).writerows(make_rows(kind, seed))
                truth[str(p.relative_to(root))] = kind
    return truth
