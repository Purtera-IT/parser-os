from __future__ import annotations

import sys
from app.core.item_identity import canonical_item_identity

examples = sys.argv[1:] or [
    "68 data drops",
    "68 comm outlets",
    "68 RJ45 terminations",
    "60 unshielded category 6 runs",
    "8 shielded Cat6 cable drops",
    "4 20 amp power locations",
    "Cable certification report exports",
    "District to verify existing raceway",
]

for text in examples:
    result = canonical_item_identity({"description": text}, allow_multi=True)
    print("\n", text)
    if not result:
        print("  NO MATCH")
        continue
    for r in result:
        print(f"  {r.canonical_key:24s} conf={r.confidence:.2f} kind={r.item_kind} by={r.matched_by}")
