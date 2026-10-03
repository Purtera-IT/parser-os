"""Print the model as threads should reason about it: spaces, heads, the
labeling opportunities each head reads, and what the fixture deal teaches.

    python -m ml.c3.card                      # architecture + opportunity map
    python -m ml.c3.card --deal path.json     # plus per-opportunity target counts
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from .schema import load_schema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deal", help="a deal in the ml/c3 JSON contract")
    ap.add_argument("--params", action="store_true", help="build the small model and count parameters")
    ap.add_argument("--guidance", help="a guidance JSON to fold into the descriptions")
    ap.add_argument("--guidance-template", action="store_true",
                    help="print a blank guidance file covering every question and answer")
    args = ap.parse_args()

    schema = load_schema(guidance=args.guidance)
    if args.guidance_template:
        import json

        from .schema import guidance_template

        print(json.dumps(guidance_template(schema), indent=2))
        return
    for key, sentence in schema.scrubbed:
        print(f"dropped from universal guidance on {key} (names company policy): {sentence}")
    counts: dict[str, int] = {}
    if args.deal:
        from .data import IGNORE, DealExample, featurize

        batch = featurize(DealExample.load(args.deal), schema)
        counts = {k: sum(v != IGNORE for v in col) for k, col in batch.targets.items()}
        for rel, pairs in batch.edges.items():
            counts[f"rel:{rel}"] = len(pairs)

    by_head: dict[str, list] = defaultdict(list)
    for o in schema.opportunities:
        by_head[o.head].append(o)
    print(f"C3 schema {schema.version}: {len(schema.opportunities)} labeling opportunities\n")
    for space, info in schema.spaces.items():
        heads = [h for h in by_head if by_head[h][0].space == space]
        if not heads:
            continue
        print(f"== {space}: {info.get('title', '')}")
        for h in heads:
            opps = by_head[h]
            print(f"   {h}  [{opps[0].layer}]")
            for o in opps:
                n = f"  targets={counts[o.key]}" if o.key in counts else ""
                print(f"      {o.key:<28} {o.kind:<9} answers={len(o.answers):<3}{n}")
        print()
    if args.params:
        from .model import C3Model

        for k, v in C3Model(schema).parameter_report().items():
            print(f"   {k:<16} {v:>12,}")


if __name__ == "__main__":
    main()
