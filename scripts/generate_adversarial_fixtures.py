from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.testing.scenarios import default_mutations, generate_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic adversarial fixture scenarios")
    parser.add_argument("--count", type=int, default=5, help="Number of scenarios to generate")
    parser.add_argument("--seed", type=int, default=1000, help="Base seed")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    created: list[dict[str, object]] = []
    for i in range(args.count):
        scenario_seed = args.seed + i
        mutation_set = default_mutations(scenario_seed)
        path = generate_scenario(scenario_seed, mutation_set, output_root=args.out)
        created.append({"seed": scenario_seed, "path": str(path), "mutations": mutation_set})

    manifest = {"count": args.count, "seed": args.seed, "scenarios": created}
    (args.out / "adversarial_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {args.count} scenarios in {args.out}")


if __name__ == "__main__":
    main()
