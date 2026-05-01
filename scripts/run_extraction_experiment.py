from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.experiments.sandbox import run_extraction_sandbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probabilistic extraction sandbox experiment.")
    parser.add_argument("--project", type=Path, required=True, help="Project directory path")
    parser.add_argument(
        "--extractor",
        required=True,
        choices=["semantic_linker", "llm_candidate_extractor", "weak_supervision_rules"],
        help="Experimental extractor to run in sandbox mode",
    )
    parser.add_argument("--extractor-version", default="exp_v1", help="Extractor version label")
    parser.add_argument("--domain-pack", default=None, help="Optional domain-pack id/path")
    parser.add_argument("--out", type=Path, required=True, help="Output experiment report JSON")
    args = parser.parse_args()

    run, report = run_extraction_sandbox(
        project_dir=args.project,
        extractor_name=args.extractor,
        extractor_version=args.extractor_version,
        domain_pack=args.domain_pack,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment_id": run.experiment_id,
                "compile_id": run.compile_id,
                "candidate_count": run.candidate_count,
                "delta_vs_baseline": run.delta_vs_baseline.model_dump(mode="json"),
            }
        )
    )


if __name__ == "__main__":
    main()
