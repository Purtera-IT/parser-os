# `scripts/` — utility CLIs

Operational utilities that sit alongside the main `parser-os` CLI.
Each script is self-contained — pass `--help` for its own usage.

## Production utilities

| Script | What it does |
|---|---|
| `fetch_stress_test_corpus.sh` | Downloads the public source PDFs / XLSXs into each `real_data_cases/STRESS_*/artifacts/` subdir. Run after a fresh clone. |
| `init_real_data_case.py` | Scaffolds a new stress case (alternative to `parser-os init`). |
| `compile_real_data_case.py` | Compiles a single `real_data_cases/<case>/` and writes the result + review folder. Thin wrapper around `parser-os compile` that adds case-specific logging. |
| `summarize_real_data_cases.py` | Prints a one-line summary of every case in `real_data_cases/` (artifact count, gold-standard presence, last-compile result). |
| `summarize_failures.py` | Aggregates compile errors across multiple cases. |
| `compare_compiles.py` | Diffs two `CompileResult` JSONs (atom-level + packet-level). |
| `compare_gold_packets.py` | Compares packet output against an explicit `gold_packets.json` (per-packet expectations rather than aggregate metrics). |

## Benchmark + perf

| Script | What it does |
|---|---|
| `run_perf_benchmark.py` | End-to-end compile timing across N cases. Prints stage-by-stage durations. |
| `run_packetizer_benchmark.py` | Times the packetizer in isolation across synthetic atom-set sizes. |
| `run_parser_benchmark.py` | Per-format parser timing on a representative file set. |

## Review + labeling

| Script | What it does |
|---|---|
| `review_packets.py` | Interactive packet-by-packet review CLI. Marks packets approved / rejected and writes back to the case's `labels/packet_reviews.json`. |
| `build_review_queue.py` | Builds a prioritized review queue from a compile result (high-risk packets first). |
| `label_packets.py` | Bulk-applies labels from a YAML to packets that match a query. |
| `inspect_packets.py` | Pretty-prints packets matching a filter (family, anchor_key prefix, atom_id, …). |
| `promote_review_to_fixture.py` | Copies a reviewed packet into `tests/fixtures/` as a regression fixture. |

## Experimentation

| Script | What it does |
|---|---|
| `run_extraction_experiment.py` | Runs an extraction-experiment sandbox (varying parser config / pack version) and records results under `experiments/`. |
| `freeze_extractor_output.py` | Snapshots the current output of an experiment to lock it in. Used by `test_freeze_*` regression tests. |
| `run_adversarial_lab.py` | Generates mutated artifacts via `app/testing/mutators.py` and runs them through the pipeline; flags any new failure modes. |
| `generate_adversarial_fixtures.py` | Mines real cases for adversarial inputs and writes them under `tests/fixtures/adversarial/`. |

## Pack maintenance

| Script | What it does |
|---|---|
| `certify_domain_pack.py` | Scores how well a pack covers a sample of cases — coverage table per `device_aliases` / `vendor` / `site` category. Run before promoting a pack from draft to production. |
| `mine_rule_suggestions.py` | Mines compile output for `RuleSuggestion` candidates (new exclusion patterns, new device aliases, new vendor surfaces). Writes them to a YAML the operator can review. |
| `apply_approved_suggestion.py` | Applies an approved `RuleSuggestion` from the suggestion YAML into the appropriate pack file. |

## ML / calibration

| Script | What it does |
|---|---|
| `train_calibrator.py` | Trains the per-parser confidence calibrator (`app/learning/calibration.py`). |
| `evaluate_calibrator.py` | Reports calibration quality (Brier score, reliability diagram bins) on a held-out set. |

## Other

| Script | What it does |
|---|---|
| `make_demo_fixtures.py` | (Re-)generates `tests/fixtures/demo_project/` from canonical templates. Run if a fixture format changes. |
| `evidence_coverage_report.py` | Reports what fraction of source-text characters became evidence atoms (per artifact, per page). |
| `overlay_extract_pages.sh` | One-off PDF utility for extracting overlay text from selected page ranges. |

## When to add a new script vs. extend the CLI

- **CLI subcommand (`parser-os <name>`)** — when the operation is part
  of the standard production workflow that operators run regularly
  (compile, compare, init, batch-compile, matrix, …).
- **`scripts/`** — when it's a developer tool, a one-off corpus utility,
  or an analysis script. Production users shouldn't have to know it
  exists.

If a `scripts/` utility becomes load-bearing, promote it to a CLI
subcommand under `app/cli.py` and keep the script as a thin wrapper for
backward compatibility.
