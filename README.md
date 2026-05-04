# Parser-OS

**Deterministic evidence compiler.** Turns messy project artifacts (RFP PDFs,
vendor quote XLSXs, addendum DOCXs, kickoff transcripts, customer emails)
into a structured graph of evidence packets that downstream consumers
(OrbitBrief, review tools, dashboards) can read.

```
artifacts → atoms → entities → graph → packets
```

Every packet ships with its source-text provenance, a confidence score, and
a stable hash. Re-running the compile on identical input always produces
byte-identical output.

## What it is

A local, deterministic Python pipeline (no LLMs in the hot path) that:

- **Parses** PDFs, XLSXs, DOCXs, emails, and transcripts into evidence atoms
- **Resolves** entities (sites, vendors, devices, part numbers, quantities)
  using domain packs (security_camera, wireless, av, bms, etc.)
- **Builds** an evidence graph with `supports`, `contradicts`, `excludes`,
  `requires`, and `same_as` edges
- **Packetizes** the graph into eleven families (scope_inclusion,
  scope_exclusion, quantity_conflict, customer_override, compliance_clause,
  meeting_decision, action_item, site_access, missing_info, vendor_mismatch,
  quantity_claim)
- **Emits** an OrbitBrief envelope (versioned `orbitbrief.input.v2` schema)
  plus a per-compile review folder

Produces explainable, regression-safe output. Every atom carries a
source-replay receipt that links it back to the exact text span (and page)
of the original artifact.

## What it isn't

No chatbot. No SOW generator. No dispatch system. No vision QC. No LLM
in the parsing path.

## Quickstart

Requires Python 3.12+.

```bash
pip install -e ".[dev]"
parser-os health         # smoke test — should print "ok"
```

Compile a project:

```bash
parser-os compile <project_dir> --out result.json --review-out reviews/
```

Run the gold-compare grid across your corpus:

```bash
parser-os matrix --cases-dir real_data_cases \
    --out matrix.json --markdown-out matrix.md
```

CLI reference: `parser-os --help` for the full surface (`compile`,
`compare`, `init`, `batch-compile`, `matrix`, `orbitbrief-envelope`,
`health`).

## Project layout

A `parser-os` project is a directory with this shape:

```
my_project/
├── project.yaml             # optional — pin a domain pack, customer name, etc.
├── SOURCE_NOTES.md          # optional — context that helps pack auto-routing
├── .parserignore            # optional — extra glob patterns to skip
├── artifacts/               # the source documents (PDFs, XLSXs, DOCXs, …)
└── labels/                  # optional — gold_standard.json for `compare`
    └── gold_standard.json
```

Scaffold a fresh one with `parser-os init <dir>`.

## Repo map

| Path | Purpose | Docs |
|---|---|---|
| `app/core/` | Pipeline stages (parse → atoms → graph → packets) | [app/core/README.md](app/core/README.md) |
| `app/parsers/` | Per-format parsers (PDF, XLSX, DOCX, email, transcript) | [app/parsers/README.md](app/parsers/README.md) |
| `app/domain/` | Domain packs + project config + auto-routing | [app/domain/README.md](app/domain/README.md) |
| `app/api/` | FastAPI routes (`/compile`, `/packets`, …) | inline docstrings |
| `app/eval/` | Gold-comparison and benchmark utilities | inline docstrings |
| `app/learning/` | Calibration, rule mining, active learning | inline docstrings |
| `tests/` | 110+ regression tests | [tests/README.md](tests/README.md) |
| `real_data_cases/` | Stress-test corpus + gold standards | [real_data_cases/README.md](real_data_cases/README.md) |
| `scripts/` | Utility CLIs (benchmarks, fixture builders, review tools) | [scripts/README.md](scripts/README.md) |

The full architecture overview lives in [app/README.md](app/README.md).

## CLI cheat sheet

```bash
# Compile a single project
parser-os compile <project_dir> --out result.json [--review-out reviews/]

# Compile many projects in one shot
parser-os batch-compile --cases-dir real_data_cases --out-dir batch_out/

# Compare a compile result against a gold standard
parser-os compare --gold real_data_cases/X/labels/gold_standard.json \
                  --compiled result.json

# Run the full case grid (regression spotter)
parser-os matrix --cases-dir real_data_cases \
                 --out matrix.json --markdown-out matrix.md

# Render the OrbitBrief input envelope from a saved compile result
parser-os orbitbrief-envelope --result result.json --out-dir envelope/

# Scaffold a new project directory
parser-os init <project_dir> --service-line security_camera --customer acme_corp
```

## Tests

```bash
pytest                                  # full suite (~110 tests)
pytest tests/test_entity_extraction.py  # one module
pytest tests/test_week6_dx.py -k compliance   # one keyword
```

See [tests/README.md](tests/README.md) for the test architecture.

## Determinism + provenance contract

Two operating guarantees the rest of the system relies on:

1. **Deterministic** — running `compile` twice on the same artifacts yields
   byte-identical `output_signature` and `compile_id` hashes. No randomness,
   no time-of-day dependency, no global state.
2. **Provenanced** — every atom has a `SourceRef` that pins it to a specific
   page / row / line in a specific artifact, plus an optional replay receipt
   that re-extracts the same text from the source on demand.

If you make a change that breaks either of those, you've broken the system
contract. The regression suite catches the obvious cases.

## License

See `pyproject.toml` for current metadata.
