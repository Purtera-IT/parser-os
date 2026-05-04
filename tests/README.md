# `tests/` — regression test suite

110+ tests across 70+ test modules. Every public stage of the pipeline
has a focused module; the `test_week*_dx.py` modules lock in the
classifier / extractor improvements made during the production-hardening
sprint.

## Running

```bash
pytest                                        # full suite
pytest tests/test_entity_extraction.py        # one module
pytest tests/test_week6_dx.py -k compliance   # by keyword
pytest -x --tb=line                           # stop on first failure, terse output
pytest -k "not adversarial and not perf"      # skip slow tests
```

`pyproject.toml` pins `pytest>=8.0,<9` because pytest 9.x triggers
`STATUS_STACK_BUFFER_OVERRUN` on Windows / CPython 3.12.3 in some
environments; remove the cap when CI moves off Windows.

## Test module layout

| Pattern | What it covers |
|---|---|
| `test_<stage>.py` | One pipeline stage (`test_packetizer`, `test_graph_builder`, `test_source_replay`, `test_entity_resolution`, …) |
| `test_<parser>_parser.py` | One parser module (`test_xlsx_parser`, `test_docx_parser`, `test_quote_parser`, …) |
| `test_<format>_adversarial.py` | (under `tests/parser_adversarial/`) edge-case inputs that historically broke a parser |
| `test_real_data_compile.py` | Smoke-test compile against `real_data_cases/` |
| `test_compiler_e2e.py` | End-to-end compile happy path |
| `test_deterministic_replay.py` | Re-running a compile yields identical output |
| `test_week<N>_dx.py` | Locks in classifier / extractor improvements from week N (sprint-era — kept as living regression coverage) |
| `test_schemas.py` | Pydantic invariants on every shape |
| `test_anchor_signatures.py` | Anchor-key shape (per packet family) |
| `test_graph_invariants.py` | Edge validator |
| `test_packet_invalidation.py` + `test_incremental_compile.py` | Re-compile semantics |
| `test_evidence_coverage.py` | What fraction of source text becomes atoms |
| `test_failure_taxonomy.py` | Maps compile failures to known buckets |
| `test_authority.py` + `test_authority_lattice.py` | Authority-class precedence |
| `test_active_learning_queue.py` + `test_promotion_pipeline.py` + `test_rule_miner.py` | Learning loop tests |
| `test_api_smoke.py` + `test_api_hardening.py` | FastAPI routes |
| `test_review_console.py` + `test_review_queue.py` | Review-workflow tests |
| `test_perf_benchmark.py` + `test_packetizer_benchmark.py` | Soft perf budgets |
| `test_domain_certification.py` + `test_domain_pack_loading.py` | Pack health |
| `test_orbitbrief_envelope.py` + `test_orbitbrief_overlay_json_pipeline.py` | OrbitBrief envelope shape |

## Fixtures (`tests/fixtures/`)

| Path | Purpose |
|---|---|
| `tests/fixtures/demo_project/` | A small synthetic project used by `test_compiler_e2e.py` and several parser tests. Has one of each artifact type. |
| `tests/fixtures/copper_identity_variants.json` | Item-identity fuzzing inputs (different surface forms of the same SKU). |
| `tests/fixtures/adversarial/` | Adversarial inputs that broke parsers in the past (used by `tests/parser_adversarial/`). |
| `tests/fixtures/gold_scenarios/demo/` | A toy gold standard for E2E comparator tests. |
| `tests/fixtures/expected/` | Expected JSON snapshots for golden-file comparisons. |

## Adding a test

1. **Pick the right module.** If you're testing one pipeline stage,
   prefer `tests/test_<stage>.py`. If you're testing a behavior that
   spans stages, write a focused module (e.g. `test_compliance_packets.py`).

2. **Use the existing fixtures** when possible — `tests/fixtures/demo_project`
   is a small but realistic project, and the helper builders in
   `tests/conftest.py` cover most boilerplate (atom + edge construction
   without all the Pydantic field padding).

3. **Class-based tests** are fine and used heavily — pytest discovers
   methods that start with `test_` on classes that start with `Test`.

4. **Determinism tests** — if you're adding a new pipeline stage,
   add a case to `test_deterministic_replay.py` that compiles twice
   and asserts byte-equality of `result.output_signature`.

5. **Adversarial fixtures** — if your fix closes a real-world parser
   gap, add a one-shot test under `tests/parser_adversarial/<format>_adversarial.py`
   with a comment quoting the problematic input. Future regressions
   surface as named test failures.

## Adversarial harness (`tests/parser_adversarial/`)

```bash
pytest tests/parser_adversarial/                  # all
pytest tests/parser_adversarial/test_xlsx_adversarial.py
```

Each adversarial test loads a JSON case from `tests/fixtures/adversarial/`,
runs it through its parser, and asserts the parser produces (a) no
crash and (b) a reasonable atom shape. The cases were collected from
real customer corpora that broke previous parser versions.

## Performance budgets (`test_perf_benchmark.py`, `test_packetizer_benchmark.py`)

These are *soft* budgets — they print warnings rather than fail the
build when violated, because hardware varies. Run with `-s` to see the
output.

If you're working on perf, prefer:

```bash
python scripts/run_perf_benchmark.py
python scripts/run_packetizer_benchmark.py
python scripts/run_parser_benchmark.py
```

These produce richer telemetry than the in-test versions.
