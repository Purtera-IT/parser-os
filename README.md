# Parser OS (Purtera Evidence Compiler)

Evidence Compiler: artifacts -> atoms -> graph -> packets

**Parser OS** — local evidence compiler: artifacts → atoms → graph → packets.

PyPI / install name: `purtera-evidence-mvp` (see `pyproject.toml`). Install from **`parser-os/`**: `cd parser-os && pip install -e ".[dev]"`.

## Current Build Status

- MVP pipeline is implemented end-to-end and working locally.
- Core flow is deterministic and test-covered:
  - artifact parsing
  - atom creation with provenance
  - entity resolution
  - evidence graph construction
  - packetization
  - quality gates
  - CLI + API inspection
- Current confidence level for beta-style internal rollout: **high for controlled pilot data**, **medium for messy production variance**.

## What This MVP Is

- A local, deterministic compiler for messy project artifacts.
- A parser + rules engine that produces explainable evidence packets.
- A CLI/API-first backend for regression-safe scope intelligence.

## What This MVP Is Not

- no chatbot
- no OrbitBrief
- no SOW generation
- no dispatch
- no VisionQC

## MVP Scope Snapshot

### Fully Implemented in This Repo

- **Core schemas and contracts**
  - Stable IDs (`sha256`, deterministic content-based)
  - `SourceRef`, `EvidenceAtom`, `EntityRecord`, `EvidenceEdge`, `EvidencePacket`, `CompileResult`
  - Enums and packet/atom family taxonomy
- **Parsers**
  - Spreadsheet/site roster parser (`.xlsx`, `.csv`)
  - Vendor quote/PO parser (`.xlsx`, `.csv`, `.txt`)
  - Email/thread parser (`.eml`, `.txt` threads)
  - DOCX parser with tracked deletion extraction (`w:del`)
  - Transcript/meeting parser (`.txt`, `.md`, `.vtt`, `.srt`, transcript-like `.json`)
- **Authority engine**
  - Deterministic rank + rule-based tie-breaks
  - Governing selection constraints (`deleted_text`, `quoted_old_email`, transcript/meeting limits)
- **Entity resolution**
  - Canonical key normalization
  - alias merging + fuzzy handling
- **Graph builder**
  - `supports`, `contradicts`, `excludes`, `requires`, `located_in`, `derived_from`, `quoted_from`
  - aggregate roster-vs-vendor quantity contradiction logic
- **Packetizer v0**
  - `quantity_conflict`, `vendor_mismatch`, `scope_exclusion`, `site_access`, `scope_inclusion`
  - transcript-driven: `meeting_decision`, `action_item`, `missing_info`
  - deterministic dedupe and review flagging
- **Persistence + API**
  - SQLite JSON-blob persistence (projects, artifacts, compile_results)
  - FastAPI endpoints for project creation, artifact upload, compile, packet/atom/edge/entity inspection
  - packet filtering by `family` and `status`
- **Quality gates**
  - compile-level validation with hard errors and warnings
  - compiler raises on hard errors unless `--allow-errors`
- **Developer experience**
  - fixture generation script
  - demo compile script
  - packet inspection utility
  - regression tests and smoke tests

### Implemented but Intentionally Minimal (MVP Stub Grade)

- **DB design**
  - JSON blob storage is intentional MVP simplification
  - not yet normalized for large-scale querying/analytics
- **Auth and multi-tenant controls**
  - no auth/permissions yet (explicitly out of scope)
- **Operational hardening**
  - no background jobs, retry queue, or artifact virus/content scanning
- **API lifecycle**
  - using FastAPI startup event (`on_event`) with deprecation warnings; lifespan migration pending
- **Parser depth**
  - robust rules exist, but still heuristic-driven and not domain ontology-complete
- **Observability**
  - no metrics/traces dashboarding yet

### Not Implemented (Out of MVP Scope)

- chatbot, OrbitBrief, SOW generation, dispatch, VisionQC
- LLM or embedding/vector retrieval features
- frontend product UX polish

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/make_demo_fixtures.py
pytest
python -m app.cli compile tests/fixtures/demo_project --out /tmp/purtera_demo_output.json
uvicorn app.main:app --reload
```

## Demo Commands

```bash
# Full demo flow
bash scripts/demo_compile.sh

# Manual compile
python scripts/make_demo_fixtures.py
python -m app.cli compile tests/fixtures/demo_project --out /tmp/purtera_demo_output.json
python scripts/inspect_packets.py /tmp/purtera_demo_output.json
```

## One-command Insane MVP Demo

Run the full MVP showcase pipeline end-to-end:

```bash
bash scripts/insane_mvp_demo.sh
```

The script produces:

```text
/tmp/purtera_mvp_demo/
  compile_result.json
  trace.json
  packetizer_benchmark.json
  adversarial_report.json
  parser_benchmark.json
  packet_summary.md
  demo_report.md
```

It executes fixture regeneration, targeted tests, compile with trace, packetizer benchmark, adversarial lab, parser benchmark, and markdown report generation in one flow.

## Beta Rollout Readiness

### Can Be Rolled Out to Internal Beta

- deterministic compile output with explainable packet reasons
- parser coverage across common artifact types in this problem space
- quality gate protections against invalid governing logic
- API + CLI interfaces stable enough for pilot workflow integration

### Beta Preconditions (Recommended Before External Pilot)

- pin and freeze runtime environment (`requirements lock`, Python minor version)
- migrate FastAPI lifecycle hook to lifespan API
- add structured logging + request IDs + compile job IDs
- baseline perf test for larger artifacts and multi-file projects
- add test fixtures for edge-case malformed inputs

## Expected Packet Output

Expected family coverage for demo fixtures:

- `quantity_conflict`
- `vendor_mismatch`
- `scope_exclusion`
- `site_access`
- `scope_inclusion`
- `meeting_decision`
- `action_item`
- `missing_info`

Expected high-level summary fixture:

- `tests/fixtures/expected/demo_summary.json`

## What Is Actually Working Today

- `python scripts/make_demo_fixtures.py` generates a full golden fixture set.
- `python -m app.cli compile ...` produces valid JSON compile output with packets.
- `python scripts/inspect_packets.py ...` prints packet table inspection view.
- API smoke flow works:
  - create project
  - upload artifacts
  - compile
  - inspect packets/atoms/edges/entities
- Full regression suite passes in project test mode.

## Architecture

```text
        +---------------------------+
        |      Project Artifacts    |
        | xlsx/csv/txt/md/eml/docx  |
        | vtt/srt/json transcripts  |
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        |        Parsers            |
        | Spreadsheet/Email/Docx/PO |
        | Transcript / Meeting      |
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        |      Evidence Atoms       |
        | + SourceRef provenance    |
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        | Entity Resolution + Graph |
        | entities + supports/contra|
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        |       Packetizer v0       |
        | decision-ready packets    |
        +-------------+-------------+
                      |
                      v
        +---------------------------+
        |   CLI / FastAPI Endpoints |
        +---------------------------+
```

## Data Contracts

- `SourceRef`
  - Artifact provenance: artifact id/type, filename, locator, extraction method, parser version.
- `EvidenceAtom`
  - Canonical evidence statement with authority, confidence, entity keys, and source refs.
- `EvidenceEdge`
  - Deterministic relation between atoms (`supports`, `contradicts`, `excludes`, etc).
- `EvidencePacket`
  - Decision-ready packet with family, anchor, governing/supporting/contradicting atoms, status, reason, flags.

## Quality Gates

- Compile-level validation runs after packetization.
- Hard errors include missing source refs, invalid governing rules, broken references.
- Warnings include low confidence, `needs_review` packets, contradictions, and vendor mismatch signals.
- Compiler raises on hard errors unless `--allow-errors` is passed in CLI.

## Deterministic Replay

- Compiler emits a versioned compile manifest with:
  - `compile_id`
  - `input_signature`
  - `output_signature`
  - parser/version fingerprint metadata
- Deterministic ordering is enforced for atoms, entities, edges, packets, and warnings.
- Re-running compile with identical artifacts and versions reproduces the same `output_signature`.

```bash
python scripts/make_demo_fixtures.py
python -m app.cli compile tests/fixtures/demo_project --out /tmp/a.json
python -m app.cli compile tests/fixtures/demo_project --out /tmp/b.json
diff <(jq -S . /tmp/a.json) <(jq -S . /tmp/b.json)
```

## Evidence Receipt Replay

- Every `EvidenceAtom` now carries `receipts[]` that record whether each `SourceRef` can be replay-verified against the original artifact.
- Replay verification enforces auditability:
  - spreadsheet locators verify cited sheet/row/columns
  - transcript/email/text locators verify line ranges
  - docx paragraph/table locators verify cited content
  - tracked deletion locators are explicitly marked `unsupported` with a reason
- Failed receipts are treated as hard errors by default. Use `--allow-unverified-receipts` only for controlled troubleshooting.

```bash
python scripts/make_demo_fixtures.py
python -m app.cli compile tests/fixtures/demo_project --out /tmp/receipts.json
python -m app.cli compile tests/fixtures/demo_project --allow-unverified-receipts --out /tmp/receipts_allow.json
```

```bash
# Inspect receipt statuses
jq '[.atoms[].receipts[]?.replay_status] | group_by(.) | map({status: .[0], count: length})' /tmp/receipts.json

# Inspect unsupported receipts with reasons
jq '.atoms[] | {atom_id: .id, receipts: [.receipts[] | select(.replay_status=="unsupported") | {source_ref_id, reason, locator}]}' /tmp/receipts.json
```

## Packet Certificates

- Every `EvidencePacket` includes a deterministic `certificate` explaining:
  - why the packet exists
  - which minimal atoms are sufficient
  - governing rationale and contradiction context
  - counterfactual impact if key atoms are removed
  - downstream blast radius for scope consumers

Example certificate shape:

```json
{
  "packet_id": "pkt_abc123",
  "certificate_version": "packet_certificate_v1",
  "existence_reason": "Created because approved_site_roster quantity 91 contradicts vendor_quote quantity 72 for device:ip_camera.",
  "governing_rationale": "approved_site_roster governs scope quantity; vendor_quote can support or contradict procurement coverage.",
  "minimal_sufficient_atom_ids": ["atm_scope_qty", "atm_vendor_qty"],
  "contradiction_summary": "2 contradicting atom(s) linked.",
  "authority_path": [{"atom_id": "atm_scope_qty", "authority_class": "approved_site_roster"}],
  "counterfactuals": [{"atom_id": "atm_scope_qty", "if_removed": "packet would not exist"}],
  "blast_radius": ["OrbitBrief.scope_truth", "SOWSmith.scope_clause", "RunbookGen.site_steps"],
  "evidence_completeness_score": 0.8,
  "ambiguity_score": 0.2
}
```

## Authority Lattice

- Authority still uses deterministic base ranks (`contractual_scope` > ... > `deleted_text`) for compatibility.
- Lattice scoring now adds explicit dimensions per atom:
  - source authority
  - recency
  - authorship
  - artifact role
  - evidence state penalties
  - review penalties
  - scope-impact context
- `authority_rank(...)` remains available as a compatibility helper.
- Packet certificates include these dimensions in `certificate.authority_path[*].dimensions` for explainability.

## Risk And PM Triage

- Every packet includes deterministic `risk` metadata for PM work ranking:
  - `risk_score` (0..1)
  - `severity` (`low|medium|high|critical`)
  - `risk_reasons`
  - `estimated_cost_exposure`
  - `operational_impact`
  - `review_priority` (1 highest, 5 lowest)
- Packet inspect output includes severity and estimated exposure so review queues can be triaged quickly.

## Adversarial Lab

- Deterministic fixture mutation lab generates ugly-but-parseable artifact bundles without external services.
- Mutation families cover spreadsheet, email, docx, transcript, and vendor quote variations.
- Lab compile loop validates invariants per scenario:
  - source refs exist
  - governing packet integrity
  - governance constraints (`deleted_text`, `quoted_old_email`)
  - expected packet-family recall
  - deterministic replay on second compile
- Run locally:

```bash
pytest tests/test_adversarial_lab.py
python scripts/run_adversarial_lab.py --count 25 --out /tmp/purtera_adversarial_report.json
```

## Packetizer Benchmark

- Gold-scenario benchmark scorecard evaluates packetizer quality with measurable metrics:
  - packet family recall
  - packet anchor recall
  - governing accuracy
  - contradiction recall
  - receipt coverage + verified receipt rate
  - false active rate
  - invalid governance count
  - determinism pass
  - compile latency, packet count, atom count
- Run benchmark:

```bash
pytest tests/test_packetizer_benchmark.py
python scripts/run_packetizer_benchmark.py --fixtures tests/fixtures/gold_scenarios --out /tmp/packetizer_benchmark.json
```

- MVP hardening targets:
  - `packet_family_recall >= 0.95`
  - `governing_accuracy >= 0.95`
  - `contradiction_recall >= 0.95`
  - `invalid_governance_count = 0`
  - `determinism_pass = true` for all scenarios
  - `false_active_rate = 0`
  - `compile_success_rate = 1.0`

## Parser Adversarial Benchmark

- Each parser has parser-specific ugly-data adversarial tests under `tests/parser_adversarial/`.
- Parser benchmark report tracks:
  - `atom_recall_by_type`
  - `source_ref_coverage`
  - `entity_key_accuracy`
  - `quantity_accuracy`
  - `authority_class_accuracy`
  - `review_flag_accuracy`
  - `parse_crash_rate`
  - `unsupported_feature_warnings`
- Synthetic target thresholds:
  - `source_ref_coverage = 1.0`
  - `parse_crash_rate = 0`
  - `quantity_accuracy >= 0.98`
  - `authority_class_accuracy >= 0.95`
  - `entity_key_accuracy >= 0.90`
  - parser adversarial atom recall `>= 0.90`

```bash
pytest tests/parser_adversarial
pytest tests/test_copper_low_voltage_adversarial.py
python scripts/run_parser_benchmark.py --out /tmp/parser_benchmark.json
```

## Anchor Signatures And Graph Invariants

- Packets now carry canonical `anchor_signature` metadata to prevent unstable or duplicate anchors.
- Canonical anchoring rules include:
  - quantity conflict/vendor mismatch by device family (for example `device:ip_camera`)
  - scope exclusion and site access by site key
  - action items by owner + normalized action topic
  - missing info by normalized question topic
  - meeting decisions by normalized decision topic
- Graph invariant checks enforce:
  - no missing atom references
  - no disallowed self loops
  - contradiction reasons present
  - excludes edges include exclusion atoms
  - requires edges include constraint atoms
  - aggregate contradiction uniqueness
  - deterministic anchor signature consistency

## Packet Invalidation Simulation

- Compile-to-compile diff simulation now reports:
  - `atom_diffs` (`added|removed|changed|unchanged`) using stable atom content hashes
  - `packet_diffs` (`added|removed|changed|unchanged|invalidated`)
  - `invalidated_packet_ids`
  - `blast_radius_summary` for downstream consumers impacted by changes
- Packet invalidation triggers when:
  - governing evidence changes or is removed
  - certificate minimal-sufficient evidence changes or is removed
  - anchor signature drifts across compiles
  - certificate minimal evidence set changes
- Supporting-only evidence drift is tracked as `changed` (not invalidated) when governing/minimal evidence remains stable.
- Compare two compile outputs:

```bash
python scripts/compare_compiles.py --before /tmp/a.json --after /tmp/b.json --out /tmp/diff.json
```

## Observability and Performance

- Every compile now emits a structured `trace` in `CompileResult` with stage-level timings and counts.
- Stage coverage includes:
  - `discover_artifacts`
  - `parse_artifacts`
  - `source_replay`
  - `entity_resolution`
  - `graph_build`
  - `packetize`
  - `packet_certificates`
  - `quality_gates`
  - `persistence` when compile is invoked with persistence enabled (API flow)
- Structured JSON logs are emitted per stage completion (`stderr`):
  - `event`
  - `compile_id`
  - `stage`
  - `duration_ms`
  - `counts`
  - `warning_count`
  - `error_count`
- CLI supports trace export:

```bash
python -m app.cli compile tests/fixtures/demo_project --out /tmp/compile.json --trace-out /tmp/trace.json
```

- Performance benchmark script:

```bash
python scripts/run_perf_benchmark.py --sites 100 --devices 1 --out /tmp/perf.json
```

- MVP local performance budgets:
  - demo project compile under 5 seconds
  - 100-site synthetic compile under 30 seconds
  - no compile stage left unmeasured

## API Hardening (Internal Beta)

- FastAPI app startup now uses lifespan instead of deprecated startup events.
- Added service metadata and health endpoints:
  - `GET /health/live`
  - `GET /health/ready`
  - `GET /version`
- Upload safety guardrails:
  - filename sanitization
  - path traversal rejection
  - extension allowlist (`.xlsx`, `.csv`, `.txt`, `.md`, `.eml`, `.docx`, `.vtt`, `.srt`, `.json`)
  - configurable max size via `PURTERA_MAX_UPLOAD_BYTES` (default 25MB)
  - empty file rejection
  - SHA256 digest and hashed storage filename
- Compile-result APIs now support pagination with metadata (`total`, `limit`, `offset`, `items`) for:
  - atoms
  - edges
  - entities
  - packets
- Packet filters:
  - `family`, `status`, `severity`, `anchor_key_contains`, `review_priority_lte`
- Atom filters:
  - `atom_type`, `authority_class`, `entity_key`, `review_status`
- Error behavior:
  - `400` invalid upload input
  - `404` unknown project
  - `422` invalid filter/query values

## Real Data Validation Packs

- The repo supports local-only real-data intake harness workflows under `real_data_cases/`.
- Safety requirements:
  - do not commit raw real artifacts
  - do not copy production unredacted files into tracked paths
  - do not require external services for validation workflows
- Case directory convention:

```text
real_data_cases/
  .gitkeep
  CASE_ID/
    artifacts/          # ignored by git
    labels/
      packet_labels.json
    outputs/
      compile_result.json
      benchmark_summary.json
    case_manifest.json
```

- Initialize a case:

```bash
python scripts/init_real_data_case.py --case-id CASE_001 --notes "Redacted telecom rollout"
```

- Compile a case locally:

```bash
python scripts/compile_real_data_case.py --case-id CASE_001
```

- Generate packet labeling skeleton:

```bash
python scripts/label_packets.py --case-id CASE_001
```

- Summarize labeled cases:

```bash
python scripts/summarize_real_data_cases.py --out /tmp/real_data_summary.json
```

## Terminal Review Console

- `scripts/review_packets.py` provides a terminal-first packet review workflow for fast PM/developer validation.
- Input: compiled JSON (`compile_result.json` shape), Output: packet review labels JSON.
- Supports filters:
  - `--family`
  - `--severity`
  - `--needs-review-only`
  - `--limit`
- For each selected packet it prints:
  - family, anchor, status, severity/risk
  - packet reason
  - certificate existence/governing rationale
  - minimal sufficient atoms with `raw_text` snippets
  - contradicting atoms with `raw_text` snippets
  - receipt replay status summaries
- Non-interactive mode can create label skeletons quickly:

```bash
python scripts/review_packets.py /tmp/purtera_demo_output.json --out /tmp/packet_reviews.json --limit 3 --non-interactive
```

## Failure Taxonomy And Error Reports

- Validation, benchmark, parser/compile crashes, and adversarial failures can be normalized into stable taxonomy records (`FailureRecord`).
- Taxonomy categories include source/ref issues, parser failures, entity/edge failures, packet quality failures, determinism, and performance budget overruns.
- Validators expose hard-error taxonomy rows via `validation_failure_records(...)`.
- Benchmark reports include per-scenario `failure_records` and top-level aggregated `failure_records`.
- Adversarial lab report now includes taxonomy failure records as well.
- Summarize failure JSON reports:

```bash
python scripts/summarize_failures.py /tmp/packetizer_benchmark.json /tmp/purtera_adversarial_report.json --out /tmp/failure_summary.json
```

## Transcript Parser

- Supported formats: `.txt`, `.md`, `.vtt`, `.srt`, transcript-like `.json`.
- Extracts: `decision`, `action_item`, `open_question`, `constraint`, `exclusion`, `scope_item`, `quantity`, `customer_instruction`.
- Transcript facts remain lower authority (`meeting_note`) and do not become contractual automatically.
- Transcript atoms feed the same entity-resolution, graph, and packetizer pipeline used by other artifacts.

```bash
python -m app.cli compile tests/fixtures/demo_project --out /tmp/purtera_demo_output.json
jq '.packets[] | select(.family=="meeting_decision" or .family=="action_item" or .family=="missing_info")' /tmp/purtera_demo_output.json
```

## Parser Plugins

- Parsers now expose self-describing `ParserCapability` metadata:
  - parser identity/version
  - supported extensions + artifact types
  - emitted atom types
  - supported domain packs
  - binary/source-replay support flags
- Router asks every registered parser for a `ParserMatch` (`confidence`, `reasons`, `artifact_type`) and selects the highest-confidence parser above threshold.
- Deterministic tie-breaking is applied for known overlaps:
  - email vs transcript (marker-driven)
  - quote vs xlsx (filename hint-driven)
- If no parser exceeds threshold, compile continues with a warning and skips that artifact.
- Routing decisions are captured in both compile manifest and compile trace (`parser_routing`) for auditability.

## CandidateAtom Layer

- The compiler now supports a bridge layer: artifact segment -> `CandidateAtom` -> adjudication -> `EvidenceAtom`.
- `CandidateAtom` represents a possible extraction, not trusted evidence by default.
- Adjudication enforces guardrails (source refs, replay checks, confidence thresholding, entity key format, span checks) before promotion.
- Scope-impacting meeting-note candidates are forced to `needs_review`, and `deleted_text` behavior remains rejected/non-governing.
- Existing deterministic parsers can continue emitting direct `EvidenceAtom`s while new extractors can emit candidates safely.
- Rejected candidates are retained as structured outcomes for future training/eval pipelines without entering packetization.

## ArtifactSegments

- Parsers now expose normalized `ArtifactSegment` shapes so universal extractors can operate on segments instead of raw files.
- Segment coverage includes spreadsheet rows/cells, email messages/lines, DOCX paragraphs/tables/tracked deletions, transcript utterances/sections, quote line items, and generic text blocks.
- Every segment carries a deterministic ID plus a `SourceRef` derived from the same locator contract used by parser atoms.
- Source replay includes `verify_segment(...)` to validate segment grounding against the original artifact.
- Existing parser outputs are preserved; segmenters are additive and safe for future candidate-based extractors.

## Semantic Linker, Not RAG

- Semantic linking proposes cross-artifact evidence neighborhoods (`same_as`, `supports`, `excludes`) as candidates; it does not answer questions or generate packets by itself.
- Deterministic fallback is char n-gram TF-IDF similarity; optional sentence-transformer scoring is behind a feature flag and disabled by default.
- Semantic similarity never creates contradiction edges and never overrides authority ranking.
- Accepted semantic links are tagged with `semantic_candidate_linker` metadata so validators and certificates can audit their provenance.
- Deterministic validators still decide whether semantic proposals are accepted, rejected, or marked `needs_review`.

## Confidence Calibration and Abstention

- Calibration is an optional post-packetization pass that maps raw atom/packet confidence to calibrated correctness probabilities using review and benchmark labels.
- The MVP uses sklearn models only (logistic regression over deterministic feature rows); no external services or neural training are required.
- If a calibrator is provided and calibrated probability falls below an abstention threshold, the system marks results `needs_review` with `calibration_abstain`.
- Compile remains fully functional without any model artifact; calibration is opt-in via CLI/model path.
- Training/evaluation scripts (`train_calibrator.py`, `evaluate_calibrator.py`) support iterative improvement as more reviewer labels accumulate.

## Active Learning Queue

- `app.learning.active_learning.build_active_learning_queue(...)` builds a deterministic, prioritized review queue from compile outputs.
- Priority combines packet risk/severity, ambiguity, low evidence completeness, novelty signals (new domain pack or unseen aliases), conflicting evidence, and failure taxonomy rows.
- Candidates from `llm_candidate` or `semantic_candidate` extractors are boosted so reviewer labels focus on high-uncertainty extraction paths.
- Queue items include reviewer-ready prompts, such as quantity-governance and scope-exclusion questions, plus alias confirmation prompts for semantic links.
- Use `python scripts/build_review_queue.py --compile-result /tmp/out.json --out /tmp/review_queue.json` to generate queue JSON for reviewer tooling.

## Weak Supervision Rule Suggestions

- `app.learning.rule_miner.mine_rule_suggestions(...)` mines repeated approved/rejected outcomes into deterministic rule suggestions for human review.
- Suggestions are output only and never auto-applied; each suggestion sets `requires_human_approval=true`.
- Current suggestion coverage includes `domain_alias`, `parser_header_alias`, `exclusion_pattern`, `constraint_pattern`, `risk_default`, and `entity_normalization_rule`.
- Confidence reflects supportive vs contradictory examples, so contested patterns are still visible but ranked lower.
- Run `python scripts/mine_rule_suggestions.py --labels real_data_cases --out /tmp/rule_suggestions.json` to generate a reviewable suggestion bundle.

## Domain Pack Certification

- `app.eval.domain_certification.certify_domain_pack(...)` runs a pre-pilot certification battery for a domain pack.
- Certification validates pack schema, alias hygiene (cross-entity duplicates + site/device collisions), pattern coverage, and risk-default completeness.
- Fixture checks ensure the pack has synthetic artifacts and parser coverage before promotion.
- Gold-scenario checks run packetizer benchmark thresholds and enforce `invalid_governance_count == 0`.
- Generate a report with:
  `python scripts/certify_domain_pack.py --domain-pack app/domain/security_camera_pack.yaml --fixtures tests/fixtures/domain/security_camera --out /tmp/cert.json`
  (use `--allow-fail` to always exit zero while iterating).

## Promotion Pipeline

- `app.learning.promotion` provides a human-approved path from review labels to fixture proposals and optional suggestion application.
- `promote_review_to_fixture(...)` turns reviewed packet/candidate outcomes into a minimal synthetic regression fixture (`project/` + `gold.json`) and emits a `PromotionArtifact` with patch preview.
- `apply_approved_suggestion(...)` never auto-applies changes; it returns a proposal unless explicit `--approve` is passed.
- Domain-pack updates are guarded: regression test/fixture files are created together with pack edits so every approved rule has a reproducible check.
- Scripts:
  - `python scripts/promote_review_to_fixture.py --review-labels /tmp/packet_reviews.json --compile-result /tmp/out.json --out-dir tests/fixtures/regression/CASE_X`
  - `python scripts/apply_approved_suggestion.py --suggestion /tmp/rule_suggestion.json --approve`

## Incremental Compile Cache

- The compiler now supports artifact-level incremental reuse to avoid reparsing unchanged files across revision cycles.
- Cache reuse key is strict: `artifact_id + sha256 + parser_name + parser_version + domain_pack_id + domain_pack_version`.
- When a cache hit occurs, parser outputs (`atoms`, `candidates`, parser warnings) are reused; downstream stages still fully rebuild entities, graph edges, packets, certificates, risk, and validation.
- Manifest includes `cache_hits`, `cache_misses`, and `reused_artifact_ids` for auditability of incremental behavior.
- CLI supports `--no-cache` to force full reparsing and prints cache hit/miss summary in compile output.

## Evidence Coverage Diagnostics

- `app.eval.coverage.build_coverage_report(...)` computes segment-level evidence coverage diagnostics from compile output.
- Coverage tracks whether each segment is:
  - `covered` (accepted atom extracted)
  - `candidate_rejected` (candidate existed but no accepted evidence)
  - `ignored` (no extraction, including intentional boilerplate ignore)
  - `unsupported` (artifact had no matched parser route)
- Boilerplate suppression includes common signatures/disclaimers plus spreadsheet total/subtotal rows.
- Reports are emitted per artifact and project-wide, with low-coverage warnings and recommended parser/domain-pack improvements.
- Generate report:
  `python scripts/evidence_coverage_report.py --compile-result /tmp/out.json --out /tmp/coverage.json`

## Probabilistic Sandbox and Freeze

- `app.experiments.sandbox.run_extraction_sandbox(...)` runs experimental extractors against the same project/segments as baseline deterministic compile, then computes hypothetical deltas without mutating production output.
- Sandbox flow: baseline compile -> experimental candidates -> dry-run adjudication -> hypothetical packetization -> diff vs baseline.
- `ExperimentRun` tracks candidate volume, accepted/rejected dry-run outcomes, and packet deltas (`new_packets_if_accepted`, `changed_packets_if_accepted`).
- `app.experiments.freeze.freeze_experiment_output(...)` freezes approved experiment outputs into deterministic artifacts (regression metadata, gold fixture scaffold, frozen candidate set), gated by explicit `--approve`.
- Scripts:
  - `python scripts/run_extraction_experiment.py --project tests/fixtures/demo_project --extractor semantic_linker --out /tmp/experiment.json`
  - `python scripts/freeze_extractor_output.py --experiment /tmp/experiment.json --approve`

## Final MVP Gauntlet v2

- Run everything in one pass with:
  `bash scripts/final_mvp_gauntlet.sh`
- Outputs are written to `/tmp/purtera_final_mvp/`, including:
  - `compile_result.json`
  - `trace.json`
  - `coverage.json`
  - `packetizer_benchmark.json`
  - `parser_benchmark.json`
  - `adversarial_report.json`
  - `domain_cert_security_camera.json`
  - `experiment_semantic_linker.json`
  - `active_learning_queue.json`
  - `final_mvp_report.md`
  - `final_mvp_report.json`
- `scripts/build_final_mvp_report.py` computes readiness thresholds and emits:
  - detailed pass/fail threshold table
  - recommended next fixes when any critical threshold fails
  - `Ready for OrbitBrief v0?` YES/NO decision.

## MVP Perfection Checklist

If the goal is a “technically insane” but still MVP-bounded production candidate, finish these:

- **Reliability**
  - replace startup hook with lifespan
  - enforce strict schema versioning in outputs
  - add compile idempotency checks by artifact hash
- **Data correctness**
  - add adversarial fixtures per parser (broken tables, mixed locales, malformed transcripts)
  - add packet-level golden snapshots for multiple scenarios beyond demo project
  - add stricter anchor consistency checks in packetizer
- **Scale**
  - introduce normalized storage path for packet querying and analytics
  - add pagination and filtering on atoms/edges/entities APIs
- **Operations**
  - add structured logs + metrics (compile latency, parser fail rate, packet family distribution)
  - add health/readiness probes and CI smoke script for demo flow
- **Security**
  - file upload size/type limits
  - artifact path sanitization and quarantine policy
  - API auth layer before external exposure
- **DX**
  - platform-specific demo wrappers (`.sh` + `.ps1`)
  - one-command “doctor” script for environment checks

## Tomorrow Test Path

```bash
# Regenerate fixtures + expected summary
python scripts/make_demo_fixtures.py

# Run all tests
pytest

# Compile demo project
python -m app.cli compile tests/fixtures/demo_project --out /tmp/purtera_demo_output.json

# Inspect packets with jq
jq '.packets[] | {family, anchor_key, status, reason, review_flags}' /tmp/purtera_demo_output.json
jq '[.packets[].family] | sort | unique' /tmp/purtera_demo_output.json
```

## Next Milestone

OrbitBrief v0 consumes packets and creates scope truth board.
