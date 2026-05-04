# `app/` — pipeline architecture

Parser-OS is built as a sequence of pure functions that pass typed data
between stages. Each stage transforms the previous stage's output into a
richer representation.

```
discover_artifacts          (find files in project_dir)
        ↓
parse_artifacts             (file → EvidenceAtom[])
        ↓
candidate_adjudication      (drop low-confidence / duplicate atoms)
        ↓
source_replay               (verify each atom against its source text)
        ↓
confidence_floor            (apply minimum-confidence gate)
        ↓
enrich_entities             (populate atom.entity_keys from text)
        ↓
entity_resolution           (merge atoms into EntityRecord[])
        ↓
graph_build                 (atoms × edges → EvidenceEdge[])
        ↓
packetize                   (atoms + edges → EvidencePacket[])
        ↓
packet_certificates         (sign each packet with a stable hash)
        ↓
quality_gates               (apply review-status flags + warnings)
```

The orchestrator is `app/core/compiler.py::compile_project`. Every stage
is a pure function that takes the previous stage's output and returns
the next. Stages are independently testable — most have a focused test
file under `tests/`.

## Top-level packages

| Package | Role | Key types |
|---|---|---|
| **`app/core/`** | Pipeline stages, schemas, the compile orchestrator | `EvidenceAtom`, `EntityRecord`, `EvidenceEdge`, `EvidencePacket`, `CompileResult`, `CompileQuality` |
| **`app/parsers/`** | One module per source format. Each parser implements a `Parser` protocol and is registered via `app/parsers/registry.py`. | `Parser`, `OrbitbriefPdfParser`, `XlsxParser`, `DocxParser`, `EmailParser`, `TranscriptParser`, `QuoteParser` |
| **`app/domain/`** | Domain packs (per-vertical vocabularies — security_camera, wireless, av, bms, …), project config schema, pack auto-routing | `DomainPack`, `ProjectConfig`, `auto_route_pack` |
| **`app/api/`** | FastAPI routes for HTTP-triggered compiles, packet inspection, project listing | `routes_compile`, `routes_packets`, `routes_artifacts`, `routes_projects`, `routes_health` |
| **`app/eval/`** | Gold comparison, benchmark scoring, failure taxonomy | `GoldScenario`, `compare_to_gold`, `domain_certification` |
| **`app/learning/`** | Confidence calibration, rule mining, active-learning queue | `Calibrator`, `RuleMiner`, `ActiveLearningQueue` |
| **`app/semantic/`** | Cross-artifact semantic linking (atoms that talk about the same thing in different words) | `propose_semantic_link_candidates` |
| **`app/storage/`** | SQLite-backed persistence for compile artifacts and reviews | `Repository`, `Models` |
| **`app/review/`** | Human-review schema + store integration | `ReviewSchema`, `ReviewStore` |
| **`app/experiments/`** | Sandboxed experiment runs + frozen-output snapshots | `run_extraction_sandbox`, `freeze_experiment_output` |
| **`app/testing/`** | Programmatic mutation + adversarial-fixture generation | `Mutators`, `Scenarios` |

## Data model

The full schema lives in `app/core/schemas.py`. Relationships:

```
EvidenceAtom (the unit of evidence)
  ├─ id: stable hash
  ├─ project_id, artifact_id
  ├─ atom_type: scope_item | exclusion | constraint | quantity |
  │              vendor_line_item | customer_instruction | decision |
  │              action_item | meeting_commitment | open_question |
  │              compliance | assumption | entity
  ├─ raw_text + normalized_text
  ├─ entity_keys: ["device:ip_camera", "site:campus", "vendor:cisco", …]
  ├─ source_refs: [SourceRef(page, block_id, locator)]
  ├─ receipts: [ReceiptRecord(replay_status)]
  ├─ authority_class: contractual_scope | customer_current_authored |
  │                   approved_site_roster | vendor_quote | meeting_note |
  │                   machine_extractor | quoted_old_email | deleted_text
  ├─ confidence: 0.0–1.0
  └─ review_status: auto_accepted | needs_review | approved | rejected

EntityRecord (resolved cross-artifact entity)
  ├─ id, entity_type, canonical_key
  ├─ entity_keys (the surface forms that resolved here)
  └─ supporting_atom_ids

EvidenceEdge (graph relation between two atoms)
  ├─ from_atom_id, to_atom_id
  ├─ edge_type: supports | contradicts | excludes | requires | same_as |
  │              located_in | derived_from | quoted_from
  ├─ reason (human-readable explanation)
  ├─ confidence
  └─ metadata: { edge_family, cross_artifact, from_artifact_id, to_artifact_id }

EvidencePacket (the unit of downstream consumption)
  ├─ id, project_id, family
  ├─ anchor_type, anchor_key, anchor_signature (hash + entity context)
  ├─ status: active | needs_review | rejected | invalidated
  ├─ confidence
  ├─ governing_atom_ids (the primary source atoms)
  ├─ supporting_atom_ids
  ├─ contradicting_atom_ids
  ├─ related_edge_ids
  └─ certificate (signed hash of contents)

CompileResult (top of envelope)
  ├─ project_id, compile_id (deterministic)
  ├─ input_signature, output_signature (deterministic hashes)
  ├─ atoms, entities, edges, packets
  ├─ manifest (input artifact registry + parser_routing)
  ├─ trace (per-stage timing + counters)
  └─ quality (CompileQuality — entity_resolution_rate, packet_specificity, …)
```

## Determinism contract

Anything that influences the output must be a function of the input. In
practice that means:

- IDs are derived from input content (`stable_id` in `app/core/ids.py`)
- No `time.time()` in the hot path; timestamps come from the manifest
- Sorted iteration when the order matters for output equality
- No `random` in the parsers / packetizer / graph builder
- Tests under `tests/test_deterministic_replay.py` re-run a compile and
  assert byte-equality of the result

If you're adding a stage, follow that pattern.

## Provenance contract

Every atom must trace back to its source text. The `source_refs` field
holds at least one `SourceRef`, and the `receipts` field holds the
`replay_status` from `source_replay` — `verified` if the parser can
re-extract the same text from the artifact, `unsupported` if the format
doesn't support replay, or `failed` otherwise. The compile fails (unless
`--allow-unverified-receipts`) if any atom has `replay_status: failed`.

## Where to start when adding a feature

| You want to… | Start here |
|---|---|
| Support a new file format | [`app/parsers/README.md`](parsers/README.md) |
| Add a new vertical (domain pack) | [`app/domain/README.md`](domain/README.md) |
| Add a new packet family | `app/core/schemas.py` (`PacketFamily` enum) + `app/core/packetizer.py` (rule) + `app/core/anchors.py` (anchor shape) |
| Add a new edge family | `app/core/schemas.py` (`EdgeType` if new edge type) + `app/core/graph_builder.py` (rule) + `app/core/graph_invariants.py` (validator) |
| Add a new entity-key prefix | `app/core/entity_extraction.py` (emitter) + `app/core/normalizers.py` (`normalize_entity_key`) |
| Add a quality metric | `app/core/quality_metrics.py` |
| Add a gold-compare check | `app/core/gold_compare.py` |
| Wire a CLI subcommand | `app/cli.py` |
| Add an HTTP endpoint | `app/api/routes_*.py` |
