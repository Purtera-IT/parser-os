# `app/core/` — pipeline stages reference

The compile pipeline is defined in `app/core/compiler.py::compile_project`.
This file documents what each stage does, what it consumes, and what it
produces.

Every stage is a pure function. Stage timing + counters are recorded in
the `CompileTrace` and surface in the `CompileResult.trace` field.

## Stage reference

### `discover_artifacts`

**File:** `compiler.py` (inline)
**Input:** `project_dir`
**Output:** `Artifact[]`

Walks the project's `artifacts/` directory, applies `.parserignore`,
honors `project.yaml::parserignore_extra`, skips built-in skip patterns
(`.derived/`, `.orbitbrief/`, `gold_standard.*`, `SOURCE_NOTES.md`,
`labels/`, …). Each surviving file becomes an `Artifact` record.

### `parse_artifacts`

**File:** `compiler.py` → dispatch to `app/parsers/`
**Input:** `Artifact[]`
**Output:** `EvidenceAtom[]`

For each artifact, picks a parser via `app/parsers/parser_router.py` and
calls its `parse()` method. Each parser produces atoms with `entity_keys=[]`
left for `enrich_entities` to populate.

### `candidate_adjudication`

**File:** `app/core/candidate_adjudicator.py`
**Input:** `EvidenceAtom[]`
**Output:** `EvidenceAtom[]` (filtered)

Drops near-duplicate atoms (same `normalized_text` + same artifact + same
locator) and atoms below a minimum-confidence floor. Keeps the highest-
confidence representative.

### `source_replay`

**File:** `app/core/source_replay.py`
**Input:** `Artifact[]`, `EvidenceAtom[]`
**Output:** `EvidenceAtom[]` (with `receipts[]` populated)

For each atom, attempts to re-extract its `raw_text` from the source
artifact at the locator (page, row, line). Sets
`receipt.replay_status` to `verified`, `unsupported`, or `failed`. The
compile aborts on `failed` unless `--allow-unverified-receipts`.

### `confidence_floor`

**File:** `compiler.py` (inline)
**Input:** `EvidenceAtom[]`
**Output:** `EvidenceAtom[]` (filtered)

Drops atoms whose calibrated confidence is below `--abstain-threshold`
(default 0.7). The dropped atoms still appear in `manifest.dropped_atom_ids`
for review.

### `enrich_entities`

**File:** `app/core/entity_extraction.py`
**Input:** `EvidenceAtom[]`, `DomainPack`
**Output:** `EvidenceAtom[]` (with `entity_keys[]` populated)

Universal entity extractor — runs after parsing so per-format parsers
don't have to know about packs. For each atom, scans the `raw_text`
against the active pack's vocabulary plus a cross-pack vendor catalog
and emits keys like `device:ip_camera`, `vendor:cisco`,
`site:perry_street_parking_deck`, `address:1700_pratt_drive`,
`part_number:cw9166i_b`, `quantity:136`, `customer:virginia_tech`,
`requirement:nfpa_72_compliance`, `qa:q12`, `spec_section:28_13_00`.

Never touches atoms that already have `entity_keys` populated by their
parser — parser-supplied keys are authoritative.

### `entity_resolution`

**File:** `app/core/entity_resolution.py`
**Input:** `EvidenceAtom[]`
**Output:** `EntityRecord[]`

Groups atoms that share the same canonical `entity_key`. Produces an
`EntityRecord` per distinct entity, populated with the supporting atom
IDs. Used downstream by `graph_build` to find atoms that share an
entity context.

### `graph_build`

**File:** `app/core/graph_builder.py`
**Input:** `EvidenceAtom[]`, `EntityRecord[]`
**Output:** `EvidenceEdge[]`

Builds the evidence graph. Inverted entity-key index keeps cross-atom
joining at O(N · √N) (down from O(N²) before Week 3). Edge families
produced:

- `value_support` (two atoms agree on a value for the same entity)
- `quantity_contradiction` / `part_number_quantity_conflict`
- `exclusion_application` (an exclusion atom rules out a target context)
- `constraint_requirement` (a constraint atom requires another)
- `device_aggregate_mismatch` / `material_aggregate_mismatch`
- `cross_artifact_co_mention` (two artifacts mention the same entity)
- `semantic_link` (proposed by `app/semantic/linker.py`)

The validator `graph_invariants.py` runs after this stage and rejects
edges that violate the contract (e.g. an `excludes` edge whose endpoints
aren't exclusion-bearing atoms).

### `packetize`

**File:** `app/core/packetizer.py`
**Input:** `EvidenceAtom[]`, `EvidenceEdge[]`
**Output:** `EvidencePacket[]`

Buckets atoms + edges into packet families. Each family has its own
selection rule + anchor shape (see `app/core/anchors.py`):

- `quantity_conflict` — pairs of `quantity` atoms with `contradicts` edges
- `vendor_mismatch` — vendor-quote vs site-roster quantity disagreements
- `scope_exclusion` — `exclusion` atoms or vendor-pollution edges
- `site_access` — `customer_instruction` atoms tagged with site-access patterns
- `meeting_decision` — `decision` / `meeting_commitment` atoms
- `action_item` — `action_item` atoms
- `compliance_clause` — `compliance` atoms (Week 6)
- `customer_override` — `customer_instruction` + `customer_current_authored`
- `missing_info` — `open_question` atoms + raceway/conduit/certification fallbacks
- `scope_inclusion` — remaining `scope_item` / `quantity` atoms

Packets carry `governing_atom_ids` (primary), `supporting_atom_ids`
(context), `contradicting_atom_ids` (review-flag triggers), and
`related_edge_ids`.

### `packet_certificates`

**File:** `app/core/packet_certificates.py`
**Input:** `EvidencePacket[]`
**Output:** `EvidencePacket[]` (with `certificate` populated)

Computes a deterministic hash over each packet's identifying fields and
attaches it as the certificate. Downstream consumers can detect packet
content changes by certificate-mismatch.

### `quality_gates`

**File:** `app/core/compiler.py` (inline) + `app/core/quality_metrics.py`
**Input:** `CompileResult` (in-flight)
**Output:** `CompileResult` (with `quality` field populated, plus warnings)

Computes `CompileQuality` (entity_resolution_rate, packet_specificity,
parser_routing_confidence_avg, parser_atom_yield_rate, atoms_per_artifact,
parsers_with_zero_atoms, parsers_with_low_confidence, stage_durations_ms,
…) and emits fail-loud warnings:

- `parsers_with_zero_atoms` — any parser ran but produced 0 atoms
- `entity_resolution_rate < threshold` (default 0.50)
- `packet_specificity < threshold` (default 0.85)
- `pack_routing_source == "default"` when SOURCE_NOTES.md exists

These are warnings, not errors — the compile succeeds, but operators
can wire the warnings into a CI gate.

## Cross-cutting modules

| Module | Responsibility |
|---|---|
| `schemas.py` | Pydantic models for every data shape: `EvidenceAtom`, `EntityRecord`, `EvidenceEdge`, `EvidencePacket`, `CompileResult`, `CompileQuality`, etc. + the enums (`AtomType`, `EdgeType`, `PacketFamily`, `AuthorityClass`, `ReviewStatus`, `PacketStatus`, `ArtifactType`). |
| `ids.py` | `stable_id(prefix, *components)` — deterministic SHA-256-based IDs. The single place where IDs are computed. |
| `normalizers.py` | `normalize_text` (case-fold + whitespace collapse + punctuation), `normalize_entity_key`. |
| `anchors.py` | Per-packet-family anchor signatures. `_topic_slug` cap (80 chars + collision-safe hash) lives here. |
| `cache.py` | Artifact-level parse cache keyed on file SHA-256 + parser version. Reused across `compile` calls when `--no-cache` is not passed. |
| `manifest.py` | Compile-time manifest assembly (artifact list, parser routing, dropped atoms, …). |
| `telemetry.py` | Structured event logging (`compile_stage_completed`, `compile_finished`). |
| `validators.py` | Cross-stage Pydantic validators for compile-result invariants. |
| `risk.py` | Per-packet risk scoring (used by review-folder and OrbitBrief). |
| `invalidation.py` | Computes which packets are invalidated by a re-compile (incremental compile support). |
| `diffing.py` | Atom + packet diffs between two compile results. |
| `item_identity.py` | Canonical material-key / part-number normalization shared by parsers and graph_builder. |
| `segments.py` | Artifact segment registry (page → block IDs). |
| `file_safety.py` | Path-traversal + symlink guards. |
| `gold_compare.py` | Per-metric pass/fail comparator vs `labels/gold_standard.json`. |
| `quality_metrics.py` | `CompileQuality` derivation + telemetry hooks. |
| `review_folder.py` | Per-compile human-review folder (`REVIEW.md` + `pack_suggestions.yaml` + per-artifact dossiers). |
| `ontology_gaps.py` | Gap-detector — surfaces unknown phrases that look like devices / vendors / sites / part-numbers and recommends pack additions. |
| `compiler.py` | The orchestrator — `compile_project()`. |
| `orbitbrief_envelope.py` | Builds the `orbitbrief.input.v2` JSON + Markdown envelope. |
| `authority.py` | Authority-class lattice + helpers. |
| `candidate_adjudicator.py` | Stage 3 — duplicate/low-confidence dedup. |
| `candidates.py` | Per-parser candidate-atom sub-shapes (pre-adjudication). |
| `source_replay.py` | Stage 4 — re-extracts atom text from source. |
| `entity_resolution.py` | Stage 7. |
| `graph_builder.py` | Stage 8. |
| `graph_invariants.py` | Edge-validator (mirrors graph_builder rules). |
| `packetizer.py` | Stage 9. |
| `packet_certificates.py` | Stage 10. |

## Adding a new pipeline stage

1. Implement it as a pure function in a new `app/core/<stage>.py`
2. Wire it into `compile_project` in `compiler.py` between the appropriate
   stages, behind a `with _stage_timer("name"):` block so it shows up in
   the trace
3. Add a focused test under `tests/test_<stage>.py`
4. If it changes any data shape, update `schemas.py` (and `tests/test_schemas.py`)
5. If it produces new warnings, surface them via `quality_metrics.py` so
   the fail-loud CI hooks pick them up
