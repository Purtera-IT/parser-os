# OrbitBrief — System README

How **parser-os** and **Orbitbrief-Core** work together to turn raw
professional-services intake (RFPs, proposals, vendor quotes, kickoff
transcripts, customer emails) into a reviewable PM brief — with every
claim traceable to a source artifact.

> Two repos, one frozen contract.

```
┌──────────────────────────┐  orbitbrief.input.v2   ┌─────────────────────────────┐
│  parser-os               │ ─────────────────────► │  Orbitbrief-Core            │
│  (file-eating side)      │   (typed JSON          │  (synthesis side)           │
│                          │    envelope)           │                             │
│  19 parsers              │                        │  13 layered packages        │
│  + atom + entity         │                        │  pack_prior → planner       │
│  + graph + packets       │                        │  → brains → validator       │
│  + source replay         │                        │  → calibrator → composer    │
│  → envelope.json         │                        │  → pm_handoff               │
└──────────────────────────┘                        └─────────────────────────────┘
                                                            │
                                                            ▼
                                                    PM_HANDOFF.json (59 fields)
                                                    SOW_DRAFT.md
                                                    RFP_DRAFT.md
                                                    review_ui /queue
```

---

## Repos

| Repo | Role | Allowed to do |
|---|---|---|
| `parser-os` | Ingestion + extraction | Read raw files. Run parsers. Build atoms/entities/edges/packets. Emit envelope. **No LLM in hot path.** |
| `Orbitbrief-Core` | Synthesis | Consume envelope (only the seam may read raw envelope JSON). Run brains via Ollama. Produce PM brief / SOW / RFP. **Never reads raw artifact files.** |

The split is enforced by `import-linter` contracts in Orbitbrief-Core
(`no-direct-pdf-libs`, `substrate-no-world-model`, etc.) and a
`tools/check_no_raw_open.py` AST scan.

---

## The contract — `orbitbrief.input.v2`

A typed JSON envelope. Producer pins `schema_version`; consumer
validates at the boundary so producer drift fails loud.

```jsonc
{
  "schema_version": "orbitbrief.input.v2",
  "project_id": "...",
  "compile_id": "cmp_<hash>",
  "generated_at": "...Z",
  "output_signature": "sha256:...",        // byte-identical across runs
  "summary":   { artifact_count, page_count, atom_count, packet_count, ... },
  "documents": [ { artifact_id, filename, artifact_type, sha256,
                   parser_name, parser_version, structured: {...}, atom_ids } ],
  "atoms":     [ EvidenceAtom rows ],      // 13 atom types
  "entities":  [ EvidenceEntity rows ],    // canonical_key + aliases
  "edges":     [ EvidenceEdge rows ],      // 8 edge types
  "packets":   [ Packet rows ],            // 11 families
  "indexes":   { atoms_by_section_path, atoms_by_atom_type,
                 atoms_by_authority, atoms_by_artifact,
                 atoms_by_entity_key, edges_by_atom,
                 entity_id_by_canonical_key },
  "manifest":  { parser_routing[*].outcome }   // A6 graceful-degradation per file
}
```

Authority is defined in [parser-os/app/core/orbitbrief_envelope.py](app/core/orbitbrief_envelope.py).
Consumer schema lives in [Orbitbrief-Core/src/orbitbrief_core/seam/envelope.py](https://github.com/Purtera-IT/Orbitbrief-Core/blob/main/src/orbitbrief_core/seam/envelope.py).

### Atom types (13)

`quantity`, `entity`, `constraint`, `exclusion`, `scope_item`,
`customer_instruction`, `vendor_line_item`, `assumption`,
`open_question`, `decision`, `action_item`, `meeting_commitment`,
`compliance`, plus B-wave additions for risk + schedule.

### Authority classes

`customer_current_authored`, `customer_historical_authored`,
`vendor_quote`, `vendor_meeting`, `meeting_note`, `machine_extractor`,
`scanned_unverified`, plus the OCR/vision-derived tiers.

### Packet families (11)

`scope_inclusion`, `scope_exclusion`, `quantity_claim`,
`quantity_conflict`, `site_access`, `missing_info`,
`customer_override`, `vendor_mismatch`, `meeting_decision`,
`action_item`, `compliance_clause`.

---

## End-to-end pipeline

### Stage 1 — parser-os (file-eating)

```
project_dir/                         compile_project()
├── *.pdf      ──► orbitbrief_pdf     ─┐
├── *.docx     ──► docx_parser        ─┤
├── *.xlsx     ──► xlsx_parser        ─┤
├── *.csv      ──► quote_parser       ─┤
├── *.md       ──► markdown_parser    ─┤      ┌──────────────────────┐
├── *.eml      ──► email_parser       ─┤      │  compiler/           │
├── *.vtt      ──► transcript_parser  ─┤      │  graph_builder/      │
├── *.html     ──► html_parser        ─┼─────►│  packet_certifier/   │ ──► envelope.json
├── *.pptx     ──► pptx_parser        ─┤      │  source_replay/      │
├── *.png/.jpg ──► image_parser (OCR) ─┤      │  validators/         │
├── *.mbox     ──► mbox_parser        ─┤      │  domain/pack_router/ │
├── *.msg      ──► msg_parser         ─┤      └──────────────────────┘
├── *.odt/.ods ──► odt_parser/ods     ─┤
├── *.rtf      ──► rtf_parser         ─┤
├── *.ics      ──► ics_parser         ─┤
├── *.zip      ──► zip_parser         ─┤
├── *.vsdx     ──► vsdx_parser        ─┤
├── *.mpp      ──► mpp_parser         ─┘
```

Outputs: `result.json` (compile result) and `envelope.json` (the v2
envelope that Orbitbrief-Core consumes). Provenance receipts let
downstream consumers re-extract any atom's source text on demand.

### Stage 2 — Orbitbrief-Core (synthesis)

```
envelope.json
    │
    ▼
00_envelope.json                                (seam/ validates and copies)
    │
    ▼
01_evidence_runtime (in-memory)                 DuckDB substrate
    │
    ▼
10_pack_prior_state.json                        which domain pack(s) activate
11_site_reality_state.json                      cluster atoms by physical site
20_retrieval_bundles/<pack>.json                4 vector indices
    │
    ▼ (Ollama)
30_brief_state.raw.json                         Qwen3-14B planner
31_brief_state.refined.json                     deterministic refiner
40_brain_outputs/<pack>.json                    per-domain brain (15 today)
50_validations/<pack>.json                      5 rule families
60_calibrations/<pack>.json                     10-signal + Platt scaling
    │
    ▼
70_review_queue/                                JSONL durable review queue
80_composed_brief.json + .md                    composer aggregates brains
90_pm_handoff/PM_HANDOFF.json                   ← UI consumer payload
              PM_HANDOFF.md
              SOW_DRAFT.md
              RFP_DRAFT.md
```

`pipeline_log.json` records a typed `StageRecord` for every stage
(status, duration, escalation reasons). `manifest.json` summarizes the
run.

---

## One-shot operator commands

```bash
# Compile + brief from a raw case directory
export PARSER_OS_ROOT=/abs/path/to/parser-os-repo
python compile_brief.py case_dir/ --out artifacts/ --ollama

# Never-skip-LLM hardened version (errors loud if Ollama is down)
./pm_handoff.sh case_dir/ [out_dir/]

# Review UI
python -m orbitbrief_core.review_ui --artifacts artifacts/ --port 8765
# → http://127.0.0.1:8765/queue
```

---

## Determinism + provenance contract

Two guarantees the entire system rests on:

1. **Deterministic compile.** Two `parser-os compile` runs on the
   same artifacts produce byte-identical `output_signature` /
   `compile_id`. No randomness, no time-of-day, no global state.
2. **Provenanced atoms.** Every atom has a `SourceRef` (file, page /
   row / line, char offset) plus an optional replay receipt that
   re-extracts the same text from the original artifact bytes on
   demand. Atoms get `verified=verified|failed|partial|unsupported|
   unverified`. Replay-failed atoms surface as validator errors
   downstream.

Breaking either guarantee breaks the system contract. The regression
suite catches the obvious cases.

---

## UI consumer payload — `PM_HANDOFF.json`

The single source of truth for the operator UI. 59 top-level fields,
all JSON-serializable, all derived from the envelope.

Highlights:

| Field group | What it gives the PM |
|---|---|
| `executive_summary`, `one_line_summary`, `metrics` | Top-of-brief situational awareness |
| `domains`, `sites`, `site_rollups`, `site_allocations` | Domain pack + physical-site lens |
| `gaps`, `customer_questions`, `customer_answer_slots` | Open questions to send back to the customer |
| `facts_by_category`, `risk_register`, `compliance_callouts` | The evidence wall |
| `schedule_phases`, `phase_dependencies`, `critical_path_chain`, `critical_path` | Mermaid Gantt + critical-path |
| `action_items`, `actions_by_week`, `acceptance_by_site`, `acceptance_checks` | What to do and when |
| `money_mentions`, `currency_mentions`, `currency_conversions`, `reconciliation_flags`, `quantity_claims`, `quantity_contradictions` | Cross-doc numeric reconciliation |
| `rfp_line_items`, `rfp_draft_markdown` | RFP draft + line items |
| `sow_draft_markdown` | SOW draft (governing-law placeholders explicit) |
| `stakeholder_pagers`, `stakeholder_contacts` | Role-lens (CFO / IT / Procurement) + contact directory |
| `parser_quality_score`, `run_telemetry`, `drift_snapshot`, `urgency_signals` | Run health + drift vs. prior compile |
| `engagement_model`, `margin_view`, `license_items`, `tax_clauses`, `sla_penalties`, `change_order_triggers` | Commercial + risk signals |
| `lead_time_flags`, `eol_flags`, `subcontractor_mentions`, `resource_conflicts`, `risk_aging` | Operational risks |

Full field catalog with concrete example values:
[OUTPUTS_FOR_UI.md](OUTPUTS_FOR_UI.md).

---

## A6 graceful degradation

When a parser fails on one of `N` artifacts, the rest of the compile
still succeeds. Per-file outcome is recorded on `manifest.parser_routing[*]`:

| Outcome | Meaning |
|---|---|
| `ok` | Parser succeeded |
| `ok_empty` | Parser ran but produced zero atoms (often instructional sheet) |
| `failed_parse` | Parser raised. `status_reason` carries the exception text |
| `skipped_no_parser` | No parser matched the suffix |
| `unknown` | Fallback |

PMHandoff renderers show degraded files in a separate callout so the
systems engineer knows which files to manually inspect instead of
silently producing a thinner envelope.

---

## Deployment

The current target is **Azure** for the production deployment, with
local Ollama (or remote vLLM) for the LLM tier. See
[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for:

* API contract + endpoint shapes
* SQL / Cosmos table schemas
* Azure Blob layout for envelopes + artifacts
* Entra ID auth + role mapping (`pm.reader`, `pm.editor`, `auditor`, `admin`)
* Dockerfile spec
* Container Apps job pattern
* First-five-deals checklist

---

## Models

All Ollama-served (local or remote via `OLLAMA_BASE_URL`):

| Model | Role | Size |
|---|---|---|
| `qwen3:14b` | Planner default tier; all brains; escalation tiebreaks | 9.3 GB |
| `qwen3:32b` | Planner escalation (contradiction density > 5 %, ambiguous pack, …) | 20 GB |
| `qwen3-embedding:8b` | All retrieval indices (4096-dim) | 4.7 GB |

Every JSON-emit system prompt includes Qwen3's `/no_think` directive
to skip reasoning overhead. Runners budget 8192 output tokens to
absorb leftover `<think>` markers.

---

## Where to look

| You want to … | Go to |
|---|---|
| Add a new file format | `parser-os/app/parsers/` + register in `registry.py` |
| Add a new domain pack | `parser-os/app/domain/*.yaml` + `pack_router.py` |
| Add a new brain | `Orbitbrief-Core/src/orbitbrief_core/brains/<name>/` |
| Add a new PM_HANDOFF field | `Orbitbrief-Core/src/orbitbrief_core/pm_handoff/models.py` + `builder.py` + `pm_intelligence.py` |
| Add a new SOW section | `Orbitbrief-Core/src/orbitbrief_core/pm_handoff/sow_draft.py` |
| Change OCR backend | `parser-os/app/parsers/_ocr_chain.py` (`available_backends()` probe) |
| Wire EoX / vendor lookups | adapters next to `pm_handoff/pm_intelligence.py` |
| Boot the UI | `python -m orbitbrief_core.review_ui --artifacts artifacts/` |
| Deploy to Azure | [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) |
| See every PM_HANDOFF field | [OUTPUTS_FOR_UI.md](OUTPUTS_FOR_UI.md) |

---

## Site schematic page parser

A separate sub-system lives in `parser-os/orbitbrief_page_os/` and
`Orbitbrief-Core/docs/site_schematic/`. It parses architectural /
electrical / network drawings into a typed page state. Not in the
main intake path today — invoked separately and exposes a different
contract. **Treat as out-of-scope for changes to the intake
pipeline.**

---

_Architecture is enforced by 12 import contracts + a raw-IO ratchet +
an AST scan + a regression suite. Drift fails loud._
