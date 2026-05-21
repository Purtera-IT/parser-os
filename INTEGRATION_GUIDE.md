# Integration Guide — UI + Azure Engineer Handoff

This guide hands off everything needed to wire parser-os +
OrbitBrief into a production UI on Azure. It covers:

1. **System architecture** — what runs where
2. **API contract** — exact endpoints to expose
3. **Data flow** — file → atoms → brief → UI
4. **Storage layout** — Azure Blob / Files / SQL plan
5. **Authentication** — Entra ID / role mapping
6. **Deployment** — container plan + Bicep / Terraform pointers
7. **Real-data wiring** — what's scaffolded and needs production data
8. **Scaling considerations**
9. **What to test on first 5 real deals**

---

## 1. System architecture

```
┌────────────────────────────────────────────────────────────┐
│                      UI (Azure App Service)                │
│      Next.js / SvelteKit reading PM_HANDOFF.json           │
└────────────────────────┬───────────────────────────────────┘
                         │ HTTPS + Entra ID
                         ▼
┌────────────────────────────────────────────────────────────┐
│            API Gateway (Azure Functions / FastAPI)         │
│  /api/compile     POST  ← user uploads deal folder         │
│  /api/brief/:id   GET   ← read PM_HANDOFF.json             │
│  /api/audit/:id   GET   ← read envelope + manifest         │
│  /api/diff        POST  ← envelope_diff CLI as service     │
│  /api/answer      POST  ← attach customer answer to slot   │
└──┬─────────────────┬───────────────────────────────────────┘
   │                 │
   ▼                 ▼
┌──────────┐  ┌──────────────────────────────────────────────┐
│ Worker   │  │  Storage (Azure Blob + Files + SQL)          │
│ pool     │  │   • Raw deal artifacts (Blob)                │
│ (Azure   │  │   • Compile outputs per deal (Blob)          │
│ Container│  │   • PM_HANDOFF.json index (SQL or Cosmos)    │
│ Apps     │  │   • corpus_history.jsonl (Files share)       │
│ jobs)    │  │   • customer_answers (SQL)                   │
└──┬───────┘  └──────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────┐
│  parser-os + OrbitBrief container                           │
│   • Runs ``python compile_brief.py <input> --out <output>`` │
│   • Reads from Blob, writes back to Blob                    │
│   • Optionally posts to Ollama (Mac Studio / H100) for LLM  │
└────────────────────────────────────────────────────────────┘
```

Worker mode: a queue triggers a compile per deal. The worker pulls
input artifacts from Blob, runs `compile_brief.py`, writes outputs
back to a per-deal Blob folder, then pushes a "compile complete"
message that the UI can poll on.

---

## 2. API contract

These are the endpoints the UI will call. All return JSON; auth via
Entra ID bearer token.

### POST /api/compile

Trigger a compile of a deal folder.

```json
// Request
{
  "deal_id": "OPTBOT_Atlanta_Office_Refresh",
  "input_uri": "blob://orbitbrief/deals/OPTBOT_2026Q2/raw/",
  "ollama_base_url": "http://100.114.102.122:11434",   // optional
  "ollama_vision_model": "llava"                       // optional
}

// Response (202 Accepted)
{
  "compile_id": "cmp_d0cfff3a0d7e556a",
  "status_uri": "/api/compile/cmp_d0cfff3a0d7e556a"
}
```

### GET /api/compile/:compile_id

Poll a compile job.

```json
// Response
{
  "compile_id": "cmp_d0cfff3a0d7e556a",
  "status": "running" | "complete" | "failed",
  "progress_pct": 78,
  "current_stage": "graph_build",
  "output_uri": "blob://orbitbrief/deals/OPTBOT_2026Q2/out/",
  "error": null
}
```

### GET /api/brief/:deal_id

Fetch the PM brief — single payload, everything inline.

```json
// Returns the full PM_HANDOFF.json content (147 KB on OPTBOT)
{
  "case_id": "...",
  "status": "red",
  "executive_summary": {...},
  "parser_quality_score": {...},
  "sow_draft_markdown": "...",
  "rfp_draft_markdown": "...",
  // ... all 58 fields per OUTPUTS_FOR_UI.md
}
```

### GET /api/audit/:deal_id

Fetch audit-side details.

```json
// Returns envelope summary + manifest + pipeline_log
{
  "manifest": {...},                  // from manifest.json
  "envelope_summary": {...},          // envelope.summary
  "documents":  [...],                // envelope.documents (drilldown)
  "atoms_sample": [...100 of 135...], // envelope.atoms paginated
  "pipeline_log": [...],              // pipeline_log.json
  "verification": {...}               // 90_inspection_report.json.verification
}
```

### POST /api/answer

Attach a customer answer to a gap question.

```json
// Request
{
  "deal_id": "...",
  "question_id": "commercial.pricing_structure",
  "answer": "Fixed fee at $1.85M",
  "answered_by": "jordan.ames@optbot.example"
}

// Response
{ "status": "ok", "slot_status": "answered" }
```

The answer persists to SQL; next compile reads back to fill the
`customer_answer_slots[*].answer` field.

### POST /api/diff

Run envelope diff between two compiles.

```json
// Request
{ "before_compile_id": "cmp_a", "after_compile_id": "cmp_b" }

// Response — markdown rendered + structured diff
{
  "markdown": "...",
  "added_files": [...],
  "removed_files": [...],
  "money_added": [...],
  "money_removed": [...],
  "risk_changes": [...]
}
```

### GET /api/portfolio

Cross-deal rollup view.

```json
// Reads all PM_HANDOFF.json in the user's tenant + corpus_history.jsonl
{
  "cases": [...],
  "totals": {
    "aggregate_deal_value": 12_500_000,
    "high_risk_count": 14,
    "compliance_callouts_total": 35
  }
}
```

---

## 3. Data flow

For one deal:

```
1. User uploads /api/compile
   └─ artifacts land in blob://orbitbrief/deals/<deal_id>/raw/

2. Worker job picks up the queued compile
   └─ docker run -v <raw>:/in -v <out>:/out parser-os-orbitbrief \
         python compile_brief.py /in --out /out

3. Compile produces /out:
   ├── 00_envelope.json          (parser-os atoms / packets / edges)
   ├── 10_pack_prior_state.json  (OrbitBrief pack activation)
   ├── 11_site_reality_state.json (OrbitBrief site clusters)
   ├── 90_inspection_report.json (funnel + verification)
   ├── 91_inspection_report.html (browsable HTML)
   ├── PM_HANDOFF.json           ← PRIMARY UI PAYLOAD
   ├── PM_HANDOFF.md / .html
   ├── SOW_DRAFT.md              (also embedded inside PM_HANDOFF.json)
   ├── RFP_DRAFT.md              (also embedded inside PM_HANDOFF.json)
   ├── PM_EXECUTIVE_SUMMARY.md / .html
   ├── SA_REVIEW_PACKET.md / .html
   ├── manifest.json
   ├── pipeline_log.json
   └── .orbitbrief_history.jsonl (append-only, shared across deals)

4. Worker writes outputs back to blob://orbitbrief/deals/<deal_id>/out/

5. UI fetches PM_HANDOFF.json + (optionally) manifest + pipeline_log
   for the audit tab.
```

---

## 4. Storage layout

### Azure Blob Storage

```
orbitbrief/
├── deals/
│   ├── OPTBOT_2026Q2/
│   │   ├── raw/
│   │   │   ├── 01_deal_overview_executive_brief.pdf
│   │   │   ├── 02_statement_of_work.docx
│   │   │   └── ... (input artifacts)
│   │   ├── out/
│   │   │   ├── PM_HANDOFF.json
│   │   │   ├── 00_envelope.json
│   │   │   ├── manifest.json
│   │   │   ├── pipeline_log.json
│   │   │   └── ... (compile outputs)
│   │   └── history/
│   │       └── cmp_<id>/              ← snapshot of each prior compile
│   └── ...
└── shared/
    └── corpus_history.jsonl           ← cross-tenant deal corpus
```

### Azure SQL / Cosmos DB

Tables:

**`deals`** — one row per deal
```
deal_id            TEXT PRIMARY KEY
display_name       TEXT
owner_user_id      TEXT
status             TEXT      (red / yellow / green)
parser_quality     INT       (0-100)
parser_grade       TEXT      (A+ / A / B / C / D / F)
deal_value_usd     BIGINT
margin_pct         REAL
blocker_count      INT
warning_count      INT
last_compile_id    TEXT
last_generated_at  TIMESTAMP
```

**`compiles`** — one row per compile run (for drift)
```
compile_id         TEXT PRIMARY KEY
deal_id            TEXT
input_signature    CHAR(64)
output_signature   CHAR(64)
created_at         TIMESTAMP
parser_quality     INT
total_duration_ms  INT
parse_outcomes     JSONB    (counts by status)
```

**`customer_answers`** — one row per answered slot
```
deal_id            TEXT
question_id        TEXT
answer             TEXT
answered_by        TEXT
answered_at        TIMESTAMP
status             TEXT     (open / answered / deferred)
PRIMARY KEY (deal_id, question_id)
```

**`audit_notes`** — auditor's persistent notes (for next month's review)
```
deal_id            TEXT
compile_id         TEXT
auditor_user_id    TEXT
note               TEXT
created_at         TIMESTAMP
```

### Azure Files share (corpus history)

`corpus_history.jsonl` lives on a Files share mounted into all
worker containers so each compile appends one row. The UI reads
the same file for `comparable_deals` lookups.

---

## 5. Authentication

- **User auth:** Entra ID (Azure AD). UI redirects to Microsoft
  login; API validates JWT with the configured tenant.
- **Worker auth:** Managed Identity. Worker reads Blob via MI;
  no secrets in container.
- **Ollama:** Tailscale-on-host for parser-os → Ollama (Mac
  Studio today; H100 box next week). Set
  ``PARSER_OS_OCR_OLLAMA_BASE_URL`` env in the container.
- **Role mapping:**
  - `pm.reader` → can read PM brief + SOW/RFP
  - `pm.editor` → above + can submit `/api/answer`
  - `auditor` → can read everything including envelope + audit notes
  - `admin` → above + can run `/api/compile`

---

## 6. Deployment plan

### Container image

`Dockerfile` in repo root (TBD — current code runs from a Python
venv; production image needs:)

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY parser-os/ /app/parser-os/
COPY Orbitbrief-Core/ /app/Orbitbrief-Core/
RUN pip install -e /app/parser-os
RUN pip install -e /app/Orbitbrief-Core

ENV PARSER_OS_OCR_OLLAMA_BASE_URL=""
ENV ORBITBRIEF_CORPUS_HISTORY=/data/corpus_history.jsonl

ENTRYPOINT ["python", "/app/Orbitbrief-Core/compile_brief.py"]
```

### Azure Container Apps job

- **Compile worker**: triggered by Service Bus queue messages. One
  message per `/api/compile` call. Container runs the compile,
  writes Blob outputs, sends completion message.
- **API service**: long-running container exposing the endpoints
  above. Stateless.
- **UI**: Azure App Service hosting the Next.js / SvelteKit app.

### Bicep / Terraform pointers

Out of scope for this doc — coordinate with Azure engineer on
existing infra patterns.

---

## 7. Real-data wiring (what's scaffolded and what needs the engineer)

| Feature | Scaffold status | Needs from engineer |
|---|---|---|
| `customer_answer_slots` | Empty slots in PM_HANDOFF.json with question_id | (1) SQL table `customer_answers`; (2) POST /api/answer; (3) re-compile reads back answers to populate `slot.answer` |
| `drift_snapshot` | First-run flag + delta shape ready | Multi-compile history persisted across runs (already writes to corpus_history.jsonl) |
| `comparable_deals` | Reader + writer wired; query by deal value + domains | Needs ≥10 real deal compiles to bootstrap a useful corpus |
| `urgency_signals` | Regex-based detection working | (Optional) LLM lens for tone / sentiment when production wants higher accuracy |
| `run_telemetry` | Reads pipeline_log + manifest at brief-time | Nothing — already works |
| Engagement model | Regex tuned; catches T&M / Fixed-fee / Subscription | Real customer language survey to add common phrasings we missed |
| Hardware EOL | 30+ SKUs static + Cisco EoX adapter when `CISCO_EOX_OAUTH_TOKEN` set | Cisco / HPE / Crestron API credentials in Key Vault |
| FX rates | Frankfurter free feed wired + offline fallback | (Optional) Bloomberg / OpenExchange wire for SOX-grade rates |
| OCR backends | 4-path chain (PyMuPDF Tess → pytesseract → easyocr → Ollama vision) | Install one: Tesseract binary in container OR pull `llava` on Ollama |
| Audit notes | Schema defined | SQL table + UI textarea |

---

## 8. Scaling considerations

- **Throughput**: a single compile on OPTBOT (7 small files) takes
  ~10 s on commodity hardware. Real enterprise deals (50+ files)
  will land 30–120 s. Plan **1 compile worker per concurrent deal**.
- **Storage**: each deal's output dir is ~1 MB compressed. 10 K deals
  = ~10 GB. Use lifecycle rules to archive completed compiles after
  90 days.
- **Cold-start**: parser-os imports take ~2 s. Use Container Apps
  warm-instance settings for the API container; worker can be
  cold-start because compiles are async.
- **LLM stages**: when LLM lenses run (Phase 1.75 `envelope_backfill_v2.py`),
  each lens × atom takes ~14 s on Mac Studio (single-GPU Ollama).
  On H100 expect 2–4× speedup with parallel batches. **Don't
  enable LLM stages on every compile**; gate behind a feature
  flag and run nightly OR on user request.
- **PM_HANDOFF.json size**: 147 KB on OPTBOT. Real deals will be
  300–700 KB. Stay well under any HTTP body cap.

---

## 9. What to test on first 5 real deals

This is the audit / readiness checklist. For each of your first
5 real deals, run a compile and verify:

### Audit-side checks

- [ ] `parser_quality_score.score` lands in 80–100 (B or better)
- [ ] `parser_quality_score.components.confidence_histogram.high_pct` ≥ 50%
- [ ] `envelope.summary.parse_outcomes.ok` == count of files dropped in
- [ ] `envelope.summary.degraded_files` is empty (or, when non-empty, the failures are explainable: scanned PDFs, password-protected, etc.)
- [ ] `pipeline_log.json` shows non-zero output for every stage
- [ ] `verification.verified_pct` ≥ 90% (low-90s is OK; <80% means parsers regressed)
- [ ] `output_signature` is stable across two compiles of the same input (deterministic)
- [ ] Per-stage durations are reasonable; no stage takes >30 s

### PM-side checks

- [ ] `executive_summary.headline` names the deal value + sites correctly
- [ ] `intake_completeness` correctly flags missing items (cross-check against your eye)
- [ ] `margin_view` either: shows real margin %, OR flags zero-margin SOW, OR clearly labels confidence as "low" when data is partial
- [ ] `stakeholder_contacts` includes the right people with role + email (not generic role like "—")
- [ ] `risk_register` has the actual risks from the source SOW
- [ ] `compliance_callouts` catches every named framework (SOC 2, HIPAA, PCI, etc.) in the source
- [ ] `reconciliation_flags` catches the real money mismatches you'd spot manually
- [ ] `action_items` doesn't repeat the same task 3 times
- [ ] `customer_answer_slots` has one entry per gap question
- [ ] `urgency_signals` correctly flags emails / messages where customer is escalating
- [ ] SOW Section 19 jurisdiction shows the `[FILL: …]` fields ready to edit
- [ ] SOW Section 21 signatures block renders correctly

### UX checks

- [ ] PM can scan the brief in <5 minutes and know the deal status
- [ ] Critical-path section either highlights real critical phases OR cleanly says "all sequential"
- [ ] Stakeholder pagers (CFO / IT / Procurement) actually filter to lens-relevant content
- [ ] SOW draft is ~80% editable (only `[FILL: …]` placeholders left)
- [ ] RFP draft has categorized vendor packets ready to send

---

## 10. What's still TBD for production

These are the cleanest pieces the Azure engineer will own and that
the current scaffolding is ready for:

1. **Auth + multi-tenant isolation**: each customer org sees only
   their deals. Implement at API gateway with Entra ID tenant
   claims.
2. **Audit notes persistence**: SQL table + textarea in UI.
3. **Customer answer ingestion**: POST /api/answer + re-compile
   trigger.
4. **HubSpot / Salesforce push-back**: write deal status changes
   back to source CRM after PM acceptance.
5. **DocuSign / Adobe Sign integration**: render SOW_DRAFT.md →
   PDF → send for e-sign.
6. **Email-thread ingestion**: receive customer emails into a
   per-deal inbox; auto-attach replies to `customer_answer_slots`.
7. **Time-tracking integration**: pull billable hours from a time
   system; compare to T&M cap from `engagement_model.tm_cap_amount`.

---

## File checklist for the engineer

When you start, you'll have these in the repo:

```
parser-os/
├── app/                          ← parser-os source
├── OUTPUTS_FOR_UI.md             ← complete field catalog
├── INTEGRATION_GUIDE.md          ← THIS FILE
└── tests/

Orbitbrief-Core/
├── src/orbitbrief_core/          ← OrbitBrief source
├── compile_brief.py              ← CLI entry point
├── tools/
│   ├── envelope_diff.py          ← deal-diff service
│   └── setup_vision_ocr.py       ← Ollama vision-model pull
└── tests/
```

That's everything. Build the API around `compile_brief.py`, wire
the UI to `PM_HANDOFF.json`, follow this guide for the rest.

Questions? The two source-of-truth docs are
[OUTPUTS_FOR_UI.md](OUTPUTS_FOR_UI.md) (every field) and this guide
(every endpoint + storage shape + auth + deploy plan).
