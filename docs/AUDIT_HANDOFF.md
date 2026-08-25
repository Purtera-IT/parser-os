# OrbitBrief Full-System Audit — Handoff & Review Path

**Written 2026-08-25 by the session that did the work. Audience: a fresh Claude chat on the MacBook/Mac Studio whose mission is to independently audit everything below, verify every receipt, and then explain the entire flow and both visualizations to Lilli in detail.**

Nothing in this document should be taken on faith — that is the point of it. Every claim carries a way to check it. Where a claim and reality disagree, reality wins and the discrepancy is a finding.

---

## 0. Mission

1. **Pull the latest code** (Section 1) and **connect to Azure** (Section 2).
2. **Verify the deployed state** matches the receipts table (Section 3) — trust the running containers, not this doc.
3. **Walk the entire flow** (Section 4) file-by-file until you can explain it without notes.
4. **Audit the audit** (Section 5): read the invariance test files as the primary record, re-run the suites, and spot-check that the pinned bleeds are really fixed.
5. **Open both visualizations** (Section 6) and check every figure on them against the code you pulled.
6. **Deliver**: a detailed spoken-level explanation of the whole system and both visualizations, plus a list of anything you found that disagrees with this document.

---

## 1. Get the code (macOS)

Three repos, three refs. Clone fresh or pull — the refs below are the truth as of writing:

```bash
mkdir -p ~/orbit-audit && cd ~/orbit-audit
git clone https://github.com/Purtera-IT/parser-os.git            # main @ f523475
git clone https://github.com/Purtera-IT/Orbitbrief-Core.git      # deploy ref = feat/neural-heads @ 914763c  (NOT main!)
cd Orbitbrief-Core && git checkout feat/neural-heads && cd ..
git clone https://github.com/Purtera-IT/purpulse-fe.git          # main (frontend; the chip lives here)
```

⚠️ **Orbitbrief-Core deploys from `feat/neural-heads`, not `main`.** Auditing core's `main` will show you stale code and phantom findings.

⚠️ **purpulse-fe pushes straight to production `main` on Lovable.** Look, don't touch, unless explicitly asked — and never `git add -A` there.

Python: both parser-os and core run pytest with plain `python -m pytest tests -q` from the repo root (core needs `src/` on path — its `pytest.ini` handles it; core's seam tests also import parser-os, so `pip install -e` parser-os or set `PYTHONPATH` to include it). The Windows-only gotchas (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, az cp1252) should not apply on macOS, but if pytest dies silently with no output, that env var is the first thing to try anyway.

---

## 2. Azure access

```bash
az login                      # account with purtera-dev-rg access
az account show               # confirm subscription
```

Everything lives in resource group **`purtera-dev-rg`**, registry **`purpulsedevacr`**:

| resource | name | role |
|---|---|---|
| Container App | `parser-os-service-dev-eus2` | HTTP service: /v1/compile/async, /feedback/correction, /v1/version |
| Container App | `parser-os-worker-warm` | warm parse worker (stage 1) |
| Container App Job | `parser-os-worker-dev-eus2` | queue-driven parse worker (stage 1) |
| Container App | `orbitbrief-core-worker-dev-eus2` | brief-gen worker (stage 2) |
| Blob storage | container `orbitbrief-artifacts` | envelopes, briefs, `_feedback/` mirror, training DBs |

The internal bearer for service endpoints:

```bash
BEARER=$(az containerapp secret show -n parser-os-service-dev-eus2 -g purtera-dev-rg \
  --secret-name bang-internal-bearer --query value -o tsv)
```

Service base URL: `https://parser-os-service-dev-eus2.whitehill-a3348ba5.eastus2.azurecontainerapps.io`

---

## 3. What is deployed right now (receipts — verify, don't trust)

| target | revision | image digest | runs code |
|---|---|---|---|
| parser-os service | `--0000088` | `sha256:38f2962c…` | parser-os `main @ f898f2e` |
| parser-os worker-warm | `--0000223` | `sha256:512f8f6f…` | parser-os `main @ f898f2e` |
| parser-os worker job | (template) | `sha256:512f8f6f…` | parser-os `main @ f898f2e` |
| orbitbrief-core worker | `--0000117` | `sha256:4ce4485d…` | core `feat/neural-heads @ 914763c` (bundles parser-os main) |

Note: parser-os `main` HEAD is `f523475`, one **docs-only** commit past the deployed `f898f2e` — deployed *code* equals main code; only `docs/` differs. If more commits have landed since, re-derive this table before relying on it.

**Verify each row yourself:**

```bash
# service: the container tells you its bundled sha
curl -sS "$SVC/v1/version" | python3 -m json.tool          # expect parser_os_sha f898f2e…

# workers: grep the actual fix out of the running container
az containerapp exec -n parser-os-worker-warm -g purtera-dev-rg \
  --command "python -c \"import inspect, app.core.packetizer as pk; print('s+included' in inspect.getsource(pk)); import os; print(os.environ.get('PARSER_OS_SHA'))\""

az containerapp exec -n orbitbrief-core-worker-dev-eus2 -g purtera-dev-rg \
  --command "python -c \"import sys; sys.path.insert(0,'/app/Orbitbrief-Core/src'); from orbitbrief_core.pm_handoff import render_markdown as rm; print(hasattr(rm,'_render_reconciliation_verdicts'))\""
```

**The deploy discipline these receipts come from** (audit it as a process): images are tagged `sha-main-<8sha>` with `PARSER_OS_SHA` taken from `git rev-parse` (never retyped — a retyped sha shipped once and was caught by the verify step), rolled by **immutable digest** (never a mutable tag — a mutable-tag roll once shipped week-old code silently), and verified **in-container** after every roll (never by CI faith).

---

## 4. The entire flow, end to end

### Stage 1 — parser-os (a file becomes evidence)

Read in this order; each step names the on-disk anchor.

1. **Routing** — `app/parsers/registry.py`. Evidence in order of strength: container magic bytes > content structure > extension > filename (scored below threshold on purpose: a name may raise a claim, never create one). Ties break on content. Nothing clears 0.50 → magic-byte sniff → abstain, recorded. 21 registered parsers, 5 families.
2. **Reading** — `app/parsers/orbitbrief_pdf.py` (~5,300 lines; page router: <80 chars → OCR/vision, ≥1200 → prose splitter; schematic cut at 25 lines/400 chars — thresholds receipted in `app/core/calibration.py`), plus OOXML / mail / text&markup / containers families.
3. **Atoms** — typed `EvidenceAtom`s (`app/core/schemas.py`): scope_item, exclusion, constraint, quantity, vendor_line_item, … each with `authority_class` (the **authority lattice**: contractual_scope 100 > pm_confirmed 95 > customer_current_authored 90 > approved_site_roster > vendor_quote 65 > meeting_note > machine_extractor > quoted_old_email > deleted_text), `entity_keys` (`site:…`, `device:…`, `quantity:…`), confidence, receipts. Coverage floors (raw_table_row / raw_utterance / plain_text_floor) run **alongside, never instead**; files nothing claims are recorded as abstains — a silent zero and a real zero must never look the same.
4. **Post-parse heads** — `app/core/site_detection.py`, `zero_miss.py` (PM-critical vocab recall sweep), `entity_extraction.py`, `vision_extraction.py`, `atom_type_sanity.py`.
5. **Graph** — `app/core/graph_builder.py` `build_edges()`: supports / contradicts / excludes / requires edges across artifacts (edge families like `quantity_contradiction`; noun-anchored quantity binding; input-order invariant — pinned).
6. **Packets** — `app/core/packetizer.py`: certified evidence packets (e.g. `quantity_conflict` requires ≥2 quantity atoms + a contradicts edge + multi-source provenance) — the PM-facing "here is a real issue with receipts" unit.
7. **Phase-2 reconciliation** — `app/core/reconcile.py`: conflict sets → the authority lattice resolves (winner + superseded + reason + edge ids) or **honestly refuses** (top-tier ties stay open). Injectable `Potential` hook so a learned model can replace the lattice when a corpus earns it. Output: `envelope["reconciliation"]` with the 41%-precision caveat attached (the contradicts-proposing rule measured 0.41 precise on 447 labelled pairs — every entry is a lead with receipts, not a verdict).
8. **Envelope** — `app/core/orbitbrief_envelope.py`: everything above serialized as `orbitbrief.input.v2` JSON → blob (`orbitbrief-artifacts/...\<deal>/latest/00_envelope.json`). Indexes (all sorted at emission — order-invariant), foreign-artifact check (a filename claiming a distant deal number is flagged, never dropped), compact atoms with `calibrated_confidence`.

### Stage 2 — Orbitbrief-Core (evidence becomes a brief)

**A core compile never re-parses** — it consumes the envelope. Two-stage topology: `parser-os-worker` parses; `orbitbrief-core-worker` briefs.

1. **Border** — `src/orbitbrief_core/seam/envelope.py`: `EnvelopeV2` pydantic model, `extra="allow"` (the producer can ship new keys first; `summary` is required). The contract doc is `contracts/orbitbrief.input.v2.yaml` (currently behind the model — known cosmetic gap).
2. **Consumers** — `pm_handoff/builder.py` (reads `service_routing` for the domain view via `_router_primary` — defensive against abstain/missing/string-dress; reads `reconciliation` via `_build_reconciliation_verdicts` — pass-through with guardrails, caps 50, counts stay uncapped), `neural_heads/_common.py` (`resolve_deal_total` — authoritative field first, string-dress coerced), 21 managed-service brains, gap/risk heads, calibrator.
3. **Render** — `pm_handoff/render_markdown.py` → PM_HANDOFF .json/.md/.html (HTML wraps the markdown). The new section: **"Cross-document conflicts — resolved by authority, and the ones that need you"** — open ties first, resolved verdicts with receipts, caveat in-line. Renders nothing on envelopes that predate the key.

### The learning loop (a PM's tap becomes training gold)

`purpulse-fe` `CorrectionChip` (on OrbitBrief page routing card) → POST `/feedback/correction` (parser-os service, `app/api/routes_feedback.py`) → `pm_correction_to_correction` (`app/core/pm_feedback.py`) → FeedbackStore + **blob mirror** (`app/core/feedback_blob.py` → `_feedback/corrections/` + a `teacher=pm` training row; the store itself is ephemeral /tmp — blob is the durable truth) → worker `sync_into_store` per compile → corrections **govern the next compile** (a pm answer becomes an EvidenceAtom at rank 95) → `_train_backbone.py` (GO#5 trainer: one command, six pinned decisions) consumes `teacher=pm` rows. GPU fine-tune rents at **≈100 real taps**; today's verified human gold count: **0** (the "218 pm rows" were graded and exposed as rank-laundered pipeline rows).

History worth knowing: this pipe was **broken in production until 2026-08-25** — the deployed service ran Aug-4 code, corrections died in /tmp, the blob mirror had never written one object. Every PM tap made a success toast and evaporated. Fixed, rolled, proven with four receipts. Grade this repair when you audit (the receipts are in the git history of that day and in `_feedback/` blob objects).

---

## 5. The seam-audit campaign (audit the audit)

**The law, 6-for-6: every bug lived at a representation boundary — two dressings of the same truth disagreeing. Zero bugs were inside the rules.** Method: hold content constant, vary the incidental dress (casing, punctuation, whitespace, line wraps, thousands separators, input order, import success), and count what changes that shouldn't.

| stop | bleeds | headline examples | pinned in | landed as |
|---|---|---|---|---|
| parsers (hijack sweep, 2,500 files) | routing hijacks fixed | filename/style-name/tab-name deciding what content should | parser fixture suites | earlier PRs |
| routers | filename-gate leak | — | router tests | earlier PRs |
| post-parse heads | **10** | "St. Louis" ≠ `st_louis` (a period); covered-term check that could never match → every covered multi-word PM term re-escalated to the LLM on every deal; PDF hyphen-wrap blind recall sweep; unguarded `quantity:` keys; vision tile-dup rows | `tests/test_head_invariance.py` (43) | parser-os PR #38 |
| graph + packetizer | **7** | `\b\d{1,5}\b` split "1,000" into 1+000 → bound nothing; `"1,000"` parsed to None → dropped from reconciliation; literal single spaces → "not\nincluded" (PDF line wrap) hid an **explicit exclusion** | `tests/test_graph_packet_invariance.py` (35; the order-invariance test asserts its own fixture is non-vacuous) | parser-os PR #39 |
| envelope | **5** | `_` is a word char → `\bCDW\b` never matched "CDW_Quote.xlsx"; leading space defeated anchored `^(\d{6})` → file exempt from misfiling check; a task-tier `continue` dropped non-quote-line task atoms from **every** entity index — only when a classifier import succeeded | `tests/test_envelope_invariance.py` (11) | parser-os PR #40 |
| the border (core) | **3 + 1 pipe** | `revenue: "48,500.00"` (legal JSON dress) → headline deal total = None; `envelope["reconciliation"]` had **zero consumers** (dead pipe, now lit) | core `tests/test_border_invariance.py` (18) + `tests/test_reconciliation_verdicts.py` (7) | core PR #64, #65 |

**What held and is pinned so it keeps holding:** `atom_type_sanity` 30/30 probes clean; `build_edges` identical edge set under any atom input order; material keys collapse "Cat6A"/"Cat 6A"/"CAT6A patch"/"cat-6a" to one identity; envelope indexes sort at emission; core's 29 domain packs ↔ FE's 29 WORKSTREAMS agree exactly; all 29 chip labels store as corrections (label-space complete even though the deployed router head only knows ~4 packs — a known coverage limitation, not a bug).

**Your audit moves:**
```bash
cd parser-os        && python -m pytest tests/test_head_invariance.py tests/test_graph_packet_invariance.py tests/test_envelope_invariance.py -q
cd ../Orbitbrief-Core && python -m pytest tests/test_border_invariance.py tests/test_reconciliation_verdicts.py -q
# then both full suites; then read those five files top to bottom — they ARE the audit record,
# each fixed bleed is a test with a comment saying what broke and why it mattered.
```
Bonus rigor: invent 2–3 new dress probes per layer that the pins DON'T cover, and run them. The campaign claims the seams are closed — try to reopen one.

---

## 6. The visualizations (in git, on main)

| file | what it is |
|---|---|
| `parser-os/docs/parser_flow.html` | **"Every path a file takes"** — the single board: evidence ladder → 5 reader families with inside-reader routing → segments → typed atoms → coverage floors → UNCOVERED → the one-rule footer → the **AFTER THE ATOMS** band (seam law / tap-pipe repair / deploy truth). Carries a **board receipts line**: code figures re-derived 2026-08-25 against `main@f898f2e`; corpus findings (84-page router probe, 2,062-file census, lost-sections, 11-atom page) are dated 2026-08 measurements. |
| `parser-os/docs/parse_map.html` | The deep narrative map — per-parser detail plus the full chapter history including "The pipe repair, and the seam audits". |

Open them in a browser (they render dark/light by theme). **Audit task:** re-derive every number — parser count from the registry, pdf-family line count (`wc -l` the 6 modules: `orbitbrief_pdf.py`, `pdf_image_gate.py`, `pdf_image_vision.py`, `vision_extraction.py`, `schematic_embedder.py`, `zero_miss.py` ≈ 8,400 as of the receipt date), threshold values against `app/core/calibration.py`. Anything drifted → finding.

---

## 7. The detailed review path (do it in this order)

1. **Setup** (Sections 1–2). Confirm refs match or note drift.
2. **Deployed-state verification** (Section 3). Every row of the receipts table, from the running containers.
3. **Visualizations first** (Section 6) — get the map before the territory.
4. **Walk stage 1** (Section 4) with the flow board open: pick one real deal envelope from blob (`az storage blob list … --prefix` under `orbitbrief-artifacts`, download `latest/00_envelope.json`) and trace actual atoms → edges → packets → reconciliation → envelope keys against the code.
5. **Walk stage 2**: same envelope through core's readers; find the PM_HANDOFF outputs on blob for that deal; confirm the reconciliation section renders (or correctly renders nothing on a 0-conflict deal).
6. **Audit the audit** (Section 5): suites, the five pin files, then your own novel probes.
7. **Audit the loop**: read `pm_feedback.py` → `feedback_blob.py` → worker sync → `_train_backbone.py`; check `_feedback/corrections/` on blob for what has actually accumulated (a synthetic tap `pm_router_f67e15bd45da` from the repair-day proof may be present; human-author rows have `@` in the author).
8. **Deliver the explanation** — the whole flow and both visualizations, in detail, at teaching level, including what YOU found that this doc got wrong or that drifted since it was written.

## 8. Open items & honest edges (know these before explaining)

- **Waiting on the first real PM tap** — verified human gold = 0; a watcher on the Windows machine polls the live store. GPU fine-tune at ≈100 taps.
- **Router coverage**: deployed head routes to ~4 of 29 packs; router gold labels are DeepSeek-circular (91% self-agreement — no trustworthy router accuracy exists yet).
- **Reconciliation section debuts** on the first deal with a real cross-doc conflict (the test deal has 0 — correct silence).
- **Contract yaml** lags the model (`reconciliation`, `service_routing` undocumented; model tolerates via `extra="allow"`).
- **Service store rehydration**: the service's ephemeral store re-seeds on every roll and does not resync from blob at boot (worker does, per compile) — queued as a task chip.
- **Core's duplicated envelope builder** (`src/orbitbrief_core/envelope.py`) has zero runtime importers — vestigial; drift risk is test/tool-only. Candidate for deletion or auto-sync.
- **Mac-specific**: the qwen3 embedder proxies on this very machine are often offline; parser-os falls back to bge-small locally (threshold ~0.62). Don't chase "embedder unreachable" as a new bug.

*Every figure above was true at f523475 / 914763c on 2026-08-25. The audit's first job is to catch where that stopped being true.*
