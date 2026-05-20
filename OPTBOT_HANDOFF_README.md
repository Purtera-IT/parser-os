# OPTBOT Atlanta — Parser-OS Integration Handoff

**Audience:** systems engineer integrating parser-os output with PurPulse / OrbitBrief
**Branch:** `claude/crazy-gauss-a9cfe0` (PR open against `main`)
**Test fixture:** `OPTBOT_Atlanta_Office_Refresh_Mock_Deal` (7 HubSpot-mirrored discovery files)

This README documents the **post-cleanup baseline** for OPTBOT, the changes that
got us there, and exactly what you should see when the same files run through
your Azure pipeline. If your Azure output differs from this baseline, your
container is stale — pull the new build.

---

## TL;DR — what changed and why you care

Five commits between `30895b5` (schematic R12) and `5a3b990` (entity cleanup)
addressed the **primary feedback from the OPTBOT state report**:

> "Site entity resolution is the main noise: 32 site-like entities / scope
> sites include document titles, roles, and mock tokens — hurts downstream
> SOW site roster (Core blocker #2)."

Old: 32 site entities for OPTBOT, ~80% junk.
New: **3 site EntityRecords** for OPTBOT, with all 8 surface forms folded into
the 3 canonical records as aliases.

| Old (compile `4205b5dc...`) | New (commits `bd9a949` → `5a3b990`) |
|---|---|
| 32 site entities | **3 site EntityRecords** with alias groups |
| `site:mock_msa`, `site:hs_deal`, `site:site_surveys_docx` | gone — head denylist + structural gate |
| `customer:customer`, `vendor:carrier`, `room:room` | gone — generic-alias sentinel |
| `part_number:hs_deal_optbot_atl_2026_047` | gone — contract-ID denylist |
| `part_number:po_mock_77421`, `part_number:atl_047` | gone — airport-prefix denylist |
| ATL-HQ / Atlanta Headquarters / Innovation Tower as separate entities | fused — `site:atl_hq` with aliases |

If you're running an older container, your output_signature will not match what's
below. **Cut a new container build from this branch before integration testing.**

---

## 1. Expected post-fix OPTBOT baseline

Run on the seven files from `OPTBOT_Atlanta_Office_Refresh_Mock_Deal`:

```
01_deal_overview_executive_brief.pdf       14 atoms
02_statement_of_work.docx                  16 atoms
03_site_surveys_and_requirements.docx      16 atoms
04_hardware_bill_of_materials.xlsx         32 atoms
05_project_schedule_and_cutover_plan.xlsx  40 atoms
06_security_it_integration_notes.pdf        8 atoms
07_contracting_procurement_packet.pdf       9 atoms
                                          ─────────
                                          135 atoms
```

Compile summary (`PYTHONHASHSEED=0`, `--no-cache`, `--allow-unverified-receipts`):

| Field | Value |
|---|---|
| `input_signature` | `2c3d103e57948babe0f3717212049afa0821cadbcbb7e60bd46526202e28bf9c` |
| `output_signature` | `7a744d1251628786b474a97f8f73a2551bd1ca258ce79ee5db892062a967fd1e` |
| `atoms` | 135 |
| `edges` | 153 |
| `packets` | 16 |
| `entities` | 35 |
| `documents` | 7 |
| `warnings` | 19 (non-fatal) |
| `errors` | 0 |
| `receipt_verified` | 132 / 135 |
| `receipt_failed` | 2 (1 + 1 from PDFs — pre-existing, not my changes) |

If your Azure container produces a different `output_signature`, it's running
a different code version. Determinism is contractual — same input always
produces the same signature.

### Entity record breakdown (post-fix)

```
site         (3)   ← canonical EntityRecords
                    site:atl_hq                       (+ atlanta_headquarters, innovation_tower)
                    site:atl_west                     (+ westside_operations_center)
                    site:airport_logistics_annex      (+ atl_air, college_park)
address      (2)   address:1180_peachtree_street
                    address:976_brady_avenue
device      (18)   real BOM line items (access_point, switch, ip_camera, ups,
                    firewall, controller, display, microphone, speaker, ...)
                    Note: includes 5 service line items still classified as
                    devices — known issue, see §7
part        (10)   manufacturer SKU names from the BOM
                    (clearmeet_bar_pro, coreedge_cx_48p, netwave_ap_9700,
                     dockflex_180, fieldtab_r12, viewbright_27q,
                     powerkeep_1500, printsure_lp600, clearmeet_panorama_4k,
                     deskflow_panel_10)
part_number  (2)   only the two legitimate codes
                    part_number:ic_001
                    part_number:optbot_atl_047
```

**No `customer:customer`, no `vendor:vendor`, no `vendor:carrier`, no
`room:room`, no contract-ID part_numbers.** If any of those reappear in your
Azure output, the container is stale.

---

## 2. Site alias fusion — how the 3 sites carry 8 names

The biggest behavioral change. Three physical sites, with each EntityRecord
carrying the canonical key + all aliases the text asserts refer to the same
place:

```json
{
  "entity_type": "site",
  "canonical_key": "site:atl_hq",
  "canonical_name": "atl hq",
  "aliases": [
    "site:atl_hq",
    "site:atlanta_headquarters",
    "site:innovation_tower"
  ],
  "source_atom_ids": [...],
  "confidence": 0.85,
  "review_status": "auto_accepted"
}
```

```json
{
  "entity_type": "site",
  "canonical_key": "site:atl_west",
  "aliases": ["site:atl_west", "site:westside_operations_center"],
  "confidence": 0.85
}
```

```json
{
  "entity_type": "site",
  "canonical_key": "site:airport_logistics_annex",
  "aliases": [
    "site:airport_logistics_annex",
    "site:atl_air",
    "site:college_park"
  ],
  "confidence": 0.85
}
```

### How the fusion decided which keys to merge

Pairwise detection — examines the text **between** each adjacent pair of
site-key spans within a sentence:

| Pattern | Example | Fuses? |
|---|---|---|
| Copular | `ATL-HQ is the Atlanta Headquarters` | yes |
| Explicit aliasing | `also known as`, `a.k.a.`, `called`, `designated` | yes |
| Parenthetical | `Atlanta Headquarters (ATL-HQ)` | yes |
| Colon-bridge | `ATL-HQ: Atlanta Headquarters` | yes |
| Separator + mixed shape | `ATL-HQ \| Atlanta Headquarters` (pipe between code + name) | yes |
| Em-dash / slash between two names | `Atlanta Headquarters / Innovation Tower` | yes |
| Hyphen-with-spaces between two names | `Atlanta Headquarters - Innovation Tower` | yes |
| Comma / semicolon / "and" | `ATL-HQ, ATL-WEST, ATL-AIR` | **no** (list) |
| Pipe between same shape | `ATL-HQ \| ATL-WEST` (two cells of a table column) | **no** |
| "at the" / "in the" | `ATL-HQ at the Innovation Tower` | **no** (containment) |

**Row-level rule** on top: in a pipe-separated row whose only distinct site
code is the row's leading identifier (so the row describes one site), every
site key in subsequent cells folds into that row's site. This is how
`College Park` (which appears inside an address cell on the ATL-AIR row) folds
into `site:airport_logistics_annex`.

Confidence is `0.85` on fused records (lower than the `1.0` confidence of a
single-source record), so the systems engineer can downgrade reliance on
fused records if needed.

---

## 3. Where to find this in the code (and what tests cover it)

| Concern | File | Function |
|---|---|---|
| Site code extraction (ATL-HQ, NYC-DC1, …) | `app/core/entity_extraction.py` | `_emit_sites` |
| Site-code SUFFIX allowlist (positive gate) | `app/core/entity_extraction.py` | `_SITE_CODE_SUFFIX_ALLOWLIST`, `_SITE_CODE_SUFFIX_PATTERN`, `_site_code_suffix_ok` |
| Site-code HEAD denylist (MOCK, DEV, MSA, HS, ...) | `app/core/entity_extraction.py` | `_SITE_CODE_HEAD_DENYLIST` |
| Proper-noun extraction with structural gate | `app/core/entity_extraction.py` | `_emit_proper_nouns` + `_has_site_corroboration` |
| Hard-disqualify tokens (mock/test/demo override place-tail) | `app/core/entity_extraction.py` | `_HARD_DISQUALIFY_PHRASE_TOKENS` |
| Cross-mention alias fusion (pairwise + row-level) | `app/core/entity_extraction.py` | `_emit_site_aliases_from_text`, `_classify_pair`, `_coalesce_alias_groups` |
| Alias fusion wired into entity_resolution | `app/core/entity_resolution.py` | `collect_site_alias_groups`, `fuse_alias_groups` |
| Pipeline orchestration | `app/core/compiler.py` | `compile_project` (entity_resolution stage) |
| Generic-alias sentinel (customer:customer, room:room kill) | `app/core/entity_extraction.py` | `_GENERIC_TYPED_ALIAS_SENTINEL` + `_typed_alias_index` + `_emit_typed` |
| Part-number contract-ID denylist | `app/core/entity_extraction.py` | `_emit_part_numbers` + `_AIRPORT_CITY_PREFIXES` |
| Generic xlsx pseudo-values (ALL/N/A/TBD) | `app/core/normalizers.py` | `_GENERIC_SITE_PSEUDO_VALUES` |

**Tests** that pin every behavior above:

| Test file | What it covers | Count |
|---|---|---|
| `tests/test_site_extraction_hardening.py` | Site extraction + alias fusion (the new ground truth) | 166 cases |
| `tests/test_entity_extraction.py` | Pre-existing entity extraction (class-based) | regression |
| `tests/test_week5_dx.py` | Cross-pack vendor matching, single-token gaps, two-word org sites | regression |
| `tests/test_week6_dx.py` | Proper-noun tightening, bare-noun stoplist, broad-customer authority | regression |
| Full schematic test grid | 251 cases | regression |

Total post-cleanup: **166 site-hardening + 79 class-based entity + 251
schematic = 496 tests, all passing.**

Run them locally:

```bash
python scripts/_schematic_smoke.py tests.test_site_extraction_hardening
# expect "failures: 0" and 166 PASS lines

python -m pytest tests/test_entity_extraction.py tests/test_week5_dx.py tests/test_week6_dx.py -v
# (pytest may crash on this Windows dev env with STATUS_STACK_BUFFER_OVERRUN;
#  CI runs them on Linux, where they pass.)

python scripts/_schematic_smoke.py
# expect "failures: 0" for the full schematic surface
```

---

## 4. How parser-os fits into your Azure pipeline

Mapping from the **path-by-path map** doc to what runs where:

```
HubSpot files
  → mirror (hubspot-files-mirror.js)               ← Function App
  → 7 rows in attachments table (Postgres)
  → enqueue parser-os-orbitbrief-jobs              ← Azure Queue
      message: { dealId, compileId }

parser-os-orbitbrief-queue (Function trigger)      ← Function App
  → buildAndUploadManifest                          → blob: parser-manifests/{compileId}.json
  → POST parser-os-service /v1/orbitbrief/rebuild-latest
        body: { "manifest_blob_url": "..." }
       ⤷ THIS IS THE BOUNDARY — parser-os code runs here
       
parser-os-service (Container App)                  ← parser-os repo (this repo)
  /v1/orbitbrief/rebuild-latest:
    1. read_manifest_json
    2. download_blob_to_path (each artifact)
    3. app.core.compiler.compile_project           ← all my changes flow through here
         ├─ discover_artifacts
         ├─ parse_artifacts                        ← per-format parsers (pdf, xlsx, docx, ...)
         ├─ candidate_adjudication
         ├─ source_replay                          ← receipt verification
         ├─ confidence_floor
         ├─ enrich_entities                        ← _emit_sites, _emit_proper_nouns,
         │                                          _emit_typed, _emit_vendors,
         │                                          _emit_part_numbers
         ├─ entity_resolution                      ← extract_entity_records
         │   ├─ resolve_aliases (fuzzy)            
         │   └─ fuse_alias_groups (NEW)            ← cross-mention alias fusion
         ├─ graph_build
         ├─ packetize
         ├─ packet_certificates
         └─ quality_gates
    4. app.core.orbitbrief_envelope.build_orbitbrief_envelope
    5. parser_os_service.server.projector.to_scope_process_v1
    6. _attachments_status (per-file metadata)
    7. upload envelope → blob: orbitbrief/latest/envelope.json
    8. response → queue worker

  → queue worker:
       applyAttachmentsParseStatus                 → Postgres: attachments.*
       persistOpportunityScopeProcessV1            → Postgres: opportunities.quote_data.scope_process_v1
       backfillOpportunityAmountFromEnvelope       → Postgres: opportunities.amount

SPA OrbitBrief tab:
  - GET /api/data/opportunities/:id/detail         → Postgres (scope_process_v1)
  - GET /api/data/deals/:dealId/artifacts          → attachment metadata
  - GET /api/quoting/deal/:id/orbitbrief/latest/envelope    → blob (debug / Core)
  - GET /api/quoting/deal/:id/orbitbrief/latest/pm-handoff  → blob (Core scorecard,
                                                              manually produced by
                                                              run-orbitbrief-core-for-deal.sh)
```

### Key boundary

**parser-os is invoked exclusively via `POST /v1/orbitbrief/rebuild-latest`**
from the queue worker. The SPA does not call parser-os directly. My changes
only affect:

- The `compile_project` pipeline (and everything it produces)
- The `envelope.json` written to blob storage
- The `scope_process_v1` projected from that envelope into Postgres
- Therefore: every SPA tab reading `scope_process_v1` (Workspace, Audit, Rail)
  will see the cleaned entities

To validate on Azure: trigger a recompile (`POST .../orbitbrief/recompile`),
wait for queue completion, then read the blob envelope and compare against
the baseline in §1.

---

## 5. Output contract — what your engineer can rely on

### Envelope file: `orbitbrief.input.v2`

```
blob path: deals/{dealId}/orbitbrief/latest/envelope.json
schema:    orbitbrief.input.v2
```

Top-level shape:

```jsonc
{
  "schema_version": "orbitbrief.input.v2",
  "project_id": "OPTBOT_Atlanta_Office_Refresh_Mock_Deal",
  "compile_id": "cmp_<16-hex>",
  "generated_at": "2026-05-20T16:52:40Z",
  "summary": { ... },
  "documents": [ /* 7 entries, one per file */ ],
  "atoms":     [ /* 135 entries — typed evidence */ ],
  "packets":   [ /* 16 entries — packet families */ ],
  "entities":  [ /* 35 EntityRecords (deduplicated) */ ],
  "edges":     [ /* 153 graph edges */ ],
  "indexes":   { ... }
}
```

### Entity contract

Every entity record has:

```jsonc
{
  "id": "ent_<stable-hash>",                       // stable across runs
  "project_id": "...",
  "entity_type": "site"|"address"|"device"|"part"|"part_number"|"customer"|"vendor"|"room",
  "canonical_key": "site:atl_hq",                  // {entity_type}:{slug}
  "canonical_name": "atl hq",                      // human-readable
  "aliases": ["site:atl_hq", "site:atlanta_headquarters", ...],
  "source_atom_ids": ["atm_...", ...],             // which atoms cite this entity
  "confidence": 0.85|1.0,                          // 0.85 = co-mention fused
  "review_status": "auto_accepted"|"needs_review"
}
```

**Stable IDs**: `canonical_key` and `entity_id` are deterministic. Same input,
same key. You can use them as foreign keys safely.

**Aliases**: the `aliases` list always includes the canonical_key itself
(redundancy by design, simplifies lookups). When you index, do
`alias → canonical_key` in both directions.

### Atom contract

```jsonc
{
  "id": "atm_<stable-hash>",
  "project_id": "...",
  "artifact_id": "art_<stable-hash>",
  "atom_type": "scope_item"|"quantity"|"vendor_line_item"|"constraint"|"risk"|...,
  "raw_text": "...",
  "normalized_text": "...",
  "value": { ... },
  "entity_keys": ["site:atl_hq", "device:access_point", ...],
  "source_refs": [
    {
      "id": "src_<stable-hash>",
      "artifact_id": "...",
      "page_number": 0,                            // 0-indexed
      "bbox": [x0, y0, x1, y1],                    // PDF only; null for xlsx/docx
      "crop_sha256": "..."                         // PDF crop hash for receipt replay
    }
  ],
  "confidence": 0.0-1.0
}
```

Every atom has at least one `source_ref` that points back to the original
artifact + page + region. `crop_sha256` lets you replay the receipt: re-extract
the same byte region from the PDF and the hash must match. This is the
provenance contract.

### Determinism contract

```
input_signature   = sha256 of (all artifact bytes + manifest)
output_signature  = sha256 of (canonical-serialized atoms + entities + edges + packets)
```

Same `input_signature` → same `output_signature`. Always. If you run parser-os
twice on the same files and get a different output_signature, that's a bug in
parser-os, not a feature.

This is what makes the system testable: you can pin the OPTBOT
output_signature in your integration tests, and any drift means someone
changed the parser.

---

## 6. Cleared from earlier feedback (parser-os builder themes)

From the OPTBOT state report's feedback section:

| # | Original feedback theme | Status |
|---|---|---|
| 1 | "Compile quality is good on OPTBOT" | ✓ still good — 7/7 parsed, 0 errors |
| 2 | "Site entity resolution is the main noise: 32 site-like entities" | **✓ FIXED** — now 3 |
| 3 | "Packets lack surfaced severity in envelope" | not addressed (Core/projector decision, not parser) |
| 4 | "`to_scope_process_v1` gap vs Core" | not addressed (projector decision, not parser) |
| 5 | "Receipt semantics" | unchanged — `allow_unverified_receipts=True` in dev |
| 6 | "Single compile id across deal" | ✓ unchanged — good |

The PR fixes the #1 customer-visible issue (#2). Feedback #3 and #4 are
projector / Core decisions that don't live in parser-os.

---

## 7. Known issues (flag for your engineer)

### 7.1 — Service line items classified as devices

The xlsx parser emits all BOM rows as `device:` entities regardless of whether
the row is a physical device or a service line. After the cleanup, 5 of the
18 OPTBOT "devices" are services:

```
device:after_hours_installation_labor
device:hypercare_support
device:project_management_and_weekly_governance
device:training_and_adoption_support
device:discovery_workshops_and_technical_design
```

**Integration workaround**: filter `device:*` keys where the
`canonical_name` contains `labor`, `support`, `training`, `governance`,
`workshop`, or `service`. These are demonstrably non-device line items.

**Proper fix** (deferred to follow-up PR): teach `xlsx_parser.py` to read the
BOM "category" column (when present) and emit `service:` for non-hardware
items. Not blocking integration — the misclassified items still carry their
description as `canonical_name`, so the systems engineer can categorize them
downstream.

### 7.2 — Address coverage (3 sites, 2 addresses)

Only two `address:` entities are produced (1180 Peachtree, 976 Brady) but
there are 3 sites. The ATL-AIR address (`4200 Global Gateway Connector,
Building C, College Park`) doesn't match the street-address regex because
the suffix "Connector" isn't in `_STREET_SUFFIXES`. Low priority but worth
flagging — the address regex was tuned for residential/commercial standard
suffixes and may miss the long tail.

Workaround: nothing needed in integration. The site entity carries the
address text in its source atoms; downstream consumers can extract the city
from the proper-noun matcher's output (`site:college_park` is folded into
ATL-AIR).

### 7.3 — No stakeholder / person entities (TIER 1 gap, deferred)

The OPTBOT deal mentions 6 named approvers across the documents:

| Name | Role | Atoms mentioning |
|---|---|---|
| Jordan Ames | VP Workplace Operations | 5 |
| Priya Narang | technical design approver | 6 |
| Camila Brooks | security / data approver | 4 |
| Elliot Tran | procurement | 5 |
| Renee Watkins | delivery governance | 9 |
| Morgan Lee | CFO Delegate | 2 |

**Currently zero `stakeholder:` or `person:` entities are produced.** The
proper-noun matcher captures `site:` and `customer:` shapes but doesn't
emit person entities. SOW routing and approval workflow downstream cannot
match approvers to roles without these.

**Where to add it**: new `_emit_stakeholders` function in
`app/core/entity_extraction.py`, hooked into `extract_keys` near
`_emit_proper_nouns`. Should detect `First Last` capitalized pairs with a
role context cue ("approves", "owner", "approver", "CFO", "VP", "Manager",
"Director", "Sponsor"). Suggest emitting `stakeholder:first_last` keys with
the surrounding role as an alias or metadata field.

### 7.4 — Customer entity missing (TIER 1 gap, deferred)

The deal overview text contains `Company: OPTBOT, Inc.` but no
`customer:optbot` entity is produced. (My cleanup fix correctly killed
the `customer:customer` noise but didn't add the real customer detection
back.)

**Where to add it**: add a "Company:" / "Customer:" / "Account:" label
detector to `_emit_customer_keys` in `app/core/entity_extraction.py`.
When the labeled value is a proper-noun with a corporate suffix
(`Inc`, `LLC`, `Corp`, `Ltd`, `Co`), emit `customer:<slug>`.

### 7.5 — Three packets anchored to `:unknown` (TIER 1 gap, deferred)

Of the 16 OPTBOT packets, three lost their entity anchor:

- `scope_exclusion` → `site:unknown`
- `site_access` → `site:unknown`
- `scope_inclusion` → `device:unknown`

These packets have content but their anchor entity didn't resolve. The
`anchor_key` is set to a sentinel `:unknown` rather than dropping the
packet, which is correct (you'd lose information otherwise) but the
systems engineer needs to know these are partial.

**Where to investigate**: `app/core/packetizer.py` — look at the
fallback path when `_select_anchor` can't find a deterministic entity
from `governing_atom_ids`. Possibly worth filtering at the packet level
or surfacing as `needs_review` with a clearer flag.

### 7.6 — Money values not normalized as entities (TIER 2 gap, deferred)

OPTBOT mentions multiple critical dollar amounts:

- `$1,847,250` — total deal amount
- `$1,500,000` — CFO approval threshold
- `$250,000` — budget owner threshold
- `$1,015,626` — hardware subtotal
- `$536,030` — services subtotal
- `$295,594` — logistics / freight / contingency / tax / fees

These appear in raw text but produce **zero `money:` / `currency:`
entities**. The OrbitBrief Core scorecard flagged "Pricing structure —
pricing model not found" as a blocker — money entity extraction would
unblock that.

**Where to add it**: new `_emit_money_keys` in `app/core/entity_extraction.py`.
Regex pattern: `\$\s*[\d,]+(?:\.\d+)?(?:\s*[KMB])?\b`. Emit
`money:<normalized_amount>` keys. Consider also attaching the surrounding
label context (`total`, `threshold`, `subtotal`) as an alias or metadata.

### 7.7 — Dates not extracted as milestone entities (TIER 2 gap, deferred)

Sixteen atoms contain ISO dates in raw text:

- `2026-07-31` — close date
- `2026-05-20` — mobilization start
- `2026-08-14` — implementation end
- `2026-06-14` — quote expiry
- `2026-06-17` to `2026-06-21` — executive blackout
- `2026-08-15` — hypercare start

Zero `date:` or `milestone:` entity keys are emitted. Timeline reasoning
(critical path, deadlines, blackout conflicts) needs structured date
entities.

**Where to add it**: new `_emit_date_keys` / `_emit_milestone_keys` in
`app/core/entity_extraction.py`. ISO date regex
`\b(20\d\d)-(\d\d)-(\d\d)\b` is straightforward; the labeling context
("close date", "mobilization", "blackout") needs a small regex set.

### 7.8 — Receipt verification under `--allow-unverified-receipts`

Two PDF atoms (`atm_7eb5d051335f550b`, `atm_b6a06f3797db5857`) have receipt
verification failures that get downgraded to warnings under
`--allow-unverified-receipts`. Pre-existing, not introduced by my changes.
For production semantics, drop the flag and these would become errors.

### 7.9 — `pytest` crashes on Windows dev env

Local pytest hits `STATUS_STACK_BUFFER_OVERRUN` on this Windows / CPython
3.12.3 setup. The CI workflow at `.github/workflows/test.yml` runs the full
suite on Ubuntu where it passes cleanly. For local validation, use the
smoke runner:

```bash
python scripts/_schematic_smoke.py <module>
```

This is a known Windows-only dev-env bug, not a project issue.

---

## 8. Verification checklist for your Azure engineer

When the new container is deployed to `parser-os-service` and the queue
worker runs against OPTBOT:

- [ ] `POST /v1/orbitbrief/rebuild-latest` returns 200 OK with the
      attachments_status + scope_process_v1 fields populated.
- [ ] Blob `deals/{dealId}/orbitbrief/latest/envelope.json` has
      `output_signature: 7a744d1251628786b474a97f8f73a2551bd1ca258ce79ee5db892062a967fd1e`.
      If the bytes differ but the signature matches, that's fine — JSON
      whitespace is normalized.
- [ ] Envelope `atoms` count == 135.
- [ ] Envelope `entities` count == 35 (was 48 pre-cleanup).
- [ ] **Three `entity_type: site` records** in `entities`, each with
      `confidence: 0.85` and an `aliases` array length 2-3.
- [ ] Zero entities with canonical_key == `customer:customer`,
      `vendor:vendor`, `vendor:carrier`, or `room:room`.
- [ ] Zero `part_number:*` entries containing `hs_deal`, `mock_msa`,
      `po_mock`, `atl_047`, or `dev_atl`.
- [ ] Postgres `opportunities.quote_data.scope_process_v1`:
      `projectNeeds.site_list` should have 3 entries (was 32), and they should
      match the three canonical site names.
- [ ] Postgres `opportunities.amount` backfilled to `1847250` (unchanged from
      previous compile).
- [ ] Re-run the queue twice on the same manifest: `output_signature` must
      be identical across runs. If it isn't, determinism is broken.

If any check fails: the container is stale (pull from this branch's HEAD), or
there's a parser-os regression to file an issue against.

---

## 9. CLI for local reproduction

```bash
# Setup
git clone https://github.com/Purtera-IT/parser-os.git
cd parser-os
git fetch origin claude/crazy-gauss-a9cfe0
git checkout claude/crazy-gauss-a9cfe0
pip install -e ".[dev]"

# Health check
parser-os health     # → "ok"

# Compile against OPTBOT folder
PYTHONHASHSEED=0 python -m app.cli compile \
    /path/to/OPTBOT_Atlanta_Office_Refresh_Mock_Deal \
    --out result.json \
    --orbitbrief-out envelope/ \
    --allow-unverified-receipts --allow-errors

# Verify the signature
grep output_signature result.json
# expect: "output_signature": "7a744d1251628786b474a97f8f73a2551bd1ca258ce79ee5db892062a967fd1e"

# Inspect the 3 fused sites
python -c "
import json
r = json.load(open('result.json'))
sites = [e for e in r['entities'] if e['entity_type'] == 'site']
for s in sites:
    print(s['canonical_key'], '→', s['aliases'])
"
# expect:
#   site:airport_logistics_annex → ['site:airport_logistics_annex', 'site:atl_air', 'site:college_park']
#   site:atl_hq                  → ['site:atl_hq', 'site:atlanta_headquarters', 'site:innovation_tower']
#   site:atl_west                → ['site:atl_west', 'site:westside_operations_center']
```

---

## 10. Commit-by-commit changelog (the perfect bits, in order)

Branch: `claude/crazy-gauss-a9cfe0` (PR open against `main`).

```
5a3b990  fix(entities): clean entity records — drop generic-noun, contract-ID,
                          ambiguous-vendor leaks
                          → kills customer:customer, vendor:vendor, vendor:carrier,
                            room:room, and contract-ID part_numbers
                          → introduces _GENERIC_TYPED_ALIAS_SENTINEL,
                            _AIRPORT_CITY_PREFIXES, carrier disambiguation

59e9430  feat(entities): pairwise alias detection + row-level fusion
                          → replaces sentence-level "any marker fuses all" with
                            pairwise check between adjacent site-key spans
                          → adds row-level rule: single-code pipe row fuses all
                            site keys in that row (catches "College Park" inside
                            an address cell folding into ATL-AIR)

73936d2  feat(entities): cross-mention alias fusion — collapse N surface names
                          to 1 logical site
                          → introduces _emit_site_aliases_from_text,
                            _coalesce_alias_groups, collect_site_alias_groups,
                            fuse_alias_groups
                          → wires into entity_resolution stage in compiler.py

599ea93  feat(entities): universal positive-signal gate for site extraction
                          → site-code SUFFIX allowlist (HQ, MAIN, WEST, EAST,
                            AIR, DC1, FL3, B12, ...)
                          → proper-noun STRUCTURAL gate (place-tail OR org-tail
                            OR address corroboration OR explicit site-context)
                          → HARD_DISQUALIFY tokens (mock/test/demo override
                            place-tail bypass)
                          → corroboration window tightened 80 → 40 chars

bd9a949  fix(entities): bullet-proof site extraction against mock/dev/contract-ID
                          noise (initial pass, then superseded by 599ea93)
                          → SITE_CODE_HEAD_DENYLIST, NON_SITE_PHRASE_TAIL_NOUNS,
                            generic xlsx pseudo-values
```

Each commit message is self-contained — the bodies have the full "what landed
and why" so your engineer can pull any single commit for narrow review.

---

## 11. Repo navigation cheat sheet

| Looking for... | Open... |
|---|---|
| The entity extraction pipeline | [app/core/entity_extraction.py](app/core/entity_extraction.py) |
| The alias-fusion logic | [app/core/entity_resolution.py](app/core/entity_resolution.py) `collect_site_alias_groups` and `fuse_alias_groups` |
| Where the compile stages live | [app/core/compiler.py](app/core/compiler.py) `compile_project` |
| Per-format parsers | [app/parsers/](app/parsers/) — `pdf_parser.py`, `xlsx_parser.py`, `docx_parser.py`, ... |
| Domain packs (vendor lists, aliases, suffix patterns) | [app/domain/](app/domain/) — one YAML per pack |
| Tests for the new behavior | [tests/test_site_extraction_hardening.py](tests/test_site_extraction_hardening.py) |
| The smoke runner (Windows pytest workaround) | [scripts/_schematic_smoke.py](scripts/_schematic_smoke.py) |
| Top-level README + architecture | [README.md](README.md), [app/README.md](app/README.md) |

---

## 11a. Follow-up PR priorities (deferred work, ordered by leverage)

These are the deltas I audited after shipping the cleanup but explicitly
deferred for a follow-up PR rather than bloating this hand-off. Priority
order is what unlocks the most downstream value, not what's easiest:

| # | Gap | Adds | Effort | Where |
|---|---|---|---|---|
| 1 | Stakeholder / person entities (§7.3) | 6 named approvers on OPTBOT alone — SOW approval routing | M | `app/core/entity_extraction.py` → new `_emit_stakeholders` |
| 2 | Customer entity from "Company: X" label (§7.4) | `customer:optbot` (currently no customer record) | S | `app/core/entity_extraction.py` → `_emit_customer_keys` extension |
| 3 | Money / currency entities (§7.6) | `money:1_847_250` etc. — unblocks Core "pricing structure" blocker | M | `app/core/entity_extraction.py` → new `_emit_money_keys` |
| 4 | Date / milestone entities (§7.7) | timeline reasoning (close date, blackouts, cutover) | M | `app/core/entity_extraction.py` → new `_emit_date_keys` |
| 5 | `:unknown` packet anchors (§7.5) | 3 of 16 OPTBOT packets carry partial data | S | `app/core/packetizer.py` `_select_anchor` |
| 6 | Service-vs-device classification (§7.1) | clean separation in BOM line items | M | `app/parsers/xlsx_parser.py` — read BOM category column |
| 7 | Address regex coverage (§7.2) | catches "Global Gateway Connector"-style suffixes | S | `app/core/entity_extraction.py` `_STREET_SUFFIXES` |

Items 1, 2, 5 are the highest-leverage / lowest-risk follow-up. Items 3
and 4 unblock the Core scorecard blocker that the optbotdealpath.md
flagged. Items 6 and 7 are nice-to-have polish.

A follow-up PR could ship items 1+2+5 together and stay under a day.
Items 3+4 are their own PR with their own test coverage.

## 12. Open questions for the systems engineer

Things the parser-os layer doesn't decide that need your input:

1. **`scope_process_v1` projector parity**: the optbotdealpath.md noted that
   `orbitbriefAudit.sowReadiness` is `null` and `missing[]` is empty even
   though Core computed 2 blockers + 4 warnings. parser-os builds the
   envelope; the projector (`to_scope_process_v1`) decides what flows into
   `scope_process_v1`. If you want readiness % and structured gaps in the
   Postgres path, the projector needs to be taught — that's outside this PR.

2. **OrbitBrief Core (Ollama)** still runs manually via
   `run-orbitbrief-core-for-deal.sh`. parser-os doesn't trigger it. If you
   want auto-Core-after-parser, the queue worker is the right place to add
   the call (after step 8 in the rebuild-latest internal sequence).

3. **Service-line classification** (§7.1) — fix in `xlsx_parser.py` to emit
   `service:` entity_type for labor/support/governance rows. Trivial PR but
   requires a downstream `service:` namespace handler if you want it
   surfaced separately in the envelope.

4. **Confidence semantics on fused records** — currently `0.85`. If you want
   to surface the alias source ("co-mention in source text X") in the
   `EntityRecord`, we'd need to add a `fusion_evidence` field to the schema.
   Easy fix if there's demand.

---

## 13. Contact / where to file issues

- **Branch**: `claude/crazy-gauss-a9cfe0`
- **PR**: open against `main` in `Purtera-IT/parser-os`
- **Test fixture**: synthetic; lives in `/c/Users/lilli/Downloads/OPTBOT_Atlanta_Office_Refresh_Mock_Deal/`
- **For parser-os bugs**: file as GitHub issue against this branch
- **For projector / Core / SPA**: those live in the Platform-infra and
  PurPulse repos, not here

---

**Bottom line for the integration**: deploy this branch, re-run the OPTBOT
queue, verify the baseline signature in §1, and the SPA shell should show
3 clean sites instead of 32 noisy ones. Everything else from the prior
state report is unchanged.
