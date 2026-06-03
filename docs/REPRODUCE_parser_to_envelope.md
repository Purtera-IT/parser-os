# parser-os: raw → parser → envelope, end to end

A from-scratch runbook so a fresh agent (or human) on another machine can
reproduce the full compile and verify the deal-kit financial work on the
`harden/messy-input-robustness` branch.

---

## 0. TL;DR

```bash
git clone https://github.com/Purtera-IT/parser-os
cd parser-os
git checkout harden/messy-input-robustness
pip install -r requirements.txt          # numpy, openpyxl, spaCy, etc.

# point at the LLM box (Tailscale → Ollama). Defaults already target it,
# so this is only needed if your IP differs:
export OLLAMA_HOST=http://100.114.102.122:11434

# put the deal inputs in a folder (NOT in git — copy them over manually):
#   _yonah3/000126 Deal Kit.xlsx, *.docx, Notes.pdf, ...

PYTHONUNBUFFERED=1 PYTHONPATH=. python -m app.cli compile "_yonah3" \
  --out "_live_yonah_gaps/envelope.json" \
  --orbitbrief-out "_live_yonah_gaps" \
  --no-cache
```

Outputs land in `_live_yonah_gaps/`:
- `envelope.json` — the **compile** envelope (atoms, entities, edges, packets, trace)
- `orbitbrief.input.json` — the **OrbitBrief** envelope (PM-facing sections; this is where `deal_header` / `deal_financials` / `bill_of_materials` live)
- `orbitbrief.input.md`, `sow.md` — rendered markdown

---

## 1. The pipeline (what "parser → envelope" actually means)

`app/core/compiler.py::compile_project()` runs ~21 sequential stages.
Each is wrapped in a telemetry context that prints one JSON line:
`{"event":"compile_stage_completed","stage":"<name>", "counts":{...}}`.
Watch those lines to see progress. Order (abridged):

1. `discover_artifacts` — walk the input dir → list of files
2. `parse_artifacts` — **the parsers run here.** xlsx/docx/pdf → `EvidenceAtom`s
3. `candidate_adjudication`, `source_replay`, `confidence_floor`,
   `prose_list_split`, `duplicate_atom_collapse`, `execution_boilerplate_drop`
4. `enrich_entities` — populate `atom.entity_keys` (⚠️ NOT wrapped in
   try/except — a raise here crashes the whole compile)
5. `typed_atom_classification` — **LLM (qwen3:14b)** promotes atoms into the
   rich taxonomy
6. `atom_type_sanity` — **our quantity-key scrub runs here** (deterministic)
7. `open_question_resolution`, geo-fallback, quantity backfill
8. `semantic_dedup` — **embeddings (qwen3-embedding:8b)** collapse dup-by-key
9. confidence recalibration → `entity_resolution` → graph build (edges) →
   evidence packets → OrbitBrief envelope assembly → truth gate

The **LLM-heavy stages** (5, 8, and parts of entity/graph work) are why a
compile takes minutes and why it needs the Ollama box reachable. Everything
else is pure Python and deterministic.

### Data model (raw → envelope)
```
Artifact ─parse─▶ EvidenceAtom (atom_type, value, entity_keys=[device:/money:/deal:/...])
         ─enrich/classify/dedup─▶ refined atoms
         ─entity_resolution─▶ EntityRecord + EvidenceEdge
         ─packetize─▶ EvidencePacket
         ─build_orbitbrief_envelope─▶ OrbitBrief envelope (PM sections)
```

---

## 2. LLM / embedding connection (the "Tailscale shit")

The parser stack calls a remote **Ollama** server over **Tailscale**. There
is no local model install required on the compile box — it's pure HTTP.

| What | Env var | Default | Used by |
|---|---|---|---|
| Ollama host | `OLLAMA_HOST` | `http://100.114.102.122:11434` | everything |
| Chat/classifier model | `OLLAMA_MODEL` | `qwen3:14b` | typed_atom_classifier, site verify, zero-miss |
| Embedding model | `OLLAMA_EMBED_MODEL` | `qwen3-embedding:8b` (4096-dim) | semantic_dedup, RAG retrieval |
| Vision model | `OLLAMA_VISION_MODEL` | `llava` | pdf/image OCR chain |
| LLM parallelism | `SOWSMITH_LLM_PARALLEL` | (impl default) | classifier batch |
| LLM timeout (s) | `SOWSMITH_LLM_TIMEOUT` | (impl default) | classifier |
| Embed timeout (s) | `SOWSMITH_EMBED_TIMEOUT` | (impl default) | embeddings |

`100.114.102.122` is the **Tailscale IP** of the GPU box running Ollama. The
other machine must be **on the same tailnet** (`tailscale up`, then
`tailscale status` should list that host) OR run its own Ollama and override
`OLLAMA_HOST` (e.g. `http://localhost:11434`) with the models pulled:
`ollama pull qwen3:14b && ollama pull qwen3-embedding:8b`.

**Reachability check before compiling:**
```bash
curl -s $OLLAMA_HOST/api/tags | python -c "import sys,json;print([m['name'] for m in json.load(sys.stdin)['models']])"
```
Should list `qwen3:14b` and `qwen3-embedding:8b`. If it times out → Tailscale
isn't up or the box is offline. Stages that need the LLM degrade/skip when
the host is unreachable (each guards with a reachability probe), so a compile
"succeeds" but with thinner classification — make sure the host is reachable
for a faithful run.

---

## 3. What this branch changed (deal-kit financials)

All universal — keys off vocabulary/structure, never a customer name.

### a. Structured financial-summary extractor — `app/parsers/xlsx_parser.py`
A deal-kit financial tab is a **2-D label→value grid**, not a row table. The
generic emitter mashed cells (`OPPTY # | 126 | Total Deal Revenue | 21560`).
New `_emit_financial_summary_rows()` reads it like a human:
- **Deal header:** `_DEAL_HEADER_LABELS` maps known labels → *stable canonical
  keys* (`opportunity_id`, `customer`, `billing_type`, …). Then a **structural
  sweep** (`_looks_like_header_label` + `_coerce_header_value`) captures *any
  other* label→value pair (PO #, Account Manager, …) so nothing is dropped.
- **P&L:** regex `(<cat>) (revenue|cost|margin)` + `Margin % on <cat>` →
  one `commercial_total` atom per category with `value.kind=="pl_line"`
  (`revenue`/`cost`/`margin`/`margin_pct`). Fractions (0.2857) → 28.57%.
- **Confidence gate:** if `<2` P&L categories with numbers AND `<3` header
  fields → fall back to the generic commercial emitter (so rate cards /
  non-P&L money grids still surface). The structural header sweep runs ONLY
  past this gate, so the fallback is never polluted.
- Routed in `_parse_sheet_rows`: `SheetRole.FINANCIAL_SUMMARY` →
  `_emit_financial_summary_rows`, else `_emit_commercial_sheet_rows`.

### b. Universal quantity-key scrub — `app/core/atom_type_sanity.py`
`scrub_nondeliverable_quantity_keys()` strips `quantity:` entity keys whose
de-slugged tail classifies as `financial`/`meta` (e.g.
`quantity:260_pmo_cost`, `quantity:28_57_margin`) so junk never reaches the
Truth Gate. Bare-numeric deliverable quantities are preserved. Wired into
`apply_type_sanity` (stage `atom_type_sanity`).

### c. PM render sections — `app/core/orbitbrief_core.py` + `orbitbrief_envelope.py`
Three builders read the structured atoms (no re-parsing) and are wired into
the OrbitBrief envelope, each gated on `present` so empty ones are omitted:
- `build_deal_header` → `envelope["deal_header"]` (merged header fields)
- `build_deal_financials` → `envelope["deal_financials"]` (ordered P&L lines + totals; deal line first)
- `build_bill_of_materials` → `envelope["bill_of_materials"]` (materials/catalog folded rows; excludes rate_card/financial_summary)

---

## 4. Verify the run

```bash
# the new PM sections live in the OrbitBrief envelope, NOT the compile envelope:
python - <<'PY'
import io, json
d = json.load(io.open("_live_yonah_gaps/orbitbrief.input.json", encoding="utf-8"))
for k in ("deal_header", "deal_financials", "bill_of_materials"):
    print(k, "->", "PRESENT" if k in d else "absent")
print("deal_financials totals:", d.get("deal_financials", {}).get("totals"))
print("header fields:", list(d.get("deal_header", {}).get("fields", {})))
PY
```

Expected for Yonah deal 126 (verified earlier against real atoms):
- `deal_header`: ~13 fields incl. `opportunity_id=126`, `customer=DCW`, `sales_rep=Dan`, `billing_type=T&M`
- `deal_financials.totals`: `{revenue:21560, cost:15660, margin:5900, margin_pct:27.37}`
- 6 P&L categories (Deal/Labor/PMO/Materials/Lift/Misc)
- Truth Gate clean of `quantity:*_cost` / `quantity:*_margin` junk entities

### Tests (no LLM needed — pure unit tests)
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest -p no:cacheprovider -q \
  tests/test_financial_summary_parser.py tests/test_deal_sections.py \
  tests/test_commercial_sheet_routing.py tests/test_atom_type_sanity.py \
  tests/test_xlsx_parser.py tests/test_orbitbrief_envelope.py
```

---

## 5. Gotchas observed on the original box

- **`MemoryError` on `import numpy`** mid-compile = the host hit a process
  memory ceiling (job-object commit cap), NOT a code bug — the traceback is
  entirely inside numpy's import machinery. A normal box with the deps and a
  few GB free won't hit it. If you do: close other heavy processes / raise
  the cap, and don't run multiple compiles + pytest concurrently.
- Read JSON with `io.open(path, encoding="utf-8")` (Windows default cp1252
  mangles the envelope).
- Never commit `_yonah*/`, `_live_*`, `*.json` artifacts, or
  `docs/parser_os_architecture.pdf`. The deal inputs are copied manually, not
  versioned.
