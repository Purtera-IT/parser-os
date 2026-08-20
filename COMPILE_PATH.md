# What happens when you run a compile

The 27 stages `compiler.py` actually runs, grouped by the layer each belongs to.
The grouping is the point: it shows which stages are commodity decoding, which
are judgments that should be learning, and which are invariants that must never
learn.

## The stages

| # | stage | layer | learns? |
|---|---|---|---|
| 1 | `discover_artifacts` | decode | no |
| 2 | `parse_artifacts` | **decode → segment** | no |
| 3 | `candidate_adjudication` | segment | — |
| 4 | `source_replay` | **receipts** | **never** |
| 5 | `confidence_floor` | segment | — |
| 6 | `prose_list_split` | segment | — |
| 7 | `duplicate_atom_collapse` | segment | — |
| 8 | `execution_boilerplate_drop` | **interpret** | should |
| 9 | `table_rollup` | segment | — |
| 10 | `enrich_entities` | **interpret** | should |
| 11 | `pre_classify_dedup` | segment | — |
| 12 | `typed_atom_classification` | **interpret** | **the type readout** |
| 13 | `atom_type_sanity` | interpret | — |
| 14 | `span_admission` | **interpret** | **the admission readout** |
| 15 | `open_question_resolution` | interpret | should |
| 16 | `site_geo_fallback` | interpret | should |
| 17 | `receipt_backfill` | **receipts** | **never** |
| 18 | `semantic_dedup` | interpret | — |
| 19 | `confidence_recalibration` | **calibration** | **conformal, later** |
| 20 | `entity_resolution` | interpret | should |
| 21 | `pm_answers` | **evidence** | — |
| 22 | `graph_build` | **graph** | the graph encoder, later |
| 23 | `packetize` | derivation | **never** |
| 24 | `packet_certificates` | derivation | **never** |
| 25 | `confidence_calibration` | calibration | conformal, later |
| 26 | `quality_gates` | **invariant** | **never** |
| 27 | `persistence` | — | — |

## Reading it against the design

**Stage 2 is where the decode seam lives.** `parse_artifacts` used to mean "a
format-specific module does extraction *and* typing *and* roster detection".
`decode/pdf.py` splits the first part out: it reports blocks, tables, figures
and locators, and decides nothing about meaning. Document Intelligence or fitz
per page, chosen by which recovered more.

**Stages 4 and 17 are the product.** Receipts never learn. Their value is being
checkable, and a model that guessed at provenance would destroy the only claim
this system makes that a competitor cannot copy.

**Stages 12 and 14 are the two readouts that already exist in skeleton.**
`typed_atom_classification` fronts the LLM with a deflector — the trained head
takes the high-confidence subset and everything else goes to the LLM, which is
the decision ladder working as designed. `span_admission` is the same shape for
"is this atom a `<relation>` item".

**Stages 8, 10, 15, 16, 20 are judgments that are currently rules.** Each is a
readout in waiting. They are not wrong as rules — a rule with a receipt beats a
classifier without one — but they are where corrections should land once there
is something to correct.

**Stages 23, 24 and 26 must never learn.** Packetization and certificates are
derivations over evidence; quality gates are invariants. A prediction here would
mean the system could be confidently wrong about what it had proven.

## What a compile costs today

Per artifact, in the order a PDF meets them:

1. **Document Intelligence `prebuilt-layout`** — one call per document, cached
   by `(path, mtime, size)`, shared by the table and text paths. Billed per
   page.
2. **`fitz`** — still used for page rendering, image extraction, page counts,
   and the schematic bbox pixel-hash. Also the table/text source on any page
   where it recovered more.
3. **OCR**, only when a page yields no text: Document Intelligence
   `prebuilt-read` first, then the legacy tesseract / easyocr / Ollama chain.
4. **DINOv2**, only on schematic pages, two forward passes per glyph crop.
5. **The LLM**, for every atom the type deflector did not take.

Item 5 is the one that matters. *"Model calls per atom"* is one of the two
numbers that separate this from a wrapper — the other being escalation rate per
reason — and it only falls when derivations, retrievers and readouts absorb work
the model used to do.

## What is NOT on this path

Worth stating, because the absence is easy to mistake for a gap:

- **No training.** A compile never trains. Heads are trained by the nightly
  retrain and promoted by an eval gate; a compile only ever *serves* a champion.
- **No corpus write, in production.** `PARSER_OS_TRAINING_LOG_DB` is unset on
  the prod worker deliberately, so prod writes no training rows until the free
  signals are captured. Turning it on before then would refill the corpus with
  the LLM grading its own homework, which is what made the first 222,000 rows
  worthless.
- **No case retrieval.** Deal-to-deal memory does not exist yet. Every compile
  starts from nothing, which is the single largest gap between what runs and
  what is designed.
