# Model Inventory

## Deterministic components used by the v54 accuracy patch

- PyMuPDF / `fitz` PDF text and table extraction.
- Office/PDF parsers already present in `app/parsers/`.
- Deterministic table-schema registry in `app/core/table_schema_registry.py`.
- Deterministic semantic-key dedup in `app/core/semantic_dedup.py`.
- TF-IDF character n-gram semantic linker in `app/semantic/vectorizer.py` when sentence-transformer mode is not enabled.

## Optional local LLM / embedding dependencies already referenced by the project brief

The handoff brief says local compile runs may use Ollama with:

- `qwen2.5:3b`
- `qwen2.5vl:7b`
- `qwen3:14b`
- `bge-m3`

The deterministic pack measurements in this patch were run with multi-entity LLM, site LLM, vision, and typed-classifier passes disabled so the results are reproducible without Ollama.

## Target model inventory for the next architecture stage

- Layout/OCR: Azure Document Intelligence or Docling for tables and section structure.
- Table schema induction: structured-output LLM with JSON schema validation.
- Fact extraction: one multi-category structured-output extractor per artifact or retrieval bundle.
- Dedup/entity resolution: embedding nearest-neighbor retrieval plus an LLM or learned pairwise equivalence judge for borderline cases.
- Confidence: learned calibrator trained on pack-level gold labels, reporting expected calibration error.
