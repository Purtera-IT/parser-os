# Parser-OS Architecture

This repository compiles a bag of customer bid artifacts into a typed evidence envelope:

```text
PDF / DOCX / XLSX artifacts
  -> artifact discovery + parser selection
  -> layout/text/table extraction
  -> typed atom construction with SourceRef + EvidenceReceipt
  -> deterministic enrichment of entity_keys
  -> table-schema enrichment for raw rows
  -> typed classification / duplicate collapse
  -> semantic deduplication and physical_site schema cleanup
  -> entity resolution
  -> graph edge construction
  -> packetization + packet certificates
  -> JSON envelope
```

## Current contract after the v54 accuracy patch

The most important invariant is that authoritative roster rows become `atom_type=physical_site` atoms with strict physical-site-shaped values. Generic `atom_type=entity` values such as `{"entity_type":"site"}` are removed once a physical-site roster exists.

Site extraction is intentionally gated. A PDF text fallback may mint physical sites only when the document declares a roster table through structural headers such as `site_id/facility/street` or `site_no/administrative_site/street/city/zip/lat_long`. Cross-document references such as “see site roster doc 08” are not enough.

Physical-site dedup runs after normal semantic key dedup and enforces:

- canonical display IDs (`ATL-HQ-01`, `ATL-WEST-02`),
- clipped ID repair (`ATL-WEST-0` -> `ATL-WEST-02` when a full ID is present),
- short alias collapse (`ATL-HQ` -> `ATL-HQ-01`),
- garbage ID rejection (`ALL`, `MOCK-*`, `PO-*`, naked years, customer/document IDs),
- strict value fields for `physical_site` so LLM bridge shape (`canonical_name`, `aliases`, `_via`) cannot create Frankenstein atoms.

## Model graph target

```text
Layout/OCR layer
  Docling / Azure Document Intelligence / PDF text / Office table readers

Table understanding layer
  Deterministic schema registry for known schemas
  + structured-output LLM schema induction for unknown table families

Extraction layer
  Structured-output multi-category extractor for prose facts
  Deterministic bridges for high-precision structures: rosters, contacts, signatures

Resolution layer
  Per-type schema validation
  Embedding candidate retrieval
  Pairwise learned/LLM equivalence judge for borderline merges
  Union-find canonicalization

Confidence layer
  Current: provenance/value/corroboration heuristics
  Target: learned calibrator with ECE checks against pack-level gold labels

Packet layer
  Graph edges from entity co-reference, contradiction, support, and semantic links
  Packet builders consume typed atoms and certified receipts
```

## Scale guardrails

The patch moves expensive generic near-duplicate and packet scans away from structured roster rows. Structured rows use semantic keys; long text fuzzy matching and semantic-link similarity are reserved for prose-scale candidates.
