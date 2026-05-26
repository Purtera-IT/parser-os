# parser-os

**Deterministic evidence compiler.** Reads raw project artifacts and emits
a typed, versioned `orbitbrief.input.v2` envelope. No LLM in the hot
path. Re-running on identical input produces a byte-identical
`output_signature`.

```
artifacts → atoms → entities → graph → packets → envelope
```

OrbitBrief-Core consumes the envelope and produces the PM brief, SOW
draft, and RFP draft. End-to-end architecture: [SYSTEM_README.md](SYSTEM_README.md).

---

## What it does

- **Parses** 19 file formats — PDF, XLSX, DOCX, PPTX, CSV, MD, EML,
  MBOX, MSG, HTML, RTF, ICS, ZIP, ODT, ODS, VSDX, MPP, transcripts
  (VTT/SRT), and images (OCR via PyMuPDF + Tesseract / vision-LLM
  fallback).
- **Extracts atoms** — every parser emits typed `EvidenceAtom` records
  with source-locator provenance (file, page/row/line, char offset).
- **Resolves entities** — sites, vendors, devices, part numbers,
  quantities, stakeholders. Coreference collapses "Renee Watkins" /
  "R. Watkins" / "Ms. Watkins" to one entity.
- **Builds an evidence graph** with `supports`, `contradicts`,
  `excludes`, `requires`, `same_as`, `located_in`, `derived_from`,
  `quoted_from` edges.
- **Certifies packets** — 11 families: `scope_inclusion`,
  `scope_exclusion`, `quantity_claim`, `quantity_conflict`,
  `site_access`, `missing_info`, `customer_override`,
  `vendor_mismatch`, `meeting_decision`, `action_item`,
  `compliance_clause`.
- **Replays source** — every atom carries a receipt that re-extracts
  the same text from the original artifact on demand.
- **Emits envelope** — the `orbitbrief.input.v2` schema is the only
  contract downstream consumers see.

## What it is NOT

No chatbot. No SOW generator. No LLM in the parsing path. (LLMs live
in OrbitBrief-Core, behind the envelope seam.)

---

## Quickstart

Requires Python 3.12+.

```bash
pip install -e ".[dev]"
parser-os health
parser-os compile <project_dir> --out result.json --review-out reviews/
```

A `parser-os` project is a directory:

```
my_project/
├── project.yaml          (optional — pin a domain pack, customer name)
├── SOURCE_NOTES.md       (optional — context for auto-routing)
├── .parserignore         (optional — extra glob patterns to skip)
├── artifacts/            (the source documents)
└── labels/               (optional — gold_standard.json for `compare`)
    └── gold_standard.json
```

Scaffold a new one: `parser-os init <dir>`.

## CLI

```bash
parser-os compile <project_dir> --out result.json [--review-out reviews/]
parser-os batch-compile --cases-dir real_data_cases --out-dir batch_out/
parser-os matrix --cases-dir real_data_cases --out matrix.json --markdown-out matrix.md
parser-os compare --gold labels/gold_standard.json --compiled result.json
parser-os orbitbrief-envelope --result result.json --out-dir envelope/
parser-os report <project_dir> --out-dir handoff/   # compile + compare + executive ZIP
parser-os init <project_dir> --service-line wireless --customer acme_corp
parser-os health
```

Full surface: `parser-os --help`.

---

## Repo map

| Path | Purpose |
|---|---|
| `app/core/` | Pipeline stages (parse → atoms → graph → packets → envelope) |
| `app/parsers/` | 19 format-specific parsers + the router + OCR chain |
| `app/domain/` | Domain packs + auto-routing + ontologies |
| `app/api/` | FastAPI routes (`/compile`, `/packets`, …) |
| `app/eval/` | Gold-comparison + benchmark utilities |
| `app/learning/` | Calibration, rule mining, active learning |
| `tests/` | Regression suite |
| `real_data_cases/` | Stress-test corpus + hand-labeled gold standards |
| `scripts/` | Utility CLIs (benchmarks, fixture builders, review tools) |
| `orbitbrief_page_os/` | Site-schematic page parser (separate sub-system) |

Per-directory READMEs:

- [app/README.md](app/README.md) — architecture overview
- [app/core/README.md](app/core/README.md) — pipeline stages
- [app/parsers/README.md](app/parsers/README.md) — parser registry + router
- [app/domain/README.md](app/domain/README.md) — domain packs
- [scripts/README.md](scripts/README.md) — utility CLIs
- [tests/README.md](tests/README.md) — test architecture

## Cross-system docs

| Doc | Purpose |
|---|---|
| [SYSTEM_README.md](SYSTEM_README.md) | End-to-end: parser-os ↔ OrbitBrief-Core ↔ UI |
| [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) | Azure deployment + API + storage layout |
| [OUTPUTS_FOR_UI.md](OUTPUTS_FOR_UI.md) | Every field in `PM_HANDOFF.json` with concrete example values |

---

## Determinism + provenance contract

Two guarantees the rest of the system depends on:

1. **Deterministic.** Two `compile` runs on the same artifacts produce
   byte-identical `output_signature` and `compile_id`. No randomness,
   no time-of-day, no global state.
2. **Provenanced.** Every atom has a `SourceRef` pinning it to a
   specific page / row / line, plus an optional replay receipt that
   re-extracts the same text on demand.

The regression suite catches the obvious breakage.

## Tests

```bash
pytest                                       # full suite
pytest tests/test_entity_extraction.py       # one module
pytest tests/test_week6_dx.py -k compliance  # one keyword
```

---

## License

See `pyproject.toml`. Closed-source today.
