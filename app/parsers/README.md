# `app/parsers/` — file format parsers

Each parser turns one file format into `EvidenceAtom[]`. Parsers don't
know about domain packs or graphs — they just produce atoms with
provenance. Domain-aware enrichment runs later in the pipeline
(`enrich_entities` stage).

## Built-in parsers

| Parser | File | Extensions | Purpose |
|---|---|---|---|
| `OrbitbriefPdfParser` | `orbitbrief_pdf.py` | `.pdf` | PDF → structured.json (sections, paragraphs, bullets, tables, notes) → atoms. The flagship parser; carries Q&A blob splitting, form-field filtering, page-footer detection, classifier overrides for compliance / customer authority / decision / action_item shapes. |
| `XlsxParser` | `xlsx_parser.py` | `.xlsx` | Spreadsheet → atoms. Tries canonical cabling/networking schedules first; falls back to a generic-row emitter. Splits Q&A rows into Q + A sub-atoms when columns include `question`/`response` shapes. |
| `DocxParser` | `docx_parser.py` | `.docx` | Word docs → paragraphs / tables / lists → atoms. |
| `EmailParser` | `email_parser.py` | `.eml`, `.msg` | Email → header (from/to/subject/date) + body line-level atoms (instruction / exclusion / open-question shapes). |
| `TranscriptParser` | `transcript_parser.py` | `.txt`, `.json` | Meeting transcripts → speaker-attributed atoms (`decision` / `action_item` / `meeting_commitment` / `assumption`). |
| `QuoteParser` | `quote_parser.py` | `.xlsx` (vendor-quote shape) | Vendor pricing schedules → `vendor_line_item` + `quantity` atoms with material-key normalization. |

## Architecture

Every parser implements `ArtifactParser` (`base.py`):

```python
class ArtifactParser(ABC):
    parser_name: str
    parser_version: str
    capability: ParserCapability   # supported_extensions, supported_artifact_types, …

    def match(self, path, sample_text, domain_pack) -> ParserMatch:
        """Score how well this parser fits the artifact (0.0–1.0)."""

    @abstractmethod
    def parse(self, path, project_id, artifact_id, domain_pack) -> ParserOutput:
        """Read the file and emit atoms. Must be deterministic."""
```

Parsers register themselves at module-import time via `register_parser()`
in `registry.py`. The router (`parser_router.py`) calls `match()` on every
registered parser and picks the highest-scoring one above `MATCH_THRESHOLD`
(0.5). When no parser scores high enough the artifact is skipped and
listed in `manifest.unsupported_artifacts`.

## Adding a new parser

1. **Subclass `ArtifactParser`** in a new module under `app/parsers/`:

   ```python
   from app.parsers.base import ArtifactParser
   from app.core.schemas import ParserCapability, ArtifactType, ParserOutput

   class MyParser(ArtifactParser):
       parser_name = "my_parser"
       parser_version = "v1"
       capability = ParserCapability(
           parser_name="my_parser",
           parser_version="v1",
           supported_extensions=[".myext"],
           supported_artifact_types=[ArtifactType.other],
           emitted_atom_types=[AtomType.scope_item, AtomType.constraint],
           supports_source_replay=True,
       )

       def parse(self, path, project_id, artifact_id, domain_pack) -> ParserOutput:
           atoms: list[EvidenceAtom] = []
           # ... emit atoms ...
           return ParserOutput(atoms=atoms, derived_files={})
   ```

2. **Register it** in `registry.py::_ensure_defaults`:

   ```python
   from app.parsers.my_parser import MyParser
   register_parser(MyParser())
   ```

3. **Bump the parser version** every time you change classification logic
   that affects atom output, so the artifact cache invalidates. The
   cache key is `sha256(file) + parser_name + parser_version`.

4. **Source-replay** — if your format supports re-extracting atom text by
   locator, set `supports_source_replay=True` and implement the replay
   path. Otherwise the compile will produce `replay_status: unsupported`
   for atoms from this parser. `app/core/source_replay.py` has the dispatch
   table.

5. **Tests** — add a test under `tests/test_<parser>.py` that exercises:
   - A representative real input → expected atoms
   - A degenerate input (empty file, malformed structure) → no crash
   - Determinism — calling `parse()` twice on the same input produces
     equal output

   Plus an adversarial test under `tests/parser_adversarial/` if you can
   construct edge-case inputs that broke previous attempts (typos, fused
   rows, page-footer leakage, multi-line cells, …).

## Atom emission contract

Every atom must:

- Have a stable `id` (`stable_id("atm", artifact_id, ...)` from `app/core/ids.py`)
- Carry at least one `source_ref` with a non-empty `locator`
- Have non-empty `raw_text` (the model rejects empty strings)
- Set `parser_version` to your parser's version string
- Leave `entity_keys=[]` (the `enrich_entities` stage populates it)
  unless you have format-specific entity knowledge worth pre-populating

Pick the right `atom_type` (see `AtomType` enum in `app/core/schemas.py`)
and `authority_class`. When the source signals customer-authoring
(addendum responses, Q&A answer markers, owner-furnished notes), set
`authority_class=customer_current_authored` so the packetizer can fire
`customer_override` packets.

## Filtering noise

Parsers should filter out content that's structurally not scope:

- Page headers / footers (`_looks_like_page_footer` in `orbitbrief_pdf.py`)
- Form-field templates (`_looks_like_form_field` — see Week 6's strong-vs-weak
  marker split in `orbitbrief_pdf.py`)
- Title-case bullet labels (`_looks_like_fragment`)
- Fused multi-row table cells (`_looks_like_fused_table_row`)
- Repeated boilerplate (page bands, signature blocks, blank form lines)

The principle: **drop, don't emit-with-low-confidence**. Low-confidence
noise atoms still pollute downstream packet counts and OrbitBrief
output; missing atoms surface in `parsers_with_zero_atoms` and the
gold-compare verdict.
