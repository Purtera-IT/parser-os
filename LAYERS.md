# Layers

`ARCHITECTURE.md` describes what the compile *does*. This describes where the
code *lives*, and why moving a file between layers is a decision rather than
housekeeping.

## The one rule

```
decode      bytes  -> blocks, tables, figures, locators   commodity, never learns
segment     blocks -> atoms carrying receipts             shared across formats
interpret   atoms  -> types, facets, roles                judgments, always learn
```

**Decoding has a correct answer; interpretation has an opinion.** A table cell
either says `77-H135` or it does not, and no PM correction generalises from
"you misread that glyph" — that is a library's job, and you buy the best one.
Whether that table is a *site roster* is a judgment, a PM can be wrong about it,
and a correction teaches something transferable.

That is the whole basis for the split, and it decides every filing question:

- **Below the seam, never learn.** Buy the best decoder. Measure it and switch
  when something reads more.
- **Above the seam, always learn.** One entry point, a confidence, an
  abstention, and a correction target.

## Why it is not cosmetic

The design calls for one shared encoder trained on *every correction from every
task, pooled* — ten tasks with a few hundred labels each are ten badly-fit
models, while the same labels in one representation are a few thousand
examples.

**Pooling is impossible while one judgment is implemented several times behind
several signatures.** "Is this a site roster?" existed three times:

| | |
|---|---|
| `site_roster_extractor.looks_like_site_roster` | PDF tables, 15 call sites |
| `table_schema_registry.identify_schema` | column patterns, 5 call sites |
| `sheet_classifier.classify_sheet` | spreadsheet sheets, 9 call sites |

A PM correcting a misread roster in a PDF taught the spreadsheet path nothing,
because there was no shared thing to correct. `app/interpret/table_kind.py` is
the single front door those collapse into.

**The seam is the precondition for the heads, not cleanup that follows them.**

## What stays symbolic, deliberately

| | why |
|---|---|
| Receipts and provenance | facts; their value is being checkable |
| Commercial form | a rule with a receipt beats a classifier without one |
| The coverage census | an invariant, not a prediction |
| Abstention | a component that cannot say "I don't know" cannot be promoted |

## Where things are

Done:

| package | holds |
|---|---|
| `app/parsers/decode/` | `base.py` (Locator/Block/Table/Figure/DecodedDoc/Decoder), `pdf.py`, `_tables.py` |
| `app/coverage/` | the census and binary-region markers — independent of the parser by design |
| `app/interpret/` | `table_kind.py` — one table judgment, one correction target |
| `app/learning/` | registry, calibration, promotion, retrain |

Not yet. `app/core/` is 113 files and ~54,900 lines, and the name says nothing
about any of them. Measured by filename against the layers above:

| group | files | lines |
|---|---|---|
| unclassified | 48 | 19,534 |
| segment (blocks → atoms) | 9 | 8,595 |
| llm access | 5 | 7,799 |
| graph / conflicts | 5 | 4,145 |
| learning (heads, registry) | 14 | 3,673 |
| interpret (atoms → meaning) | 8 | 2,830 |
| compile orchestration | 5 | 2,763 |
| retrieval / memory | 7 | 2,717 |
| receipts / provenance | 4 | 1,959 |
| schematics | 8 | 882 |

The groups are real and mostly separable. They are **not** being moved in one
change: 113 files at once is unreviewable, would collide with everything in
flight, and a mechanical sweep of that size has already produced two silent
breakages in this codebase — an import placed above `from __future__` (which
`ast.parse` accepts and `compile()` rejects), and an `import os as _os` alias
left stranded by a regex. Both passed a weaker check than the one that runs the
code.

So the map exists first, and files move a group at a time, each verified
against the full suite. `app/coverage/` and `decode/_tables.py` are what that
looks like: a move, a re-export so call sites are untouched, and a suite whose
failure set is byte-identical before and after.

## Adding a format

Write a decoder. That is meant to be the whole job:

```python
class MyDecoder:
    name = "myformat"
    def can_decode(self, path: Path) -> bool: ...
    def decode(self, path: Path) -> DecodedDoc: ...
```

Report what the document says and where it sits. `Block.kind` is layout
(`heading`, `list_item`), never meaning (`requirement`, `physical_site`). If a
decoder wants to assign meaning, that judgment belongs above the seam where
every format's corrections reach it.

The reason `orbitbrief_pdf.py` is 10,006 lines is that it never had this seam.
