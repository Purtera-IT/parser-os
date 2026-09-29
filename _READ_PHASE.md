# The READ phase

What happens between "a deal has files" and "the compile has atoms", what it
costs, and what was wrong with it. Every number here was measured on live
artifacts pulled from `orbitbrief-artifacts`, on this branch.

---

## 1. The headline: the same deal now parses the same way every time

It did not before. `parse_artifacts` has run a thread pool since #247, and
**four separate pieces of shared state** were being written by several
documents at once. None of them raised an exception. They changed the words in
the atoms, and atom text is three quarters of `label_key`, so gold labels
attached or detached depending on thread timing.

| # | What was shared | How it showed up |
|---|---|---|
| 1 | One `pysbd.Segmenter` for the whole process | Sentences silently DROPPED, so a paragraph's three atoms became one |
| 2 | `DocxParser._table_lead_in` / `_para_lead_in` / `_structure_idxs` | A table took its lead-in from a DIFFERENT document |
| 3 | `XlsxParser._sheet_profiles` / `_sheet_supplies` / `_block_detection_failures` / `_coverage_backstop_note` | One workbook's reset discarded another's pending failures |
| 4 | PyMuPDF's single global MuPDF context | The same SOW produced 156 atoms serially and 190 / 153 / 157 under threads |

### 1.1 The sentence segmenter

`pysbd.Segmenter.segment` is not reentrant. It parks the text on the instance
(`self.original_text = text`) and reads it back in `sentences_with_char_spans`
to locate each sentence it just produced. A second thread overwrites that
buffer while the first is still searching it, so the first looks for ITS
sentences inside the OTHER thread's document, finds nothing, and the sentence
is dropped from the returned list with no error.

`_expand_lines_to_sentences` keeps a line whole unless it gets back two or more
substantial pieces, so a dropped sentence collapses a paragraph into one atom.
Live 010094:

    serial   "As a follow up, AZ would like to see that attached built out."
             "As they know that costs may vary by location, they ask that we please..."
             "Is this something you may be able to get back to me?"      = 3 atoms
    parallel "As a follow up, AZ would like to see that attached built out. As they..."
                                                                         = 1 atom

Measured directly: **3 of 400 splits wrong under 8 threads, 3 sentences lost.**
Fixed with a thread-local segmenter -- each thread builds at most one, and
construction is microseconds.

### 1.2 One parser instance, many documents

`registry._REGISTERED` holds exactly ONE instance of each parser and
`choose_parser` hands that same object to every thread. Any parser that parks
per-document state on `self` is therefore sharing it.

On 010237 -- the deal I labelled -- the Workato work-order table was given the
lead-in *"The following response and resolution targets apply to incidents
logged through ServiceNow..."*. That sentence belongs to a **different file**
in the deal (`NMC-SLA-Purtera IT-Workato_Middleware`). Parsed on its own the
Workato table has no lead-in at all. It was fabricated context on a contract
table, and it appeared only when the two files happened to parse together.

Fixed with `PerThreadState` (`app/parsers/base.py`): per-thread, per-instance
storage that still raises `AttributeError` when unset, so the
`getattr(self, name, default)` reads throughout the parsers keep their exact
present behaviour. Verified the feature still works where it belongs: the
NMC-SLA file still gets its 11 lead-ins.

The xlsx case was the more dangerous one. `_block_detection_failures`
accumulates through a parse and is drained and reset at the end; a second
workbook resetting it first throws the first one's failures away, so a sheet
whose block detection crashed is never reported and **reads as empty** -- the
exact failure `_note_block_failure` exists to prevent.

### 1.3 PyMuPDF

PyMuPDF binds a single global MuPDF context; separate `fitz.Document` objects
in separate threads still share it. The same file, same bytes, PyMuPDF
1.27.2.3:

    serially    156 atoms, 156, 156, 156      (identical every run)
    4 threads   190 atoms, 153, 157, 157      (every run different)

A statement of work moving by up to 24% depending on what else was parsing.
The parser already suspected this within one parse -- `_build_low_text_page`
re-opens the document "to keep fitz state isolated from the outer page-loop"
-- but nothing isolated one artifact from another.

Fixed by serialising PDF parses (`app/parsers/_pdf_lock.py`). Every other
artifact type still parses in parallel, and the corpus says that is nearly all
of it -- 361 PDFs against 60,642 emails, 662 `.docx` and 559 `.xlsx`. The lock
covers 0.6% of the artifacts and the other 99.4% keep their overlap.

### 1.4 Proof

Eight live deals, widths 1 / 4 / 8, repeated runs, comparing the **text of
every atom** and not just counts:

    01491cca  c065bfc4  02557291  044a1e0f  4edf04d3  01fdcdb3  005a4c6b  abcfdf7a
       all DETERMINISTIC: every width and run produced identical atoms

Also verified with every network path removed (OCR stubbed, LLM off, semantic
rules forced to lexical) so that "shared state" and "a model answered
differently" could be told apart. Both are clean.

---

## 2. Speed

### 2.1 How many parse workers

The default was 8. It is now **4**, and the reason is not the one I first
guessed. Pinned to 4 CPUs to match the container:

| workers | OCR stubbed at 0s | OCR stubbed at 1.5s |
|---|---|---|
| 1 | 12.26s (1.00x) | 171.2s (1.00x) |
| 2 | 12.42s (0.99x) | 87.0s (1.97x) |
| 4 | 12.39s (0.99x) | **47.0s (3.64x)** |
| 8 | 12.47s (0.98x) | 45.8s (3.74x) |
| 12 | 12.41s (0.99x) | -- |

Reading a file is zip inflate, XML and regex -- GIL-bound, so threads buy
**nothing** for it. The entire benefit is overlapping the Document
Intelligence round trip. Four takes 97% of the available speedup; the next
four are worth 1.03x between them, and cost double the concurrent OCR calls
(429s there are what latch `_llm_unreachable`) and double the peak memory.

### 2.1a "Four at a time" is two different things

`SOWSMITH_PARSE_WORKERS=4` is four ARTIFACTS at once inside ONE deal. It is not
four deals at once.

Four deals in flight together is a replica count: the queue hands one message
to one worker, so N deals in parallel needs N replicas (or one replica with
enough CPU to be N workers, which the GIL argues against). Those two numbers
multiply -- 4 replicas each running 4 parse threads is 16 concurrent Document
Intelligence calls, and 429s there are what latch `_llm_unreachable` and
degrade the rest of a compile.

So if the queue UI is set to run four deals at once, the thing to watch is the
OCR concurrency, not the CPU.

### 2.2 Spreadsheets got 3.5x faster, with byte-identical output

`_map_canonical_header` called `_header_cell_tokens(cell)` **inside** its loop
over `HEADER_ALIASES`, recomputing the identical token set once per alias key.
On the 1.8 MB CDW pricing sheet that was **2,104,569 calls** driving 2.08
million regex substitutions -- 60% of the file's parse time spent recomputing
answers it already had.

Hoisted the call out of the loop and memoised the tokenizer (it is a pure
function of the cell string, and a spreadsheet repeats its headers):

| file | before | after | atoms | output |
|---|---|---|---|---|
| 010319-CDW Pricing Sheet | 5.79s | 3.29s | 4114 -> 4114 | identical sha |
| Collegiate Pricing Sheet | 6.58s | 0.69s | 564 -> 564 | identical sha |
| Holbrook Pricing Sheet | 3.87s | 0.66s | 656 -> 656 | identical sha |
| **total** | **16.24s** | **4.57s (3.6x)** | | **byte-identical** |

This is the shared header path, so `.xlsx` and `.csv` get it too, not just the
legacy formats. The legacy bridge itself was never the cost: converting
`.xls` to `.xlsx` through calamine takes **0.65s**.

### 2.3 OCR

`_ocr_cid_part` OCRs every inline image when the body references no CID,
because HTML-only `cid:` refs are invisible to a plain-text scan. That
fallback has to stay -- I tried replacing it with an HTML cid scan, measured
it, and it made things **worse** (010180: 5 -> 33 calls), so it was reverted.

What it costs is handled by two guards instead:

* a byte floor -- an image too small to hold text is a signature logo, not
  content. Measured on 46 live emails across three deals: **99 billed calls
  become 31, a 69% reduction.** PDFs are exempt; a small PDF is a page of text.
* a sha256 cache with a per-key lock, so the logo repeated down a thread is
  paid for once rather than once per message per worker.

---

## 3. Files that used to read as empty

| format | before | after | what was in it |
|---|---|---|---|
| `.xls` | 1 atom | **4,114 atoms** | CDW pricing sheet: materials, labour, riser breakdown |
| `.dotx` | 0 atoms | **45 atoms** | TV Install Change Order -- scope and money |

Live corpus counts: 5 `.xls` and 2 `.dotx`. `.xlsb`, `.ods`, `.odt` and `.dotm`
do not appear at all today, so the bridges cover them defensively rather than
actively -- and note that a `.ods` would currently be claimed by the existing
`OdsParser` at 0.95 before the calamine bridge is asked at 0.58. That ordering
only needs revisiting if one ever arrives.

Both are bridges, not new parsers: convert the unreadable container into the
readable one and hand it to code that is already tested.

* `.xls` / `.xlsb` / `.ods` become a temp `.xlsx` via python-calamine.
* `.dotx` / `.dotm` rewrite ONE string in `[Content_Types].xml`
  (`wordprocessingml.template.main+xml` to `...document.main+xml`), which is
  the only reason python-docx refused the file.

Both ask `can_read()` at **routing** time, so a file the bridge cannot open
keeps its visible `unread` marker instead of silently producing zero atoms.
The `unread` parser's hint was kept and reworded so a PM still gets an action:
*"legacy Excel that could not be opened. Ask for a .xlsx"*.

---

## 4. Routing

**Routing costs 16ms per artifact -- 0.04% of a compile.** A routing head would
save nothing. You suggested one and the measurement says don't build it: there
is no time there to win, and a head that picks the parser would add a model's
nondeterminism to the one decision that is currently exact.

`choose_parser` asks all 22 registered parsers for a confidence and takes the
best. The cost is dominated by `sniff()` reading the first bytes, which has to
happen anyway -- an `.eml` attachment saved as `message` has no suffix, and
without content sniffing the text parsers never see its body.

---

## 5. What is still not deterministic, and why

Everything above makes a compile **self-consistent** and reproducible on a
fixed environment. Two things remain, and both are honest limits rather than
bugs:

1. **Vision and OCR.** A model asked to read a scanned page can answer
   differently on different days. The sha256 cache makes it consistent
   *within* a run; it cannot make two runs agree.
2. **The embedder.** `SemanticRule.fires()` consulted a live network probe on
   *every* call, so one compile could judge one document with embeddings and
   the next with the lexical fallback. That is now resolved **once per
   compile**, under a lock, and frozen (`semantic_backend_available`), and
   `compile_project` resets it when a compile begins. A deal parsed while the
   embedder is up still reads differently from the same deal parsed while it
   is down -- pin `SOWSMITH_SEMANTIC_RULES` to settle that permanently.

---

## 6. Quality gate

The measure that matters is gold-label re-attachment: a `label_key` is
`sha256(deal | filename | page | text)`, so a label stops resolving the moment
the parser words a fact differently. Old tree against new, same files:

| deal | gold labels | resolved BEFORE | resolved AFTER |
|---|---|---|---|
| 1bb4a199 | 3,163 | 1,016 | 1,016 |
| c79db726 | 441 | 244 | 244 |
| c065bfc4 (010237) | 197 | 63 | 63 |
| **total** | **3,801** | **1,323** | **1,323** |

**LOST 0.**

(A label resolving at all depends on the whole compile, not just the parse --
these runs parse the artifacts and stop, so post-parse stages that reword an
atom are not represented. That is why the absolute number is not 3,801. The
figure that matters is that BEFORE and AFTER are the same number, on the same
files, through the same harness.)

---

## 7. The next thing to do here

Lift OCR out of the PDF lock. `ocr_pdf_page` needs the open page, so a scanned
PDF currently pays its Document Intelligence round trips inside the lock.
Rasterising the page under the lock and OCRing the bytes outside it would give
scanned PDFs the same overlap every other artifact type already has. It does
not change any of the correctness reasoning above.

---

## 8. Library upgrades available, deliberately NOT taken

Each of these changes how a file is read, so each needs its own impact run
(`_tools/_label_impact.py` on both trees, plus a sha256 comparison of every
atom). I did not take them overnight, because "the parser reads this document
differently now" is exactly the change that must be looked at before it ships.

| package | installed | available | why it matters |
|---|---|---|---|
| PyMuPDF | 1.27.2.3 | 1.28.2 | moves PDF text extraction across the whole corpus |
| extract-msg | 0.55.0 | 0.56.1 | `.msg` bodies and attachments |
| openpyxl | 3.1.2 | 3.1.5 | patch-level, lowest risk of the four |
| lxml | 6.0.2 | 6.1.3 | HTML email bodies |
| beautifulsoup4 | 4.12.3 | 4.15.0 | HTML email bodies |

PyMuPDF also prints a standing hint on every PDF: *"Consider using the
pymupdf_layout package for a greatly improved page layout analysis."* That is
a different extraction strategy rather than a version bump, and the PDFs are
where the scope lives, so it is worth evaluating properly -- on the gold-label
corpus, not on a sample.
