"""One PDF through MuPDF at a time.

PyMuPDF binds a SINGLE global MuPDF context. Separate ``fitz.Document``
objects in separate threads still share it, so concurrent parses corrupt each
other's extraction -- silently, with no exception. Measured on live 01491cca's
``Attachment+1_B704+Fitness+Center+P.W.+SOW.pdf``, PyMuPDF 1.27.2.3, the SAME
file parsed four times:

    serially     156 atoms, 156, 156, 156   (identical, every run)
    4 threads    190 atoms, 153, 157, 157   (every run different)

A statement of work is the document the scope comes from, and this moved its
content by up to 24% depending on what else happened to be parsing. It also
made ``label_key`` unstable, so gold labels attached or detached by luck.

``parse_artifacts`` has run a thread pool since #247, so this has been live.
The parser already knew the risk within one parse -- ``_build_low_text_page``
re-opens the document "to keep fitz state isolated from the outer page-loop"
-- but nothing isolated one artifact's parse from another's.

Serialising PDFs costs the overlap on PDF work only; every other artifact type
still parses in parallel, and a deal is mostly email and spreadsheets. The
cost lands hardest on a scanned PDF, whose OCR round trips are inside the lock
because ``ocr_pdf_page`` needs the open page. Lifting that out -- rasterise
under the lock, OCR the bytes outside it -- is the optimisation to make next,
and it does not change any of the reasoning above.

Re-entrant, because a PDF parse re-enters through nested helpers.
"""
from __future__ import annotations

import threading

#: Held for the whole of one PDF parse. Import and use this lock rather than
#: making another one: two locks around one global context protect nothing.
PDF_PARSE_LOCK = threading.RLock()
