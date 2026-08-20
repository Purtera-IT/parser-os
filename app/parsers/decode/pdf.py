"""PDF decoding: Document Intelligence where it reads more, fitz where it doesn't.

PDF is the only format worth a decoder fight. A spreadsheet declares its own
cells and a docx declares its own runs, but a PDF describes ink on a page and
every piece of structure above that is inference. That is why measured on real
deal documents:

    Phillips Connect K298   fitz  2 tables /  53 cells / 433 lines
                            DI    4 tables / 233 cells / 942 lines
    APS_fiber_Attachment_B  fitz  2 tables /  13 cells
                            DI    4 tables / 964 cells   (a 159-site roster)
    B704 Fitness SOW        fitz  raised "No common ancestor in structure tree"

and equally why fitz still wins sometimes:

    CFC_Conference_Rooms_BOM   fitz 134 cells   DI 130
    Livano Virginia Beach      fitz  76 cells   DI  77

Neither is the better reader in general, so the choice is made per page on what
each actually recovered, ties going to fitz because its output has already been
through the transposition and underscore repairs.
"""

from __future__ import annotations

import os

from app.core.env import env_get
import warnings
from pathlib import Path
from typing import Any

from app.parsers.decode._tables import _table_rows_repaired
from app.parsers.decode.base import Block, DecodedDoc, Figure, Locator, Table

_PDF_MAGIC = b"%PDF-"


def _disabled() -> bool:
    return env_get("PARSER_OS_PDF_TABLES_DOC_INTEL", "1").strip().lower() in {
        "0", "false", "no", "off",
    }


def _di_pages(path: Path) -> list[dict[str, Any]]:
    """``prebuilt-layout`` for the whole PDF, or [] when unavailable.

    One call serves text, tables and figures. Empty on any failure so the
    caller silently keeps fitz -- an outage costs recall, never a compile.
    """
    if _disabled():
        return []
    try:
        from app.core.doc_intel_ocr import doc_intel_available, extract_pdf_pages

        if not doc_intel_available():
            return []
        return extract_pdf_pages(path.read_bytes()) or []
    except Exception:
        return []


def _lines(text: str) -> int:
    return sum(1 for ln in (text or "").splitlines() if ln.strip())


def _di_tables_for(page: dict[str, Any]) -> list[list[list[str]]]:
    out: list[list[list[str]]] = []
    for table in page.get("tables") or []:
        cells = table.get("cells") or []
        if not cells:
            continue
        n_rows = max(int(c.get("row", 0)) for c in cells) + 1
        n_cols = max(int(c.get("col", 0)) for c in cells) + 1
        grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for c in cells:
            grid[int(c.get("row", 0))][int(c.get("col", 0))] = c.get("text") or ""
        out.append(grid)
    return out


def _cells_in(grids: list[list[list[str]]]) -> int:
    return sum(1 for g in grids for row in g for c in row if str(c or "").strip())


class PdfDecoder:
    """Reads a PDF. Decides nothing about what it means."""

    name = "pdf"

    def can_decode(self, path: Path) -> bool:
        try:
            with open(path, "rb") as fh:
                return fh.read(len(_PDF_MAGIC)) == _PDF_MAGIC
        except Exception:
            return False

    def decode(self, path: Path) -> DecodedDoc:
        path = Path(path)
        try:
            import fitz  # type: ignore[import-not-found]
        except Exception:
            return DecodedDoc(path=path)

        di_by_page: dict[int, dict[str, Any]] = {}
        for p in _di_pages(path):
            try:
                n = int(p.get("page_number") or 0)
            except Exception:
                continue
            if n > 0:
                di_by_page[n - 1] = p

        blocks: list[Block] = []
        tables: list[Table] = []
        figures: list[Figure] = []
        backends = {"text": "fitz", "tables": "fitz"}
        order = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                doc = fitz.open(path)
            except Exception:
                return DecodedDoc(path=path)
            try:
                for page_index in range(doc.page_count):
                    try:
                        page = doc.load_page(page_index)
                    except Exception:
                        continue
                    di = di_by_page.get(page_index, {})

                    # ---- text: whichever carried more, fitz on a tie -------
                    try:
                        fitz_text = page.get_text() or ""
                    except Exception:
                        fitz_text = ""
                    di_text = di.get("text") or ""
                    if _lines(di_text) > _lines(fitz_text):
                        chosen, src = di_text, "doc_intel"
                        backends["text"] = "doc_intel"
                    else:
                        chosen, src = fitz_text, "fitz"
                    for line in chosen.splitlines():
                        if line.strip():
                            blocks.append(Block(
                                text=line,
                                kind="paragraph",
                                locator=Locator(page=page_index + 1,
                                                extra={"reader": src}),
                                order=order,
                            ))
                            order += 1

                    # ---- tables: same rule, measured in cells --------------
                    di_grids = _di_tables_for(di) if di else []
                    try:
                        fitz_tabs = list(
                            getattr(page.find_tables(), "tables", []) or []
                        )
                    except Exception:
                        fitz_tabs = []

                    fitz_grids: list[tuple[list[list[str]], Any]] = []
                    for t in fitz_tabs:
                        try:
                            rows = _table_rows_repaired(page, t)
                        except Exception:
                            try:
                                rows = t.extract()
                            except Exception:
                                continue
                        fitz_grids.append(
                            ([[str(c or "") for c in r] for r in rows],
                             getattr(t, "bbox", None))
                        )

                    if di_grids and _cells_in(di_grids) > _cells_in(
                        [g for g, _ in fitz_grids]
                    ):
                        backends["tables"] = "doc_intel"
                        for g in di_grids:
                            tables.append(Table(
                                rows=g,
                                locator=Locator(page=page_index + 1,
                                                extra={"reader": "doc_intel"}),
                                order=order,
                            ))
                            order += 1
                    else:
                        for g, bbox in fitz_grids:
                            tables.append(Table(
                                rows=g,
                                locator=Locator(
                                    page=page_index + 1,
                                    bbox=tuple(bbox) if bbox else None,
                                    extra={"reader": "fitz"},
                                ),
                                order=order,
                            ))
                            order += 1

                    # ---- figures ------------------------------------------
                    try:
                        for img in page.get_images(full=True) or []:
                            figures.append(Figure(
                                locator=Locator(page=page_index + 1),
                                image_ref=str(img[0]),
                                order=order,
                            ))
                            order += 1
                    except Exception:
                        pass

                return DecodedDoc(
                    path=path,
                    blocks=blocks,
                    tables=tables,
                    figures=figures,
                    page_count=doc.page_count,
                    backends=backends,
                )
            finally:
                try:
                    doc.close()
                except Exception:
                    pass
