"""Read the spreadsheet formats openpyxl cannot, by handing them to it anyway.

``.xls`` is the pre-2007 binary BIFF format. openpyxl reads OOXML only, so
``load_workbook`` raises ``InvalidFileException`` on it and the file falls
through routing to the ``unread`` parser. Live corpus: five ``.xls`` files
across four deals, and they are CDW **pricing sheets** -- one of them 1.9 MB
producing a single atom that says nobody read it.

Rather than teach 1,577 lines of XlsxParser a second file format, this converts
the workbook to a temporary ``.xlsx`` and lets the parser that already works do
the work. Every downstream behaviour -- sheet routing, column roles, table
rollup, the quote/xlsx tie-break -- is the code that is already tested.

Why calamine for the read
-------------------------
``python-calamine`` is a Rust reader that handles ``.xls``, ``.xlsb``, ``.ods``
and ``.xlsx`` through one interface. Benchmarked on this corpus:

    anywAIR CALC.xlsx        0.9MB   openpyxl 0.14s   calamine 0.01s   (14x)
    CDW Pricing Sheet.xls    1.9MB   openpyxl FAILED  calamine 0.06s   2,685 rows

It is an optional dependency: without it, ``.xls`` behaves exactly as it does
today rather than raising. A missing reader should narrow what the parser can
do, never break what it already does.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

#: Formats openpyxl cannot open but calamine can. ``.ods`` is included because
#: the corpus has an OdsParser that has never fired -- if one arrives, this is a
#: better answer than the text fallback.
CONVERTIBLE = {".xls", ".xlsb", ".ods"}

#: A cell value long enough to be prose rather than a field. Truncated on write
#: so one pathological cell cannot make the temporary workbook enormous; the
#: parser only ever reads a prefix of a cell for its atoms anyway.
_MAX_CELL_CHARS = 32_000


def calamine_available() -> bool:
    try:
        import python_calamine  # noqa: F401
        return True
    except Exception:
        return False


def needs_conversion(path: Path) -> bool:
    return path.suffix.lower() in CONVERTIBLE


def can_read(path: Path) -> bool:
    """True when a reader can actually open this workbook.

    Asked at ROUTING time, not parse time, and the distinction is the point. If
    the spreadsheet parser claims every ``.xls`` and then cannot open one, the
    file produces zero atoms and no error -- it exists in the manifest and
    nowhere else, which is the silent miss `unread` was built to prevent. A
    workbook this cannot open is left to `unread`, which says so out loud in an
    atom somebody can see.

    Opening costs ~0.06s on a 1.9MB book, against a parse that costs seconds.
    """
    if not needs_conversion(path):
        return False
    try:
        import python_calamine
    except Exception:
        return False
    try:
        python_calamine.CalamineWorkbook.from_path(str(path))
        return True
    except Exception:
        return False


def to_xlsx(path: Path) -> Path | None:
    """A temporary ``.xlsx`` with the same sheets, or None if it cannot be made.

    Returns None rather than raising: the caller's job is to parse a deal, and
    a format it cannot convert should degrade to today's behaviour rather than
    fail the artifact.
    """
    if not needs_conversion(path):
        return None
    try:
        import python_calamine
        from openpyxl import Workbook
    except Exception as exc:
        log.info("legacy spreadsheet %s left unconverted: %s", path.name, exc)
        return None

    try:
        book = python_calamine.CalamineWorkbook.from_path(str(path))
    except Exception as exc:
        log.warning("calamine could not open %s: %s: %s",
                    path.name, type(exc).__name__, exc)
        return None

    wb = Workbook()
    wb.remove(wb.active)          # drop the sheet Workbook() creates for us
    wrote_any = False
    for name in book.sheet_names:
        try:
            rows = book.get_sheet_by_name(name).to_python()
        except Exception as exc:
            log.warning("sheet %r of %s unreadable: %s", name, path.name, exc)
            continue
        # Excel caps a sheet title at 31 characters and forbids []:*?/\ -- a
        # legacy book can carry names openpyxl will refuse to write.
        safe = "".join(ch for ch in str(name) if ch not in "[]:*?/\\")[:31] or "Sheet"
        ws = wb.create_sheet(title=safe)
        for row in rows:
            ws.append([_cell(v) for v in row])
        wrote_any = True

    if not wrote_any:
        return None

    out = Path(tempfile.mkdtemp(prefix="xls2xlsx_")) / (path.stem + ".xlsx")
    try:
        wb.save(out)
    except Exception as exc:
        log.warning("could not write converted %s: %s", path.name, exc)
        return None
    return out


def _cell(value: object) -> object:
    """A value openpyxl will accept.

    calamine returns real Python types -- including ``datetime``, ``date``,
    ``time`` and ``timedelta``. openpyxl writes the first three natively; a
    ``timedelta`` it does not, and one duration cell would otherwise fail the
    whole conversion.
    """
    import datetime as _dt

    if isinstance(value, _dt.timedelta):
        return str(value)
    if isinstance(value, str) and len(value) > _MAX_CELL_CHARS:
        return value[:_MAX_CELL_CHARS]
    return value
