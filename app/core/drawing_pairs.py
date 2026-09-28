"""When a sheet arrives twice, only one copy of it is a source.

A deal almost never gets a drawing once. The architect issues SP-6.dwg and
exports SP-6.pdf, and both land in the same mailbox, so the parser reads the
same sheet twice by two different methods:

    the DWG   62 atoms, with the program summary, every layer and every
              coordinate -- the counts, the partition footage, the NIC regions
    the PDF   12 atoms from OCR and a vision pass: "IT", "JAN", "ADA RR",
              "WOMEN'S RESTROOM", and two paragraphs of prose about a picture

The second set is not a second opinion. It is a worse reading of the same
sheet, and every row of it is a room word that decides nothing. On live 010180
those twelve rows sat in the workspace looking exactly as authoritative as the
106-workstation count.

So the PDF's rows are demoted, not deleted -- the same rule the substance gate
follows. A PM who opens the drawing still finds the PDF's own account of it,
marked as the export of a sheet we read properly, and a head sees a labelled
example of the distinction rather than a gap where the evidence used to be.

Deleting them would also be wrong in the one case that matters: when the DWG
fails to convert, the PDF is the ONLY reading of the sheet, and it is promoted
back automatically because there is nothing to supersede it.
"""
from __future__ import annotations

import re
from typing import Any

#: The flag that says "a better reading of this same sheet exists".
SUPERSEDED_FLAG = "superseded_by_drawing"

#: A sheet code as a draughtsman writes it in a filename: SP-6, A-101, T1.2.
#:
#: `\b` cannot open this. A filename writes "12 FL_SP-6", and `_` is a word
#: character, so there is no word boundary in front of the code at all -- the
#: pattern matched nothing on the one filename it was written for.
_SHEET_CODE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z]{1,3})[\s_-]?(\d{1,3}(?:[.\-]\d{1,2})?)(?![A-Za-z0-9])",
    re.I)

_DRAWING_EXT = (".dwg", ".dxf")
_EXPORT_EXT = (".pdf",)

#: Words that carry no identity -- a date stamp, the issuer, "floor", "plan".
_NOISE = re.compile(r"^(\d{1,4}|\d{2}[.\-]\d{2}[.\-]\d{2,4}|fl|floor|plan|plans|"
                    r"drawing|dwg|pdf|rev|sheet|copy|final|new)$", re.I)


def _stem(name: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]+$", "", name or "")


def sheet_code(filename: str) -> str:
    """The sheet's own code, e.g. 'SP6' from '...12 FL_SP-6.dwg'.

    Taken from the END of the stem, because that is where a sheet code sits and
    because an address like '7 PENN PLAZA' matches the same shape.
    """
    codes = _SHEET_CODE.findall(_stem(filename))
    if not codes:
        return ""
    letters, digits = codes[-1]
    return f"{letters}{digits}".upper().replace("-", "").replace(".", "")


def _tokens(filename: str) -> set[str]:
    raw = re.split(r"[^A-Za-z0-9]+", _stem(filename))
    return {t.upper() for t in raw if t and not _NOISE.match(t)}


def same_sheet(a: str, b: str) -> bool:
    """Are these two filenames the same sheet in two formats?

    Both tests have to pass. The sheet code alone would pair every SP-6 on
    every deal; the token overlap alone would pair two sheets of one set,
    which are different drawings of the same building.
    """
    ca, cb = sheet_code(a), sheet_code(b)
    if not ca or ca != cb:
        return False
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.6


def _artifact(atom: Any) -> str:
    return str(getattr(atom, "artifact_id", "") or "")


def _filename(atom: Any) -> str:
    for ref in (getattr(atom, "source_refs", None) or []):
        name = getattr(ref, "filename", None)
        if name:
            return str(name)
    return str(getattr(atom, "filename", "") or "")


def _demote(atom: Any, sheet: str) -> None:
    """Take away the claim, leave the words. (As `atom_substance_gate` does.)"""
    flags = getattr(atom, "review_flags", None)
    if isinstance(flags, list) and SUPERSEDED_FLAG not in flags:
        flags.append(SUPERSEDED_FLAG)
    try:
        from app.core.schemas import AtomType, ReviewStatus  # noqa: PLC0415

        if getattr(atom, "atom_type", None) is not AtomType.deal_metadata:
            atom.atom_type = AtomType.deal_metadata
        # Same reason the substance gate does this: an atom that survives only
        # because nobody has judged it IS awaiting review, and the compile
        # validator holds that an abstaining atom must say so.
        if getattr(atom, "review_status", None) is ReviewStatus.auto_accepted:
            atom.review_status = ReviewStatus.needs_review
    except Exception:  # noqa: BLE001
        pass
    note = (f"The export of a sheet we read from the drawing itself ({sheet}). "
            f"The DWG gives this room with its layer, its coordinates and the "
            f"program summary's count; this row is the same room read off a "
            f"picture of the same sheet, so it corroborates and does not state.")
    try:
        if not getattr(atom, "reviewer_note", None):
            atom.reviewer_note = note
    except Exception:  # noqa: BLE001
        pass


def demote_export_duplicates(atoms: list[Any]) -> list[Any]:
    """Demote a PDF's rows when the same sheet's DWG parsed.

    Returns the same list -- nothing is removed. When the drawing produced
    nothing (no converter, a file that will not open), the export is the only
    reading there is and is left exactly as it was.
    """
    drawings: dict[str, str] = {}      # filename -> artifact_id
    for atom in atoms:
        name = _filename(atom)
        if name.lower().endswith(_DRAWING_EXT):
            drawings.setdefault(name, _artifact(atom))
    if not drawings:
        return atoms

    #: A drawing that yielded only its "awaiting conversion" marker has not
    #: been read, and cannot supersede anything.
    read: dict[str, int] = {name: 0 for name in drawings}
    for atom in atoms:
        name = _filename(atom)
        if name in read and (getattr(atom, "value", None) or {}).get("kind") != "cad_marker":
            read[name] += 1

    for atom in atoms:
        name = _filename(atom)
        if not name.lower().endswith(_EXPORT_EXT):
            continue
        for dwg, _ in drawings.items():
            if read.get(dwg, 0) >= 2 and same_sheet(name, dwg):
                _demote(atom, dwg)
                break
    return atoms
