"""Read a Word TEMPLATE, which is a Word document that says it is a template.

``.dotx`` is OOXML, identical in structure to ``.docx``. The only difference
that matters is one string in ``[Content_Types].xml``::

    .docx   application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml
    .dotx   application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml

python-docx checks that declaration and raises ``ValueError: ... is not a Word
file``, so the file reached the text fallback and produced nothing. Live corpus:
two of them, and both are CHANGE ORDERS --

    010195- TV Install Change Order 8.20 v1.dotx
    00051- Merrill Gardens (Change Order).dotx

A change order is scope and money. Losing one because a content-type string
says "template" is the cheapest kind of content loss there is.

This rewrites that one declaration into a temporary copy and hands the copy to
the parser that already works, rather than teaching DocxParser a second format.
The original is never modified.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

TEMPLATE_SUFFIXES = {".dotx", ".dotm"}

_TEMPLATE_CT = "wordprocessingml.template.main+xml"
_DOCUMENT_CT = "wordprocessingml.document.main+xml"

#: A .dotm carries macros. The macro part is left in place -- nothing here
#: executes it, and stripping parts risks breaking relationships the document
#: body refers to. Only the content-type declaration is touched.


def is_word_template(path: Path) -> bool:
    return path.suffix.lower() in TEMPLATE_SUFFIXES


def to_docx(path: Path) -> Path | None:
    """A temporary ``.docx`` with the template declaration rewritten.

    Returns None when the file is not a template, is not a readable zip, or
    does not actually declare itself one -- in every case the caller should do
    exactly what it does today rather than fail the artifact.
    """
    if not is_word_template(path):
        return None
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "[Content_Types].xml" not in names:
                return None
            declared = z.read("[Content_Types].xml").decode("utf-8", errors="replace")
    except Exception as exc:
        log.warning("word template %s is not a readable zip: %s: %s",
                    path.name, type(exc).__name__, exc)
        return None

    if _TEMPLATE_CT not in declared:
        # Already declares itself a document, or is something else wearing a
        # .dotx name. Either way this rewrite has nothing to do.
        return None

    out_dir = Path(tempfile.mkdtemp(prefix="dotx2docx_"))
    out = out_dir / (path.stem + ".docx")
    rewritten = declared.replace(_TEMPLATE_CT, _DOCUMENT_CT)

    try:
        # Copy every part across unchanged except the one declaration. Rewriting
        # in place on a copy of the archive would be shorter, but zipfile cannot
        # replace an entry, and appending a second [Content_Types].xml leaves a
        # file with two of them -- which some readers accept and some do not.
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(
            out, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "[Content_Types].xml":
                    data = rewritten.encode("utf-8")
                dst.writestr(item, data)
    except Exception as exc:
        log.warning("could not rewrite %s: %s: %s", path.name, type(exc).__name__, exc)
        shutil.rmtree(out_dir, ignore_errors=True)
        return None
    return out


def can_read(path: Path) -> bool:
    """True when the rewrite produces something python-docx will open.

    Asked at ROUTING time. A parser that claims every ``.dotx`` and then fails
    on one produces zero atoms and no error -- the file lands in the manifest
    and nowhere else, which is the silent miss the marker parser exists to
    prevent.
    """
    out = to_docx(path)
    if out is None:
        return False
    try:
        import docx

        docx.Document(str(out))
        return True
    except Exception:
        return False
