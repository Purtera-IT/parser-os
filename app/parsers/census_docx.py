"""Independent region census for a .docx artifact.

This is the *independent reader* the content census needs: it inventories
**every region of every kind** in a Word file straight from the OOXML zip —
deliberately NOT via python-docx, whose ``document.paragraphs`` /
``document.tables`` only see body direct children and silently miss content
controls (``w:sdt``), textboxes (``w:txbxContent``), headers/footers,
footnotes/endnotes, comments, and embedded media.

Because the denominator comes from a reader the extractor can't bias, the
coverage invariant (every region COVERED or MARKED, never UNCOVERED) can catch
the *never-detected* loss class — content that exists in the file but the
parser's own field of view can't even see.

Usage::

    census = census_docx(path, artifact_id="art")
    census.reconcile(parser_atoms)
    print(census.report())
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from app.core.content_census import ContentCensus, Region, RegionKind

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _local(tag: str) -> str:
    """Strip the ``{namespace}`` prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _para_text(p: ET.Element) -> str:
    """Text of a single ``w:p``: runs concatenated with NO separator.

    Word frequently splits one word across multiple ``w:r``/``w:t`` runs for
    formatting (e.g. ``$`` + ``98.00``). python-docx's ``paragraph.text`` joins
    them with nothing, so we must too — otherwise the census would read
    ``$ 98.00`` and never reconcile against the parser's ``$98.00``.
    """
    return "".join(t.text or "" for t in p.iter(f"{{{_W}}}t")).strip()


def _text_of(el: ET.Element) -> str:
    """Visible text of a block element (``w:p`` / cell / table).

    Concatenates runs within each paragraph (no separator, matching
    python-docx) and joins separate paragraphs with a space.
    """
    paras = [_para_text(p) for p in el.iter(f"{{{_W}}}p")]
    paras = [p for p in paras if p]
    if paras:
        return " ".join(paras).strip()
    # Fallback for elements with no w:p children (rare).
    return "".join(t.text or "" for t in el.iter(f"{{{_W}}}t")).strip()


def _row_text(tr: ET.Element) -> str:
    """One table row rendered as ``cell | cell | cell`` (matches parser)."""
    cells: list[str] = []
    for tc in tr.iter(f"{{{_W}}}tc"):
        txt = _text_of(tc).strip()
        if txt:
            cells.append(txt)
    return " | ".join(cells)


def _ancestor_kinds(el: ET.Element, parent: dict[ET.Element, ET.Element]) -> set[str]:
    """Local tag names of every ancestor (for sdt / txbx / tbl detection)."""
    out: set[str] = set()
    cur = parent.get(el)
    while cur is not None:
        out.add(_local(cur.tag))
        cur = parent.get(cur)
    return out


def _inventory_body(
    census: ContentCensus,
    artifact_id: str,
    xml_bytes: bytes,
) -> None:
    """Inventory paragraphs + tables in word/document.xml, incl. sdt/txbx."""
    root = ET.fromstring(xml_bytes)
    parent = {child: par for par in root.iter() for child in par}

    p_seen = 0
    tbl_seen = 0
    # Tables first, so we can skip the paragraphs they contain. We inventory
    # one region PER ROW (not per table): the parser emits a per-row atom, so
    # per-row granularity reconciles honestly and surfaces a single dropped
    # row as partial loss rather than hiding it behind a whole-table match.
    for tbl in root.iter(f"{{{_W}}}tbl"):
        anc = _ancestor_kinds(tbl, parent)
        # Skip nested tables (a tbl inside a tbl) — the outer one covers it.
        if "tbl" in anc:
            continue
        where = "sdt/tbl" if "sdtContent" in anc else ("txbx/tbl" if "txbxContent" in anc else "body/tbl")
        for row_idx, tr in enumerate(tbl.iter(f"{{{_W}}}tr")):
            row_txt = _row_text(tr).strip()
            if not row_txt:
                continue
            census.register(Region(
                region_id=f"{artifact_id}:tbl{tbl_seen}.r{row_idx}",
                artifact=artifact_id,
                kind=RegionKind.TABLE,
                location=f"{where}{tbl_seen}.r{row_idx}",
                text=row_txt,
            ))
        tbl_seen += 1

    for p in root.iter(f"{{{_W}}}p"):
        anc = _ancestor_kinds(p, parent)
        if "tbl" in anc:
            continue  # counted within its table region
        txt = _text_of(p)
        if not txt:
            continue
        if "sdtContent" in anc:
            where, kind = "sdt/p", RegionKind.TEXT
        elif "txbxContent" in anc:
            where, kind = "txbx/p", RegionKind.TEXT
        else:
            where, kind = "body/p", RegionKind.TEXT
        census.register(Region(
            region_id=f"{artifact_id}:p{p_seen}",
            artifact=artifact_id,
            kind=kind,
            location=f"{where}{p_seen}",
            text=txt,
        ))
        p_seen += 1


def _inventory_part(
    census: ContentCensus,
    artifact_id: str,
    xml_bytes: bytes,
    *,
    part_name: str,
    kind: RegionKind,
) -> None:
    """Inventory a secondary part (header/footer/footnotes/comments)."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return
    idx = 0
    for p in root.iter(f"{{{_W}}}p"):
        txt = _text_of(p)
        if not txt:
            continue
        census.register(Region(
            region_id=f"{artifact_id}:{part_name}:p{idx}",
            artifact=artifact_id,
            kind=kind,
            location=f"{part_name}/p{idx}",
            text=txt,
        ))
        idx += 1


def census_docx(path: str | Path, *, artifact_id: str = "") -> ContentCensus:
    """Build a ContentCensus by inventorying a .docx straight from the zip."""
    path = Path(path)
    artifact_id = artifact_id or path.stem
    census = ContentCensus(artifact=path.name)

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

        if "word/document.xml" in names:
            _inventory_body(census, artifact_id, zf.read("word/document.xml"))

        # Headers / footers — separate parts python-docx's body view never sees.
        for n in sorted(names):
            if n.startswith("word/header") and n.endswith(".xml"):
                _inventory_part(census, artifact_id, zf.read(n),
                                part_name=n[len("word/"):-4], kind=RegionKind.HEADER_FOOTER)
            elif n.startswith("word/footer") and n.endswith(".xml"):
                _inventory_part(census, artifact_id, zf.read(n),
                                part_name=n[len("word/"):-4], kind=RegionKind.HEADER_FOOTER)

        # Notes: footnotes / endnotes / comments.
        for part, fname in (("footnotes", "word/footnotes.xml"),
                            ("endnotes", "word/endnotes.xml"),
                            ("comments", "word/comments.xml")):
            if fname in names:
                _inventory_part(census, artifact_id, zf.read(fname),
                                part_name=part, kind=RegionKind.NOTE)

        # Binary regions: embedded media (images) and OLE/embedded objects.
        for n in sorted(names):
            if n.startswith("word/media/"):
                rel = n[len("word/"):]  # e.g. "media/image1.png"
                size = zf.getinfo(n).file_size
                census.register(Region(
                    region_id=f"{artifact_id}:{rel}",
                    artifact=artifact_id,
                    kind=RegionKind.IMAGE,
                    location=rel,
                    note=f"{size} bytes",
                ))
            elif n.startswith("word/embeddings/"):
                rel = n[len("word/"):]
                size = zf.getinfo(n).file_size
                census.register(Region(
                    region_id=f"{artifact_id}:{rel}",
                    artifact=artifact_id,
                    kind=RegionKind.EMBEDDED_OBJECT,
                    location=rel,
                    note=f"{size} bytes",
                ))

    return census
