"""Universal content census — an independent region inventory for EVERY
supported file format, not just .docx.

``census(path)`` dispatches by extension to a format-specific reader that
inventories **every region of every kind** straight from the bytes/zip/XML —
deliberately independent of the production parser. Reconciling that inventory
against the parser's emitted atoms (``ContentCensus.reconcile``) enforces the
coverage invariant: every region must be COVERED (produced an atom) or MARKED
(produced a needs-review marker), never UNCOVERED (silent loss).

Readers, by family:
  * OOXML zip  — docx (delegated), pptx (slides + notes), xlsx (cells)
  * ODF  zip   — odt (paragraphs/cells), ods (cells)
  * PDF        — per-line page text + image XObjects
  * plain text — txt / md / csv (per line)
  * email      — eml (headers + body + attachments), msg (best-effort)
  * mbox       — per message
  * html       — visible text blocks + img/iframe/object
  * rtf        — stripped text lines
  * ics        — per VEVENT
  * transcript — vtt / srt cues
  * zip        — per archive member
  * vsdx / mpp — best-effort shape/whole-file regions

Binary regions (images/charts/drawings/embedded objects/attachments) are
registered with a ``location`` equal to the ``region_ref`` the parser's
markers use (see ``app/parsers/binary_markers.py``), so a marked binary region
reconciles as MARKED.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from app.core.content_census import ContentCensus, Region, RegionKind
from app.parsers.census_docx import census_docx

# OOXML/ODF binary dir -> region kind. Locations match binary_markers refs.
_BIN_KINDS = {
    "media/": RegionKind.IMAGE,
    "embeddings/": RegionKind.EMBEDDED_OBJECT,
    "charts/": RegionKind.CHART,
    "drawings/": RegionKind.SHAPE,
    "Pictures/": RegionKind.IMAGE,
    "ObjectReplacements/": RegionKind.EMBEDDED_OBJECT,
}


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _is_content_line(s: str) -> bool:
    """A line is a *content* region only if it carries at least one
    alphanumeric character. Pure separators / rules (``----``, ``====``,
    ``***``) are not lost content, so they are not counted as regions."""
    return any(ch.isalnum() for ch in s)


def _all_text(el: ET.Element, sep: str = "") -> str:
    """Concatenate every descendant text node (sep="" mimics run-join)."""
    return sep.join(t for t in el.itertext() if t).strip()


# --- binary inventory shared by zip-based formats --------------------------

def _inventory_zip_binaries(
    census: ContentCensus, artifact_id: str, zf: zipfile.ZipFile, *, family: str
) -> None:
    for name in sorted(zf.namelist()):
        if name.endswith("/"):
            continue
        rel = name
        if family == "ooxml" and "/" in name:
            rel = name.split("/", 1)[1]
        kind = None
        for prefix, k in _BIN_KINDS.items():
            if rel.startswith(prefix) or name.startswith(prefix):
                kind = k
                break
        if kind is None:
            continue
        try:
            size = zf.getinfo(name).file_size
        except KeyError:  # pragma: no cover
            size = 0
        census.register(Region(
            region_id=f"{artifact_id}:{rel}",
            artifact=artifact_id,
            kind=kind,
            location=rel,
            note=f"{size} bytes",
        ))


# --- OOXML: pptx -----------------------------------------------------------

_A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _census_pptx(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

        def _slide_num(n: str) -> int:
            m = re.search(r"(\d+)\.xml$", n)
            return int(m.group(1)) if m else 0

        slides = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=_slide_num)
        for n in slides:
            sn = _slide_num(n)
            try:
                root = ET.fromstring(zf.read(n))
            except ET.ParseError:
                continue
            para_idx = 0
            for para in root.iter(f"{{{_A}}}p"):
                txt = "".join(t.text or "" for t in para.iter(f"{{{_A}}}t")).strip()
                if not txt:
                    continue
                census.register(Region(
                    region_id=f"{artifact_id}:slide{sn}.p{para_idx}",
                    artifact=artifact_id, kind=RegionKind.TEXT,
                    location=f"slide{sn}/p{para_idx}", text=txt,
                ))
                para_idx += 1

        notes = sorted((n for n in names if re.match(r"ppt/notesSlides/notesSlide\d+\.xml$", n)), key=_slide_num)
        for n in notes:
            sn = _slide_num(n)
            try:
                root = ET.fromstring(zf.read(n))
            except ET.ParseError:
                continue
            np = 0
            for para in root.iter(f"{{{_A}}}p"):
                txt = "".join(t.text or "" for t in para.iter(f"{{{_A}}}t")).strip()
                if not txt:
                    continue
                census.register(Region(
                    region_id=f"{artifact_id}:notes{sn}.p{np}",
                    artifact=artifact_id, kind=RegionKind.NOTE,
                    location=f"notesSlide{sn}/p{np}", text=txt,
                ))
                np += 1

        _inventory_zip_binaries(census, artifact_id, zf, family="ooxml")
    return census


# --- OOXML: xlsx -----------------------------------------------------------

_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _census_xlsx(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            try:
                sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in sroot.iter(f"{{{_S}}}si"):
                    shared.append("".join(t.text or "" for t in si.iter(f"{{{_S}}}t")))
            except ET.ParseError:
                shared = []

        def _sheet_num(n: str) -> int:
            m = re.search(r"(\d+)\.xml$", n)
            return int(m.group(1)) if m else 0

        sheets = sorted((n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n)), key=_sheet_num)
        for n in sheets:
            snum = _sheet_num(n)
            try:
                root = ET.fromstring(zf.read(n))
            except ET.ParseError:
                continue
            for row in root.iter(f"{{{_S}}}row"):
                r = row.get("r", "")
                cells: list[str] = []
                for c in row.iter(f"{{{_S}}}c"):
                    t = c.get("t")
                    v_el = c.find(f"{{{_S}}}v")
                    if t == "s" and v_el is not None and v_el.text is not None:
                        try:
                            cells.append(shared[int(v_el.text)])
                        except (ValueError, IndexError):
                            pass
                    elif t == "inlineStr":
                        is_el = c.find(f"{{{_S}}}is")
                        if is_el is not None:
                            cells.append("".join(tt.text or "" for tt in is_el.iter(f"{{{_S}}}t")))
                    elif v_el is not None and v_el.text is not None:
                        cells.append(v_el.text)
                cells = [c for c in (s.strip() for s in cells) if c]
                if not cells:
                    continue
                # Inventory per CELL, not per joined row. The xlsx parser
                # consumes row 0 as column headers and re-emits data rows as
                # ``Header: value | Header: value`` — so a per-row joined
                # string never appears verbatim, but every individual cell
                # value (and header label) does. Per-cell is therefore the
                # robust independent denominator that reconciles by substring.
                for ci, cell_txt in enumerate(cells):
                    census.register(Region(
                        region_id=f"{artifact_id}:sheet{snum}.r{r}.c{ci}",
                        artifact=artifact_id, kind=RegionKind.TABLE,
                        location=f"sheet{snum}/r{r}/c{ci}", text=cell_txt,
                    ))
        _inventory_zip_binaries(census, artifact_id, zf, family="ooxml")
    return census


# --- ODF: odt / ods --------------------------------------------------------

def _census_odf(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    with zipfile.ZipFile(path) as zf:
        if "content.xml" in zf.namelist():
            try:
                root = ET.fromstring(zf.read("content.xml"))
            except ET.ParseError:
                root = None
            if root is not None:
                # The ODF parser emits one atom per table CELL and one per
                # text:p / text:h (the paragraph loop iterates ALL paragraphs,
                # including those inside cells), so we inventory at the same
                # granularity: every cell is a TABLE region, every paragraph /
                # heading is a TEXT region. Both reconcile by substring.
                cidx = 0
                for cell in root.iter():
                    if _localname(cell.tag) in ("table-cell", "covered-table-cell"):
                        ct = _all_text(cell, sep=" ")
                        if ct:
                            census.register(Region(
                                region_id=f"{artifact_id}:cell{cidx}",
                                artifact=artifact_id, kind=RegionKind.TABLE,
                                location=f"table/cell{cidx}", text=ct,
                            ))
                            cidx += 1
                pidx = 0
                for el in root.iter():
                    if _localname(el.tag) in ("p", "h"):
                        txt = _all_text(el, sep="")
                        if txt:
                            census.register(Region(
                                region_id=f"{artifact_id}:p{pidx}",
                                artifact=artifact_id, kind=RegionKind.TEXT,
                                location=f"body/p{pidx}", text=txt,
                            ))
                            pidx += 1
        _inventory_zip_binaries(census, artifact_id, zf, family="odf")
    return census


# --- PDF -------------------------------------------------------------------

def _census_pdf(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        import fitz  # PyMuPDF
    except Exception:
        fitz = None
    if fitz is not None:
        try:
            doc = fitz.open(str(path))
        except Exception:
            doc = None
        if doc is not None:
            with doc:
                for pno in range(doc.page_count):
                    page = doc.load_page(pno)
                    text = page.get_text("text") or ""
                    for li, line in enumerate(text.splitlines()):
                        line = line.strip()
                        if line:
                            census.register(Region(
                                region_id=f"{artifact_id}:p{pno}.l{li}",
                                artifact=artifact_id, kind=RegionKind.TEXT,
                                location=f"page{pno}/line{li}", text=line,
                            ))
                    try:
                        images = page.get_images(full=True)
                    except Exception:
                        images = []
                    for ii, img in enumerate(images):
                        xref = img[0] if img else ii
                        census.register(Region(
                            region_id=f"{artifact_id}:page{pno}.img{xref}",
                            artifact=artifact_id, kind=RegionKind.IMAGE,
                            location=f"page{pno}/image{xref}",
                            note="pdf image xobject",
                        ))
            return census
    # Fallback: pdfplumber text only.
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for pno, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                for li, line in enumerate(text.splitlines()):
                    line = line.strip()
                    if line:
                        census.register(Region(
                            region_id=f"{artifact_id}:p{pno}.l{li}",
                            artifact=artifact_id, kind=RegionKind.TEXT,
                            location=f"page{pno}/line{li}", text=line,
                        ))
    except Exception:
        pass
    return census


# --- plain text / markdown / csv -------------------------------------------

def _census_text(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return census
    is_md = path.suffix.lower() in (".md", ".markdown")
    for li, line in enumerate(data.splitlines()):
        s = line.strip()
        if not _is_content_line(s):
            continue
        if is_md:
            # Match the markdown parser's atom granularity: strip the
            # heading / bullet / numbered markers it removes, and skip
            # table-separator rows it never emits, so the census region
            # text is the same content the parser turns into an atom.
            if re.match(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$", s):
                continue  # table separator — not content
            mh = re.match(r"^#{1,6}\s+(.*)$", s)
            if mh:
                s = mh.group(1).strip()
            else:
                mb = re.match(r"^[-*+]\s+(.*)$", s) or re.match(r"^\d+[.)]\s+(.*)$", s)
                if mb:
                    s = mb.group(1).strip()
            if not s:
                continue
        census.register(Region(
            region_id=f"{artifact_id}:l{li}",
            artifact=artifact_id, kind=RegionKind.TEXT,
            location=f"line{li}", text=s,
        ))
    # Markdown image references are binary regions.
    if is_md:
        for mi, m in enumerate(re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", data)):
            census.register(Region(
                region_id=f"{artifact_id}:img{mi}",
                artifact=artifact_id, kind=RegionKind.IMAGE,
                location=f"image/{m.group(1)}", note="markdown image ref",
            ))
    return census


# --- email (.eml) ----------------------------------------------------------

def _census_eml(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        from email import policy
        from email.parser import BytesParser
        msg = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    except Exception:
        return census
    for field in ("from", "to", "cc", "subject", "date"):
        val = msg.get(field)
        if val:
            census.register(Region(
                region_id=f"{artifact_id}:hdr.{field}",
                artifact=artifact_id, kind=RegionKind.TEXT,
                location=f"header/{field}", text=str(val),
            ))
    try:
        body = msg.get_body(preferencelist=("plain", "html"))
        text = body.get_content() if body else ""
    except Exception:
        text = ""
    if text:
        import re as _re
        text = _re.sub(r"<[^>]+>", " ", text)  # crude tag strip for html bodies
        for li, line in enumerate(text.splitlines()):
            s = line.strip()
            if _is_content_line(s):
                census.register(Region(
                    region_id=f"{artifact_id}:body.l{li}",
                    artifact=artifact_id, kind=RegionKind.TEXT,
                    location=f"body/line{li}", text=s,
                ))
    for ai, att in enumerate(msg.iter_attachments()):
        name = att.get_filename() or f"attachment{ai}"
        census.register(Region(
            region_id=f"{artifact_id}:attachment/{name}",
            artifact=artifact_id, kind=RegionKind.EMBEDDED_OBJECT,
            location=f"attachment/{name}", note=att.get_content_type(),
        ))
    return census


def _census_mbox(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        import mailbox
        box = mailbox.mbox(str(path))
    except Exception:
        return census
    for mi, message in enumerate(box):
        for field in ("from", "subject", "date"):
            val = message.get(field)
            if val:
                census.register(Region(
                    region_id=f"{artifact_id}:m{mi}.hdr.{field}",
                    artifact=artifact_id, kind=RegionKind.TEXT,
                    location=f"msg{mi}/header/{field}", text=str(val),
                ))
        try:
            payload = message.get_payload(decode=True)
            text = payload.decode("utf-8", "replace") if payload else (
                message.get_payload() if isinstance(message.get_payload(), str) else ""
            )
        except Exception:
            text = ""
        for li, line in enumerate((text or "").splitlines()):
            s = line.strip()
            if _is_content_line(s):
                census.register(Region(
                    region_id=f"{artifact_id}:m{mi}.body.l{li}",
                    artifact=artifact_id, kind=RegionKind.TEXT,
                    location=f"msg{mi}/body/line{li}", text=s,
                ))
        # Attachments — inventory as embedded objects so a per-message
        # attachment marker can reconcile them as MARKED.
        try:
            for part in message.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                fn = part.get_filename()
                if part.get("Content-Disposition", "").lower().startswith("attachment") or fn:
                    name = fn or "(unnamed)"
                    census.register(Region(
                        region_id=f"{artifact_id}:m{mi}.attachment/{name}",
                        artifact=artifact_id, kind=RegionKind.EMBEDDED_OBJECT,
                        location=f"attachment/{name}", note="mbox attachment",
                    ))
        except Exception:
            pass
    return census


def _census_msg(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        import extract_msg
        m = extract_msg.Message(str(path))
    except Exception:
        # No lib: register the whole file as one region the msg_marker covers.
        census.register(Region(
            region_id=f"{artifact_id}:msg",
            artifact=artifact_id, kind=RegionKind.OTHER,
            location="msg", note="outlook .msg (needs extract_msg)",
        ))
        return census
    for field, val in (("from", m.sender), ("subject", m.subject), ("date", m.date)):
        if val:
            census.register(Region(
                region_id=f"{artifact_id}:hdr.{field}",
                artifact=artifact_id, kind=RegionKind.TEXT,
                location=f"header/{field}", text=str(val),
            ))
    for li, line in enumerate((m.body or "").splitlines()):
        s = line.strip()
        if s:
            census.register(Region(
                region_id=f"{artifact_id}:body.l{li}",
                artifact=artifact_id, kind=RegionKind.TEXT,
                location=f"body/line{li}", text=s,
            ))
    for ai, att in enumerate(getattr(m, "attachments", []) or []):
        name = getattr(att, "longFilename", None) or getattr(att, "shortFilename", None) or f"attachment{ai}"
        census.register(Region(
            region_id=f"{artifact_id}:attachment/{name}",
            artifact=artifact_id, kind=RegionKind.EMBEDDED_OBJECT,
            location=f"attachment/{name}", note="msg attachment",
        ))
    return census


# --- html ------------------------------------------------------------------

def _census_html(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return census
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(data, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        bi = 0
        for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]):
            txt = el.get_text(separator=" ", strip=True)
            if txt:
                census.register(Region(
                    region_id=f"{artifact_id}:b{bi}",
                    artifact=artifact_id, kind=RegionKind.TEXT,
                    location=f"block{bi}", text=txt,
                ))
                bi += 1
        for ii, img in enumerate(soup.find_all(["img", "iframe", "object", "embed"])):
            ref = img.get("src") or img.get("data") or f"media{ii}"
            census.register(Region(
                region_id=f"{artifact_id}:img{ii}",
                artifact=artifact_id, kind=RegionKind.IMAGE,
                location=f"media/{ref}", note=img.name,
            ))
    except Exception:
        # Regex fallback.
        text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", data, flags=re.I)
        text = re.sub(r"<[^>]+>", "\n", text)
        for li, line in enumerate(text.splitlines()):
            s = line.strip()
            if s:
                census.register(Region(
                    region_id=f"{artifact_id}:l{li}",
                    artifact=artifact_id, kind=RegionKind.TEXT,
                    location=f"line{li}", text=s,
                ))
        for ii, m in enumerate(re.finditer(r"<img[^>]+src=[\"']([^\"']+)", data, re.I)):
            census.register(Region(
                region_id=f"{artifact_id}:img{ii}",
                artifact=artifact_id, kind=RegionKind.IMAGE,
                location=f"media/{m.group(1)}", note="img",
            ))
    return census


# --- rtf / ics / transcript ------------------------------------------------

def _census_rtf(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return census
    # Strip RTF groups/control words crudely.
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", raw)
    text = re.sub(r"[{}]", " ", text)
    for li, chunk in enumerate(re.split(r"\n|\.\s", text)):
        s = chunk.strip()
        if len(s) >= 3:
            census.register(Region(
                region_id=f"{artifact_id}:l{li}",
                artifact=artifact_id, kind=RegionKind.TEXT,
                location=f"line{li}", text=s,
            ))
    return census


def _census_ics(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return census
    events = re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", raw, re.S)
    for ei, ev in enumerate(events):
        for field in ("SUMMARY", "LOCATION", "DESCRIPTION", "DTSTART", "ORGANIZER"):
            m = re.search(rf"{field}[^:]*:(.+)", ev)
            if m:
                census.register(Region(
                    region_id=f"{artifact_id}:e{ei}.{field}",
                    artifact=artifact_id, kind=RegionKind.TEXT,
                    location=f"vevent{ei}/{field}", text=m.group(1).strip(),
                ))
    return census


def _census_transcript(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return census
    idx = 0
    for line in raw.splitlines():
        s = line.strip()
        # Skip vtt/srt timing + index-only lines.
        if not s or s == "WEBVTT" or s.isdigit() or "-->" in s:
            continue
        census.register(Region(
            region_id=f"{artifact_id}:c{idx}",
            artifact=artifact_id, kind=RegionKind.TEXT,
            location=f"cue{idx}", text=s,
        ))
        idx += 1
    return census


# --- zip (recurse one level by member) -------------------------------------

def _census_zip(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        zf = zipfile.ZipFile(path)
    except Exception:
        return census
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            # The zip parser emits one atom per entry (its name) — register
            # each member as a region keyed on its name so it reconciles.
            census.register(Region(
                region_id=f"{artifact_id}:member/{info.filename}",
                artifact=artifact_id, kind=RegionKind.OTHER,
                location=f"member/{info.filename}", text=info.filename,
                note=f"{info.file_size} bytes",
            ))
    return census


# --- vsdx / mpp (best-effort) ----------------------------------------------

def _census_vsdx(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    try:
        zf = zipfile.ZipFile(path)
    except Exception:
        census.register(Region(
            region_id=f"{artifact_id}:file", artifact=artifact_id,
            kind=RegionKind.OTHER, location="vsdx", note="legacy .vsd / unreadable",
        ))
        return census
    with zf:
        pages = [n for n in zf.namelist() if re.search(r"visio/pages/page\d+\.xml$", n)]
        for n in sorted(pages):
            try:
                root = ET.fromstring(zf.read(n))
            except ET.ParseError:
                continue
            for ti, el in enumerate(root.iter()):
                if _localname(el.tag) == "Text":
                    txt = _all_text(el, sep="")
                    if txt:
                        census.register(Region(
                            region_id=f"{artifact_id}:{Path(n).stem}.t{ti}",
                            artifact=artifact_id, kind=RegionKind.SHAPE,
                            location=f"{Path(n).stem}/text{ti}", text=txt,
                        ))
        _inventory_zip_binaries(census, artifact_id, zf, family="ooxml")
    return census


def _census_mpp(path: Path, artifact_id: str) -> ContentCensus:
    census = ContentCensus(artifact=path.name)
    # MS Project is an opaque binary; without mpxj we can't read it
    # independently, so register the whole file as one region the mpp marker
    # (or task atoms) account for.
    census.register(Region(
        region_id=f"{artifact_id}:project", artifact=artifact_id,
        kind=RegionKind.EMBEDDED_OBJECT, location="mpp/project",
        note="ms project schedule (needs mpxj)",
    ))
    return census


# --- dispatcher ------------------------------------------------------------

_READERS = {
    ".docx": lambda p, a: census_docx(p, artifact_id=a),
    ".pptx": _census_pptx,
    ".xlsx": _census_xlsx,
    ".odt": _census_odf,
    ".ods": _census_odf,
    ".pdf": _census_pdf,
    ".txt": _census_text,
    ".md": _census_text,
    ".markdown": _census_text,
    ".csv": _census_text,
    ".eml": _census_eml,
    ".mbox": _census_mbox,
    ".msg": _census_msg,
    ".html": _census_html,
    ".htm": _census_html,
    ".xhtml": _census_html,
    ".rtf": _census_rtf,
    ".ics": _census_ics,
    ".ical": _census_ics,
    ".vtt": _census_transcript,
    ".srt": _census_transcript,
    ".zip": _census_zip,
    ".vsdx": _census_vsdx,
    ".vsd": _census_vsdx,
    ".mpp": _census_mpp,
}


def census(path: str | Path, *, artifact_id: str = "") -> ContentCensus:
    """Build a ContentCensus for any supported file, by extension.

    Unknown extensions fall back to the plain-text line reader (every text
    line is a region), so even an unrecognized format gets an honest
    independent denominator rather than silently having none.
    """
    path = Path(path)
    artifact_id = artifact_id or path.stem
    reader = _READERS.get(path.suffix.lower(), _census_text)
    try:
        return reader(path, artifact_id)
    except Exception:
        # A reader must never crash the pipeline; return an empty census and
        # let the caller treat "no denominator" as a soft signal.
        return ContentCensus(artifact=path.name)


def reconciled_census(
    paths,
    atoms,
    *,
    artifact: str = "project",
) -> ContentCensus:
    """Inventory every region across ``paths`` and reconcile it against ``atoms``.

    This is the deal-level independent denominator that
    :func:`app.core.complaint_router.route` consumes for its NEEDS_EXTRACTOR
    bucket: a region left ``UNCOVERED`` after reconciliation is content that
    exists in the source files but no emitted atom represents — never-detected
    loss that only extractor code can recover.

    Each file is read by its format-specific reader (independent of the
    production parser), and every region is folded into one combined census.
    Region ids are already namespaced by ``artifact_id`` (the file stem) inside
    :func:`census`, so cross-file collisions are not a concern. Reconciliation
    is a single substring pass over the full atom set, so the census stays an
    honest, parser-independent check. Never raises — a bad reader is skipped.

    Args:
        paths: iterable of source file paths (e.g. the compiler's
            ``_iter_artifacts`` output) — the same files the deal compiled from.
        atoms: the emitted atoms to reconcile against (``result.atoms``).
        artifact: a label for the combined census (cosmetic).

    Returns:
        A reconciled :class:`ContentCensus`; ``.uncovered()`` lists the
        never-detected regions.
    """
    combined = ContentCensus(artifact=artifact)
    for p in paths:
        try:
            c = census(p, artifact_id=Path(p).stem)
        except Exception:
            continue
        for region in c.regions.values():
            combined.register(region)
    try:
        combined.reconcile(list(atoms or []))
    except Exception:
        pass
    return combined


__all__ = ["census", "census_docx", "reconciled_census"]
