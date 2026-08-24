"""Image regions, OCR fallback, and the scan for content the text layer missed.

Marks every region nobody has read yet, and reads the page as an image when the text layer is absent or lying. A marker is coverage bookkeeping about the artifact, never a question for a human.
"""

from __future__ import annotations

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import ArtifactType
from app.core.schemas import AtomType
from app.core.schemas import AuthorityClass
from app.core.schemas import EvidenceAtom
from app.core.schemas import ReviewStatus
from app.core.schemas import SourceRef
from app.parsers.binary_markers import region_marker
from app.parsers.pdf._shared import _NEW_TABLE_HEADER_RE
from app.parsers.pdf._shared import _is_image_field_label
from app.parsers.pdf._shared import _is_photo_request
from app.parsers.pdf.forms import _literal_x_checkbox_atoms_from_line
from app.parsers.pdf.forms import _pdf_header_field_atoms_from_text
from app.parsers.pdf.tables import _vertical_table_atoms_from_text
from pathlib import Path
import re


def _pdf_image_markers(
    *,
    path: Path,
    project_id: str,
    artifact_id: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Emit one located marker per embedded image XObject in the PDF.

    Detection is total and cheap (reads only the image table, never decodes
    pixels). The ``region_ref`` (``page{n}/image{xref}``) matches the content
    census so each image region reconciles as MARKED. An OCR/vision atom that
    later covers the same region is preferred; this is the floor that
    guarantees no image silently vanishes.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover — env-specific
        return []
    out: list[EvidenceAtom] = []
    try:
        doc = fitz.open(str(path))
    except Exception:  # pragma: no cover — unreadable PDF
        return []
    # Crop each embedded image out to a sidecar file so a later OCR / vision
    # pass can read it (the marker points AT the saved file instead of "0
    # bytes"). Save dir is env-overridable; defaults to a project-local folder.
    import os as _os
    img_root = Path(_os.environ.get("SOWSMITH_IMAGE_DIR", "_extracted_images")) / _safe_stem(path.stem)
    saved_by_xref: dict[int, tuple[str, int]] = {}  # xref -> (saved_path, size); same image reused across pages
    emitted_xrefs: set[int] = set()  # one marker atom per UNIQUE image, not per page
    # The most recent "Upload N photos showing X" instruction — a field-report's
    # photos answer the request that precedes them, often spanning pages ("Upload
    # 4 Photos" -> 2 on this page, 2 on the next). Carry it forward so each photo
    # is captioned with what it should show.
    current_request: str | None = None
    try:
        for page_index in range(doc.page_count):
            try:
                page = doc.load_page(page_index)
                images = page.get_images(full=True)
            except Exception:
                continue
            # Collect photo-request BLOCKS with their vertical position. Reading
            # blocks (not raw lines) joins a request wrapped across lines
            # ("Upload photo showing Battery \nCharger Mounting" -> one request)
            # and skips footer page numbers a line-split would leak.
            page_requests: list[tuple[float, str]] = []   # (y0, request text) — carry-forward
            page_captions: list[tuple[float, str]] = []   # requests + field labels — per-image caption
            try:
                for b in page.get_text("blocks"):
                    joined = re.sub(r"\s+", " ", (b[4] or "")).strip()
                    if not joined:
                        continue
                    if _is_photo_request(joined):
                        page_requests.append((float(b[1]), joined[:160]))
                        page_captions.append((float(b[1]), joined[:160]))
                    elif _is_image_field_label(joined):
                        # a labelled field whose value is the image below it
                        # ('Signature' -> the signature image) — caption the image
                        # with the field, not the far-off photo request above.
                        page_captions.append((float(b[1]), joined[:160]))
            except Exception:
                pass
            page_requests.sort(key=lambda r: r[0])
            page_captions.sort(key=lambda r: r[0])
            # Map each image xref to its top-edge Y so we can pair it to the
            # request directly above it. A field report stacks request-then-photo
            # down the page; two requests + two photos must NOT all collapse onto
            # the last request (the old line-scan kept only the most recent one).
            img_y: dict[int, float] = {}
            try:
                for info in page.get_image_info(xrefs=True):
                    xr = info.get("xref")
                    bb = info.get("bbox")
                    if xr and bb:
                        img_y[xr] = float(bb[1])
            except Exception:
                pass
            for ii, img in enumerate(images):
                xref = img[0] if img else ii
                # A logo/letterhead embedded once but referenced on every page is
                # ONE image — emit a single marker for it (on first sight), not a
                # duplicate "needs_extractor" atom per page (was flooding scan-heavy
                # PDFs with 10+ identical logo markers).
                if xref in emitted_xrefs:
                    continue
                emitted_xrefs.add(xref)
                saved_path: str | None = None
                size = 0
                if xref in saved_by_xref:
                    saved_path, size = saved_by_xref[xref]
                else:
                    try:
                        info = doc.extract_image(xref) or {}
                        data = info.get("image") or b""
                        size = len(data)
                        if data:
                            img_root.mkdir(parents=True, exist_ok=True)
                            ext = (info.get("ext") or "png").lstrip(".")
                            fn = img_root / f"page{page_index}_image{xref}.{ext}"
                            with open(fn, "wb") as fh:
                                fh.write(data)
                            saved_path = str(fn).replace("\\", "/")
                            saved_by_xref[xref] = (saved_path, size)
                    except Exception:
                        saved_path, size = None, 0  # degrade to plain marker
                # Pair this image to the request directly above it: the last
                # request whose block top is at/above the image top. No request
                # above (photo continues a request from a prior page) -> use the
                # carried request. No position info -> fall back to order/last.
                caption = current_request
                y = img_y.get(xref)
                if page_captions and y is not None:
                    # A request/label introduces the image(s) BELOW it, so an image
                    # is owned by the nearest caption (photo request OR field label
                    # like 'Signature') AT OR ABOVE its top edge — that's the thing
                    # it illustrates. If nothing is above it on this page, the image
                    # precedes every caption here — it CONTINUES the carried request
                    # from a prior page; never grab a caption below it (that one
                    # introduces later images: a cable-test photo at the page top
                    # vs a 'POS 3' request lower down).
                    above = [r for r in page_captions if r[0] <= y + 2.0]
                    if above:
                        caption = above[-1][1]
                elif page_requests:
                    caption = page_requests[min(ii, len(page_requests) - 1)][1]
                out.append(region_marker(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    artifact_type=ArtifactType.pdf,
                    parser_version=parser_version,
                    region_ref=f"page{page_index}/image{xref}",
                    kind="image_marker",
                    label="image",
                    size=size,
                    saved_path=saved_path,
                    caption=caption,
                ))
            # Carry the LAST request on this page forward: a multi-page request
            # ("Upload 4 Photos" -> 2 here, 2 on the next page) captions the
            # following page's photos when that page has no request line of its own.
            if page_requests:
                current_request = page_requests[-1][1]
    finally:
        doc.close()
    return out

def _safe_stem(stem: str) -> str:
    """Filesystem-safe folder name from an artifact stem."""
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "artifact"

def _ocr_fallback_atoms(
    *,
    path: Path,
    project_id: str,
    artifact_id: str,
    parser_version: str,
    already_emitted: list[EvidenceAtom],
) -> list[EvidenceAtom]:
    """For each page that produced ZERO atoms via the structured
    pipeline AND is text-poor (likely a scanned image), run OCR and
    emit one scope_item atom per recovered text block.

    Returns [] when Tesseract isn't installed or every page already
    contributed atoms.
    """
    try:
        from orbitbrief_page_os.segmentation.schematic.ocr import is_available
        from orbitbrief_page_os.segmentation.schematic.raster import (
            is_text_poor_page,
            render_page_to_ndarray,
        )
    except Exception:
        return []
    if not is_available():
        return []
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        from orbitbrief_page_os.segmentation.schematic.ocr import ocr_words
    except Exception:
        return []

    # Pages that already have atoms
    pages_with_atoms: set[int] = set()
    for a in already_emitted:
        if not a.source_refs:
            continue
        loc = a.source_refs[0].locator if a.source_refs[0] else {}
        if isinstance(loc, dict) and loc.get("page") is not None:
            try:
                pages_with_atoms.add(int(loc["page"]))
            except (TypeError, ValueError):
                continue

    out: list[EvidenceAtom] = []
    try:
        doc = fitz.open(str(path))
    except Exception:
        return []
    try:
        for page_index in range(doc.page_count):
            if page_index in pages_with_atoms:
                continue
            try:
                page = doc.load_page(page_index)
            except Exception:
                continue
            try:
                if not is_text_poor_page(page):
                    continue
            except Exception:
                continue
            try:
                arr = render_page_to_ndarray(page, dpi=200)
            except Exception:
                arr = None
            if arr is None:
                continue
            try:
                words = ocr_words(arr)
            except Exception:
                words = []
            if not words:
                continue
            # Group OCR words into lines by y-coordinate buckets.
            lines: dict[int, list] = {}
            for w in words:
                y_bucket = round(w.bbox[1] / 12.0) * 12
                lines.setdefault(y_bucket, []).append(w)
            recovered_text_blocks: list[str] = []
            for y in sorted(lines):
                sorted_words = sorted(lines[y], key=lambda w: w.bbox[0])
                line_text = " ".join(w.text for w in sorted_words).strip()
                if len(line_text) >= 6:
                    recovered_text_blocks.append(line_text)
            if not recovered_text_blocks:
                continue
            page_text = " ".join(recovered_text_blocks)
            atom_id = stable_id(
                "atm", project_id, artifact_id, "ocr_fallback",
                page_index, page_text
            )
            src = SourceRef(
                id=stable_id("src", atom_id),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.pdf,
                filename=path.name,
                locator={
                    "page": page_index,
                    "block_kind": "ocr_fallback",
                    "extraction": "pdf_ocr_fallback_v1",
                    "word_count": len(words),
                },
                extraction_method="pdf_ocr_fallback_v1",
                parser_version=parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.scope_item,
                    raw_text=page_text[:2000],
                    normalized_text=page_text[:2000].lower(),
                    value={
                        "kind": "ocr_recovered",
                        "page": page_index,
                        "word_count": len(words),
                        "lines": len(recovered_text_blocks),
                    },
                    entity_keys=[],
                    source_refs=[src],
                    receipts=[],
                    authority_class=AuthorityClass.contractual_scope,
                    confidence=0.60,
                    confidence_raw=0.60,
                    calibrated_confidence=0.60,
                    review_status=ReviewStatus.needs_review,
                    review_flags=["ocr_recovered"],
                    parser_version=parser_version,
                )
            )
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out

_CHECKBOX_RE = re.compile(
    r"(?P<mark>☒|☑|✓|✔|\[x\]|\[X\]|\(x\)|\(X\)|☐|□|\[\s\]|\(\s\))"
    r"\s*(?P<label>[^|;\n]+)"
)

_WORKFLOW_STEP_RE = re.compile(
    r"\b(detect|triage|contain|escalate|recover|remediate|notify|"
    r"dispatch|close|improve)\b",
    re.I,
)
_LOW_TEXT_VISUAL_THRESHOLD = 80

# 5B — Form grid (multi-line / multi-column "x Foo" tables).
_FORM_GROUP_HEADINGS: dict[str, dict[str, frozenset[str]]] = {
    "monitoring tool intake": {
        "known_options": frozenset(
            {
                "LogicMonitor",
                "Microsoft Sentinel",
                "ServiceNow Event Mgmt",
                "Aruba Central",
                "Meraki Dashboard",
                "Genetec Security Center",
                "PRTG",
                "Datadog",
            }
        )
    },
}

def _split_form_grid_line(line: str) -> list[tuple[str, bool]]:
    cells = [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]
    out: list[tuple[str, bool]] = []
    for cell in cells:
        selected = bool(re.match(r"^[xX]\s+", cell))
        label = re.sub(r"^[xX]\s+", "", cell).strip()
        if label:
            out.append((label, selected))
    return out

def _form_grid_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """5B — when a line names a known form-group heading (e.g.
    "Monitoring Tool Intake"), scan the next ~12 lines for option
    labels. Emit one ``form_option_state`` atom per known option,
    with ``selected=True`` if the cell starts with literal ``x ``,
    else ``selected=False``."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[EvidenceAtom] = []
    for i, line in enumerate(lines):
        group_name = normalize_text(line)
        group_config = _FORM_GROUP_HEADINGS.get(group_name)
        if group_config is None:
            continue
        known_options = group_config["known_options"]
        option_index = 0
        for j in range(i + 1, min(i + 12, len(lines))):
            candidate = lines[j].strip()
            if not candidate:
                break
            for label, selected in _split_form_grid_line(candidate):
                if label not in known_options:
                    continue
                source_ref = SourceRef(
                    id=stable_id(
                        "src", artifact_id, "pdf", page_number, "form_grid",
                        group_name, option_index,
                    ),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.pdf,
                    filename=filename,
                    locator={
                        "page": page_number,
                        "group": group_name,
                        "line_index": j,
                        "option_index": option_index,
                    },
                    extraction_method="pdf_form_grid_v1",
                    parser_version=parser_version,
                )
                out.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm", project_id, artifact_id, "form_grid",
                            page_number, group_name, option_index,
                            selected, label,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.form_option_state,
                        raw_text=(
                            f"{'Selected' if selected else 'Not selected'} "
                            f"{group_name}: {label}"
                        ),
                        normalized_text=normalize_text(label),
                        value={
                            "kind": "form_option_state",
                            "group": group_name,
                            "label": label,
                            "selected": selected,
                            "page": page_number,
                        },
                        entity_keys=[],
                        source_refs=[source_ref],
                        receipts=[],
                        authority_class=AuthorityClass.customer_current_authored,
                        confidence=0.90 if selected else 0.70,
                        review_status=ReviewStatus.auto_accepted,
                        review_flags=[]
                        if selected
                        else ["form_option_unselected", "do_not_certify_as_exclusion"],
                        parser_version=parser_version,
                    )
                )
                option_index += 1
        break
    return out

# 5D — field checklist row.
_FIELD_CHECKLIST_ROW_RE = re.compile(
    r"^\s*(?P<num>\d{1,2})\s{2,}"
    r"(?P<item>.+?)\s{2,}"
    r"(?P<status>OPEN|N/A|NA|PASS|FAIL|BLOCKED|CLOSED|PENDING)\s{2,}"
    r"(?P<area>[A-Za-z0-9 /_-]{2,60})\s{2,}"
    r"(?P<note>.+?)\s*$",
    re.I,
)

def _field_checklist_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """5D — emit one atom per field-checklist row when a page
    contains the literal phrase 'field checklist'."""
    if "field checklist" not in normalize_text(text):
        return []
    out: list[EvidenceAtom] = []
    for line_idx, line in enumerate(text.splitlines()):
        m = _FIELD_CHECKLIST_ROW_RE.match(line)
        if not m:
            continue
        item_no = m.group("num")
        item = m.group("item").strip()
        status = m.group("status").strip()
        area = m.group("area").strip()
        note = m.group("note").strip()
        atom_type = (
            AtomType.constraint
            if status.upper() in {"OPEN", "BLOCKED", "PENDING"}
            else AtomType.scope_item
        )
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, "pdf", page_number, "field_check", item_no),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={
                "page": page_number,
                "line_index": line_idx,
                "field_check_item": item_no,
            },
            extraction_method="pdf_field_checklist_row_v1",
            parser_version=parser_version,
        )
        out.append(
            EvidenceAtom(
                id=stable_id(
                    "atm", project_id, artifact_id, "field_checklist",
                    page_number, item_no, item, status,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=atom_type,
                raw_text=f"Field checklist {item_no}: {item} | {status} | {area} | {note}",
                normalized_text=normalize_text(item),
                value={
                    "kind": "field_checklist_row",
                    "item_no": item_no,
                    "item": item,
                    "status": status,
                    "area": area,
                    "note": note,
                    "page": page_number,
                },
                entity_keys=[],
                source_refs=[source_ref],
                receipts=[],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.86,
                review_status=ReviewStatus.auto_accepted,
                review_flags=[],
                parser_version=parser_version,
            )
        )
    return out

# 5E — horizontal workflow (Detect | Triage | Contain | Escalate | Recover | Improve).
_WORKFLOW_ORDER = ["Detect", "Triage", "Contain", "Escalate", "Recover", "Improve"]

def _horizontal_workflow_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: list[EvidenceAtom] = []
    heading_idx = None
    for i, line in enumerate(lines):
        if all(step.lower() in line.lower() for step in _WORKFLOW_ORDER):
            heading_idx = i
            break
    if heading_idx is None:
        return out
    # PR5 — descriptions can be:
    #   (a) one line per step (already array-aligned), OR
    #   (b) ONE line with all step descriptions separated by ≥2 spaces.
    # Try (b) first when the very next line splits into N pieces.
    raw_descs: list[str] = []
    if heading_idx + 1 < len(lines):
        candidate = lines[heading_idx + 1]
        cells = [c.strip() for c in re.split(r"\s{2,}", candidate.strip()) if c.strip()]
        if len(cells) == len(_WORKFLOW_ORDER):
            raw_descs = cells
    if not raw_descs:
        raw_descs = lines[heading_idx + 1 : heading_idx + 1 + len(_WORKFLOW_ORDER)]
    for idx, step in enumerate(_WORKFLOW_ORDER):
        desc = raw_descs[idx] if idx < len(raw_descs) else ""
        source_ref = SourceRef(
            id=stable_id(
                "src", artifact_id, "pdf", page_number, "workflow_horizontal", idx,
            ),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={
                "page": page_number,
                "workflow_step_index": idx,
                "layout": "horizontal",
            },
            extraction_method="pdf_horizontal_workflow_v1",
            parser_version=parser_version,
        )
        out.append(
            EvidenceAtom(
                id=stable_id(
                    "atm", project_id, artifact_id, "workflow_horizontal",
                    page_number, idx, step, desc,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.action_item,
                raw_text=f"Workflow step {idx + 1} {step}: {desc}".strip(),
                normalized_text=normalize_text(f"{step} {desc}"),
                value={
                    "kind": "workflow_step",
                    "step_index": idx,
                    "step_name": step,
                    "description": desc,
                    "page": page_number,
                    "layout": "horizontal",
                },
                entity_keys=[],
                source_refs=[source_ref],
                receipts=[],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.82,
                review_status=ReviewStatus.needs_review,
                review_flags=["layout_derived_workflow"],
                parser_version=parser_version,
            )
        )
    return out

def _checkbox_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Extract checked / unchecked checkbox state from page text.

    Checked boxes (☒, ☑, ✓, ✔, [x], (X)) emit a ``scope_item`` atom
    with ``value.checked=true``. Unchecked boxes (☐, □, [ ], ( ))
    emit an ``exclusion`` atom with ``value.checked=false`` and the
    review flag ``unchecked_checkbox_not_scope`` so the calibrator
    flags it for human review — unchecked is *evidence of exclusion*,
    not silent absence.
    """
    atoms: list[EvidenceAtom] = []
    for idx, m in enumerate(_CHECKBOX_RE.finditer(text)):
        mark = m.group("mark")
        label = m.group("label").strip(" :-\t")
        if not label:
            continue
        checked = mark in {"☒", "☑", "✓", "✔", "[x]", "[X]", "(x)", "(X)"}
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, "pdf", page_number, "checkbox", idx),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={"page": page_number, "checkbox_index": idx},
            extraction_method="pdf_checkbox_state_v1",
            parser_version=parser_version,
        )
        # Revised checkbox semantics (post-PR7 review). Checked
        # boxes are evidence the option WAS selected → scope_item.
        # Unchecked boxes are AMBIGUOUS — they can mean "not selected",
        # "not applicable", "blank option", or "not answered". So
        # unchecked emits ``form_option_state`` (a neutral marker) and
        # is left for the packetizer to combine with explicit
        # exclusion language elsewhere if appropriate. Never auto-
        # certify an unchecked box as a contractual exclusion.
        atom_type = AtomType.scope_item if checked else AtomType.form_option_state
        atoms.append(
            EvidenceAtom(
                id=stable_id(
                    "atm",
                    project_id,
                    artifact_id,
                    "checkbox",
                    page_number,
                    idx,
                    checked,
                    label,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=atom_type,
                raw_text=f"{'Selected' if checked else 'Not selected'} checkbox: {label}",
                normalized_text=normalize_text(label),
                value={
                    "kind": "checkbox",
                    "label": label,
                    "checked": checked,
                    "page": page_number,
                },
                entity_keys=[],
                source_refs=[source_ref],
                receipts=[],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.90 if checked else 0.55,
                review_status=ReviewStatus.auto_accepted
                if checked
                else ReviewStatus.needs_review,
                review_flags=[]
                if checked
                else [
                    "unchecked_checkbox_ambiguous",
                    "do_not_certify_as_exclusion",
                ],
                parser_version=parser_version,
            )
        )
    return atoms

def _workflow_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Emit one ``action_item`` atom per workflow step on a page that
    contains 3+ workflow verbs (detect / triage / contain / escalate /
    recover / remediate / notify / dispatch / close / improve).

    Page text is split on common arrow / pipe glyphs (→ -> › > / |)
    so ``Detect → Triage → Contain → Recover`` becomes 4 atoms."""
    if len(_WORKFLOW_STEP_RE.findall(text)) < 3:
        return []
    chunks = re.split(r"\s*(?:→|->|›|>|/|\|)\s*", text)
    atoms: list[EvidenceAtom] = []
    for idx, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk or not _WORKFLOW_STEP_RE.search(chunk):
            continue
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, "pdf", page_number, "workflow", idx),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.pdf,
            filename=filename,
            locator={"page": page_number, "workflow_step_index": idx},
            extraction_method="pdf_workflow_step_v1",
            parser_version=parser_version,
        )
        atoms.append(
            EvidenceAtom(
                id=stable_id(
                    "atm",
                    project_id,
                    artifact_id,
                    "workflow",
                    page_number,
                    idx,
                    chunk,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.action_item,
                raw_text=chunk,
                normalized_text=normalize_text(chunk),
                value={
                    "kind": "workflow_step",
                    "step_index": idx,
                    "page": page_number,
                },
                entity_keys=[],
                source_refs=[source_ref],
                receipts=[],
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.86,
                review_status=ReviewStatus.auto_accepted,
                review_flags=[],
                parser_version=parser_version,
            )
        )
    return atoms

def _visual_review_atom(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    parser_version: str,
    reason: str,
) -> EvidenceAtom:
    """Mark a low-text page as carrying visual evidence the structured
    pipeline could not extract (rack diagrams, floor plans, OCR-only
    pages). Surfaces as ``open_question`` with
    ``review_flags=['visual_evidence_not_fully_extracted']`` so the
    review UI surfaces the page instead of letting it disappear."""
    source_ref = SourceRef(
        id=stable_id("src", artifact_id, "pdf", page_number, "visual_review"),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.pdf,
        filename=filename,
        locator={"page": page_number},
        extraction_method="pdf_visual_page_marker_v1",
        parser_version=parser_version,
    )
    return EvidenceAtom(
        id=stable_id(
            "atm", project_id, artifact_id, "visual_review", page_number, reason
        ),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.open_question,
        raw_text=(
            f"PDF page {page_number} appears to contain visual / table / "
            "diagram evidence that was not fully extracted."
        ),
        normalized_text="visual evidence requires review",
        value={
            "kind": "visual_page_marker",
            "page": page_number,
            "reason": reason,
        },
        entity_keys=[],
        source_refs=[source_ref],
        receipts=[],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.60,
        review_status=ReviewStatus.needs_review,
        review_flags=["visual_evidence_not_fully_extracted"],
        parser_version=parser_version,
    )

# Workflow-specific stop tokens — applied only by the vertical-workflow
# extractor when collecting the description for the FINAL step
# ("Improve"). These are bare single nouns that legitimately appear as
# data cells inside other tables, so we never use them in
# _NEW_TABLE_HEADER_RE.
_WORKFLOW_STOP_RE = re.compile(
    r"^(runbook|trigger|owner|status|evidence|"
    r"cyber\s*/\s*logging\s+notes|notes?)\s*$",
    re.I,
)

# =====================================================================
# Boss-review F6 — vertical workflow steps.
# =====================================================================
def _vertical_workflow_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Emit one ``action_item`` atom per workflow step when steps are
    listed vertically (each step name on its own line followed by a
    short description that may span 1-2 lines).

    Trigger phrase: ``Incident and Vulnerability Response Workflow`` or
    a sequence where ``Detect`` and ``Triage`` appear on consecutive
    non-empty lines (a strong vertical signal).
    """
    lines = [ln.strip() for ln in text.splitlines()]
    out: list[EvidenceAtom] = []
    n = len(lines)
    # Locate the first occurrence of "Detect" on its own line where
    # "Triage" appears within the next 3 non-empty lines.
    for i in range(n):
        if lines[i].lower() != "detect":
            continue
        # Confirm Triage appears within the next ~6 non-empty lines.
        seen: list[int] = []
        j = i + 1
        while j < n and len(seen) < 6:
            if lines[j]:
                seen.append(j)
            j += 1
        if not any(lines[k].lower() == "triage" for k in seen):
            continue
        # Collect step boundaries by scanning forward.
        steps_lower = ["detect", "triage", "contain", "escalate", "recover", "improve"]
        anchor_indices: dict[str, int] = {}
        cursor = i
        for step in steps_lower:
            while cursor < n and lines[cursor].lower() != step:
                cursor += 1
            if cursor >= n:
                break
            anchor_indices[step] = cursor
            cursor += 1
        if len(anchor_indices) < 4:
            return out
        # For each step, the description is everything between its
        # anchor and the next anchor (or up to 4 lines).
        step_keys = [s for s in steps_lower if s in anchor_indices]
        anchors_ordered = [anchor_indices[s] for s in step_keys]
        anchors_ordered.append(min(n, anchors_ordered[-1] + 6))
        for idx, step in enumerate(step_keys):
            start = anchors_ordered[idx] + 1
            end = anchors_ordered[idx + 1]
            desc_lines: list[str] = []
            for k in range(start, end):
                ln = lines[k]
                if not ln:
                    continue
                # Boss-review v8 F5 — stop description collection when
                # the next table header begins (Runbook | Trigger |
                # Owner | Status | Evidence on noc_soc page 2 was
                # bleeding into "Improve").
                if _NEW_TABLE_HEADER_RE.match(ln) or _WORKFLOW_STOP_RE.match(ln):
                    break
                desc_lines.append(ln)
            desc = " ".join(desc_lines).strip()
            step_name = step.title()
            source_ref = SourceRef(
                id=stable_id(
                    "src", artifact_id, "pdf", page_number, "workflow_vertical", idx,
                ),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.pdf,
                filename=filename,
                locator={
                    "page": page_number,
                    "workflow_step_index": idx,
                    "layout": "vertical",
                },
                extraction_method="pdf_vertical_workflow_v1",
                parser_version=parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm", project_id, artifact_id, "workflow_vertical",
                        page_number, idx, step_name, desc,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.action_item,
                    raw_text=f"Workflow step {idx + 1} {step_name}: {desc}".strip(),
                    normalized_text=normalize_text(f"{step_name} {desc}"),
                    value={
                        "kind": "workflow_step",
                        "step_index": idx,
                        "step_name": step_name,
                        "description": desc,
                        "page": page_number,
                        "layout": "vertical",
                    },
                    entity_keys=[],
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=0.84,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=parser_version,
                )
            )
        return out
    return out

# =====================================================================
# Boss-review F5 — group-aware unchecked form-option detection.
# =====================================================================
_FORM_OPTION_GROUP_HEADERS: tuple[str, ...] = (
    "connection availability / field checks",
    "connection availability",
    "field checks",
    "site survey - access checklist",
    "site survey access checklist",
)
_FORM_OPTION_END_MARKERS: tuple[str, ...] = (
    # Boss-review v9 C001-F2/C002-F2 — substring matchers that ALWAYS
    # indicate a real section break. We removed bare single nouns
    # like "port" / "table" because they appeared inside legitimate
    # option labels ("Network port available", "Patch panel
    # accessible") and were stopping the parser at row 5 of 8.
    "margin note",
    "synthetic planning",
    "field checklist - pathway",
    "rack elevation",
    "open rfis",
    "open rfi",
    "working measurements",
    "as-built exception",
    "required signatures",
    "page 1",
    "page 2",
    "incident workflow",
)

def _group_form_option_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Emit ``form_option_state`` atoms for a known checkbox group.

    Boss-review F5: the parser already emits checked options from
    lines starting with ``x`` (via _SINGLE_LINE_X_RE), but unchecked
    options have no leading sentinel. We anchor on a known group
    header (e.g., 'Connection Availability / Field Checks') and treat
    the next contiguous run of single-line items as form options,
    selecting=true if the line starts with ``x``.
    """
    out: list[EvidenceAtom] = []
    lines = [ln.rstrip() for ln in text.splitlines()]
    n = len(lines)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        norm = normalize_text(line).strip()
        if norm not in _FORM_OPTION_GROUP_HEADERS:
            continue
        # Collect up to 12 following non-empty lines as candidate options.
        opts: list[tuple[int, str]] = []
        j = i + 1
        while j < n and len(opts) < 12:
            ln = lines[j].strip()
            if not ln:
                j += 1
                continue
            normln = normalize_text(ln)
            if any(end in normln for end in _FORM_OPTION_END_MARKERS):
                break
            # Skip pure section labels.
            if ln.endswith(":") or len(ln.split()) > 12:
                break
            opts.append((j, ln))
            j += 1
        if not opts:
            continue
        for idx, (line_idx, raw) in enumerate(opts):
            selected = bool(re.match(r"^\s*x\s+\S", raw, re.I))
            label = re.sub(r"^\s*x\s+", "", raw, flags=re.I).strip()
            if not label:
                continue
            source_ref = SourceRef(
                id=stable_id(
                    "src", artifact_id, "pdf", page_number, "form_option_group", idx,
                ),
                artifact_id=artifact_id,
                artifact_type=ArtifactType.pdf,
                filename=filename,
                locator={
                    "page": page_number,
                    "line_index": line_idx,
                    "form_group": norm,
                    "option_index": idx,
                },
                extraction_method="pdf_group_form_option_v1",
                parser_version=parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm", project_id, artifact_id, "form_option_grouped",
                        page_number, idx, label, "selected" if selected else "unselected",
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.form_option_state,
                    raw_text=("Selected option: " if selected else "Unselected option: ") + label,
                    normalized_text=normalize_text(label),
                    value={
                        "kind": "form_option_state",
                        "group": norm,
                        "label": label,
                        "selected": selected,
                        "page": page_number,
                    },
                    entity_keys=[],
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=0.84,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=parser_version,
                )
            )
    return out

def _scan_pdf_for_extras(
    *,
    project_id: str,
    artifact_id: str,
    path: Path,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Single fitz pass — emit checkbox / workflow / visual atoms.

    Errors are swallowed by the caller so a malformed PDF can't kill
    the structured pipeline; this whole pass is best-effort enrichment.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return []

    out: list[EvidenceAtom] = []
    with fitz.open(str(path)) as doc:
        for page_idx in range(len(doc)):
            try:
                page_text = doc[page_idx].get_text("text") or ""
            except Exception:
                page_text = ""
            stripped = page_text.strip()
            if len(stripped) < _LOW_TEXT_VISUAL_THRESHOLD:
                # Only flag "visual evidence" when the page ACTUALLY has images
                # or vector drawings. A near-empty page (a trailing line, a
                # short final page) is sparse text, not a scanned diagram —
                # don't emit a bogus visual-review marker for it.
                pg = doc[page_idx]
                has_raster = bool(pg.get_images(full=True))
                has_vector = bool(pg.get_drawings())
                # Raster images on the page already become captioned image
                # markers (saved_path + "Upload N photos…" caption) AND drive the
                # vision pass via find_visual_pages_from_image_markers. Emitting a
                # second "visual evidence not fully extracted" atom for the same
                # page is redundant noise the reviewer sees stacked on the photos.
                # Keep the marker ONLY for a vector-drawing page with no raster
                # (a vector floor-plan / rack diagram that NO image marker covers).
                if has_vector and not has_raster:
                    out.append(
                        _visual_review_atom(
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            page_number=page_idx + 1,
                            parser_version=parser_version,
                            reason=f"low_text_page_{len(stripped)}_chars",
                        )
                    )
                continue
            out.extend(
                _checkbox_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            # RF2 — literal "x Foo x Bar" line scan for PDFs whose
            # text extraction lost the unicode checkbox glyphs.
            for line_idx, line in enumerate(page_text.splitlines()):
                out.extend(
                    _literal_x_checkbox_atoms_from_line(
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        page_number=page_idx + 1,
                        line=line,
                        line_index=line_idx,
                        parser_version=parser_version,
                    )
                )
            # PR5 (post-v3) — header KV / form grid / field checklist /
            # horizontal workflow.
            out.extend(
                _pdf_header_field_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            out.extend(
                _form_grid_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            out.extend(
                _field_checklist_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            # Boss-review F3+F4 — vertical-listed table v2 (each cell
            # on its own line, common with hand-form PDFs).
            out.extend(
                _vertical_table_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            # Boss-review F5 — group-aware form options (selected=true
            # AND selected=false) under known group headers.
            out.extend(
                _group_form_option_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            )
            # Prefer the horizontal six-step workflow if the page has
            # one; otherwise try the vertical workflow (each step name
            # on its own line); fall back to the original verb-density
            # workflow extractor.
            horizontal = _horizontal_workflow_atoms_from_text(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                page_number=page_idx + 1,
                text=page_text,
                parser_version=parser_version,
            )
            vertical_workflow: list[EvidenceAtom] = []
            if not horizontal:
                vertical_workflow = _vertical_workflow_atoms_from_text(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    page_number=page_idx + 1,
                    text=page_text,
                    parser_version=parser_version,
                )
            if horizontal:
                out.extend(horizontal)
            elif vertical_workflow:
                out.extend(vertical_workflow)
            else:
                out.extend(
                    _workflow_atoms_from_text(
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        page_number=page_idx + 1,
                        text=page_text,
                        parser_version=parser_version,
                    )
                )
    return out
