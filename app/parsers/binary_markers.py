"""Universal binary-region markers — turn silently-skipped binary content
(images, charts, drawings, embedded OLE objects, email attachments) into
*located markers* so it can never silently vanish.

Detection of a binary region is total and guaranteed even when we cannot yet
read it: the region becomes a marker atom the PM sees ("image/object/attachment
here, vision / OLE / manual pass required"). Extraction quality is a separate,
improving frontier. Every marker carries a ``region_ref`` in its ``value`` so
the content census reconciles the region as MARKED rather than UNCOVERED.

This module is parser-agnostic: docx/pptx/xlsx/odt/ods all share the OOXML/ODF
zip layout, and email/mbox/msg share the MIME-attachment shape, so one helper
each covers the whole family. No keyword lists, no per-deal tuning — a binary
part is a binary part.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

# Zip sub-paths (by leading directory) that hold binary regions, and how we
# label them. Keys are matched against the path AFTER the top container dir is
# stripped (word/ ppt/ xl/ -> ""), so they work across the whole OOXML family.
_OOXML_BINARY_DIRS: tuple[tuple[str, str, str], ...] = (
    ("media/", "image_marker", "image"),
    ("embeddings/", "embedded_object_marker", "embedded object"),
    ("charts/", "chart_marker", "chart"),
    ("drawings/", "drawing_marker", "drawing / shape"),
)

# ODF binary locations.
_ODF_BINARY_DIRS: tuple[tuple[str, str, str], ...] = (
    ("Pictures/", "image_marker", "image"),
    ("ObjectReplacements/", "embedded_object_marker", "embedded object"),
    ("Object ", "embedded_object_marker", "embedded object"),
)


def _marker_atom(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    artifact_type: ArtifactType,
    parser_version: str,
    region_ref: str,
    kind: str,
    label: str,
    size: int,
    saved_path: str | None = None,
    caption: str | None = None,
) -> EvidenceAtom:
    if saved_path:
        # The bytes have been cropped out and written to disk — a later OCR /
        # vision pass reads that file. The marker now points AT the saved image
        # instead of reporting "0 bytes", so nothing is lost and the region is
        # ready for downstream extraction.
        text = (
            f"[{label.capitalize()} extracted - saved to {saved_path} "
            f"({size:,} bytes), awaiting OCR / vision] {region_ref} in {filename}."
        )
    else:
        text = (
            f"[{label.capitalize()} awaiting OCR / vision / OLE extraction] "
            f"{region_ref} in {filename} — {size:,} bytes. A vision or embedded-"
            f"object pass is required to recover its content."
        )
    # The expected content of the image — e.g. the 'Upload N photos showing X'
    # form instruction this photo answers. Gives the reviewer (and the vision
    # pass) what the photo SHOULD show instead of a bare 'awaiting OCR' marker.
    if caption:
        text = text.rstrip(".") + f' — expected: "{caption}".'
    atom_id = stable_id("atm", artifact_id, "binary_marker", region_ref)
    src = SourceRef(
        id=stable_id("src", atom_id),
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        filename=filename,
        locator={"region_ref": region_ref, "extraction": "binary_region_marker_v1",
                 **({"saved_path": saved_path} if saved_path else {})},
        extraction_method="binary_region_marker_v1",
        parser_version=parser_version,
    )
    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.open_question,
        raw_text=text,
        normalized_text=normalize_text(text),
        value={"kind": kind, "region_ref": region_ref, "size_bytes": size,
               **({"saved_path": saved_path} if saved_path else {}),
               **({"expected_content": caption} if caption else {})},
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.meeting_note,
        confidence=0.5,
        confidence_raw=0.5,
        calibrated_confidence=0.5,
        review_status=ReviewStatus.needs_review,
        review_flags=["binary_region_marker"],
        parser_version=parser_version,
    )


def region_marker(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    artifact_type: ArtifactType,
    parser_version: str,
    region_ref: str,
    kind: str = "image_marker",
    label: str = "image",
    size: int = 0,
    saved_path: str | None = None,
    caption: str | None = None,
) -> EvidenceAtom:
    """A located marker for one referenced (non-zip-embedded) binary region.

    Used by parsers whose binary content is *referenced* rather than packed in
    a zip — HTML ``<img>``/``<iframe>``, markdown ``![](...)`` image refs, PDF
    page images. The ``region_ref`` must match the census region's ``location``
    (e.g. ``media/<src>`` for HTML, ``page3/image7`` for PDF) so the content
    census reconciles the region as MARKED rather than UNCOVERED.
    """
    return _marker_atom(
        project_id=project_id,
        artifact_id=artifact_id,
        filename=filename,
        artifact_type=artifact_type,
        parser_version=parser_version,
        region_ref=region_ref,
        kind=kind,
        label=label,
        size=size,
        saved_path=saved_path,
        caption=caption,
    )


def emit_zip_binary_markers(
    *,
    path: Path,
    project_id: str,
    artifact_id: str,
    filename: str,
    artifact_type: ArtifactType,
    parser_version: str,
    family: str = "ooxml",
) -> list[EvidenceAtom]:
    """Emit a located marker for every binary part in a zip-based document.

    ``family="ooxml"`` strips the leading ``word/`` / ``ppt/`` / ``xl/`` so the
    ``region_ref`` is ``media/imageN.ext`` regardless of format; ``family="odf"``
    keeps ``Pictures/...`` paths. Reads only the central directory — no content
    is decoded, so this is cheap and never fails on an unreadable binary part.
    """
    dirs = _OOXML_BINARY_DIRS if family == "ooxml" else _ODF_BINARY_DIRS
    atoms: list[EvidenceAtom] = []
    try:
        zf = zipfile.ZipFile(path)
    except Exception:  # pragma: no cover - not a zip / unreadable
        return []
    with zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/"):
                continue
            # Strip the single top-level container dir for OOXML.
            rel = name
            if family == "ooxml" and "/" in name:
                rel = name.split("/", 1)[1]
            matched: tuple[str, str] | None = None
            for prefix, kind, label in dirs:
                if rel.startswith(prefix) or name.startswith(prefix):
                    matched = (kind, label)
                    break
            if matched is None:
                continue
            kind, label = matched
            try:
                size = zf.getinfo(name).file_size
            except KeyError:  # pragma: no cover
                size = 0
            atoms.append(_marker_atom(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                artifact_type=artifact_type,
                parser_version=parser_version,
                region_ref=rel,
                kind=kind,
                label=label,
                size=size,
            ))
    return atoms


def attachment_marker(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    artifact_type: ArtifactType,
    parser_version: str,
    attachment_name: str,
    size: int = 0,
    content_type: str = "",
) -> EvidenceAtom:
    """A located marker for one email/message attachment (the attachment file
    content itself is not extracted here — it becomes a visible region)."""
    ref = f"attachment/{attachment_name}"
    ctype = f" ({content_type})" if content_type else ""
    text = (
        f"[Attachment present, content not extracted] {attachment_name}{ctype} "
        f"in {filename} — {size:,} bytes. The attachment is a separate artifact; "
        f"extract or attach it directly to recover its content."
    )
    atom_id = stable_id("atm", artifact_id, "attachment_marker", attachment_name)
    src = SourceRef(
        id=stable_id("src", atom_id),
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        filename=filename,
        locator={"region_ref": ref, "extraction": "attachment_marker_v1"},
        extraction_method="attachment_marker_v1",
        parser_version=parser_version,
    )
    return EvidenceAtom(
        id=atom_id,
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.open_question,
        raw_text=text,
        normalized_text=normalize_text(text),
        value={"kind": "attachment_marker", "region_ref": ref,
               "attachment_name": attachment_name, "content_type": content_type,
               "size_bytes": size},
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.meeting_note,
        confidence=0.5,
        confidence_raw=0.5,
        calibrated_confidence=0.5,
        review_status=ReviewStatus.needs_review,
        review_flags=["attachment_marker"],
        parser_version=parser_version,
    )
