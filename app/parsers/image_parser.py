"""Image artifact parser — emits a needs-OCR placeholder per image.

Real MSP intakes carry HEIC / PNG / JPG site survey photos and
annotated network diagrams. Without an OCR / vision model these
pixels can't yield text atoms, but the brief still needs to
surface "this file was in the intake and could not be auto-
extracted" so the PM doesn't think a deliverable is missing —
it's just waiting on a manual review or a vision pass.

Each image becomes ONE atom of type ``open_question`` carrying:
  * the filename
  * pixel dimensions (when readable)
  * file size
  * a "manual review or vision-LLM extraction required" note

The PM_HANDOFF "Files requiring manual review" section (A6)
picks these up automatically because the parse_outcome is
``ok_empty`` from the consumer side — there's no atom shortage,
but the atom carries a marker explaining what's still needed.

If pytesseract is installed, the parser tries a best-effort OCR
text extraction and surfaces the recovered text as a second atom.
This is opportunistic — when tesseract isn't available, we just
emit the marker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ParserCapability,
    ParserMatch,
    ParserOutput,
    ReviewStatus,
    SourceRef,
)
from app.domain.schemas import DomainPack
from app.parsers.base import BaseParser


_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".heic", ".heif",
    ".webp", ".tiff", ".tif", ".bmp",
}


class ImageParser(BaseParser):
    parser_name = "image"
    parser_version = "image_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=sorted(_IMAGE_EXTENSIONS),
        supported_artifact_types=[ArtifactType.image],
        emitted_atom_types=[AtomType.open_question, AtomType.scope_item],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=False,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        suffix = path.suffix.lower()
        confidence = 0.95 if suffix in _IMAGE_EXTENSIONS else 0.0
        reasons = [f"image_extension:{suffix}"] if confidence else []
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=ArtifactType.image,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact("unknown_project", artifact_id, artifact_path)

    def parse_artifact(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> list[EvidenceAtom]:
        return self.parse_artifact_full(
            project_id=project_id,
            artifact_id=artifact_id,
            path=path,
            domain_pack=domain_pack,
        ).atoms

    def parse_artifact_full(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        del domain_pack
        atoms: list[EvidenceAtom] = []
        # Try to read dimensions + size for the placeholder atom.
        size_bytes = 0
        width, height = 0, 0
        try:
            size_bytes = path.stat().st_size
        except OSError:
            pass
        try:
            from PIL import Image
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            # PIL not available / corrupt / HEIC without pillow-heif — fall
            # through and emit placeholder with whatever metadata we have.
            pass

        marker_text = (
            f"[Image artifact awaiting OCR or vision-LLM extraction] "
            f"{path.name} — {width}×{height}px, {size_bytes:,} bytes. "
            f"Manual review or a vision pass is required to recover the "
            f"text / annotations in this image."
        )
        marker = self._make_atom(
            project_id=project_id,
            artifact_id=artifact_id,
            filename=path.name,
            text=marker_text,
            atom_type=AtomType.open_question,
            value_extra={
                "kind": "image_marker",
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
            },
        )
        atoms.append(marker)

        # Opportunistic OCR — fires only when pytesseract is
        # available. The recovered text is emitted as a second
        # scope_item atom so the PM_HANDOFF can show what was
        # recoverable without OCR being a hard dependency.
        ocr_text = self._try_ocr(path)
        if ocr_text:
            atoms.append(self._make_atom(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                text=f"[OCR-recovered text] {ocr_text[:2000]}",
                atom_type=AtomType.scope_item,
                value_extra={"kind": "image_ocr_text"},
                confidence=0.55,  # OCR is lower-confidence than text-layer
            ))
        return ParserOutput(atoms=atoms, derived_files=[])

    def _try_ocr(self, path: Path) -> str:
        """Multi-backend OCR via ``_ocr_chain``.

        Tries PyMuPDF Tesseract → pytesseract → easyocr → Ollama vision
        in order. Returns the recovered text or "" when nothing fires.
        """
        try:
            from app.parsers._ocr_chain import ocr_image_file
            result = ocr_image_file(path)
            return (result.get("text") or "").strip()
        except Exception:
            return ""

    def _make_atom(
        self,
        *,
        project_id: str,
        artifact_id: str,
        filename: str,
        text: str,
        atom_type: AtomType,
        value_extra: dict[str, Any],
        confidence: float = 0.85,
    ) -> EvidenceAtom:
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, atom_type.value, text[:80]),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.image,
            filename=filename,
            locator={"kind": "image_artifact"},
            extraction_method="image_marker_or_ocr",
            parser_version=self.parser_version,
        )
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, atom_type.value, text[:120]),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=atom_type,
            raw_text=text,
            normalized_text=text.lower(),
            value=value_extra,
            entity_keys=[],
            source_refs=[source_ref],
            authority_class=AuthorityClass.customer_current_authored,
            confidence=confidence,
            review_status=ReviewStatus.needs_review,
            parser_version=self.parser_version,
        )
