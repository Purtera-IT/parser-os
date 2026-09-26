"""A CAD drawing is a document, and right now it is not even a file.

Deal 010180 ships the floor plan of 7 Penn Plaza twice -- once as a PDF and
twice as the AutoCAD source -- and both `.dwg` files come back
``skipped_no_parser``. Nothing is extracted, nothing is shown, and the previewer
says "DWG files can't be previewed inline". For a structured-cabling job the plan
is the single most informative artifact there is: it carries the rooms, their
names, their positions and the pathways between them, and it carries them as
TEXT entities with coordinates rather than as ink a vision pass has to guess at.

Two halves, because they fail independently:

**The preview.** A DWG embeds a thumbnail of the sheet in its header, so this
costs nothing and needs no CAD library at all. 010180's `SP-6.dwg` yields a
valid 256x139 PNG. Its sibling `SP-6-1.dwg` does not -- the preview sentinel
does not match, and walking its image table produces 63 entries at impossible
offsets. So the table is never trusted: the PNG is found by signature and
validated by walking its own chunks to IEND, and a file without one simply has
no preview rather than a crash or a garbage image.

**The text.** No pure-Python DWG reader exists. `ezdxf` reads DXF natively, and
DWG reaches it through a converter -- LibreDWG's `dwg2dxf`, or the ODA File
Converter. Neither ships here, so the conversion is a seam: when a converter is
on PATH the entities become atoms, and when it is not the drawing still arrives
as a marker the PM can see, with its preview attached. A file nobody can read is
a worse outcome than a file nobody has converted yet, and today we have the
former.

`.dxf` needs none of that and is handled directly.
"""
from __future__ import annotations

import base64
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ParserCapability,
    ParserDerivedFile,
    ParserMatch,
    ParserOutput,
    ReviewStatus,
    SourceRef,
)
from app.domain.schemas import DomainPack
from app.parsers.base import BaseParser

_DWG_EXTENSIONS = {".dwg", ".dxf"}

#: The AutoCAD release that wrote the file. Worth carrying: a converter that
#: handles R14 may refuse AC1032, and "which release" is the first question
#: anyone asks when a drawing will not open.
_RELEASES = {
    "AC1009": "R11/R12", "AC1012": "R13", "AC1014": "R14",
    "AC1015": "AutoCAD 2000", "AC1018": "AutoCAD 2004",
    "AC1021": "AutoCAD 2007", "AC1024": "AutoCAD 2010",
    "AC1027": "AutoCAD 2013", "AC1032": "AutoCAD 2018",
}

#: Converters that turn a DWG into a DXF, best first. `dwg2dxf` is LibreDWG's
#: and is the one a Linux image can apt-install; ODA's is the reference
#: implementation and handles the newest releases most reliably.
_CONVERTERS = ("dwg2dxf", "ODAFileConverter")

#: Layers that carry the drafting apparatus rather than the building: title
#: blocks, dimensions, revision clouds. Their text is real but it is about the
#: sheet, not the site.
_APPARATUS_LAYERS = frozenset({
    "defpoints", "dim", "dims", "dimensions", "title", "titleblock",
    "tblk", "revision", "revisions", "rev", "notes-rev", "xref",
})


def preview_png(data: bytes) -> bytes | None:
    """The thumbnail a DWG carries in its header, or None.

    Found by signature rather than by the header's image table. On 010180's
    `SP-6-1.dwg` that table claims 63 images at offsets past the end of the
    file, so trusting its count is how this crashes on the second drawing
    anyone gives it. A PNG that walks cleanly to its own IEND is a PNG.
    """
    start = 0
    while True:
        start = data.find(b"\x89PNG\r\n\x1a\n", start)
        if start < 0:
            return None
        pos = start + 8
        while pos + 8 <= len(data):
            length = struct.unpack(">I", data[pos:pos + 4])[0]
            kind = data[pos + 4:pos + 8]
            if length > len(data) - pos:
                break
            pos += 12 + length
            if kind == b"IEND":
                return data[start:pos]
        start += 8


def _png_size(png: bytes) -> tuple[int, int]:
    try:
        return struct.unpack(">II", png[16:24])
    except Exception:  # noqa: BLE001 - a preview we cannot measure still shows
        return (0, 0)


def release_of(data: bytes) -> tuple[str, str]:
    """(version tag, human release) from the first six bytes."""
    tag = data[:6].decode("ascii", errors="replace")
    return tag, _RELEASES.get(tag, "unknown release")


def find_converter() -> str | None:
    """A DWG-to-DXF converter on PATH, or one named by ``DWG_TO_DXF``."""
    explicit = os.environ.get("DWG_TO_DXF", "").strip()
    if explicit and (Path(explicit).exists() or shutil.which(explicit)):
        return explicit
    for name in _CONVERTERS:
        found = shutil.which(name)
        if found:
            return found
    return None


def to_dxf(path: Path, converter: str, timeout: int = 120) -> Path | None:
    """Convert one DWG, returning the DXF path inside a temp dir, or None."""
    out_dir = Path(tempfile.mkdtemp(prefix="dwg2dxf_"))
    target = out_dir / (path.stem + ".dxf")
    name = Path(converter).name.lower()
    if name.startswith("odafileconverter"):
        # ODA works on directories: in, out, version, type, recurse, audit.
        cmd = [converter, str(path.parent), str(out_dir), "ACAD2018", "DXF", "0", "0"]
    else:
        cmd = [converter, "-o", str(target), str(path)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except Exception:  # noqa: BLE001 - a converter that fails is a missing one
        return None
    if target.exists() and target.stat().st_size:
        return target
    found = sorted(out_dir.glob("*.dxf"))
    return found[0] if found else None


def text_entities(dxf_path: Path) -> list[dict[str, Any]]:
    """Every TEXT/MTEXT on the drawing, with its layer and where it sits.

    Position is the point of doing this from CAD rather than from a picture of
    it: "IT CLOSET" at a known x/y is a room a cable has to reach, while the
    same words off an OCR pass are a string.
    """
    try:
        import ezdxf  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - not installed is not an error here
        return []
    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception:  # noqa: BLE001 - a DXF we cannot read yields nothing
        return []
    out: list[dict[str, Any]] = []
    spaces = [doc.modelspace()]
    # Paper space holds the title block and the sheet's own notes. Worth
    # reading -- the drawing number and revision live there -- and separable
    # afterwards by layer.
    try:
        spaces.extend(doc.layouts.get(name) for name in doc.layouts.names()
                      if name.lower() != "model")
    except Exception:  # noqa: BLE001 - model space alone is still an answer
        pass
    for space in spaces:
        for entity in space:
            kind = entity.dxftype()
            if kind not in ("TEXT", "MTEXT", "ATTRIB"):
                continue
            try:
                raw = entity.plain_text() if kind == "MTEXT" else str(entity.dxf.text)
            except Exception:  # noqa: BLE001
                continue
            text = " ".join(str(raw or "").split())
            if not text:
                continue
            # TEXT carries `insert`; `align_point` is only set when the string
            # is aligned or fitted, in which case `insert` can be (0, 0).
            x = y = 0.0
            for attr in ("insert", "align_point"):
                try:
                    point = getattr(entity.dxf, attr)
                    x, y = float(point[0]), float(point[1])
                    if (x, y) != (0.0, 0.0):
                        break
                except Exception:  # noqa: BLE001 - a label without a point is still a label
                    continue
            layer = str(getattr(entity.dxf, "layer", "") or "")
            out.append({"text": text, "layer": layer, "x": x, "y": y, "kind": kind})
    return out


def is_apparatus(layer: str) -> bool:
    """True when the layer is about the sheet rather than the building."""
    name = layer.strip().lower()
    return any(part in _APPARATUS_LAYERS for part in name.replace("-", " ").split())


class DwgParser(BaseParser):
    parser_name = "dwg"
    parser_version = "dwg_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=sorted(_DWG_EXTENSIONS),
        supported_artifact_types=[ArtifactType.image],
        emitted_atom_types=[AtomType.open_question, AtomType.site_attribute,
                            AtomType.deal_metadata],
        supported_domain_packs=["*"],
        requires_binary=True,
        supports_source_replay=False,
    )

    def match(self, path: Path, sample_text: str | None,
              domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        suffix = path.suffix.lower()
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=0.95 if suffix in _DWG_EXTENSIONS else 0.0,
            reasons=[f"cad_extension:{suffix}"] if suffix in _DWG_EXTENSIONS else [],
            artifact_type=ArtifactType.image,
        )

    def parse(self, artifact_path: Path) -> Any:
        return self.parse_artifact_full(
            project_id="unknown_project",
            artifact_id=stable_id("art", str(artifact_path)),
            path=artifact_path,
        )

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path,
                       domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        return self.parse_artifact_full(project_id=project_id, artifact_id=artifact_id,
                                        path=path, domain_pack=domain_pack).atoms

    def parse_artifact_full(self, project_id: str, artifact_id: str, path: Path,
                            domain_pack: DomainPack | None = None) -> ParserOutput:
        del domain_pack
        atoms: list[EvidenceAtom] = []
        derived: list[ParserDerivedFile] = []
        warnings: list[str] = []

        try:
            data = path.read_bytes()
        except OSError as exc:
            return ParserOutput(atoms=[], warnings=[f"dwg: unreadable ({exc})"])

        is_dxf = path.suffix.lower() == ".dxf"
        tag, release = ("", "DXF") if is_dxf else release_of(data)

        # The preview, first, because it never fails for a reason worth fixing.
        png = None if is_dxf else preview_png(data)
        if png:
            width, height = _png_size(png)
            derived.append(ParserDerivedFile(
                relative_path=f"{path.stem}.preview.json",
                content_kind="json",
                content_json={
                    "kind": "cad_preview",
                    "source": path.name,
                    "width": width,
                    "height": height,
                    "mime": "image/png",
                    "base64": base64.b64encode(png).decode("ascii"),
                },
            ))
        elif not is_dxf:
            warnings.append("dwg: no embedded preview in this drawing")

        # The text, if anything here can reach it.
        dxf_path: Path | None = path if is_dxf else None
        converter = None
        if not is_dxf:
            converter = find_converter()
            if converter:
                dxf_path = to_dxf(path, converter)
                if dxf_path is None:
                    warnings.append(f"dwg: {Path(converter).name} produced no DXF")

        labels = text_entities(dxf_path) if dxf_path else []
        building = [x for x in labels if not is_apparatus(x["layer"])]

        for label in building:
            atoms.append(self._make_atom(
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                text=label["text"],
                atom_type=AtomType.site_attribute,
                value_extra={
                    "kind": "cad_label",
                    "layer": label["layer"],
                    "x": label["x"],
                    "y": label["y"],
                    "entity": label["kind"],
                },
            ))

        # ...and a marker either way, so the drawing is never invisible. This is
        # the whole complaint: today a .dwg is `skipped_no_parser` and the PM
        # never learns the plan was in the intake at all.
        if building:
            marker = (f"[CAD drawing] {path.name} — {release}, "
                      f"{len(building)} labels recovered from "
                      f"{len(labels)} text entities.")
            marker_type = AtomType.deal_metadata
        else:
            missing = "no DWG-to-DXF converter available" if not converter else \
                      "conversion produced no readable text"
            marker = (f"[CAD drawing awaiting conversion] {path.name} — {release}, "
                      f"{len(data):,} bytes. The sheet's text was not recovered: "
                      f"{missing}. "
                      + ("A preview image was extracted." if png else
                         "This drawing carries no embedded preview."))
            marker_type = AtomType.open_question
        atoms.insert(0, self._make_atom(
            project_id=project_id,
            artifact_id=artifact_id,
            filename=path.name,
            text=marker,
            atom_type=marker_type,
            value_extra={
                "kind": "cad_marker",
                "version_tag": tag,
                "release": release,
                "size_bytes": len(data),
                "has_preview": bool(png),
                "converter": Path(converter).name if converter else None,
                "label_count": len(building),
            },
        ))
        return ParserOutput(atoms=atoms, derived_files=derived, warnings=warnings)

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
        locator: dict[str, Any] = {"kind": "cad_drawing"}
        for key in ("layer", "x", "y", "entity"):
            if key in value_extra:
                locator[key] = value_extra[key]
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, atom_type.value, text[:80]),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.image,
            filename=filename,
            locator=locator,
            extraction_method="cad_label_or_marker",
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
