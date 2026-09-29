# -*- coding: utf-8 -*-
"""A field the model declares survives the legacy shape, or the cache lies.

`EvidenceAtom.from_legacy_shape` and `Packet.from_legacy_shape` rebuild the
model from a HAND-LISTED set of keys. A field added to the model afterwards is
not in that list, so it is silently dropped the moment anything round-trips
through the legacy shape -- and the compile logs success.

`confidence_raw` was one of them, and the consequence was not cosmetic. The
artifact cache serialises atoms with `atom_id`, which is exactly what selects
the legacy path, so:

    compile 1 (fresh parse)  -> output_signature 418c4ece...
    compile 2 (cache read)   -> output_signature 4a8bd499...
    compile 3 (cache read)   -> output_signature 4a8bd499...

Two compiles of one unchanged project did not agree. It converges after the
first, so it never looked like randomness -- it looked like a stable answer
that happened to be a different stable answer than the first run's.

This was latent: it only reaches `output_signature` for atoms that survive to
the envelope, and the phase-3 dedup fixes changed which atoms do. Finding it
took a bisect against HEAD, because my first reading was that I had introduced
the nondeterminism rather than exposed it.
"""
from __future__ import annotations

from pathlib import Path

from app.core.compiler import compile_project
from app.core.schemas import ArtifactType, EvidenceAtom, EvidencePacket, SourceRef


def _ref() -> SourceRef:
    return SourceRef(
        id="src_1",
        artifact_id="art_1",
        artifact_type=ArtifactType.txt,
        filename="totals.xlsx",
        locator={"line_start": 1, "line_end": 1},
        extraction_method="deterministic_rule",
        parser_version="test_parser_v1",
    )


def test_a_declared_field_survives_the_legacy_shape() -> None:
    """The legacy path is selected by `atom_id`, which is how the cache writes."""
    atom = EvidenceAtom.model_validate({
        "atom_id": "a1",
        "claim": "Total Labor Revenue: $121,519",
        "entity": "labor",
        "project_id": "p",
        "artifact_id": "art_1",
        "confidence": 0.8,
        "confidence_raw": 0.8,
        "source_refs": [_ref()],
    })
    assert atom.confidence_raw == 0.8, "the legacy shape dropped a declared field"


def test_an_undeclared_key_is_still_ignored() -> None:
    """Carrying fields over must not become a way to smuggle junk in."""
    atom = EvidenceAtom.model_validate({
        "atom_id": "a1", "claim": "x", "entity": "e",
        "project_id": "p", "artifact_id": "art_1",
        "not_a_field_on_this_model": "should not appear",
        "source_refs": [_ref()],
    })
    assert not hasattr(atom, "not_a_field_on_this_model")


def test_a_packet_keeps_its_declared_fields_too() -> None:
    packet = EvidencePacket.model_validate({
        "packet_id": "p1", "topic": "scope", "project_id": "p",
        "atom_ids": ["a1"], "confidence": 0.7,
        "governing_atom_ids": ["a1"],
    })
    assert packet.id == "p1"


def test_the_cache_read_agrees_with_the_fresh_parse(demo_project: Path) -> None:
    """Three compiles of one unchanged project, the first parsing and the rest
    reading the cache, must produce one signature."""
    sigs = [
        compile_project(demo_project, project_id="demo_project").manifest.output_signature
        for _ in range(3)
    ]
    assert len(set(sigs)) == 1, f"the cache changed the answer: {sigs}"


def test_the_cache_agrees_with_no_cache_at_all(demo_project: Path) -> None:
    cached = compile_project(demo_project, project_id="demo_project")
    fresh = compile_project(demo_project, project_id="demo_project", use_cache=False)
    assert cached.manifest.output_signature == fresh.manifest.output_signature
