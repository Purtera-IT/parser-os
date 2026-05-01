from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.loader import DOMAIN_DIR, load_domain_pack


def test_load_copper_cabling_resolves_pack_id_and_version() -> None:
    pack = load_domain_pack("copper_cabling")
    assert pack.pack_id == "copper_cabling"
    assert pack.version == "0.4.0-generated"
    assert pack.name
    onto = DOMAIN_DIR / "ontology" / "copper_low_voltage_ontology.yaml"
    if onto.is_file():
        assert pack.reference_ontology_path == "ontology/copper_low_voltage_ontology.yaml"
    else:
        assert pack.reference_ontology_path is None


def test_load_default_pack_unchanged() -> None:
    pack = load_domain_pack("default_pack")
    assert pack.pack_id == "default_pack"
    assert pack.reference_ontology_path is None


def test_invalid_yaml_that_is_not_reference_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad_pack.yaml"
    bad.write_text("not_json: [broken\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_pack(bad)
