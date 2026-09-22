"""app/core/atom_types.json is THE list of atom types; every other list must agree.

Each assertion names the file to change, so adding a type is: add it to the
JSON with status "v2", run this file, and follow the failures.
"""
from __future__ import annotations

from app.core.atom_type_registry import KEEP, coarse_of, facet_of, load_registry, type_names
from app.core.schemas import AtomType
from app.core.typed_atom_classifier import _TAXONOMY
from runpod_detector.taxonomy import FACETS, MICRO_TO_FACET

REG = load_registry()
ENUM = {a.value for a in AtomType}


def test_names_are_unique_and_well_formed():
    names = [t["name"] for t in REG["types"]]
    assert len(names) == len(set(names)), "duplicate type in atom_types.json"
    for t in REG["types"]:
        assert t["name"] == t["name"].strip().lower() and " " not in t["name"]
        assert t["coarse"] in REG["coarse"], f"{t['name']}: coarse {t['coarse']!r} not in registry coarse list"
        assert t["facet"] in REG["facets"], f"{t['name']}: facet {t['facet']!r} not in registry facets"
        assert t["status"] in {"live", "v2"}
        assert t["label_space"] in {"head", "extractor"}
        assert t["desc"].strip(), f"{t['name']}: a labeler needs a definition"


def test_facets_match_runpod_taxonomy():
    assert tuple(REG["facets"]) == FACETS, "runpod_detector/taxonomy.py FACETS drifted"


def test_head_label_space_matches_micro_to_facet():
    head = {t["name"]: t["facet"] for t in REG["types"] if t["label_space"] == "head"}
    trained = {k: v for k, v in MICRO_TO_FACET.items() if k != KEEP}
    missing = sorted(set(trained) - set(head))
    extra = sorted(set(head) - set(trained) - set(_TAXONOMY))
    assert not missing, f"in runpod_detector/taxonomy.py MICRO_TO_FACET but not atom_types.json: {missing}"
    assert not extra, f"head types in atom_types.json that MICRO_TO_FACET does not train: {extra}"
    for name, facet in trained.items():
        assert head[name] == facet, f"{name}: facet {head[name]} here vs {facet} in MICRO_TO_FACET"


def test_live_types_are_in_the_prod_enum_and_v2_types_are_not():
    live_missing = sorted(n for n in type_names(status="live") if n not in ENUM)
    assert not live_missing, f"status=live but not in app/core/schemas.py AtomType: {live_missing}"
    v2_in_enum = sorted(n for n in type_names(status="v2") if n in ENUM)
    assert not v2_in_enum, f"in AtomType now -- flip status to 'live' in atom_types.json: {v2_in_enum}"


def test_every_prompt_type_is_registered():
    missing = sorted(set(_TAXONOMY) - {t["name"] for t in REG["types"]})
    assert not missing, f"typed_atom_classifier._TAXONOMY teaches types the registry lacks: {missing}"


def test_lookup_helpers():
    assert facet_of("work_scope_item") == "WORK"
    assert coarse_of("physical_site") == "site"
    assert coarse_of(KEEP) == KEEP
    assert facet_of("nope") is None


def test_hint_chips_are_declared():
    keys = [h["key"] for h in REG["context_hints"]]
    assert len(keys) == len(set(keys)) and "own_words" in keys and "neighbor_above" in keys
