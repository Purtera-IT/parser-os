"""A JSON key/value field means what its key path says.

An intake form's "logistics.escort_required: yes" reached the substance gate as
prose: the snake_case path is not dictionary words, so the line scored as OCR
debris and was dropped as a context-free fragment. The fact -- an escort is
required -- went with it. A keyed field with a short value is a stated fact;
free prose with no key is judged exactly as before.
"""

from __future__ import annotations

import json

from app.core.atom_substance_gate import (
    apply_substance_gate,
    drop_contact_chrome,
    drop_nonsubstantive_fragments,
    drop_unreadable_text,
)
from app.core.schemas import AtomType
from app.parsers.json_parser import JsonParser

_FORM = {
    "logistics": {
        "escort_required": "yes",
        "esc_reqd": "yes",
        "ppe_reqd": "no",
        "notify_lead_on_arrival": "yes",
        "lift_needed": "no",
        "badge_pickup": "y",
        "after_hours_ok": "n",
        "dock_access": True,
        "liftgate_truck": False,
        "trip_count": 2,
    },
    "coordination": {"gc_walks_tech": "yes", "callback_window": "2026-01-05"},
    "people": [{"label": "dispatch", "phone": "(555) 010-0199"}],
}
_YESNO = {
    "logistics.escort_required", "logistics.esc_reqd", "logistics.ppe_reqd",
    "logistics.notify_lead_on_arrival",
    "logistics.lift_needed", "logistics.badge_pickup", "logistics.after_hours_ok",
    "logistics.dock_access", "logistics.liftgate_truck",
    "coordination.gc_walks_tech",
}


def _atoms(tmp_path, data=_FORM):
    p = tmp_path / "form.json"
    p.write_text(json.dumps(data))
    return JsonParser().parse_artifact("p", "art_form", p)


def _paths(atoms):
    return {a.value.get("key_path") for a in atoms}


def test_yes_no_fields_survive_the_gate_with_their_key_path(tmp_path):
    atoms = _atoms(tmp_path)
    kept, dropped = apply_substance_gate(list(atoms))
    assert not (_paths(dropped) & _YESNO)
    survivors = {a.value["key_path"]: a for a in kept if a.value.get("key_path") in _YESNO}
    assert set(survivors) == _YESNO
    for path, atom in survivors.items():
        assert atom.raw_text.startswith(f"{path}: ")
        assert atom.source_refs[0].locator["key_path"] == path
        # Live, not demoted out of scope.
        assert atom.atom_type is not AtomType.deal_metadata


def test_every_field_of_the_form_survives(tmp_path):
    atoms = _atoms(tmp_path)
    kept, dropped = apply_substance_gate(list(atoms))
    assert dropped == []
    assert _paths(kept) >= _paths(atoms)
    # Nor demoted out of scope: a dated or phone field under its key is a fact.
    assert all(a.atom_type is not AtomType.deal_metadata for a in kept), sorted(
        a.value.get("key_path") for a in kept if a.atom_type is AtomType.deal_metadata
    )


def test_each_shape_pass_keeps_keyed_short_fields(tmp_path):
    atoms = _atoms(tmp_path)
    for gate in (drop_unreadable_text, drop_contact_chrome, drop_nonsubstantive_fragments):
        kept, dropped = gate(list(atoms))
        assert dropped == [], gate.__name__
        assert len(kept) == len(atoms)


def test_keyed_long_value_is_still_judged_on_its_words(tmp_path):
    debris = "xq zzv brrk tnnp qwxv plmk vvrt"
    atoms = _atoms(tmp_path, {"notes": {"scan_text": debris}})
    _, dropped = drop_unreadable_text(list(atoms))
    assert _paths(dropped) == {"notes.scan_text"}


def _prose(text, atom_type=AtomType.scope_item):
    from app.core.schemas import (
        ArtifactType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef,
    )

    return EvidenceAtom(
        id=f"atm_{abs(hash(text))}",
        project_id="p",
        artifact_id="art_mail",
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value={},
        entity_keys=[],
        source_refs=[SourceRef(
            id="src_x", artifact_id="art_mail", artifact_type=ArtifactType.email,
            filename="m.eml", locator={}, extraction_method="t", parser_version="t",
        )],
        authority_class=AuthorityClass.customer_current_authored,
        confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        parser_version="t",
    )


def test_context_free_prose_fragment_is_still_dropped():
    yes = _prose("Yes.")
    _, dropped = drop_nonsubstantive_fragments([yes])
    assert dropped == [yes]


def test_unkeyed_debris_is_still_dropped():
    junk = _prose("xq zzv brrk tnnp qwxv plmk vvrt")
    _, dropped = drop_unreadable_text([junk])
    assert dropped == [junk]

