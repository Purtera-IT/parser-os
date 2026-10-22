"""Whose call a line's keep is: the type head drops chatter (small_talk / _keep
at its SURE bar) and co_action cannot rescue it, because co_action was never
labeled on those lines; co_action decides among substantive lines, and its
keep still beats a parser or regex reject. Made-up atoms; 010388's chatter
wording is the audit's example."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

import app.core.c3_type_base as c3
from app.core import c3_brief, c3_sow
from app.core.admission_chatter import mark_admission_chatter
from app.core.deal_chatter import retype_small_talk
from app.core.head_fold import fold_by_heads
from app.core.schemas import AtomType
from app.core.suppression_ledger import SURVIVOR_KEY

GAME = "Yesterday's game was nuts."
WORK = "Mount the four lobby displays on the north wall."


@pytest.fixture(autouse=True)
def _deal(monkeypatch):
    monkeypatch.setattr(c3, "_DEAL", {})
    c3.set_deal([])
    monkeypatch.setenv("SOWSMITH_C3_SURE", "0.5")
    monkeypatch.setenv("SOWSMITH_C3_BRIEF_TRUST", "read:co_action,read:sow_section")
    c3._DEAL["types"].update({"g": ["small_talk", 0.94], "k": ["_keep", 0.8], "w": ["work_scope_item", 0.9],
                              "u": ["small_talk", 0.4], "r": ["scope_item", 0.7]})
    yield


def test_the_type_heads_chatter_read_is_a_drop():
    assert c3.type_head_drops("g") and c3.type_head_drops("k")
    assert not c3.type_head_drops("w") and not c3.type_head_drops("u") and not c3.type_head_drops("nope")


def test_brief_co_action_cannot_rescue_type_head_chatter_but_decides_substantive_lines():
    atoms = [NS(id=i, atom_type=t, raw_text=x, review_status="approved") for i, t, x in (
        ("g", "deal_metadata", GAME), ("k", "scope_item", "So day game."), ("w", "scope_item", WORK),
        ("u", "small_talk", "Thanks, and the rack is in the MDF."), ("r", "small_talk", "Cable runs are about 120 ft."))]
    reads = {a.id: {"read:co_action": ["keep", 0.99], "read:sow_section": ["scope_of_work", 0.9]} for a in atoms}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    kept = {x["id"] for v in b["sections"].values() for x in v}
    assert kept == {"w", "u", "r"}  # u: type head unsure; r: a rule's small_talk, co_action keep beats it
    assert b["left_out"]["by_reason"] == {"type head small_talk": 1, "type head _keep": 1}


def test_sow_rejects_type_head_chatter_even_when_the_brief_kept_it():
    env = {"atoms": [{"id": "g", "artifact_id": "m", "text": GAME, "atom_type": "deal_metadata",
                      "structured": {"c3_read": {"type": "small_talk", "p": 0.94}}, "locator": {}},
                     {"id": "x", "artifact_id": "m", "text": "Conducts the on-site wireless RF survey at each site.",
                      "atom_type": "small_talk", "structured": {"admission_reject": True,
                                                                "c3_read": {"type": "work_scope_item", "p": 0.6}},
                      "locator": {}}],
           "c3_brief": {"sections": {"scope_of_work": [{"id": "g", "text": GAME, "by": "purtera:read:co_action"},
                                                       {"id": "x", "text": "Conducts the on-site wireless RF survey at each site.",
                                                        "by": "purtera:read:co_action"}]},
                        "also_mentioned": [], "kept_unsure": [{"id": "g", "text": GAME, "p": 0.4}],
                        "decided_by": {"heads_used": ["read:co_action"]}}}
    rej = c3_sow._rejected_atoms(env)
    assert "g" in rej and "x" not in rej
    assert c3_sow._keep_gate(env) == {"x"}
    assert {f["id"] for f in c3_sow.facts(env)} == {"x"}


def test_head_fold_never_keeps_a_copy_the_type_head_drops():
    s = NS(id="s", raw_text=GAME, artifact_id="a", value={}, review_flags=[], source_refs=[], receipts=[], entity_keys=[])
    c = NS(id="c", raw_text=GAME, artifact_id="b", review_flags=["suppressed:quoted_history_dedup"],
           source_refs=[], receipts=[], entity_keys=[],
           value={"_suppression": {"stage": "quoted_history_dedup"}, SURVIVOR_KEY: {"atom_id": "s"}})
    reads = {"s": ("reject", 0.9), "c": ("keep", 0.99)}
    atoms, swaps = fold_by_heads([s], [c], reading=lambda a, h: reads.get(a), type_drops=lambda a: a == "c")
    assert swaps == [] and atoms == [s]
    atoms, swaps = fold_by_heads([s], [c], reading=lambda a, h: reads.get(a), type_drops=lambda a: False)
    assert atoms == [c]


def test_a_held_line_the_type_head_reads_as_chatter_is_small_talk():
    a = mark_admission_chatter(NS(id="k", raw_text="Best regards,", atom_type=AtomType.deal_metadata, value={},
                                  review_flags=[], source_refs=[], artifact_id="m"), "signoff")
    assert retype_small_talk([a]) == 1 and a.atom_type.value == "small_talk"


def test_type_reads_are_stamped_on_every_line_read():
    a, b = NS(id="w", value={}), NS(id="zz", value={})
    assert c3.stamp_type_reads([a, b]) == 1
    assert a.value["c3_read"] == {"type": "work_scope_item", "p": 0.9} and "c3_read" not in b.value
