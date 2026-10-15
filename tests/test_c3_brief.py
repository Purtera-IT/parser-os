"""Brief v2: heads decide where the registry trusts them, rules otherwise."""
from types import SimpleNamespace as NS

from app.core import c3_brief
from c3_record import heads


def _atom(i, t, text="x"):
    return NS(id=f"a{i}", atom_type=t, raw_text=text, review_status="approved")


def test_registry_layers_and_trust():
    reg = heads.registry()
    assert reg, "c3_record/heads.json ships with the package"
    assert heads.layer("read:co_action") == "purtera"
    assert heads.layer("read:sow_section") == "base"
    assert heads.trusted("read:co_action")
    assert not heads.trusted("col:wants")  # one-sided labels: not trusted


def test_no_read_falls_back_to_rules():
    atoms = [_atom(1, "scope_item", "Install 4 cameras"), _atom(2, "small_talk", "thanks!"),
             _atom(3, "open_question", "Who owns the lift?")]
    b = c3_brief.build(atoms, service_fallback="security_camera")
    assert [x["id"] for x in b["sections"]["scope_of_work"]] == ["a1"]
    assert b["service"] == {"value": "security_camera", "by": "rule:domain_pack", "base_says": None}
    assert [q["id"] for q in b["questions"]] == ["a3"]
    assert b["decided_by"]["heads_used"] == []


def test_trusted_heads_decide_and_say_so():
    atoms = [_atom(1, "scope_item"), _atom(2, "scope_item"), _atom(3, "task")]
    reads = {"a1": {"read:co_action": ["keep", 0.9], "read:sow_section": ["deliverables", 0.8]},
             "a2": {"read:co_action": ["reject", 0.7], "read:co_reason": ["hw_price_not_ours", 0.6]},
             "a3": {"read:co_action": ["keep", 0.4]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert b["sections"]["deliverables"][0]["by"] == "purtera:read:co_action"
    assert b["sections"]["deliverables"][0]["section_by"] == "base:read:sow_section"
    assert [x["id"] for x in b["also_mentioned"]] == ["a3"]
    assert b["left_out"]["by_reason"] == {"hw_price_not_ours": 1}
    assert {t["id"] for t in b["label_next"]} == {"a3"}


def test_brief_survives_result_round_trip():
    from app.core.schemas import CompileResult

    r = CompileResult(project_id="p", atoms=[], entities=[], edges=[], packets=[], project_dir="/x",
                      c3_brief={"version": 2})
    assert CompileResult.model_validate_json(r.model_dump_json()).c3_brief == {"version": 2}
