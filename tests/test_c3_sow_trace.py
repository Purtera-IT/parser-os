from app.core import c3_sow_trace


def _env():
    atoms = [
        {"id": "atm_00000000aaaa0001", "artifact_id": "art_1", "atom_type": "task", "confidence": 0.7, "text": "Before",
         "locator": {"page": 1}},
        {"id": "atm_00000000aaaa0002", "artifact_id": "art_1", "atom_type": "task", "confidence": 0.8,
         "text": "Mount two TVs in room A", "locator": {"page": 1, "line_start": 4},
         "structured": {"c3_label": {"type": "scope_item", "p": 0.91, "was": "task"},
                        "said_by": {"name": "Pat", "company": "cdw.com", "side": "theirs"}}},
        {"id": "atm_00000000aaaa0003", "artifact_id": "art_1", "atom_type": "assumption", "text": "After"},
    ]
    return {
        "atoms": atoms,
        "documents": [{"artifact_id": "art_1", "filename": "scope.pdf"}],
        "c3_brief": {
            "sections": {"scope_of_work": [{"id": "atm_00000000aaaa0002", "keep_p": 0.97, "by": "purtera:read:co_action",
                                            "section_by": "base:read:sow_section", "supplier": ["us", 0.88]}]},
            "questions": [{"id": "q:x", "evidence": [{"id": "atm_00000000aaaa0003"}]}],
            "label_next": [{"id": "atm_00000000aaaa0003", "head": "read:co_action", "base_says": "keep", "p": 0.49}],
        },
        "c3_sow": {
            "scope": [{"text": "Mount two (2) TVs in room A.", "cites": ["atm_00000000aaaa0002"]}],
            "writer_draft": {"assumptions": [{"text": "x", "cites": ["['atm_00000000aaaa0003']"]}]},
            "rule_decisions": [{"rule": "r", "head": "sow_keep", "atom_ids": ["atm_00000000ffff0009"], "value": "drop"}],
        },
    }


def test_trace_covers_every_cited_atom_with_its_head_calls():
    env = _env()
    c3_sow_trace.attach(env)
    t = env["c3_sow"]["atom_trace"]
    assert list(t) == ["atm_00000000aaaa0002", "atm_00000000aaaa0003", "atm_00000000ffff0009"]
    a = t["atm_00000000aaaa0002"]
    assert a["type"] == "scope_item" and a["type_p"] == 0.91 and a["type_was"] == "task"
    assert a["section"] == "scope_of_work" and a["keep_p"] == 0.97 and a["supplier"] == ["us", 0.88]
    assert a["file"] == "scope.pdf" and a["where"] == "p. 1 · line 4"
    assert [x["text"] for x in a["around"]] == ["Before", "After"]
    assert a["said_by"]["company"] == "cdw.com"
    b = t["atm_00000000aaaa0003"]  # cited only through a stringified writer-draft list
    assert b["section"] is None and b["brief"] == ["questions", "label_next"]
    assert b["unsure"] == ["read:co_action", "keep", 0.49]
    assert t["atm_00000000ffff0009"] == {"missing": True}


def test_attach_skips_an_errored_sow():
    env = {"c3_sow": {"error": "X"}}
    c3_sow_trace.attach(env)
    assert "atom_trace" not in env["c3_sow"]
