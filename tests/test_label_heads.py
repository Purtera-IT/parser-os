"""app/core/label_heads.json maps every label field to exactly one head.

Each assertion names what to change: a new reading, relation, judgment head or
training task fails here until it is given a head.
"""
from __future__ import annotations

from collections import Counter

from app.core.atom_type_registry import load_registry
from app.core.label_heads import (
    head,
    head_of_judgment,
    head_of_read,
    head_of_relation,
    head_of_task,
    heads,
    load_heads,
    space_of_task,
)
from app.core.pm_feedback import HEAD_REGISTRY
from app.learning.multitask_table import COMPANY_PROFILES, tasks_for

REG = load_registry()
H = load_heads()
SPACES = [s["key"] for s in H["spaces"]]
LABELED_COLUMNS = {"label_type", "about", "wants", "supplier", "entity_keys", "hints", "hint_refs",
                   "note", "rejected", "weight_tier"}

#: Judgment surfaces the page has that parser-os HEAD_REGISTRY does not: the
#: suppression card is aliased onto `admission`, and the rule card feeds the
#: SemanticRule trainer.
PAGE_ONLY_JUDGMENTS = {"rule", "suppression"}


def _all(field: str) -> list[str]:
    return [v for h in heads() for v in h.get(field, [])]


def test_heads_are_well_formed():
    keys = [h["key"] for h in heads()]
    assert len(keys) == len(set(keys)), "duplicate head key in label_heads.json"
    for h in heads():
        assert h["layer"] in {"universal", "company", "meta"}, h["key"]
        assert h["status"] in {"label", "self_supervised", "planned"}, h["key"]
        assert h["space"] in SPACES or (h["layer"] == "meta" and h["space"] is None), \
            f"{h['key']}: space {h['space']!r} not in label_heads.json spaces"
        assert h["key"].split(".")[0] in SPACES + ["structure", "meta"], h["key"]
        assert h["label"].strip() and h["question"].strip() and h["output"].strip(), h["key"]
        assert set(h["columns"]) <= LABELED_COLUMNS, f"{h['key']}: unknown column"
        if h["space"] == "conduct":
            assert h["layer"] == "company", f"{h['key']}: conduct is the company layer"
        if h["layer"] == "company":
            assert h["space"] == "conduct", f"{h['key']}: a company head lives in conduct"


def test_every_reading_has_exactly_one_head_of_its_own_layer():
    reads = {r["key"]: r for r in REG["reads"]}
    owned = Counter(_all("reads"))
    twice = sorted(k for k, n in owned.items() if n > 1)
    assert not twice, f"readings with two heads in label_heads.json: {twice}"
    missing = sorted(set(reads) - set(owned))
    assert not missing, f"readings in atom_types.json with no head in label_heads.json: {missing}"
    unknown = sorted(set(owned) - set(reads))
    assert not unknown, f"label_heads.json names readings atom_types.json lacks: {unknown}"
    for k, r in reads.items():
        h = head(head_of_read(k))
        want = "meta" if r["layer"] in {"meta", "staging"} else r["layer"]
        assert h["layer"] == want, f"{k}: reading layer {r['layer']} but head {h['key']} is {h['layer']}"


def test_every_relation_has_exactly_one_head():
    rels = [r["key"] for r in REG["relations"]]
    owned = Counter(_all("relations"))
    assert sorted(owned) == sorted(rels), "label_heads.json relations differ from atom_types.json relations"
    assert all(n == 1 for n in owned.values()), "a relation sits under two heads"


def test_every_judgment_head_has_exactly_one_head():
    owned = Counter(_all("judgments"))
    assert all(n == 1 for n in owned.values()), "a judgment head sits under two heads"
    missing = sorted(set(HEAD_REGISTRY) - set(owned))
    assert not missing, f"pm_feedback.HEAD_REGISTRY heads with no head in label_heads.json: {missing}"
    extra = sorted(set(owned) - set(HEAD_REGISTRY) - PAGE_ONLY_JUDGMENTS)
    assert not extra, f"label_heads.json judgments nobody defines: {extra}"


def test_the_note_splits_into_one_universal_and_one_company_part():
    parts = {h["key"]: h.get("note_part") for h in heads() if "note" in h["columns"]}
    assert sorted(parts.values()) == ["company", "universal"], parts
    assert head(next(k for k, v in parts.items() if v == "universal"))["layer"] == "universal"
    assert head(next(k for k, v in parts.items() if v == "company"))["layer"] == "company"


def test_every_training_task_has_a_head():
    profiles = ("base",) + tuple(COMPANY_PROFILES)
    tasks = {t for p in profiles for t in tasks_for(p)}
    tasks |= {"rationale:atom", "rationale:evidence", "rationale:edge", "rationale:deal"}
    tasks |= {f"rationale:policy:{c}" for c in COMPANY_PROFILES}
    missing = sorted(t for t in tasks if head_of_task(t) is None)
    assert not missing, f"training tasks with no head in label_heads.json: {missing}"


def test_base_trains_no_company_head_except_the_listed_legacy_tasks():
    legacy = set(H["legacy"]["base_tasks_in_company_heads"])
    wrong = sorted(t for t in tasks_for("base")
                   if head(head_of_task(t))["layer"] != "universal" and t not in legacy)
    assert not wrong, f"base tasks whose head is company or meta: {wrong}"


def test_lookups():
    assert head_of_read("co_action") == "conduct.action"
    assert head_of_read("stage_std") == "context.stage"
    assert head_of_relation("governs") == "links.hierarchy"
    assert head_of_relation("derived_from") == "links.support"
    assert head_of_judgment("gap") == "consequence.questions"
    assert head_of_task("policy:purtera") == "conduct.action"
    assert head_of_task("rationale:policy:acme") == "conduct.action"
    assert head_of_task("gap_valid_reason") == "consequence.questions"
    assert head_of_task("reads:superseded") == "beliefs.tracker"
    assert head_of_task("nope") is None
    assert space_of_task("atom_type") == "content"


def test_multitask_rows_carry_their_head_and_space(tmp_path):
    import sqlite3

    from app.learning.multitask_table import MultitaskTable, TaskRow

    table = MultitaskTable(rows=[
        TaskRow("atom_type", "Install 4 displays in the lobby", "work_scope_item", "d1", "train", "human", "x.db"),
        TaskRow("policy:purtera", "I'll send the revised quote by EOD", "reject", "d1", "train", "human", "x.db"),
    ])
    out = tmp_path / "m.db"
    table.write(out)
    got = sqlite3.connect(out).execute("select task, head, space from multitask_rows order by task").fetchall()
    assert got == [("atom_type", "content.type", "content"), ("policy:purtera", "conduct.action", "conduct")]
    assert table.per_head() == {"content.type": 1, "conduct.action": 1}


def test_every_registered_relation_trains_the_edge_head():
    from app.learning.human_labels import _LINK_TO_EDGE

    assert set(_LINK_TO_EDGE) == {r["key"] for r in REG["relations"]}
    assert _LINK_TO_EDGE["derived_from"] == "derived_from"


def test_the_deal_kit_shape_judgment_is_company_layer():
    """`commercial` is the shape of OUR kit -- billing type, PM/PC hours, travel
    days (relation commercial_terms). Nothing in a deal's documents states it,
    so it is Purtera's constant, not a universal claim, and never trains the base."""
    key = head_of_judgment("commercial")
    assert key == "conduct.constants"
    assert head(key)["layer"] == "company"
