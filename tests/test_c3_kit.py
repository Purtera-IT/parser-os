"""The deal kit model reads atoms from before the kit was attached.

Invented rows only. Each atom is scored from itself, the atoms linked to it,
and one summary per other section. A filled section teaches the atoms left
off it. A site the tab kept is a job-site mark for the atom model. A reason
is stored and is not an input. The five held-out kits never train.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.kit import (KIT_HOLDOUT, KitExample, KitModel, _targets,  # noqa: E402
                       billing_history, decision_reads, kit_loss, load_kit,
                       load_kit_checkpoint, lock_kit_split, prepare_kit_run,
                       run_kit, save_kit, section_vote, site_judgments, slot_matrix)
from ml.c3.schema import load_schema  # noqa: E402


def _payload():
    return {
        "deal_id": "synthetic-kit",
        "attached_at": "2026-06-01T12:00:00Z",
        "billing": "t_and_m",
        "atoms": [
            {"key": "s1", "atom_id": "at-s1", "text": "North office on Elm Street.",
             "type": "physical_site", "entered_at": "2026-05-01T10:00:00Z", "doc_kind": "email"},
            {"key": "s2", "atom_id": "at-s2", "text": "A vendor billing address on Main.",
             "type": "physical_site", "entered_at": "2026-05-02T10:00:00Z", "doc_kind": "email"},
            {"key": "t1", "atom_id": "at-t1", "text": "Install four displays in the lobby.",
             "type": "task", "entered_at": "2026-05-03T10:00:00Z", "doc_kind": "email",
             "links": [{"relation": "context", "to": "sheet"}]},
            {"key": "b1", "atom_id": "at-b1", "text": "Four displays, part UB-4.",
             "type": "bom_line", "entered_at": "2026-05-04T10:00:00Z", "doc_kind": "email"},
            {"key": "late", "text": "A note written after the kit existed.",
             "type": "task", "entered_at": "2026-06-02T10:00:00Z", "doc_kind": "email"},
            {"key": "sheet", "text": "Our own kit row.", "type": "task",
             "entered_at": "2026-05-01T10:00:00Z", "doc_kind": "deal_kit"},
            {"key": "hidden", "text": "A later document the cut already closed.",
             "type": "physical_site", "entered_at": "2026-05-01T10:00:00Z", "doc_kind": "email",
             "reader_scope": {"deal_kit": {"visible": False, "why": "postdates"}}},
        ],
        "lines": [
            {"section": "sites", "atom_key": "s1", "pc_hours": 4, "travel_days": 1},
            {"section": "labor", "atom_key": "t1", "hours": 8, "reason": "Ladder work, so longer."},
            {"section": "hardware", "atom_key": "b1", "qty": 4},
            {"section": "materials", "qty": 2},
        ],
        "dropped_keys": ["s2"],
        "links": [
            {"src": "t1", "dst": "s1", "relation": "context"},
            {"src": "b1", "dst": "t1", "relation": "supports"},
            {"src": "t1", "dst": "late", "relation": "context"},
            {"from_key": "s1", "to_label_key": "s2", "relation": "near_miss"},
        ],
    }


def test_context_is_atoms_from_before_the_kit_routed_by_type():
    ex = KitExample.from_dict(_payload())
    keys = [a.key for a in ex.atoms]
    assert keys == ["s1", "s2", "t1", "b1"]
    assert [a.section for a in ex.atoms] == ["sites", "sites", "labor", "hardware"]
    assert "Ladder work" not in " ".join(ex.context_texts())
    assert ex.reasons == ("Ladder work, so longer.",)


def test_a_kept_site_marks_the_atom_site_head_and_a_left_off_site_does_not():
    ex = KitExample.from_dict(_payload())
    marks = site_judgments(ex)
    assert len(marks) == 1
    assert marks[0]["verdict"] == "job_site"
    assert marks[0]["target"]["site"]["evidence"]["atomId"] == "at-s1"
    schema = load_schema()
    atom = {"label_key": "s1", "atom_id": "at-s1", "text": "North office on Elm Street.",
            "doc_id": "email-1", "doc_kind": "email", "section": "body",
            "entered_at": "2026-05-01T10:00:00Z", "speaker_role": "customer",
            "speaker_side": "customer"}
    deal = DealExample.from_training_blob(
        {"labels": [{"label_key": "s1", "label_type": "physical_site", "labeler": "max@purtera-it.com"}],
         "judgments": marks},
        [atom], deal_id="synthetic-kit")
    batch = featurize(deal, schema)
    opp = schema.by_key()["jdg:site_role"]
    assert batch.targets["jdg:site_role"][0] == opp.index("job_site")


def test_every_atom_is_scored_from_its_links_and_one_summary_per_other_section():
    ex = KitExample.from_dict(_payload())
    assert {a.key for a in ex.atoms} == {"s1", "s2", "t1", "b1"}
    assert all("late" not in edge and "sheet" not in edge for edge in ex.links)
    reads = {r.key: r for r in decision_reads(ex)}
    assert set(reads) == {"s1", "s2", "t1", "b1"}
    task = reads["t1"]
    assert task.evidence == ("s1", "b1")
    assert task.siblings == ()
    assert dict(task.channels)["sites"] == ("s2",)
    assert dict(task.channels)["hardware"] == ()
    site = reads["s1"]
    assert site.evidence == ("t1",)
    assert site.differ == ("s2",)
    assert site.siblings == ()
    assert dict(site.channels)["hardware"] == ("b1",)
    for read in reads.values():
        seen = [read.key, *read.evidence, *read.same, *read.differ, *read.siblings]
        for _, keys in read.channels:
            seen.extend(keys)
        assert sorted(seen) == ["b1", "s1", "s2", "t1"]
        assert len(read.channels) == 3
    vote = section_vote(ex)
    assert set(vote) == {"sites", "labor", "hardware"}
    assert vote["sites"] == pytest.approx(vote["labor"])
    assert vote["labor"] == pytest.approx(vote["hardware"])
    pred = KitModel()(ex, torch.zeros(4))
    assert tuple(pred["keep"].shape) == (4,)
    assert tuple(pred["hours"].shape) == (4,)
    assert int(torch.count_nonzero(pred["hours"])) == 0


def test_a_filled_section_teaches_the_atoms_left_off_it_and_an_empty_section_does_not():
    raw = _payload()
    raw["atoms"] = list(raw["atoms"]) + [
        {"key": "t2", "atom_id": "at-t2", "text": "Ship the boxes to the lobby.",
         "type": "task", "entered_at": "2026-05-03T11:00:00Z", "doc_kind": "email"},
        {"key": "n1", "atom_id": "at-n1", "text": "Mornings are better for the crew.",
         "type": "preference", "entered_at": "2026-05-03T12:00:00Z", "doc_kind": "email"},
    ]
    raw["lines"] = [ln for ln in raw["lines"] if ln["section"] != "hardware"]
    ex = KitExample.from_dict(raw)
    keep = dict(zip((a.key for a in ex.atoms), _targets(ex)["keep"]))
    assert keep["t1"] == 1.0
    assert keep["t2"] == 0.0
    assert keep["s1"] == 1.0
    assert keep["s2"] == 0.0
    assert keep["b1"] != keep["b1"]
    assert keep["n1"] != keep["n1"]
    raw["dropped_keys"] = ["s2", "b1"]
    marked = KitExample.from_dict(raw)
    dropped = dict(zip((a.key for a in marked.atoms), _targets(marked)["keep"]))
    assert dropped["b1"] == 0.0
    assert [m["target"]["site"]["evidence"]["atomId"] for m in site_judgments(marked)] == ["at-s1"]


def test_a_labor_line_cannot_point_at_a_site_atom():
    raw = _payload()
    raw["lines"][1]["atom_key"] = "s1"
    with pytest.raises(ValueError, match="labor line points at a sites atom"):
        KitExample.from_dict(raw)


def test_the_fixed_holdout_cannot_train_and_history_excludes_it():
    with pytest.raises(ValueError, match="Holdout kit"):
        lock_kit_split([KIT_HOLDOUT[0]], KIT_HOLDOUT)
    train = KitExample.from_dict({**_payload(), "deal_id": "train-a", "billing": "fixed"})
    other = KitExample.from_dict({**_payload(), "deal_id": "train-b", "billing": "t_and_m"})
    held = KitExample.from_dict({**_payload(), "deal_id": KIT_HOLDOUT[0]})
    hist = billing_history([train, other, held], "train-a", KIT_HOLDOUT)
    assert hist.tolist() == [0.0, 1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="Holdout kit"):
        run_kit([held], [])


def test_one_kit_step_trains_on_the_saved_tab_and_scores_the_holdout_once():
    train = KitExample.from_dict(_payload())
    hold = KitExample.from_dict({**_payload(), "deal_id": "synthetic-hold"})
    out = run_kit([train], [hold], steps=1)
    assert out["train_ids"] == ("synthetic-kit",)
    assert out["holdout"][0]["deal_id"] == "synthetic-hold"
    assert out["ready_for_training"] is False
    loss = kit_loss(KitModel(), train, billing_history([train], train.deal_id))
    assert torch.isfinite(loss)


def test_attention_tags_every_other_atom_and_the_tag_learns():
    ex = KitExample.from_dict(_payload())
    keys = [a.key for a in ex.atoms]
    grid = slot_matrix(ex)
    t1, s1, s2, b1 = (keys.index(k) for k in ("t1", "s1", "s2", "b1"))
    assert grid[t1][t1] == "self"
    assert grid[t1][s1] == "evidence"
    assert grid[t1][b1] == "evidence"
    assert grid[t1][s2] == "sites"
    assert grid[s1][s2] == "differ"
    assert grid[s2][t1] == "labor"
    model = KitModel()
    loss = kit_loss(model, ex, torch.zeros(4))
    loss.backward()
    grad = model.blocks[0].slot.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all())
    assert int(torch.count_nonzero(grad)) > 0


def test_a_live_split_keeps_the_five_and_the_checkpoint_stays_unreleased(tmp_path):
    import json

    train = KitExample.from_dict({**_payload(), "deal_id": "train-a"})
    held = [KitExample.from_dict({**_payload(), "deal_id": deal_id, "billing": "fixed"})
            for deal_id in KIT_HOLDOUT]
    with pytest.raises(ValueError, match="missing a fixed kit"):
        prepare_kit_run([train], held[:4], live=True)
    prepare_kit_run([train], held, live=True)
    with pytest.raises(ValueError, match="Duplicate deal"):
        prepare_kit_run([train, train], held, live=True)
    raw = tmp_path / "kit.json"
    raw.write_text(json.dumps({**_payload(), "deal_id": "train-a"}), encoding="utf-8")
    assert load_kit(raw).deal_id == "train-a"
    with pytest.raises(ValueError, match="deal_id"):
        load_kit_path = tmp_path / "blank.json"
        load_kit_path.write_text("{}", encoding="utf-8")
        load_kit(load_kit_path)
    out = run_kit([train], held[:1], steps=1)
    path = tmp_path / "kit.pt"
    save_kit(path, out["model"], step=1)
    fresh = KitModel()
    assert load_kit_checkpoint(path, fresh) == 1
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert blob["ready_for_training"] is False
