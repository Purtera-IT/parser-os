"""ml/c3/quote_heads.py follows the heads (AUDIT_dealkit_heads items 2, 3, 12): a sure head reading
decides hours, crew, level and billing; regex and family medians only as a marked fallback; old Deal Kit
atoms and lines the heads left out never feed the counts."""
import json

import pytest

from ml.c3.quote_heads import QuoteHeads


@pytest.fixture
def qh(tmp_path, monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_QUOTE_WORDING", "off")
    fam = {"deals": 9, "median_hours_per_tech": 8.0, "per_unit_crew_hours": 4.0, "usual_techs": {"2": 5, "1": 4},
           "usual_rate_line": "L2 Networking", "usual_price": 100}
    b = {"version": "t", "families": {"camera": dict(fam), "av": dict(fam)}, "descriptions": {"camera": "cams", "av": "tvs"},
         "description_embeddings": [[1.0, 0.0], [0.0, 1.0]], "embed_model": "e", "wording_model": None,
         "quantity_weights": [0, 1, 1, 1, 1, 0, 0, 0, 0, 0], "profiles": {}, "count_scaled_families": ["camera"]}
    path = tmp_path / "b.json"
    path.write_text(json.dumps(b))
    q = QuoteHeads(str(path))
    monkeypatch.setattr(q, "_embed", lambda texts: [[1.0, 0.0] for _ in texts])
    return q


def _deal(atoms, **kw):
    d = {"deal_name": "000001 - Camera install", "account": "", "atoms": atoms,
         "c3_brief": {"sections": {"scope_of_work": [{"id": a["id"]} for a in atoms]}, "job_details": []}}
    d.update(kw)
    return d


def test_sure_heads_decide_hours_crew_level_and_billing(qh):
    atoms = [
        {"id": "a1", "type": "task", "text": "Install cameras across the warehouse"},
        {"id": "a2", "type": "requirement", "text": "Plan on 20 hours each for the team",
         "reads": {"read:co_hours_estimate": ["present", 0.92]}},
        {"id": "a3", "type": "requirement", "text": "send three technicians",
         "reads": {"read:crew_size": ["present", 0.9]}},
        {"id": "a4", "type": "requirement", "text": "Needs an L3 engineer for the NVR",
         "reads": {"read:tech_level": ["present", 0.95]}},
    ]
    out = qh.quote(_deal(atoms, heads={"asked": True, "deal": {"jdg:billing_type": ["fixed", 0.93]}}), word=False)
    line = out["lines"][0]
    assert line["techs"] == 3 and line["by"]["techs"] == "heads:read:crew_size"
    # 20 h is past the regex's old 16 h cap; the head's line stands (snapped to the hours grid)
    assert line["hours_per_tech"] == 24 and line["by"]["hours"] == "heads:read:co_hours_estimate"
    assert line["rate_line"] == "L3 Networking" and line["by"]["rate"] == "heads:read:tech_level"
    assert any("check the L3 Networking rate" in c for c in line["caveats"])
    assert out["billing"] == {"answer": "fixed", "p": 0.93, "by": "heads:jdg:billing_type", "atom_ids": []}
    assert {"a2", "a3", "a4"} <= set(line["atom_ids"])
    assert out["heads"]["used"]["hours"]["atom_id"] == "a2"


def test_a_value_the_read_sends_wins_over_the_line_text(qh):
    atoms = [{"id": "a1", "type": "task", "text": "Install cameras, about 2 hours",
              "reads": {"read:hours_stated": ["true", 0.9, 6.0]}}]
    line = qh.quote(_deal(atoms, heads={"asked": True}), word=False)["lines"][0]
    assert line["hours_per_tech"] == 6 and line["by"]["hours"] == "heads:read:hours_stated"


def test_unsure_heads_fall_back_to_regex_and_say_so(qh):
    atoms = [{"id": "a1", "type": "task", "text": "Install cameras, approximately 6 hours with 2 techs",
              "reads": {"read:labor_hours": ["present", 0.4], "read:crew_size": ["_absent", 0.99]}}]
    line = qh.quote(_deal(atoms, heads={"asked": True}), word=False)["lines"][0]
    assert line["hours_per_tech"] == 6 and line["by"]["hours"] == "fallback:regex_head_unsure"
    assert line["techs"] == 2 and line["by"]["techs"] == "fallback:regex_head_unsure"
    assert any(c.startswith("hours not stated on any line the heads read") for c in line["caveats"])


def test_no_heads_sent_is_marked_not_asked(qh):
    atoms = [{"id": "a1", "type": "task", "text": "Install cameras"}]
    out = qh.quote(_deal(atoms), word=False)
    line = out["lines"][0]
    assert line["by"]["hours"] == "fallback:family_median" and line["by"]["techs"] == "fallback:family_usual"
    assert out["heads"]["asked"] is False and out["billing"]["by"] == "none" and out["billing"]["reason"] == "heads not asked"
    assert line["caveats"] == []


def test_labor_hours_answer_is_used_but_marked(qh):
    atoms = [{"id": "a1", "type": "task", "text": "Install cameras: 10 hrs",
              "reads": {"read:labor_hours": ["present", 0.95]}}]
    line = qh.quote(_deal(atoms, heads={"asked": True}), word=False)["lines"][0]
    assert line["by"]["hours"] == "heads:read:labor_hours" and line["hours_per_tech"] == 10
    assert any("read:labor_hours scores below live" in c for c in line["caveats"])


def test_quantity_never_reads_kit_or_left_out_lines(qh):
    atoms = [
        {"id": "a1", "type": "task", "text": "Install 3 cameras at the front door"},
        {"id": "kit", "type": "task", "text": "Install 40 cameras", "deal_kit": True},
        {"id": "rej", "type": "vendor_line_item", "text": "CDW quote: 40 cameras"},
        {"id": "rej2", "type": "task", "text": "40 cameras from the old quote", "left_out": True},
    ]
    d = _deal(atoms)
    d["c3_brief"]["left_out"] = ["rej"]
    out = qh.quote(d, word=False)
    assert out["quantity"]["count"] == 3 and out["quantity"]["atom_id"] == "a1"
    # with nothing but kit / left-out lines, there is no count at all
    out = qh.quote(_deal([a for a in atoms if a["id"] != "a1"], c3_brief={"left_out": ["rej"], "sections": {}}), word=False)
    assert out["quantity"] is None


def test_heads_site_count_and_regex_count_disagree_is_a_flag_not_a_replacement(qh):
    atoms = [{"id": "a1", "type": "task", "text": "Install cameras at 12 stores"}]
    sites = [{"site": "site:a", "atom_ids": ["s1"]}, {"site": "site:b", "atom_ids": ["s2"]}, {"site": "site:c", "atom_ids": ["s3"]}]
    out = qh.quote(_deal(atoms, site_count=3, sites=sites), word=False)
    assert out["sites"] == {"count": 3, "by": "heads:physical_site", "atom_ids": ["s1", "s2", "s3"]}
    assert "documents mention 12 sites, the heads list 3" in out["flags"]


def test_output_shape_matches_the_platform_contract(qh):
    """Platform-infra PR #738 reads exactly these fields (governProposal / headsBilling / day type)."""
    atoms = [{"id": "a1", "type": "task", "text": "Mount 3 cameras overnight, about 4 hours",
              "reads": {"read:hours_stated": ["true", 0.9]}}]
    d = _deal(atoms, site_count=2, sites=[{"site": "s:a", "atom_ids": ["s1"]}, {"site": "s:b", "atom_ids": ["s2"]}],
              heads={"asked": True, "deal": {"jdg:billing_type": ["per_site", 0.85]}})
    d["c3_brief"]["job_details"] = [{"cites": ["a1"], "text": "3 cameras"}]
    out = qh.quote(d, word=False)
    assert out["site_count"] == 2 and out["after_hours"] is True
    assert set(out["billing"]) >= {"answer", "p", "by"} and out["billing"]["answer"] == "per_site"
    line = out["lines"][0]
    assert set(line["by"]) == {"techs", "hours", "rate", "day_type"}
    for v in line["by"].values():
        assert v.startswith(("heads", "fallback:", "rule:")), v
    assert line["day_type"] == "ot" and line["by"]["day_type"] == "rule:after_hours_text" and line["after_hours"] is True
    assert line["atom_ids"] and "a1" in line["atom_ids"] and line["cites"] == ["a1"]
    assert line["source_text"].startswith("Mount 3 cameras")
    # no after-hours wording: "reg" is a marked default, never silent
    plain = qh.quote(_deal([{"id": "b", "type": "task", "text": "Mount cameras"}]), word=False)["lines"][0]
    assert (plain["day_type"], plain["by"]["day_type"], plain["after_hours"]) == ("reg", "fallback:default", False)
