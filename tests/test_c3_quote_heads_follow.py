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
    survey = dict(fam, usual_rate_line="L1 Cabling", usual_price=115, usual_techs={"1": 3})
    b = {"version": "t", "families": {"camera": dict(fam), "av": dict(fam), "survey": survey},
         "descriptions": {"camera": "cams", "av": "tvs", "survey": "surveys"},
         "description_embeddings": [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], "embed_model": "e", "wording_model": None,
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


def test_the_heads_roster_is_the_site_count_and_wording_counts_only_without_one(qh):
    """Owner 2026-10-28: site count comes from the heads' roster when it exists; text heuristics only
    when there is none, and marked."""
    atoms = [{"id": "a1", "type": "task", "text": "Install cameras at 12 stores"}]
    sites = [{"site": "site:a", "atom_ids": ["s1"]}, {"site": "site:b", "atom_ids": ["s2"]}, {"site": "site:c", "atom_ids": ["s3"]}]
    out = qh.quote(_deal(atoms, site_count=3, sites=sites), word=False)
    assert {k: out["sites"][k] for k in ("count", "by", "atom_ids")} == {
        "count": 3, "by": "heads:physical_site", "atom_ids": ["s1", "s2", "s3"]}
    assert not any("documents mention" in f for f in out["flags"])
    out = qh.quote(_deal(atoms, site_count=0, sites=[]), word=False)
    assert out["sites"]["by"] == "rule:sites_said_text (no roster)" and out["sites"]["said"] == 12
    assert any("documents mention 12 sites (rule:sites_said_text (no roster))" in f for f in out["flags"])
    # a declared-count deal head answers before the wording
    out = qh.quote(_deal(atoms, site_count=0, sites=[], heads={"asked": True, "deal": {
        "deal:declared_site_count": ["5", 0.9]}}), word=False)
    assert out["sites"]["by"] == "heads:deal:declared_site_count" and out["sites"]["said"] == 5


def test_per_site_counts_tunnels_and_floor_plans_are_not_site_counts(qh):
    atoms = [{"id": "a1", "type": "task", "text": "Quote for 5 tanks per site, sensors on each"},
             {"id": "a2", "type": "requirement", "text": "10 Site-to-Site IPsec VPN Tunnels"},
             {"id": "a3", "type": "site_infrastructure", "text": "[OCR-recovered text] STOR\n202\nOFFICE\n225 OFFICES"}]
    out = qh.quote(_deal(atoms, site_count=0, sites=[]), word=False)
    assert not (out["quantity"] and out["quantity"]["unit"] == "site"), out["quantity"]
    assert out["sites"]["said"] is None


def test_after_hours_and_lift_flags_respect_negation_and_are_marked(qh):
    atoms = [{"id": "a1", "type": "requirement", "text": "There are no lift requirements and no after-hours work"},
             {"id": "a2", "type": "task", "text": "Install cameras"}]
    out = qh.quote(_deal(atoms), word=False)
    assert out["after_hours"] is False and not any("lift" in f for f in out["flags"])
    atoms = [{"id": "a1", "type": "requirement", "text": "Work must be done after hours; a scissor lift is needed"}]
    out = qh.quote(_deal(atoms), word=False)
    assert out["lines"][0]["by"]["day_type"] == "rule:after_hours_text"
    assert out["sites"]["rule_atom_ids"] == {"after_hours": ["a1"], "height": ["a1"]}
    assert "height / lift access mentioned (rule:height_text)" in out["flags"]


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


def test_the_service_head_picks_the_job_family(qh):
    atoms = [{"id": "a1", "type": "task", "text": "Install cameras across the warehouse"}]
    out = qh.quote(_deal(atoms, heads={"asked": True, "deal": {"deal:primary_service": ["audio_visual", 0.81]}}), word=False)
    assert out["job_type"] == "av" and out["job_type_by"] == "heads:deal:primary_service" and out["job_type_p"] == 0.81
    # unsure (run O: 0.15-0.58 everywhere): recorded, not used
    out = qh.quote(_deal(atoms, heads={"asked": True, "deal": {"deal:primary_service": ["audio_visual", 0.31]}}), word=False)
    assert out["job_type"] == "camera" and out["job_type_by"] == "fallback:embedding_family"
    assert out["service"] == {"answer": "audio_visual", "p": 0.31}
    out = qh.quote(_deal(atoms, heads={"asked": True, "deal": {"deal:primary_service": ["_absent", 0.9]}}), word=False)
    assert out["job_type"] == "camera" and out["job_type_by"] == "fallback:embedding_family"
    # a service with no family in this bundle (wireless -> ap is not in the test bundle) stays the embedding's
    out = qh.quote(_deal(atoms, heads={"asked": True, "deal": {"deal:primary_service": ["wireless", 0.9]}}), word=False)
    assert out["job_type_by"] == "fallback:embedding_family"


def test_chatter_and_questions_cast_no_billing_vote(qh):
    atoms = [{"id": "a1", "type": "small_talk", "text": "we did this for CDW last year",
              "reads": {"read:billing_type": ["t_and_m", 0.99]}},
             {"id": "a2", "type": "open_question", "text": "how do you usually bill these?",
              "reads": {"read:billing_type": ["t_and_m", 0.99]}},
             {"id": "a3", "type": "payment_term", "text": "Fixed fee per site, invoiced on completion",
              "reads": {"read:billing_type": ["fixed", 0.9]}}]
    out = qh.quote(_deal(atoms, heads={"asked": True}), word=False)
    assert out["billing"]["answer"] == "fixed" and out["billing"]["atom_ids"] == ["a3"]


def test_a_company_named_one_is_no_crew(qh):
    atoms = [{"id": "a1", "type": "task", "text": "Access One engineer will meet our technicians on site",
              "reads": {"read:crew_size": ["present", 0.9]}}]
    line = qh.quote(_deal(atoms, heads={"asked": True}), word=False)["lines"][0]
    assert not line["by"]["techs"].startswith("heads:")


def test_the_work_window_head_decides_the_day_type(qh):
    sat = {"id": "a1", "type": "task", "text": "The drop will be installed on a Saturday",
           "reads": {"read:work_window": [["weekend"], {"weekend": 0.91, "_absent": 0.05}]}}
    out = qh.quote(_deal([sat], heads={"asked": True}), word=False)
    assert out["lines"][0]["day_type"] == "ot" and out["lines"][0]["by"]["day_type"] == "heads:read:work_window"
    biz = {"id": "a1", "type": "task", "text": "Work after hours if the store asks",
           "reads": {"read:work_window": [["business_hours"], {"business_hours": 0.9}]}}
    out = qh.quote(_deal([biz], heads={"asked": True}), word=False)
    assert out["lines"][0]["day_type"] == "reg" and out["lines"][0]["by"]["day_type"] == "heads:read:work_window"
    # no head sure: the wording rule, marked
    out = qh.quote(_deal([{"id": "a1", "type": "task", "text": "Work after hours only"}], heads={"asked": True}), word=False)
    assert out["lines"][0]["by"]["day_type"] == "rule:after_hours_text"


def _sites(*texts):
    return [{"site": f"site:s{i}", "text": t, "atom_ids": [f"s{i}"]} for i, t in enumerate(texts)]


def test_a_roster_across_areas_flags_travel_and_never_answers_it(qh):
    """010358: a 101-site WLAN survey across Houston, Austin, Tyler, ... got travel 'unsure'. Roster geography was right
    on 6 of 14 fires on the kits, so it is a flag for a PM, not the answer."""
    atoms = [{"id": "a1", "type": "task", "text": "Install cameras"}]
    sites = _sites("1200 Main St, Houston, TX 77002", "500 Congress Ave, Austin, TX 78701", "Tyler, TX 75701")
    out = qh.quote(_deal(atoms, site_count=3, sites=sites, heads={"asked": True, "deal": {"jdg:travel_scope": ["none", 0.55]}}), word=False)
    t = out["travel"]
    assert (t["answer"], t["p"], t["by"], t["reason"]) == (None, None, "none", "travel head unsure")
    assert t["check"]["rule"] == "rule:roster_geography" and t["check"]["decides"] is False
    assert "not by distance" in t["check"]["note"]
    assert "sites span Houston, Austin, Tyler: check travel (rule:roster_geography)" in out["flags"]
    # a sure travel head answers, and no flag is raised
    out = qh.quote(_deal(atoms, site_count=3, sites=sites, heads={"asked": True, "deal": {"jdg:travel_scope": ["none", 0.9]}}), word=False)
    assert out["travel"] == {"answer": "none", "p": 0.9, "by": "heads:jdg:travel_scope"}
    assert not any("check travel" in f for f in out["flags"])
    # two states, read off the site keys alone
    keyed = [{"site": "site:chicago_il_60675", "atom_ids": ["s1"]}, {"site": "site:new_york_ny_10014", "atom_ids": ["s2"]}]
    out = qh.quote(_deal(atoms, site_count=2, sites=keyed), word=False)
    assert out["travel"]["answer"] is None and out["travel"]["check"]["rule"] == "rule:roster_geography"


def test_one_area_or_no_places_leaves_travel_to_the_head(qh):
    atoms = [{"id": "a1", "type": "task", "text": "Install cameras"}]
    one_city = _sites("1200 Main St, Houston, TX 77002", "5000 Westheimer Rd, Houston, TX 77056")
    out = qh.quote(_deal(atoms, site_count=2, sites=one_city, heads={"asked": True}), word=False)
    assert out["travel"] == {"answer": None, "p": None, "by": "none", "reason": "travel head unsure"}
    # one state, two cities, one ZIP3: compared by ZIP3, a suburb in the same ZIP3 is the same area
    # (Dallas 752 / Richardson 750 would count as two: no distance table, so ZIP3 is the proxy)
    suburb = _sites("Dallas, TX 75201", "University Park, TX 75205")
    assert "check" not in qh.quote(_deal(atoms, site_count=2, sites=suburb), word=False)["travel"]
    # "Virginia Beach" is a city, not the state of Virginia; a sentence ending in a state is no place
    assert qh._roster_places(_sites("INDUSTRIAL PARK VIRGINIA BEACH, VA 23452")) == [("Park Virginia Beach", "VA", "23452")]
    assert qh._roster_places(_sites("The information contained herein, GA", "Bangalore, India")) == []


def test_pc_hours_follow_max_curve_over_the_jobs_total_crew_hours(qh):
    """Max (QUOTE_READINESS.md): 1 PC h up to about 10 h of work, 2 h at 16-24 h, 3-4 h at 36-48 h; big jobs by hand."""
    def run(hours, techs, sites):
        atoms = [{"id": "a1", "type": "task", "text": f"Install cameras, {hours} hours each with {techs} techs",
                  "reads": {"read:hours_stated": ["true", 0.9, float(hours)], "read:crew_size": ["present", 0.9]}}]
        return qh.quote(_deal(atoms, site_count=sites, sites=_sites(*["Houston, TX 77002"] * sites), heads={"asked": True}), word=False)
    out = run(8, 1, 1)
    assert out["pc_hours"] == {"hours": 1, "by": "rule:max_pc_curve", "total_crew_hours": 8, "reason": "8 crew hours of work"}
    assert out["total"] == {"crew_hours": 8, "sites": 1, "by": "line"}
    # the line is one site's work: three roster sites x 2 techs x 4 h = 24 crew hours -> 2 PC h
    out = run(4, 2, 3)
    assert out["total"] == {"crew_hours": 24, "sites": 3, "by": "rule:per_site_x_sites (heads:physical_site)"}
    assert out["pc_hours"]["hours"] == 2
    assert run(8, 2, 2)["pc_hours"]["hours"] == 3  # 32 h
    big = run(8, 2, 4)  # 64 h: past the curve, set by hand
    assert big["pc_hours"]["hours"] is None and big["pc_hours"]["by"] == "rule:max_pc_curve"
    assert any(f.startswith("PC hours: 64 crew hours is past the PC curve") for f in big["flags"])


def test_a_wireless_survey_takes_max_wifi_survey_line_not_the_cabling_one(qh):
    """The survey family learned L1 Cabling from three cabling / site-assessment kits; every Wi-Fi survey kit
    (010322, 010225, 010142) was priced on the L3 engineer / Wi-Fi survey line."""
    audit = {"asked": True, "deal": {"deal:primary_service": ["audit", 0.9]}}
    wlan = [{"id": "a1", "type": "scope_item", "text": "Perform a predictive and passive WLAN survey with Ekahau at each clinic"}]
    out = qh.quote(_deal(wlan, deal_name="010358 - Houston clinics", heads=audit), word=False)
    line = out["lines"][0]
    assert out["job_type"] == "survey"
    assert (line["rate_line"], line["by"]["rate"], line["price"]) == ("L3 Networking", "rule:survey_kind", 150)
    assert "PS-L3-ENG-LABOR-ONSITE" in line["why_rate"] and "a1" in line["atom_ids"]
    assert any(c.startswith("wireless survey: check the PS-L3-ENG-LABOR-ONSITE rate") for c in line["caveats"])
    # a cabling survey, or a Wi-Fi survey someone else does, keeps the family's line
    cable = [{"id": "a1", "type": "scope_item", "text": "Survey the cabling pathways and document each MDF/IDF"}]
    assert qh.quote(_deal(cable, heads=audit), word=False)["lines"][0]["by"]["rate"] == "fallback:family_usual"
    other = [{"id": "a1", "type": "scope_item", "text": "Wireless RF, predictive, or validation surveys are addressed under a separate SOW"},
             {"id": "a2", "type": "scope_item", "text": "Assess the campus network and deploy Wi-Fi hardware"}]
    assert qh.quote(_deal(other, heads=audit), word=False)["lines"][0]["rate_line"] == "L1 Cabling"
    # the deal name alone says it ("WLAN survey")
    out = qh.quote(_deal([{"id": "a1", "type": "scope_item", "text": "101 sites, report per site"}],
                         deal_name="010358 - Houston WLAN survey", heads=audit), word=False)
    assert out["lines"][0]["by"]["rate"] == "rule:survey_kind"
    # a sure level head still decides the level
    wlan.append({"id": "a2", "type": "requirement", "text": "An L2 tech is fine for the walk", "reads": {"read:tech_level": ["present", 0.95]}})
    line = qh.quote(_deal(wlan, heads=audit), word=False)["lines"][0]
    assert line["rate_line"] == "L2 Networking" and line["by"]["rate"] == "heads:read:tech_level"
