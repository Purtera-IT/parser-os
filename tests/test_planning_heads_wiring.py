"""Run O planning-heads wiring (audit 2026-10-28, run 8). Synthetic deals only."""
from types import SimpleNamespace as NS

from app.core import c3_type_base


def test_the_quote_is_sent_the_service_head(monkeypatch):
    seen = {}
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    monkeypatch.setattr(c3_type_base, "_DEAL", {"reads": {}, "view": "planning",
                                                "deal": {"deal:primary_service": ["imac", 0.83],
                                                         "jdg:billing_type": ["t_and_m", 0.9]}})
    monkeypatch.setattr(c3_type_base, "_request", lambda p, b, timeout: seen.update(body=b) or {"lines": []})
    atoms = [NS(id="w1", atom_type="task", review_flags=[], value={}, source_refs=[],
                raw_text="Unbox and connect 40 laptops at the clinic")]
    c3_type_base.quote(atoms, {"deal": {}, "sections": {"scope_of_work": [{"id": "w1"}]}})
    assert seen["body"]["deal"]["heads"]["deal"]["deal:primary_service"] == ["imac", 0.83]


def test_a_partners_signed_sow_is_not_our_paper_and_sets_no_cut():
    """010300: the incumbent partner's signed PSOW, uploaded by our seller on a note, read as our sow_draft
    by its "SOW" heading; its arrival cut the quoting view to 41 of 559 lines."""
    from c3_record import provenance as prov

    T0, DAY = 1_780_000_000.0, 86400.0
    lines = [
        {"label_key": "c1", "atom_id": "c1", "doc_id": "m1", "text": "Can you take over the phone rollout?",
         "entered_at": T0, "doc_kind": "email", "speaker_side": "", "section": ""},
        {"label_key": "p1", "atom_id": "p1", "doc_id": "psow", "text": "Provider Name: NewBold LLC",
         "entered_at": T0 + DAY, "doc_kind": "", "speaker_side": "", "section": "STATEMENT OF WORK"},
        {"label_key": "p2", "atom_id": "p2", "doc_id": "psow", "text": "Customer: Dentistry Example | Fee: $550 per site",
         "entered_at": T0 + DAY, "doc_kind": "", "speaker_side": "", "section": "STATEMENT OF WORK"},
        {"label_key": "c2", "atom_id": "c2", "doc_id": "m2", "text": "Here is what we need at each office.",
         "entered_at": T0 + 2 * DAY, "doc_kind": "email", "speaker_side": "", "section": ""}]
    docinfo = {"m1": {"direction": "inbound", "kind": "email", "message": True, "sender": "b@example.com"},
               "m2": {"direction": "inbound", "kind": "email", "message": True, "sender": "b@example.com"},
               "psow": {"direction": "internal", "kind": "sow_draft", "filename": "PSOW SIGNED.pdf"}}
    P = prov.decide(lines, {k: dict(v) for k, v in docinfo.items()}, {})
    assert P.doc["psow"]["ours"]  # before: our sow_draft, the cut
    third = c3_type_base._third_party_papers(lines)
    assert third == {"psow": "NewBold LLC"}
    docinfo["psow"]["third_party"] = third["psow"]
    P = prov.decide(lines, docinfo, {})
    assert not P.doc["psow"]["ours"] and P.cut == float("inf")
    assert [lines[i]["label_key"] for i in prov.quoting_inputs(P, lines)] == ["c1", "p1", "p2", "c2"]


def test_our_own_sow_naming_us_as_provider_stays_ours():
    lines = [{"label_key": "s1", "atom_id": "s1", "doc_id": "sow", "text": "Provider: PurTera-IT LLC", "doc_kind": ""},
             {"label_key": "s2", "atom_id": "s2", "doc_id": "sow", "text": "Customer: Example Corp", "doc_kind": ""}]
    assert c3_type_base._third_party_papers(lines) == {}


def test_a_planning_compile_quotes_only_from_lines_before_our_first_pricing(monkeypatch):
    """Owner rule: the quote's facts come from the pre-quote record; our SOW, the PO and post-award mail
    feed delivery (planning brief, SowSmith), never the quote."""
    seen = {}
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    monkeypatch.setattr(c3_type_base, "_DEAL", {"reads": {}, "deal": {}, "view": "planning",
                                                "quote_out_why": {"sow1": "ours", "po1": "after_cut"}})
    monkeypatch.setattr(c3_type_base, "view_name", lambda: "planning")
    monkeypatch.setattr(c3_type_base, "_request", lambda p, b, timeout: seen.update(body=b) or {"lines": []})
    atoms = [NS(id=i, atom_type="task", review_flags=[], value={}, source_refs=[], raw_text=t)
             for i, t in (("rq1", "Please quote 12 AP swaps at the clinic"), ("sow1", "Provider will install 12 APs; fixed fee $9,800"),
                          ("po1", "PO 4471 issued for the AP install"))]
    c3_type_base.quote(atoms, {"deal": {}, "sections": {"scope_of_work": [{"id": a.id} for a in atoms]}})
    rows = {r["id"]: r for r in seen["body"]["deal"]["atoms"]}
    assert not rows["rq1"].get("left_out")
    assert rows["sow1"]["left_out"] and rows["sow1"]["out_of_view"] == "ours"
    assert rows["po1"]["left_out"] and rows["po1"]["out_of_view"] == "after_cut"
    monkeypatch.setenv("SOWSMITH_C3_QUOTE_PREQUOTE_ONLY", "off")
    assert c3_type_base.out_of_quote() == {}


def test_set_aside_lines_are_never_typed_by_the_llm(monkeypatch):
    kinds = {"k1": "deal_kit", "c1": "estimating_calc", "e1": "email", "o1": "email"}
    monkeypatch.setattr(c3_type_base, "doc_kind", lambda a, sha=None: kinds[a.id])
    monkeypatch.setattr(c3_type_base, "_DEAL", {"view_out": {"o1"}})
    monkeypatch.setattr(c3_type_base, "view_name", lambda: "quoting")
    a = {i: NS(id=i) for i in kinds}
    assert c3_type_base.set_aside(a["k1"]) == ""  # the kit stays typed (read as structure downstream)
    assert c3_type_base.set_aside(a["c1"]) == "estimating_calc"
    assert c3_type_base.set_aside(a["o1"]) == "out_of_view"
    assert c3_type_base.set_aside(a["e1"]) == ""
    # a planning read keeps its out-of-view lines typed (the planning brief may use them)
    monkeypatch.setattr(c3_type_base, "view_name", lambda: "planning")
    assert c3_type_base.set_aside(a["o1"]) == ""


def test_a_shared_read_reports_the_document_reads_time(monkeypatch):
    monkeypatch.setattr(c3_type_base, "_DOC_STATUS", {"state": "read", "read_ms": 4321.0})
    monkeypatch.setattr(c3_type_base, "_DEAL", {"types": {"a1": ["task", 0.9]}})
    monkeypatch.setattr(c3_type_base, "_STATUS", {"state": "off"})
    out = c3_type_base._remote("d", [{}], ["a1"], [0])
    assert out == {0: ("task", 0.9)}
    assert c3_type_base._STATUS["read_ms"] == 4321.0 and c3_type_base._STATUS["read_reused_from"] == "document_job_scope"


def test_a_date_is_not_a_place():
    from app.core.site_keys_follow_heads import is_place_key

    assert not is_place_key("site:august_17th", "Install on August 17th")
    assert not is_place_key("site:17th_of_march", "the 17th of March")
    assert not is_place_key("site:oct_3_2026", "Oct 3 2026")
    assert is_place_key("site:augusta_hq", "Augusta HQ, 12 Main St")
    assert is_place_key("site:may_street_depot", "May Street Depot")


def test_a_partial_address_key_is_the_same_quote_site():
    roster = {
        "site:basking_ridge_nj": {"site": "site:basking_ridge_nj", "keys": ["site:basking_ridge_nj"], "atom_ids": ["a1"]},
        "site:basking_ridge_nj_07920": {"site": "site:basking_ridge_nj_07920", "keys": ["site:basking_ridge_nj_07920"], "atom_ids": ["a2"]},
        "site:6046_n_beltline_road": {"site": "x", "keys": ["site:6046_n_beltline_road"], "atom_ids": ["a3"]},
        "site:6046_n_beltline_road_irving_tx_75038": {"site": "y", "keys": ["site:6046_n_beltline_road_irving_tx_75038"], "atom_ids": ["a4"]},
        "site:30_river_rd_essex_jct_vt_05453": {"site": "z", "keys": ["site:30_river_rd_essex_jct_vt_05453"], "atom_ids": ["a5"]},
        "site:38_river_rd_essex_jct_vt_05453": {"site": "w", "keys": ["site:38_river_rd_essex_jct_vt_05453"], "atom_ids": ["a6"]},
        "site:essex_jct": {"site": "v", "keys": ["site:essex_jct"], "atom_ids": ["a7"]},
    }
    c3_type_base._fold_contained(roster)
    assert sorted(roster) == ["site:30_river_rd_essex_jct_vt_05453", "site:38_river_rd_essex_jct_vt_05453",
                              "site:6046_n_beltline_road_irving_tx_75038", "site:basking_ridge_nj_07920"]
    assert roster["site:basking_ridge_nj_07920"]["atom_ids"] == ["a2", "a1"]


def test_the_quote_says_which_fields_a_pm_confirms():
    q = {"billing": {"answer": "t_and_m", "p": 0.99, "by": "heads:read:billing_type"},
         "job_type": "cabling", "job_type_by": "fallback:embedding_family",
         "sites": {"count": 3, "by": "heads:physical_site"},
         "lines": [{"by": {"techs": "heads:read:crew_size", "hours": "fallback:family_median", "rate": "fallback:account_profile"}}]}
    out = c3_type_base._pm_confirms(q)["pm_confirms"]
    assert "unproven" in out["billing"] and "embedding" in out["job_type"]
    assert out["hours"].startswith("not from a head") and "crew_size" in out["techs"]
    assert "sites" not in out  # the type head (live) read the roster
    assert c3_type_base._pm_confirms({"degraded": True}) == {"degraded": True}


def test_a_quantity_line_is_not_overwritten_into_a_site_by_an_unsure_read():
    ap = NS(value={"quantity": 200, "unit": "each", "normalized_item": "access point"}, entity_keys=["quantity:200"],
            atom_type="bom_line")
    assert c3_type_base.overwrite_bar(ap, "physical_site", 0.5) == 0.85
    assert c3_type_base.overwrite_bar(ap, "task", 0.5) == 0.5          # a task keeps its quantity
    assert c3_type_base.overwrite_bar(ap, "bom_line", 0.5) == 0.5      # the same type
    plain = NS(value={"text": "Our site is in Plano"}, entity_keys=[], atom_type="scope_item")
    assert c3_type_base.overwrite_bar(plain, "physical_site", 0.5) == 0.5
