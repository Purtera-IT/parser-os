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
