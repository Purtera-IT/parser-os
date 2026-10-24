"""c3_record.provenance and c3_record.view: the quoting cut, the planning end, the heads per view, the
stage clamp and the view tag, shared by training's views and the compile's read."""
from c3_record import provenance as prov
from c3_record import view as V

DAY = 86400.0
T0 = 1_780_000_000.0


def _doc(direction="", kind="", message=False, carrier=None, carrier_at=0.0, role="", filename="", sender=""):
    return {"direction": direction, "sender": sender, "role": role, "kind": kind, "carrier": carrier,
            "carrier_at": carrier_at, "message": message, "filename": filename}


def _atom(key, doc, text, at, kind="email", side=""):
    return {"label_key": key, "atom_id": key, "doc_id": doc, "text": text, "entered_at": at, "doc_kind": kind,
            "speaker_side": side, "section": ""}


def _deal():
    atoms = [
        _atom("a1", "rfp", "We need 40 access points installed at 3 sites.", T0, kind="email"),
        _atom("a2", "sow", "Provider will install 40 access points.", T0 + 2 * DAY, kind="sow_draft", side="ours"),
        _atom("a3", "sow", "Price: $12,000 per site.", T0 + 2 * DAY, kind="sow_draft", side="ours"),
        _atom("a4", "late", "Customer: can you also cover the warehouse?", T0 + 5 * DAY, kind="email"),
        _atom("k1", "kit", "Labor | $95/hr", T0 + DAY, kind="deal_kit", side="ours"),
    ]
    docinfo = {"rfp": _doc("inbound", "email", True, sender="buyer@example.com"),
               "sow": _doc("purtera_outbound", "sow_draft", role="SOW provided by PurTera-IT"),
               "late": _doc("inbound", "email", True, sender="buyer@example.com"),
               "kit": _doc("internal", "deal_kit")}
    return atoms, docinfo


def test_our_sow_sets_the_cut_and_is_never_an_input():
    atoms, docinfo = _deal()
    P = prov.decide(atoms, docinfo, {})
    # Our Deal Kit (internal, a day in) is our first paper: it cuts before the SOW goes out.
    assert P.cut == T0 + DAY and P.doc["sow"]["ours"] and P.doc["kit"]["ours"]
    keep = prov.quoting_inputs(P, atoms, exclude=prov.pricing_workbook_keys(P, atoms))
    assert [atoms[i]["label_key"] for i in keep] == ["a1"]


def test_as_of_moves_the_cut_earlier_never_later():
    atoms, docinfo = _deal()
    P = prov.decide(atoms, docinfo, {})
    assert prov.quoting_inputs(P, atoms, as_of=T0 + 30 * DAY) == [0]
    P = prov.decide(atoms, docinfo, {})
    assert prov.quoting_inputs(P, atoms, as_of=T0 - DAY) == [] and P.cut == T0 - DAY


def test_our_priced_outbound_message_sets_the_cut():
    atoms = [_atom("c1", "m1", "We have 12 stores to refresh.", T0),
             _atom("o1", "m2", "Please see the attached quote: $4,500 per store.", T0 + DAY),
             _atom("c2", "m3", "Thanks, reviewing.", T0 + 2 * DAY)]
    docinfo = {"m1": _doc("inbound", "email", True, sender="b@example.com"),
               "m2": _doc("purtera_outbound", "email", True, sender="rep@purtera-it.com"),
               "m3": _doc("inbound", "email", True, sender="b@example.com")}
    P = prov.decide(atoms, docinfo, {})
    assert P.cut == T0 + DAY and "o1" in P.ours_atoms
    assert prov.quoting_inputs(P, atoms) == [0]


def test_owner_ruling_theirs_keeps_a_document_an_input():
    d = "art_4e0b98f29cdd4175"
    atoms = [_atom("x1", d, "Kiosk install steps", T0, kind="sow_draft", side="ours")]
    P = prov.decide(atoms, {d: _doc("purtera_outbound", "sow_draft", role="provided by PurTera-IT")}, {})
    assert not P.doc[d]["ours"] and prov.is_input(P, atoms[0])


def test_a_priced_line_to_a_supplier_is_our_cost_side_not_the_quote():
    atoms = [_atom("c1", "m1", "We need two techs.", T0),
             _atom("s1", "m2", "What is your rate? We can do $80/hr.", T0 + DAY)]
    docinfo = {"m1": _doc("inbound", "email", True, sender="b@example.com"),
               "m2": _doc("purtera_outbound", "email", True, sender="rep@purtera-it.com")}
    to_of = {"s1": [{"email": "ops@subco-example.com"}]}
    P = prov.decide(atoms, docinfo, {}, to_of=to_of, suppliers={"subco-example.com"})
    assert "s1" in P.to_supplier and P.cut == float("inf")


def test_heads_per_view():
    keys = ["col:label_type", "read:co_action", "read:sow_section", "deal:primary_service", "read:task_owner_role",
            "read:needed_by"]
    assert V.heads("quoting", keys) == ["col:label_type", "read:co_action", "read:sow_section", "deal:primary_service"]
    assert V.heads("planning", keys) == ["col:label_type", "read:co_action", "read:task_owner_role", "read:needed_by"]
    assert V.heads("", keys) == keys
    assert V.head_scale("pre_quote", keys) == {"read:task_owner_role": 0.25, "read:needed_by": 0.25}
    assert V.head_scale("planning", keys) == {"read:sow_section": 0.0, "deal:primary_service": 0.0}
    assert V.runs_quote("quoting") and not V.runs_quote("planning")
    assert V.of_run("2026-10-01T00:00:00Z") == "quoting" and V.of_run(None) == "planning"


def test_stage_clamp_and_view_tag():
    assert V.clamp_stage("quoted", "pre_quote", crm=False, before_cut=False) == "scoping"
    assert V.clamp_stage("contracting", "quoting", crm=True, before_cut=False) == "quoted"
    assert V.clamp_stage("contracting", "planning", crm=True, before_cut=False) == "contracting"
    assert V.clamp_stage("quoted", "planning", crm=False, before_cut=True) == "scoping"
    assert V.clamp_stage("", "planning", crm=False, before_cut=False) == "quoted"
    assert V.clamp_stage("quoted", "", crm=False, before_cut=False) == "quoted"
    rows = [{"entered_at": T0, "stage_at": "quoted"}, {"entered_at": T0 + 3 * DAY, "stage_at": "contracting"}]
    V.stamp(rows, "planning", crm=False, cut=T0 + DAY)
    assert [r["stage_at"] for r in rows] == ["scoping", "contracting"] and {r["view"] for r in rows} == {"planning"}
    assert V.tag("pre_quote") == "view:quoting"


def test_planning_end_is_our_kickoff_after_the_deal_is_won():
    rows = [{"text": "Once approved, we'll schedule the installation.", "entered_at": T0 + DAY, "doc_kind": "email",
             "speaker_side": "ours"},
            {"text": "Kickoff call is set; the install is scheduled for Monday.", "entered_at": T0 + 9 * DAY,
             "doc_kind": "email", "speaker_side": "ours"},
            {"text": "The techs are onsite.", "entered_at": T0 + 12 * DAY, "doc_kind": "email", "speaker_side": "ours"},
            {"text": "Kickoff call notes", "entered_at": T0 + 10 * DAY, "doc_kind": "email", "speaker_side": "theirs"}]
    assert V.planning_end(rows, won_at=T0 + 5 * DAY) == T0 + 9 * DAY
    assert V.planning_end(rows, won_at=None) == float("inf")              # not won: no end
    assert V.planning_end(rows, timeline_marks=[(T0 + 10 * DAY, "contracting")]) == T0 + 12 * DAY
    keep = V.planning_inputs([dict(r, label_key=str(i), doc_id="d") for i, r in enumerate(rows)], T0 + 9 * DAY)
    assert keep == [0]


def test_the_view_tag_is_an_input_only_for_a_base_trained_on_it(tmp_path):
    import torch
    from ml.c3.data import DealExample, featurize
    from ml.c3.gates import load_checkpoint, save_checkpoint
    from ml.c3.loop import SMALL
    from ml.c3.model import C3Model
    from ml.c3.schema import load_schema
    from dataclasses import replace

    schema = load_schema()
    rows = [{"key": "a", "text": "We need 4 APs.", "entered_at": "2026-05-01T00:00:00Z", "doc_id": "d",
             "doc_kind": "email", "stage_at": "scoping", "view": "quoting"},
            {"key": "b", "text": "Install by June.", "entered_at": "2026-05-02T00:00:00Z", "doc_id": "d",
             "doc_kind": "email", "stage_at": "scoping", "view": "quoting"}]
    batch = featurize(DealExample.from_dict({"deal_id": "x", "atoms": rows}), schema)
    assert batch.inputs()["view"] == ["quoting", "quoting"]
    torch.manual_seed(0)
    old = C3Model(schema, replace(SMALL, tag_dropout=0.0)).eval()
    inp = batch.inputs()
    with torch.no_grad():
        a = old.encoder.embed_atoms(inp)
        b = old.encoder.embed_atoms(dict(inp, view=["planning", "planning"]))
    assert torch.equal(a, b)                       # an older base never reads the tag
    old.cfg.view_tag = True
    with torch.no_grad():
        c = old.encoder.embed_atoms(dict(inp, view=["planning", "planning"]))
    assert not torch.equal(a, c)                   # hashed like stage: once trained on it
    path = tmp_path / "ck.pt"
    save_checkpoint(path, old, step=1, schema_sha256="s")
    fresh = C3Model(schema, replace(SMALL, tag_dropout=0.0))
    load_checkpoint(path, fresh, schema_sha256="s")
    assert fresh.cfg.view_tag is True


def test_a_leading_undated_document_arrives_at_the_deals_latest_time():
    from c3_record import document_times

    docs = [{"doc_id": "old", "kind": "file", "authored_at": "2020-01-01T00:00:00Z", "pos": 0},
            {"doc_id": "m1", "kind": "email", "sent_at": "2026-05-01T00:00:00Z", "authored_at": "2026-05-01T00:00:00Z",
             "pos": 1},
            {"doc_id": "f", "kind": "file", "authored_at": "2026-05-02T00:00:00Z", "pos": 2},
            {"doc_id": "m2", "kind": "email", "sent_at": "2026-06-01T00:00:00Z", "authored_at": "2026-06-01T00:00:00Z",
             "pos": 3}]
    got = document_times(docs)
    assert got["old"] == "2026-06-01T00:00:00Z"      # no evidence: the latest time, never the first
    assert got["f"] == "2026-05-01T00:00:00Z"        # after m1 in reading order: m1's arrival
    assert got["m2"] == "2026-06-01T00:00:00Z"
    files = [{"doc_id": "a", "kind": "file", "authored_at": "2026-01-01T00:00:00Z"},
             {"doc_id": "b", "kind": "file", "authored_at": "2026-03-01T00:00:00Z"}]
    assert set(document_times(files).values()) == {"2026-03-01T00:00:00Z"}
    assert document_times([{"doc_id": "x", "kind": "file"}]) == {}


def test_our_notes_restatement_of_their_asks_is_theirs():
    """Owner Q66 (e): in our own priced note, the unpriced lines of a "<them> confirmed:" / "requested:" block
    are the other side's words (inputs); the BOM, our own "(you) noted:" block and the takeaways stay ours."""
    texts = ["SOW JSON", "BOM: 4 x camera $250 each", "Pat confirmed:", "Install at 3 sites", "Weekend work only",
             "Lee requested:", "Labels on every cable", "Price per site $1,200", "Chase (you) noted:",
             "Check lift access", "Key Takeaways", "Send the quote Friday", "Sam (you) confirmed:", "Our crew of 2"]
    atoms = [_atom(f"n{i}", "note1", t, T0, kind="note", side="ours") | {"order": i} for i, t in enumerate(texts)]
    atoms.append(_atom("c1", "m1", "We need cameras at 3 sites.", T0 - DAY))
    docinfo = {"note1": _doc("internal", "crm_note", True, sender="chase@purtera-it.com"),
               "m1": _doc("inbound", "email", True, sender="pat@example.com")}
    P = prov.decide(atoms, docinfo, {})
    assert P.their_asks == {"n3", "n4", "n6"}
    assert {"n1", "n7", "n9", "n11", "n13"} <= P.ours_atoms
    assert all(prov.is_input(P, a) for a in atoms if a["label_key"] in P.their_asks)
