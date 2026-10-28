"""Site keys follow the heads (audit 2026-10-28, wave 12 W4). Synthetic fixtures."""
from __future__ import annotations

from types import SimpleNamespace as NS

from app.core.site_keys_follow_heads import settle, slug_of_line
from app.core.site_provenance_join import join_atoms_to_document_site


def _atom(aid, text, kind="scope_item", keys=(), *, art="art_doc", line=None, flags=(), value=None,
          artifact_type="docx", filename="sow.docx", prov=None):
    loc = {"line_start": line} if line is not None else {}
    return NS(id=aid, raw_text=text, atom_type=kind, entity_keys=list(keys), artifact_id=art,
              review_flags=list(flags), value=dict(value or {}), decision_provenance=prov,
              source_refs=[NS(artifact_type=artifact_type, filename=filename, locator=loc,
                              extraction_method="docx_text_and_ooxml")])


def _site_keys(a):
    return [k for k in a.entity_keys if k.startswith("site:")]


def test_staffing_slot_and_column_keys_no_site_line_backs_are_dropped():
    rate = _atom("a1", "Site: Tech #1 | Country: USA | Billing Type: Per Hour", "commercial_total",
                 ["site:tech_1", "site:usa_cost", "money:94"], filename="Deal Kit.xlsx", artifact_type="xlsx")
    tool = _atom("a2", "Battery powered drill and impact driver", "deliverable", ["site:tech_1"])
    real = _atom("a3", "Install at 40 Harbor Road, Springfield, IL 62701", "physical_site",
                 ["site:40_harbor_road"], line=3)
    types = {"a1": ["commercial_total", 0.9], "a2": ["deliverable", 0.7], "a3": ["physical_site", 0.98]}
    rec = settle([rate, tool, real], types)
    assert _site_keys(rate) == [] and _site_keys(tool) == []
    assert "money:94" in rate.entity_keys, "only site keys are touched"
    assert _site_keys(real) == ["site:40_harbor_road"]
    dropped = {d["site"]: d["lines"] for d in rec["dropped"]}
    assert dropped == {"site:tech_1": 2, "site:usa_cost": 1}, "every dropped key is listed with its line count"


def test_inferred_site_line_needs_the_heads_to_call_it_a_site():
    # A site atom backfilled from an entity key onto a line that only mentions "joining".
    ghost = _atom("g", "Pat said she was going to be joining.", "physical_site", ["site:join_url"],
                  flags=["site_entity_backfill"], value={"inferred": True})
    invite = _atom("i", "Meeting invite: Join URL https://meet.example/abc", "deal_metadata", ["site:join_url"])
    settle([ghost, invite], {"g": ["stakeholder", 0.41], "i": ["deal_metadata", 0.9]})
    assert _site_keys(ghost) == [] and _site_keys(invite) == []
    # The same inferred line, read as a site by the heads, keeps its key.
    town = _atom("t", "Springfield, IL", "physical_site", ["site:springfield_il"],
                 flags=["geo_mention_site"], value={"inferred": True})
    settle([town], {"t": ["physical_site", 0.93]})
    assert _site_keys(town) == ["site:springfield_il"]


def test_one_address_over_three_lines_is_one_key():
    name = _atom("n", "200 Lakeview, Suite 400", "physical_site", ["site:200_lakeview"], line=10)
    street = _atom("s", "200 SW 9th Court", "physical_site", [], line=11,
                   value={"c3_label": {"type": "physical_site", "p": 0.6}})
    city = _atom("c", "Springfield, Illinois 62701", "physical_site", ["site:springfield_illinois_62701"], line=12)
    other = _atom("o", "75 Elm Street", "physical_site", ["site:75_elm_street"], line=20)
    types = {"n": ["physical_site", 0.87], "c": ["physical_site", 0.77], "o": ["physical_site", 0.9]}
    rec = settle([name, street, city, other], types)
    assert _site_keys(name) == _site_keys(street) == _site_keys(city) == ["site:200_lakeview"]
    assert _site_keys(other) == ["site:75_elm_street"], "a line further down is another site"
    assert rec["address_lines_joined"] == 2
    assert "address_lines_joined" in city.review_flags


def test_a_list_of_towns_is_not_joined_into_one_site():
    a = _atom("a", "Springfield, IL", "physical_site", ["site:springfield_il"], line=1)
    b = _atom("b", "Peoria, IL", "physical_site", ["site:peoria_il"], line=2)
    settle([a, b], {"a": ["physical_site", 0.9], "b": ["physical_site", 0.9]})
    assert _site_keys(a) == ["site:springfield_il"] and _site_keys(b) == ["site:peoria_il"]


def test_head_site_heading_without_address_gets_a_key():
    heading = _atom("h", "ORD / Chicago {(may not be required)}", "physical_site", [],
                    value={"c3_label": {"type": "physical_site", "p": 0.91, "act": "authority"}})
    field = _atom("f", "site.address.city: Springfield", "physical_site", [],
                  value={"c3_label": {"type": "physical_site", "p": 0.9}})
    rec = settle([heading, field], {"h": ["physical_site", 0.91], "f": ["physical_site", 0.9]})
    assert _site_keys(heading) == ["site:ord_chicago"]
    assert heading.value["site_id"] == "ord_chicago" and heading.value["name"].startswith("ORD / Chicago")
    assert _site_keys(field) == [], "a table field is not a place's name"
    assert rec["keys_minted_from_head"] == 1
    assert slug_of_line("Location: Building 7 - north dock") == "building_7"


def test_a_quoted_copy_of_a_keyed_site_line_takes_its_key():
    note = _atom("n", "Customer Address: 117 Main Way, Townsville, Washington 98000", "physical_site",
                 ["site:main_way"])
    copy = _atom("c", "Customer Address:  117 Main Way, Townsville, Washington 98000", "physical_site", [],
                 value={"c3_label": {"type": "physical_site", "act": "fill"}})
    settle([note, copy], {"n": ["physical_site", 0.99]})
    assert _site_keys(copy) == ["site:main_way"]


def test_nothing_changes_without_a_heads_read():
    a = _atom("a", "Site: Tech #1", "commercial_total", ["site:tech_1"])
    assert settle([a], {})["by"].startswith("none")
    assert _site_keys(a) == ["site:tech_1"]


def test_provenance_join_skips_conversations():
    sig = _atom("s", "100 Commerce Dr, Springfield, IL 62701", "physical_site", ["site:100_commerce_dr"],
                art="art_mail", artifact_type="email", filename="deal-hs-email-1.eml")
    hdr = _atom("h", "From: a@x.example | To: b@y.example", "deal_metadata", [], art="art_mail",
                artifact_type="email", filename="deal-hs-email-1.eml")
    note_site = _atom("ns", "Customer Address: 9 Oak Way, Springfield, IL", "physical_site", ["site:9_oak_way"],
                      art="art_note", artifact_type="txt", filename="deal-hs-note-22-Intake.txt")
    note_line = _atom("nl", "note_id=22 | author=someone", "deal_metadata", [], art="art_note",
                      artifact_type="txt", filename="deal-hs-note-22-Intake.txt")
    sow_site = _atom("ps", "Site: Springfield Middle School", "physical_site", ["site:springfield_ms"], art="art_sow")
    sow_row = _atom("pr", "Days of Operation | Monday - Friday", "constraint", [], art="art_sow")
    n = join_atoms_to_document_site([sig, hdr, note_site, note_line, sow_site, sow_row])
    assert n == 1
    assert _site_keys(hdr) == [] and _site_keys(note_line) == []
    assert _site_keys(sow_row) == ["site:springfield_ms"], "a per-site document still speaks for its rows"


def test_a_signature_address_the_heads_read_as_a_person_is_no_site():
    sig = _atom("s", "3800 Pecan Trail, Austin, TX 78700", "physical_site", ["site:3800_pecan_trail"],
                artifact_type="email", filename="deal-hs-email-9.eml")
    job = _atom("j", "Install at 12 Dock Street, Springfield, IL 62701", "physical_site", ["site:12_dock_street"])
    rec = settle([sig, job], {"s": ["stakeholder", 0.99], "j": ["physical_site", 0.97]})
    assert _site_keys(sig) == [] and _site_keys(job) == ["site:12_dock_street"]
    assert [d["site"] for d in rec["dropped"]] == ["site:3800_pecan_trail"]


def test_an_unbacked_site_line_makes_no_roster_row_and_no_quote_site():
    from app.core.c3_type_base import quote_sites
    from app.core.orbitbrief_core import build_site_readiness

    sig = _atom("s", "3800 Pecan Trail, Austin, TX 78700", "physical_site", ["site:3800_pecan_trail"],
                value={"id": "SHI-HQ", "site_id": "SHI-HQ", "name": "Reseller HQ"})
    job = _atom("j", "Install at 12 Dock Street, Springfield, IL 62701", "physical_site", ["site:12_dock_street"],
                value={"id": "dock-st", "site_id": "dock-st", "name": "12 Dock Street"})
    for a in (sig, job):
        a.project_id, a.review_status, a.confidence = "p", "auto_accepted", 0.9
    settle([sig, job], {"s": ["stakeholder", 0.99], "j": ["physical_site", 0.97]})
    rows = build_site_readiness(atoms=[sig, job], edges=[])
    keys = {r.get("site") or r.get("site_key") for r in rows.get("sites") or []}
    assert not any("pecan" in str(k) or "shi_hq" in str(k) for k in keys), keys
    assert any("dock" in str(k) for k in keys), keys
    assert [q["site"] for q in quote_sites([sig, job])] == ["site:12_dock_street"]


def test_a_street_line_inside_a_roster_row_is_that_site():
    row = _atom("r", "facility: Northgate | address: 55 Main Rd | city: Springfield", "physical_site",
                ["site:northgate"], flags=["note_site_roster_v1"])
    street = _atom("s", "55 Main Rd", "physical_site", ["site:55_main_rd"])
    other = _atom("o", "70 Oak Rd", "physical_site", ["site:70_oak_rd"])
    settle([row, street, other], {"r": ["physical_site", 0.99], "s": ["physical_site", 0.8], "o": ["physical_site", 0.8]})
    assert _site_keys(street) == ["site:northgate"]
    assert _site_keys(other) == ["site:70_oak_rd"]
