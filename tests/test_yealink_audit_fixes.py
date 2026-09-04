"""Deal 010300 (CDW / Dentistry For Children) audit, each finding pinned to its fix.

Every fixture below is the LIVE shape from the 2026-09-03 envelope, not an
idealised one -- live records win over fixtures.
"""
from __future__ import annotations

from pathlib import Path

from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


# --- Y-03: people from signature blocks and "Name Title" list lines ----------

def test_signature_cluster_yields_a_full_contact_record():
    from app.parsers.signature_block import people_from_signature_lines
    lines = [
        "All the best,",
        "",
        "Carl Painter| Sr. Account Manager| carlpai@cdw.com|<mailto:carlpai@cdw.com%7C> CDW",
        "Direct:847-968-9740 | Mobile: 847-363-7372",
        "Toll free :(877) 434-6268 | Fax:847-465-6845",
        "200 N Milwaukee Ave|Vernon Hills, IL 60061",
    ]
    people = people_from_signature_lines(lines)
    assert len(people) == 1, people
    p = people[0]
    assert p["name"] == "Carl Painter"
    assert p["email"] == "carlpai@cdw.com"
    assert p["role"] == "Sr. Account Manager"
    assert p["phone"] == "847-968-9740"
    assert p["phones"]["Mobile"] == "847-363-7372"


def test_stacked_signature_reads_title_and_phone_on_following_lines():
    from app.parsers.signature_block import people_from_signature_lines
    lines = ["Thanks,", "", "Patrick Kelly", "", "Account Executive", "", "patrick@purtera-it.com", "", "770.769.7311", "", "PurTera-IT.com<https://x>"]
    people = people_from_signature_lines(lines)
    assert people and people[0]["name"] == "Patrick Kelly"
    assert people[0]["role"] == "Account Executive"
    assert people[0]["email"] == "patrick@purtera-it.com"
    assert people[0]["phone"] == "770.769.7311"


def test_a_bare_name_followed_by_prose_is_not_a_signature():
    from app.parsers.signature_block import people_from_signature_lines
    assert people_from_signature_lines(["Carl Painter", "asked us to move the date to Friday.", "The site opens at 7."]) == []


def test_name_role_list_line_splits_name_from_title():
    from app.parsers.signature_block import name_and_role_from_list_line
    assert name_and_role_from_list_line("Rhonda Sharp Professional Services Manager") == {
        "kind": "person", "name": "Rhonda Sharp", "role": "Professional Services Manager"}
    assert name_and_role_from_list_line("Jacob Long Project Manager Phone and ITAD")["name"] == "Jacob Long"
    assert name_and_role_from_list_line("Install the clocks at all ten schools") is None
    assert name_and_role_from_list_line("Carl Painter") is None


def test_signature_cluster_lines_are_marked_as_chrome():
    from app.parsers.signature_block import signature_line_indexes
    lines = ["Thanks,", "", "Patrick Kelly", "", "Account Executive", "", "patrick@purtera-it.com", "", "770.769.7311", "Next steps below."]
    idx = signature_line_indexes(lines)
    assert 2 in idx and 4 in idx and 6 in idx and 8 in idx
    assert 0 not in idx and 9 not in idx


def test_salutation_and_sign_off_are_not_a_signature():
    from app.parsers.signature_block import people_from_signature_lines
    assert people_from_signature_lines(["Hi Hiran,", "We need to swap 12 access points.", "Thanks,", "Chase", "Office: 555-123-4567"]) == []


def test_nameless_or_party_phrase_stakeholders_are_dropped():
    from app.core.atom_substance_gate import drop_contextless_stakeholders
    atoms = [
        _atom(AtomType.stakeholder, "Each Party will appoint a person", kind="person", name=None, role="approver"),
        _atom(AtomType.stakeholder, "The Buyer", kind="person", name="The Buyer", role="approver"),
        _atom(AtomType.stakeholder, "Step 5: Appoint a contact person for each party", role="contact person"),
        _atom(AtomType.stakeholder, "Carl Painter | Sr. Account Manager", kind="person", name="Carl Painter", role="Sr. Account Manager", email="carlpai@cdw.com"),
        _atom(AtomType.stakeholder, "Jacob Long Project Manager Phone and ITAD", kind="person", name="Jacob Long", role="Project Manager Phone and ITAD"),
    ]
    atoms.append(_atom(AtomType.stakeholder, "both parties | contact person", kind="party", name="both parties", role="contact person"))
    kept, dropped = drop_contextless_stakeholders(atoms)
    assert [a.value.get("name") for a in dropped] == [None, "The Buyer", None, "both parties"]
    assert [a.value.get("name") for a in kept] == ["Carl Painter", "Jacob Long"]


def test_party_page_sites_become_party_addresses():
    from app.core.party_address_veto import veto_party_page_sites
    def _p(atom_type, text, page, **value):
        a = _atom(atom_type, text, **value)
        a.source_refs = [SourceRef(id=f"s_{page}_{abs(hash(text))}", artifact_id="d1", artifact_type="pdf", filename="x.pdf", locator={"page": page}, extraction_method="t", parser_version="t")]
        return a
    site = _p(AtomType.physical_site, "200 N. Milwaukee Ave., Vernon Hills, IL 60061", 6, kind="physical_site")
    site.entity_keys = ["site:vernon_hills_il_60061"]
    hq = _p(AtomType.physical_site, "2970 Brandywine Rd, Suite 200, Atlanta, GA 30341", 4, kind="physical_site")
    hq.entity_keys = ["site:atlanta_ga_30341"]
    sig = _p(AtomType.signatory, "CDW Technologies LLC: By: Mike Murphy", 6)
    assert veto_party_page_sites([site, hq, sig]) == 1
    assert site.atom_type == AtomType.deal_metadata and site.value["kind"] == "party_address" and site.entity_keys == []
    assert hq.atom_type == AtomType.physical_site and hq.entity_keys == ["site:atlanta_ga_30341"]
    assert veto_party_page_sites([site, hq, sig]) == 0, "idempotent: it runs before dedup and again before the join"


def test_party_address_does_not_pollute_the_real_site_through_dedup():
    """The veto runs before the location-bucket merge, so the HQ keeps its
    city, state and ZIP and never inherits the party's aliases."""
    from app.core.party_address_veto import veto_party_page_sites
    from app.core.semantic_dedup import semantic_dedup_atoms
    def _site(text, page, **v):
        a = _atom(AtomType.physical_site, text, kind="physical_site", **v)
        a.source_refs = [SourceRef(id=f"s{page}{abs(hash(text))}", artifact_id="d1", artifact_type="pdf", filename="x.pdf", locator={"page": page}, extraction_method="t", parser_version="t")]
        return a
    hq = _site("facility: HQ | address: 2970 Brandywine Rd, STE 200", 4, id="HQ", site_id="HQ", name="HQ", street_address="2970 Brandywine Rd, STE 200", city="Atlanta", state="GA", zip="30641")
    hq.entity_keys = ["site:hq"]
    party = _site("200 N. Milwaukee Ave., Vernon Hills, IL 60061", 6, id="VERNON", site_id="VERNON", name="Vernon Hills Office", street_address="200 N. Milwaukee Ave.", city="Vernon Hills", state="IL", zip="60061", aliases=["vernon hills"])
    party.entity_keys = ["site:vernon_hills_il_60061"]
    sig = _atom(AtomType.signatory, "CDW Technologies LLC: By: Mike Murphy")
    sig.source_refs = [SourceRef(id="ssig", artifact_id="d1", artifact_type="pdf", filename="x.pdf", locator={"page": 6}, extraction_method="t", parser_version="t")]
    atoms = [hq, party, sig]
    veto_party_page_sites(atoms)
    out = semantic_dedup_atoms(atoms)
    sites = [a for a in out if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1 and sites[0].value["city"] == "Atlanta" and sites[0].value["zip"] == "30641"
    assert "vernon hills" not in (sites[0].value.get("aliases") or [])


def test_quoted_author_display_name_resolves_through_the_signature():
    from app.core.orbitbrief_envelope import _annotate_quoted_message_scope
    timeline = {"transitions": [{"label": "Open- Awaiting Scope", "order": 1}, {"label": "Submitted for Quoting", "order": 2}]}
    docs = [{"artifact_id": "reply", "filename": "reply.eml",
             "deal_stage": {"stage_at_arrival": "Submitted for Quoting", "admissible_for": "label"},
             "reader_scope": {"deal_kit": {"visible": False, "why": "label"}, "sow": {"visible": False, "why": ""}, "orbitbrief": {"visible": True, "why": ""}, "atlas": {"visible": True, "why": ""}}}]
    env = {"atoms": [
        {"artifact_id": "reply", "atom_type": "deal_metadata", "text": "There are about 170+ sites and growing across the US.",
         "structured": {"quoted": True, "author": "Carl Painter Jr", "message_index": 3}},
        {"artifact_id": "reply", "atom_type": "stakeholder", "text": "Carl Painter | Sr. Account Manager | carlpai@cdw.com",
         "structured": {"kind": "person", "name": "Carl Painter", "email": "carlpai@cdw.com", "message_index": 3, "quoted": True, "author": "Carl Painter Jr"}},
    ]}
    assert _annotate_quoted_message_scope(env, docs, timeline) == 2
    assert env["atoms"][0]["reader_scope"]["deal_kit"]["visible"] is True
    assert env["atoms"][0]["structured"]["author_resolved"] == "carlpai@cdw.com"


def test_email_sender_becomes_a_contact_even_without_a_signature(tmp_path):
    from app.parsers.email_parser import EmailParser
    eml = ("From: Charlie Magee <charlie.magee@cdw.com>\nTo: t@purtera-it.com\nDate: Thu, 03 Sep 2026 17:37:05 +0000\n"
           "Subject: Re: 010300 Partner Swap\nMessage-ID: <c@x>\n\nTT - clean up aisle Purtera.\n\nGet Outlook for Mac<https://aka.ms/GetOutlookForMac>\n")
    p = tmp_path / "c.eml"; p.write_text(eml, encoding="utf-8")
    atoms = EmailParser().parse(Path(p))
    people = [a.value for a in atoms if str(getattr(a.atom_type, "value", a.atom_type)) == "stakeholder"]
    assert any(v.get("name") == "Charlie Magee" and v.get("email") == "charlie.magee@cdw.com" for v in people), people


def test_a_rendered_person_record_is_not_reread_as_another_person():
    from app.core.entity_extraction import _structural_people_atoms
    rec = _atom(AtomType.stakeholder, "Patrick Kelly | Account Executive | patrick@purtera-it.com | 770.769.7311",
                kind="person", name="Patrick Kelly", role="Account Executive", email="patrick@purtera-it.com")
    rec.entity_keys = ["stakeholder:patrick_kelly"]
    out = _structural_people_atoms([rec], project_id="p")
    assert not any(a.value.get("name") == "Account Executive" for a in out), [a.value for a in out]


def test_ocr_debris_is_unreadable_and_prose_is_not():
    from app.core.text_quality import is_unreadable, readability
    debris = [
        "‘Tes aks wilenur tht projetcompen mee egutements olin inthe ope seve Minasthe towing cine",
        "Psat iy Wa ns ache A VIA, 2 Tlpmsnapenen nem ori pr ni ptm deSo Suunchnayrlirgity",
        "44 Marware and materats ae ot nclded ia this scope oping.",
        "tonmnuinunvionetenatucnnihnapusstsnse tet oa",
    ]
    prose = [
        "Customer has approximately 169 locations (network closets) listed in Exhibit A.",
        "Hardware and materials are not included in this scope or pricing.",
        "1 × 1 Tech onsite for 4 Hours – Per Item = $570.00",
        "PSOW for D4C: MP58-WHE2-TEAMS and MP56-E2-TEAMS phones on PoE.",
        "If work i cancelled with less han 48 hours noice, Seller wilbe responsible 100% of the aso fe.",  # OCR typos, still mostly words
    ]
    for t in debris:
        assert is_unreadable(t), (t, readability(t))
    for t in prose:
        assert not is_unreadable(t), (t, readability(t))


def test_capitalised_debris_is_unreadable_but_names_are_allowed():
    from app.core.text_quality import is_unreadable, readability
    debris = [
        "Dut Su nee, es ae a Stoeger see ene Scoala ny Syne tae",
        "IC Tes Pia a OE SPR Seep a france",
        "Sips tensts te Deg ao [eo area ae",
        "“ShuulsSthamn uated nate tmnt Sra",
        "eprint tia Shcttrtininsr mame",
        "Pepa Cota he? COW TesterLE",
    ]
    # garbled but mostly words: an OCR'd bullet list stays readable
    assert not is_unreadable("(© Change conto and management (© Statusmeetings and reporting ‘SCOPE")
    keep = [
        "Nisha Ngyuen Project Manager Networking",
        "Carl Painter | Sr. Account Manager | carlpai@cdw.com",
        "Re: 010300 CDW/ Dentistry For Children Partner Swap",
        "Step 4: Box up old phones in the new equipment boxes and leave them onsite with the location MOD.",
        "+ MPS6-E2-TEAMS + Mss MPs2-E2-TEAMS For each location, Customer is responsible for requesting either two (2) hours of onsite support",
    ]
    for t in debris:
        assert is_unreadable(t), (t, readability(t))
    for t in keep:
        assert not is_unreadable(t), (t, readability(t))


def test_header_lines_are_not_banners_and_footers_count_across_documents():
    from app.parsers.email_parser import _is_title_case_banner
    assert not _is_title_case_banner("From: Carl Painter Jr")
    assert not _is_title_case_banner("Subject: Fw: CDW/ Dentistry For Children Partner Swap")
    from app.core.atom_type_sanity import strip_document_chrome
    def _p(aid, text, page):
        a = _atom(AtomType.deal_metadata, text)
        a.artifact_id = aid
        a.source_refs = [SourceRef(id=f"s{aid}{page}", artifact_id=aid, artifact_type="pdf", filename=f"{aid}.pdf", locator={"page": page}, extraction_method="t", parser_version="t")]
        return a
    atoms = [_p("a", "Page 2 CDW Technologies LLC", 2), _p("a", "Page 3 CDW Technologies LLC", 3), _p("b", "Page 6 CDW Technologies LLC", 6), _p("b", "Install the switch in closet 2.", 1)]
    strip_document_chrome(atoms)
    assert [a.raw_text for a in atoms] == ["Install the switch in closet 2."]


def test_vendor_rate_midrow_and_total_sentence():
    from app.core.atom_type_sanity import enrich_vendor_line_items
    a = _atom(AtomType.vendor_line_item, "48 Hour Cancellation or Turnaway Fee – Per $500.00 Item – Per Item | 1 $500.00")
    b = _atom(AtomType.vendor_line_item, "Services Fees will be calculated on a TIME AND MATERIALS basis. The invoiced amount of Services Fees will equal the rate applicable for a unit of a service. Services Fees of $93,583.25 is merely an estimate and does not represent a fixed fee.")
    assert enrich_vendor_line_items([a, b]) == 2
    assert (a.value["rate"], a.value["units"], a.value["subtotal"]) == (500.0, 1, 500.0)
    assert b.atom_type == AtomType.commercial_total and b.value["amount"] == 93583.25


def test_a_bare_list_marker_line_is_a_pending_bullet():
    from app.parsers.orbitbrief_pdf import _BARE_ENUM_RE, _BULLET_LINE_RE
    for marker in ("a.", "c. ", " 3.", "iv)"):
        assert _BARE_ENUM_RE.match(marker), marker
    for not_marker in ("a. Switch", "Switch", "Camera ", "2. Each location will have different needs", "ok."):
        assert not _BARE_ENUM_RE.match(not_marker), not_marker
    assert _BULLET_LINE_RE.match("b. Access Point (Indoor Only)").group(2) == "Access Point (Indoor Only)"


def test_signature_rows_merge_into_one_record_per_party():
    from app.core.atom_type_sanity import merge_signature_rows
    def _p(text):
        a = _atom(AtomType.signatory, text)
        a.source_refs = [SourceRef(id=f"s{abs(hash(text))}", artifact_id="d1", artifact_type="pdf", filename="x.pdf", locator={"page": 6}, extraction_method="t", parser_version="t")]
        return a
    atoms = [
        _p("CDW Technologies LLC: | NewBold LLC: Shelly Lewis"),
        _p("CDW Technologies LLC: By: Mike Murphy (Mar 26, 2025 10:24 EDT) | NewBold LLC: By: Shelly Lewis (Mar 25, 2025 11:49 EDT)"),
        _p("CDW Technologies LLC: Title: Professional Services Manager | NewBold LLC: Title:"),
        _p("Shelly Lewis"),
        _p("NewBold LLC: EVP & COO"),
        _p("CDW Technologies LLC: Mar 26, 2025 | NewBold LLC: Mar 25, 2025"),
        # live shapes that had minted junk parties "By", "Name", "Title"
        _p("By: Shelly Lewis (Mar 25, 2025 11:49 EDT)"),
        _p("Name: Mike Murphy Name: Shelly Lewis Title: Professional Services Manager Title:"),
    ]
    assert merge_signature_rows(atoms) == 7
    assert len(atoms) == 1
    signers = {s["party"]: s for s in atoms[0].value["signers"]}
    assert set(signers) == {"CDW Technologies LLC", "NewBold LLC"}, signers
    assert signers["CDW Technologies LLC"]["name"] == "Mike Murphy"
    assert signers["CDW Technologies LLC"]["title"] == "Professional Services Manager"
    assert signers["CDW Technologies LLC"]["signed_at"].startswith("Mar 26, 2025")
    assert signers["NewBold LLC"]["name"] == "Shelly Lewis"
    assert signers["NewBold LLC"]["title"] == "EVP & COO"


def test_caption_note_gives_an_upload_its_provenance():
    from app.core.orbitbrief_envelope import _link_caption_notes
    timeline = {"transitions": [{"label": "Open- Awaiting Scope", "order": 1}, {"label": "Submitted for Quoting", "order": 2}]}
    docs = [
        {"artifact_id": "note", "artifact_type": "txt", "filename": "hs-note-psow from current partner.txt", "authored_at": "2026-09-03T17:05:52.424Z", "direction": "internal"},
        {"artifact_id": "pdf", "artifact_type": "pdf", "filename": "NEWBOLD PSOW.pdf", "authored_at": "2026-09-03T17:06:05.843Z", "direction": None,
         "deal_stage": {"stage_at_arrival": "Submitted for Quoting", "admissible_for": None}, "lifecycle": {"admissible_for": None}},
        {"artifact_id": "late", "artifact_type": "pdf", "filename": "other.pdf", "authored_at": "2026-09-03T19:00:00Z", "direction": None, "deal_stage": {"stage_at_arrival": "Submitted for Quoting", "admissible_for": None}},
    ]
    # live shape: a caption note carries only its provenance atom (title, no author)
    env = {"atoms": [{"artifact_id": "note", "atom_type": "deal_metadata", "structured": {"field_name": "hubspot_note_provenance", "title": "psow from current partner", "source": "hubspot_note"}}]}
    assert _link_caption_notes(docs, env, timeline) == 1
    pdf = docs[1]
    assert pdf["delivered_by"] is None and pdf["caption"] == "psow from current partner"
    assert pdf["direction"] == "internal" and pdf["deal_stage"]["admissible_for"] == "evidence"
    assert docs[2].get("delivered_by") is None


def test_cross_document_same_name_and_nameless_phone_rows_fold():
    from app.core.semantic_dedup import dedupe_stakeholder_atoms
    def mk(aid, name, **v):
        a = _atom(AtomType.stakeholder, name or "Job Title | Sr. CSDA | Phone Number | 404-918-0783", kind="person", name=name, **v)
        a.artifact_id = aid
        return a
    out = dedupe_stakeholder_atoms([
        mk("e1", "Quinton James", role="Senior Client Executive, Northeast Majors", phone="708.288.8778"),
        mk("e2", "Quinton James", role="Client Executive | Northeast Majors", email="quinton.james@cdw.com"),
        mk("e3", "Quinton James", email="someone.else@other.com"),  # conflicting identity: stays apart
        mk("s1", "Bernard Donnelly", email="bernie.donnelly@sodexo.com", phone="404-918-0783"),
        mk("s1", None, role="Sr. CSDA", phone="404-918-0783"),
        mk("s2", None, role="Sr. CSDA", phone="404-918-0783"),  # another SOW's row: same full phone
    ])
    names = [(a.artifact_id, a.value.get("name")) for a in out]
    assert names.count(("e2", "Quinton James")) + names.count(("e1", "Quinton James")) == 1
    assert ("e3", "Quinton James") in names
    bern = [a for a in out if a.value.get("name") == "Bernard Donnelly"]
    assert len(bern) == 1 and bern[0].value.get("title") == "Sr. CSDA"
    assert not any(a.value.get("name") is None for a in out)


def test_caption_note_provenance_pointer_is_metadata_but_a_real_note_keeps_its_type():
    from app.core.note_provenance_backfill import _mint_provenance_atom
    dup = _atom(AtomType.scope_item, "SOW")
    cap = _mint_provenance_atom(project_id="p", artifact_id="n1", filename="n1.txt", note_body="SOW",
                                parsed={"note_id": "1", "title": "SOW"}, duplicate_atom=dup)
    assert cap.atom_type == AtomType.deal_metadata
    dup2 = _atom(AtomType.scope_item, "Please remove the West Wing from scope for now")
    real = _mint_provenance_atom(project_id="p", artifact_id="n2", filename="n2.txt", note_body="Please remove the West Wing from scope for now",
                                 parsed={"note_id": "2", "title": "West Wing"}, duplicate_atom=dup2)
    assert real.atom_type == AtomType.scope_item


def test_a_sow_between_other_parties_is_third_party_terms():
    from app.core.document_parties import annotate_document_parties, our_org_tokens
    assert "purtera" in our_org_tokens()
    docs = [{"artifact_id": "pdf", "filename": "NEWBOLD PSOW.pdf", "authored_at": "2025-03-05"},
            {"artifact_id": "scan", "filename": "NEWBOLD scanned.pdf"},
            {"artifact_id": "ours", "filename": "PurTera SOW.docx"}]
    env = {"summary": {}, "atoms": [
        {"artifact_id": "pdf", "atom_type": "deal_metadata", "text": "col_0: Provider Name: | D4C Site Assessment and Implementation Program – Phase 1: NewBold LLC FKA NewBold Corporation"},
        {"artifact_id": "pdf", "atom_type": "deal_metadata", "text": "col_0: Customer Name: | D4C Site Assessment and Implementation Program – Phase 1: DENTISTRY FOR CHILDREN"},
        {"artifact_id": "pdf", "atom_type": "deal_metadata", "text": "Seller: CDW"},
        {"artifact_id": "pdf", "atom_type": "signatory", "text": "CDW Technologies LLC: Mike Murphy | NewBold LLC: Shelly Lewis",
         "structured": {"signers": [{"party": "CDW Technologies LLC", "name": "Mike Murphy"}, {"party": "NewBold LLC", "name": "Shelly Lewis"}]}},
        {"artifact_id": "pdf", "atom_type": "vendor_line_item", "text": "1 × 1 Tech onsite for 4 Hours – Per Item = $570.00"},
        {"artifact_id": "scan", "atom_type": "deal_metadata", "text": "|ProviderName:|NewBold LLC FKA NewBold Corporation"},
        {"artifact_id": "ours", "atom_type": "deal_metadata", "text": "Provider: PurTera IT LLC"},
        {"artifact_id": "ours", "atom_type": "deal_metadata", "text": "Customer: Marion County School District"},
    ]}
    assert annotate_document_parties(docs, env) == 3
    pdf, scan, ours = docs
    assert scan["third_party_terms"] is True and scan["terms_owner"].startswith("NewBold")
    assert pdf["third_party_terms"] is True and pdf["terms_owner"] == "NewBold LLC FKA NewBold Corporation"
    assert pdf["parties"]["roles"]["customer"] == "DENTISTRY FOR CHILDREN" and pdf["parties"]["roles"]["seller"] == "CDW"
    assert pdf["our_role"] is None
    assert env["atoms"][4]["decision_provenance"]["terms_owner"].startswith("NewBold")
    assert ours["third_party_terms"] is False and ours["our_role"] == "provider"
    assert env["summary"]["third_party_terms"][0]["provider"].startswith("NewBold")


def test_scanned_header_parties_come_from_page_text_and_org_names_are_not_people():
    from app.core.document_parties import annotate_document_parties, parties_from_page_text
    page = "D4C Teams Phone Implementation Onsite Support | Seller Representative:\nProject\nName:\nCarl Painter\nDENTISTRY FOR CHILDREN\n|ProviderName:|NewBold LLC FKA NewBold Corporation\ncarlpai@edw.com\nCDW Technologies LLC\n"
    assert parties_from_page_text(page)["provider"].startswith("NewBold")
    docs = [{"artifact_id": "scan", "filename": "NEWBOLD scanned.pdf"}]
    env = {"summary": {}, "atoms": [{"artifact_id": "scan", "atom_type": "assumption", "text": "Provider assumes the phones are plug and play."}]}
    assert annotate_document_parties(docs, env, {"scan": page}) == 1
    assert docs[0]["third_party_terms"] is True
    from app.core.entity_extraction import _emit_stakeholders
    assert "stakeholder:newbold" not in _emit_stakeholders("Provider: NewBold")
    assert "stakeholder:sves_engagement_advisor" not in _emit_stakeholders("Coo Title: Sves Engagement Advisor Tithe: Mar 6, 2025")


def test_phone_in_the_email_field_still_identifies_the_person():
    from app.core.semantic_dedup import dedupe_stakeholder_atoms
    def mk(aid, name, **v):
        a = _atom(AtomType.stakeholder, name or "Job Title | Sr. CSDA | Phone Number | 404-918-0783", kind="person" if name else "table_row", name=name, **v)
        a.artifact_id = aid
        return a
    out = dedupe_stakeholder_atoms([
        mk("s1", "Bernard Donnelly", email="bernie.donnelly@sodexo.com", phone="404-918-0783"),
        mk("s2", None, role="Sr. CSDA", email="404-918-0783"),  # live shape: the phone landed in the email field
    ])
    assert [a.value.get("name") for a in out] == ["Bernard Donnelly"] and out[0].value.get("title") == "Sr. CSDA"


def test_email_prose_without_a_scope_object_is_context():
    from app.core.atom_substance_gate import _has_scope_object
    assert not _has_scope_object("TT - clean up aisle Purtera.", [])
    assert not _has_scope_object("We have not notified customer yet that we will be moving partners until we have the sorted on the back end with you.", [])
    assert _has_scope_object("One more call out while you work on PSOW for the Phone Project. The Phone project is for a Yealink phones that is being deployed on an Nextiva platform", [])
    assert _has_scope_object("There are about 170+ sites and growing across the US.", [])
    assert _has_scope_object("Please remove the west wing from scope.", ["site:west_wing"])
    assert _has_scope_object("Install the new clocks at each school.", [])


def test_verified_high_confidence_atoms_leave_the_review_queue():
    from app.core.confidence_recalibration import accept_verified_high_confidence
    from app.core.schemas import ReviewStatus, EvidenceReceipt
    def mk(text, *, conf=0.9, receipt="verified", flags=None, method="pdf_prose_v1", authority=AuthorityClass.contractual_scope):
        a = _atom(AtomType.scope_item, text)
        a.confidence = conf; a.calibrated_confidence = conf
        a.review_status = ReviewStatus.needs_review
        a.review_flags = list(flags or [])
        a.authority_class = authority
        a.source_refs = [SourceRef(id=f"s{abs(hash(text))}", artifact_id="d1", artifact_type="pdf", filename="x.pdf", locator={"page": 1}, extraction_method=method, parser_version="t")]
        if receipt:
            a.receipts = [EvidenceReceipt(atom_id=a.id, artifact_id="d1", filename="x.pdf", source_ref_id=a.source_refs[0].id,
                                          replay_status=receipt, reason="t", verifier_version="t")]
        return a
    verified = mk("Hardware and materials are not included in this scope or pricing.")
    uncalibrated = mk("Cabling is not included in this quote.")
    uncalibrated.calibrated_confidence = 0.0  # never recalibrated: not "zero confidence"
    demoted = mk("Any additional onsite labor will be invoiced at $115.00 per hour.", flags=["unearned_contract_authority_demoted", "task"])
    low = mk("This quote is valid for thirty days.", conf=0.6)
    unsupported = mk("Provider assumes technicians will be granted all access required.", receipt="unsupported")
    flagged = mk("Step 4: Box up old phones.", flags=["calibration_abstain"])
    derived = mk("This image outlines the scope.", method="pdf_image_vision_describe")
    quoted = mk("There are about 170+ sites and growing.", authority=AuthorityClass.quoted_old_email)
    assert accept_verified_high_confidence([verified, uncalibrated, demoted, low, unsupported, flagged, derived, quoted]) == 3
    assert verified.review_status == ReviewStatus.auto_accepted and verified.review_flags == ["accepted_verified_receipt"]
    assert uncalibrated.review_status == ReviewStatus.auto_accepted
    assert demoted.review_status == ReviewStatus.auto_accepted and "unearned_contract_authority_demoted" in demoted.review_flags
    assert all(a.review_status == ReviewStatus.needs_review for a in (low, unsupported, flagged, derived, quoted))


def test_short_list_items_are_atoms():
    from app.parsers.orbitbrief_pdf import _atoms_for_bullet
    out = []
    for i, text in enumerate(["Switch", "Access Point (Indoor Only)", "Camera", "Firewall", "•"]):
        out += list(_atoms_for_bullet(
            item={"text": text}, depth=1, path_indices=[i], project_id="p", artifact_id="d1",
            filename="x.pdf", parser_version="t", base_locator={"page": 2, "block_kind": "bullet_list"},
            section_path=["SITE VISIT 1", "install any of the following four items while onsite for Visit 1"],
        ))
    assert [a.raw_text for a in out] == ["Switch", "Access Point (Indoor Only)", "Camera", "Firewall"]


def test_a_numbered_sentence_starting_with_a_negator_is_not_a_heading():
    from app.parsers.orbitbrief_pdf import _split_runon_numbered_clause
    assert _split_runon_numbered_clause("1. No Provider Pre-Existing Materials are included in any Work Product unless identified as such in the SOW.") is None
    got = _split_runon_numbered_clause("8. Contract Award and Interpretations ACE may accept or reject any proposal at its discretion.")
    assert got and got[0] == "Contract Award and Interpretations"


def test_document_chrome_is_stripped_by_shape():
    from app.core.atom_type_sanity import strip_document_chrome
    def _p(atom_type, text, page):
        a = _atom(atom_type, text)
        a.source_refs = [SourceRef(id=f"s{page}{abs(hash(text))}", artifact_id="d1", artifact_type="pdf", filename="x.pdf", locator={"page": page}, extraction_method="t", parser_version="t")]
        return a
    atoms = [
        _p(AtomType.scope_item, "Page 1 CDW Technologies LLC", 1),
        _p(AtomType.deal_metadata, "Page 2 CDW Technologies LLC", 2),
        _p(AtomType.deal_metadata, "Page 3 CDW Technologies LLC", 3),
        _p(AtomType.scope_item, "Seller will provide Services benefiting the following locations. Page 5 CDW Technologies LLC", 5),
        _p(AtomType.signatory, "CDW Technologies LLC: {{Sig_es_:signer3:signature}} | NewBold LLC: {{Sig_es_:signer1:signature}} Shelly Lewis", 6),
        _p(AtomType.signatory, "CDW Technologies LLC: Date: | NewBold LLC: Date:", 6),
        _p(AtomType.scope_item, "Access Point (Indoor Only) c.", 2),
        _p(AtomType.scope_item, "COMPREHENSIVE PROJECT PLANNING AND MANAGEMENT FRAMEWORK o", 1),
        _p(AtomType.scope_item, "Project Name:", 1),
        _p(AtomType.scope_item, "Install the switch in closet 2.", 2),
    ]
    strip_document_chrome(atoms)
    texts = [a.raw_text for a in atoms]
    assert "Page 1 CDW Technologies LLC" not in texts and "Page 2 CDW Technologies LLC" not in texts
    assert "Seller will provide Services benefiting the following locations." in texts
    assert "CDW Technologies LLC: | NewBold LLC: Shelly Lewis" in texts
    assert "CDW Technologies LLC: Date: | NewBold LLC: Date:" not in texts
    assert "Access Point (Indoor Only)" in texts
    assert "COMPREHENSIVE PROJECT PLANNING AND MANAGEMENT FRAMEWORK" in texts
    assert "Project Name:" not in texts
    assert "Install the switch in closet 2." in texts


def test_vendor_line_items_get_rate_units_subtotal_by_shape():
    from app.core.atom_type_sanity import enrich_vendor_line_items
    a = _atom(AtomType.vendor_line_item, "Additional Onsite Hours – Per Hour – Per Hour $115.00 | 1 $115.00")
    b = _atom(AtomType.vendor_line_item, "Unit Type Unit Rate Billable Units Subtotal First Site Visit - Site Assessment with Report & $550.00 169 $92,950.00 Installation – Per Si – Per Item")
    c = _atom(AtomType.vendor_line_item, "1 × 1 Tech onsite for 4 Hours – Per Item = $570.00")
    assert enrich_vendor_line_items([a, b, c]) == 3
    assert (a.value["rate"], a.value["units"], a.value["subtotal"]) == (115.0, 1, 115.0)
    assert (b.value["rate"], b.value["units"], b.value["subtotal"]) == (550.0, 169, 92950.0)
    assert (c.value["units"], c.value["subtotal"], c.value["rate"]) == (1, 570.0, 570.0)


def test_broken_anchor_and_url_tail_and_banner_are_chrome():
    from app.parsers.email_parser import _is_link_only_line, _is_title_case_banner
    assert _is_link_only_line("Report Suspicious<https://us-phishalarm-ewt.proofpoint.com/EWT/v1/HUqgN_M!")
    assert _is_link_only_line("-Qbq_7kcmFNziBwcstvtkGDG0MwYN6d8iclpycVjKwWxeNBNpmBFHBO3GBNQLuitRgx9OhruWntvfRzV-aAeLybnKnCZunNpjArA0YtzfwpTdXTSUXUCy_maqQ6HQnLgAX1L24bwKfCEbvp78g$>")
    assert _is_title_case_banner("This Message Is From an External Sender")
    assert not _is_title_case_banner("Please remove the West Wing from scope.")
    assert not _is_title_case_banner("Marion County School District needs 10 clocks")


# --- Y-05: link-only lines ----------------------------------------------------

def test_link_only_lines_are_chrome():
    from app.parsers.email_parser import _is_link_only_line
    assert _is_link_only_line("PurTera-IT.com<https://nam13.safelinks.protection.outlook.com/?url=https%3A%2F%2Fx>")
    assert _is_link_only_line("url=https%3A%2F%2Furldefense.proofpoint.com%2Fv2%2Furl%3Fu%3Dhttp-3A__purtera-2Dit.com_")
    assert _is_link_only_line("t@purtera-it.com<mailto:T@purtera-it.com>")
    assert _is_link_only_line("Get Outlook for Mac<https://aka.ms/GetOutlookForMac>")  # an anchor: label glued to its link
    assert not _is_link_only_line("Please see the PSOW at https://cdw.com/psow before Friday.")


# --- Y-02: party-block addresses are not sites --------------------------------

def test_address_inside_a_signature_cluster_is_a_party_address():
    from app.parsers.email_parser import _is_party_address_context
    lines = [
        "Carl Painter| Sr. Account Manager| carlpai@cdw.com| CDW",
        "Direct:847-968-9740 | Mobile: 847-363-7372",
        "200 N Milwaukee Ave|Vernon Hills, IL 60061",
    ]
    assert _is_party_address_context(lines, 2)
    body = ["Please dispatch to:", "2970 Brandywine Rd, Suite 200, Atlanta, GA 30341", "between 7am and 6pm."]
    assert not _is_party_address_context(body, 1)


# --- Y-04: an exclusion needs a negation --------------------------------------

def _atom(atom_type, text, **value):
    return EvidenceAtom(
        id=f"a_{abs(hash(text))}", project_id="p", artifact_id="d1", atom_type=atom_type,
        raw_text=text, normalized_text=text.lower(), value=value or {},
        authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.auto_accepted, entity_keys=[], parser_version="t",
    )


def test_exclusions_without_negation_go_back_to_scope():
    from app.core.atom_type_sanity import demote_exclusions_without_negation
    atoms = [
        _atom(AtomType.exclusion, "There is a site assessment survey that gets completed after each install."),
        _atom(AtomType.exclusion, "We have you covered on an A+ PM and qualified technicians."),
        _atom(AtomType.exclusion, "Hardware and materials are not included in this scope or pricing."),
        _atom(AtomType.exclusion, "Services not specified in this SOW are considered out of scope."),
        _atom(AtomType.exclusion, "Cabling", list_section="exclude"),
    ]
    assert demote_exclusions_without_negation(atoms) == 2
    assert [a.atom_type for a in atoms] == [AtomType.scope_item, AtomType.scope_item, AtomType.exclusion, AtomType.exclusion, AtomType.exclusion]
    assert "exclusion_without_negation" in atoms[0].review_flags


# --- Y-05: conversational prose ----------------------------------------------

def test_conversational_email_prose_is_not_scope():
    from app.core.atom_substance_gate import _is_conversational_prose
    assert _is_conversational_prose("Excited to knock this out of the park with y'all.", [])
    assert _is_conversational_prose("This has been received and we are on it.", [])
    assert _is_conversational_prose("Thank you for the opportunity!", [])
    assert not _is_conversational_prose("The customer pivoted to Nextiva so update the PSOW.", [])
    assert not _is_conversational_prose("There are about 170+ sites and growing across the US.", [])
    assert not _is_conversational_prose("Please remove the west wing from scope.", ["site:west_wing"])


# --- Y-01: quoted external messages get their own scope -----------------------

def test_quoted_customer_message_inside_an_outbound_reply_is_visible():
    from app.core.orbitbrief_envelope import _annotate_quoted_message_scope
    timeline = {"transitions": [{"label": "Open- Awaiting Scope", "order": 1}, {"label": "Submitted for Quoting", "order": 2}]}
    docs = [{
        "artifact_id": "reply", "filename": "reply.eml",
        "deal_stage": {"stage_at_arrival": "Submitted for Quoting", "admissible_for": "label"},
        "reader_scope": {"deal_kit": {"visible": False, "why": "label"}, "sow": {"visible": False, "why": "label"}, "orbitbrief": {"visible": True, "why": ""}, "atlas": {"visible": True, "why": ""}},
    }]
    env = {"atoms": [
        {"artifact_id": "reply", "atom_type": "deal_metadata", "text": "There are about 170+ sites and growing across the US.",
         "structured": {"quoted": True, "author": "Carl Painter Jr <carlpai@cdw.com>", "message_index": 3}},
        {"artifact_id": "reply", "atom_type": "scope_item", "text": "We have you covered on an A+ PM.",
         "structured": {"quoted": False, "author": "patrick@purtera-it.com", "message_index": 0}},
        {"artifact_id": "reply", "atom_type": "scope_item", "text": "Load this up and reply to them.",
         "structured": {"quoted": True, "author": "Trent Torrence <t@purtera-it.com>", "message_index": 1}},
    ]}
    n = _annotate_quoted_message_scope(env, docs, timeline)
    assert n == 1, env["atoms"]
    carl = env["atoms"][0]
    assert carl["reader_scope"]["deal_kit"]["visible"] is True
    assert carl["decision_provenance"]["source"] == "quoted_message"
    assert carl["decision_provenance"]["admissible_for"] == "evidence"
    assert "reader_scope" not in env["atoms"][1]  # our own authored line: file decides
    assert "reader_scope" not in env["atoms"][2]  # quoting ourselves does not make it the customer's


# --- Y-07: a document's own date outranks the upload time --------------------

def _row(text, page=1):
    return EvidenceAtom(
        id=f"r_{abs(hash(text))}", project_id="p", artifact_id="pdf", atom_type=AtomType.deal_metadata,
        raw_text=text, normalized_text=text.lower(), value={}, authority_class=AuthorityClass.contractual_scope,
        confidence=0.8, review_status=ReviewStatus.auto_accepted, entity_keys=[], parser_version="t",
        source_refs=[SourceRef(id="s", artifact_id="pdf", artifact_type="pdf", filename="x.pdf", locator={"page": page}, extraction_method="t", parser_version="t")],
    )


def test_document_header_date_is_read_from_a_date_label_on_page_one():
    from app.core.orbitbrief_envelope import _document_header_date
    atoms = [_row("col_0: Date: | D4C Site Assessment and Implementation Program – Phase 1: March 21, 2025"),
             _row("Mike Murphy (Mar 26, 2025 10:24 EDT)", page=7),
             _row("Customer Name: | DENTISTRY FOR CHILDREN")]
    assert _document_header_date(atoms) == "2025-03-21"
    assert _document_header_date([_row("Customer Name: | DENTISTRY FOR CHILDREN")]) is None
    # the file's own first page, when the header table never became an atom
    page = "STATEMENT OF WORK\nProject Name: D4C Site Assessment\nCustomer Name: DENTISTRY FOR CHILDREN\nDate:                          March 21, 2025\nThis statement of work ... dated the 15th day of October, 2024."
    assert _document_header_date([], page) == "2025-03-21"
    assert _document_header_date([], "Drafted By: Sasha Beard\nMSA dated the 15th day of October, 2024.") is None
    # a scanned header whose "Date:" label OCR'd as "foe:" but whose value stands alone on its header line
    scanned = "D4C Teams Phone Implementation Onsite Support | Seller Representative:\nProject\nName:\nv2\nCarl Painter\nDENTISTRY FOR CHILDREN\n+1 (847) 9689740\n|ProviderName:|NewBold LLC FKA NewBold Corporation\ncarlpai@edw.com\nCDW Technologies LLC\nDrafted By:\nfoe:\nMarch 05, 2025\nRyan Adamski\nThis statement of work (\"Statement of Work\" or \"SOW\") is made and entered into on the date this SOW is signed by both\nparties (the \"SOW Effective Date\") ... dated the 15th day of October, 2024."
    assert _document_header_date([], scanned) == "2025-03-05"


# --- Y-01/Y-03/Y-05 end to end through the email parser ----------------------

def test_email_parser_reads_signature_people_and_skips_chrome(tmp_path):
    from app.parsers.email_parser import EmailParser
    eml = (
        "From: Patrick Kelly <patrick@purtera-it.com>\nTo: carlpai@cdw.com\n"
        "Date: Thu, 03 Sep 2026 17:04:37 +0000\nSubject: Re: 010300 Partner Swap\nMessage-ID: <a@x>\n\n"
        "Hello Carl and CDW team-\n\nThis has been received and we are on it.\n\nThanks,\n\nPatrick Kelly\n\nAccount Executive\n\n"
        "patrick@purtera-it.com\n\n770.769.7311\n\nPurTera-IT.com<https://nam13.safelinks.protection.outlook.com/?url=x>\n\n"
        "________________________________\nFrom: Carl Painter Jr <carlpai@cdw.com>\nSent: Thursday, September 3, 2026 10:38 AM\n"
        "To: Trent Torrence <t@purtera-it.com>\nSubject: CDW/ Dentistry For Children Partner Swap\n\n"
        "There are about 170+ sites and growing across the US.\n\nWe will need:\n\n"
        "  *   Rhonda Sharp Professional Services Manager\n  *   Jacob Long Project Manager Phone and ITAD\n\n"
        "Carl Painter| Sr. Account Manager| carlpai@cdw.com| CDW\nDirect:847-968-9740 | Mobile: 847-363-7372\n"
        "200 N Milwaukee Ave|Vernon Hills, IL 60061\n"
    )
    p = tmp_path / "t.eml"; p.write_text(eml, encoding="utf-8")
    atoms = EmailParser().parse(Path(p))
    people = {a.value.get("name"): a.value for a in atoms if str(getattr(a.atom_type, "value", a.atom_type)) == "stakeholder"}
    assert people["Carl Painter"]["email"] == "carlpai@cdw.com" and people["Carl Painter"]["phone"] == "847-968-9740"
    assert people["Patrick Kelly"]["role"] == "Account Executive"
    assert people["Rhonda Sharp"]["role"] == "Professional Services Manager"
    assert people["Jacob Long"]["role"] == "Project Manager Phone and ITAD"
    sites = [a for a in atoms if str(getattr(a.atom_type, "value", a.atom_type)) == "physical_site"]
    assert not sites, [a.raw_text for a in sites]
    texts = [a.raw_text for a in atoms]
    assert not any("safelinks" in t for t in texts), texts
    quoted = [a for a in atoms if "170+" in a.raw_text]
    assert quoted and quoted[0].value.get("author", "").lower().endswith("carlpai@cdw.com>") or "carlpai@cdw.com" in str(quoted[0].value.get("author", "")).lower()


# ───────────────────────── round 22: wrapped bullet tails ─────────────────────


def test_wrapped_bullet_tail_joins_whatever_letter_it_opens_with():
    """Live 010300 signed PSOW: 'Each location will have different needs, ...
    communicated by Customer and' | 'Seller, and Provider will be made notified
    ...' — the PDF wrapped the clause and the tail opened with a capital, so it
    became its own (mistyped) paragraph. A full-width line without terminal
    punctuation continues on the next line regardless of case."""
    from app.parsers.orbitbrief_pdf import _text_rich_sections

    txt = (
        "SITE VISIT 1:\n"
        "    2.   Each location will have different needs, and what exactly is to be installed will be communicated by Customer and\n"
        "         Seller, and Provider will be made notified prior to any work being scheduled. Seller will have all hardware to be\n"
        "         installed already shipped to site.\n"
        "    3.   For some sites where a redundancy solution is being installed, new cabling will be required to be added.\n"
    )
    items = [
        it.get("text") or ""
        for s in _text_rich_sections(txt)
        for b in s.get("blocks", [])
        if b.get("kind") == "bullet_list"
        for it in b.get("items", [])
    ]
    assert len(items) == 2, items
    assert items[0].startswith("Each location will have different needs")
    assert "Seller, and Provider will be made notified" in items[0]
    assert items[0].endswith("shipped to site.")


def test_wrapped_tail_needs_layout_evidence_not_just_a_capital():
    from app.parsers.orbitbrief_pdf import _is_wrapped_tail

    full = "Customer is responsible for providing a Sign Off List of all items to be confirmed by Technician prior to leaving an"
    lines = [full, "Implementation Site Visit which will be shared via a report to Customer and Seller.", "Next item here."]
    assert _is_wrapped_tail(lines, 1)
    # Previous line ended a sentence: no wrap.
    assert not _is_wrapped_tail([full + ".", lines[1]], 1)
    # Previous line is short (a list item that just ends without a period).
    assert not _is_wrapped_tail(["Switch", "Access Point (Indoor Only)", full], 1)
    # An ALLCAPS heading is never a tail.
    assert not _is_wrapped_tail([full, "ASSUMPTIONS"], 1)
    assert not _is_wrapped_tail(lines, 0)


def test_ocr_bullet_glyph_class_reads_as_list_marker():
    """Scanned PSOW 156648: tesseract rendered bullets as '©', '¢', '«'. One
    non-word symbol + space + text is a list marker; content openers are not."""
    from app.parsers.orbitbrief_pdf import _BULLET_LINE_RE

    def item(line):
        m = _BULLET_LINE_RE.match(line)
        return m.group(2) if m else None

    assert item("© Solution and Technical Architecture Review and planning") == "Solution and Technical Architecture Review and planning"
    assert item("¢ External Project Meeting") == "External Project Meeting"
    assert item("• Status meetings and reporting") == "Status meetings and reporting"
    assert item("$500 fee applies") is None
    assert item("(a) something") is None
    assert item("“Quoted” start of a sentence") is None
    # tesseract also renders a bullet as "=" ("= Internal Project Technical
    # Planning"); no sentence opens with "= ".
    assert item("= Internal Project Technical Planning") == "Internal Project Technical Planning"
    assert item("Normal sentence here.") is None


def test_confidence_floor_skips_provenance_records():
    """Quoted-message headers are emitted auto_accepted at 0.45 and the floor
    (0.50) flipped every one of them into the review queue — six per thread
    on live 010300. A record about the document's plumbing is not a claim."""
    from types import SimpleNamespace
    from app.core.compiler import _is_provenance_record

    hdr = SimpleNamespace(value={"kind": "quoted_message_header", "sender": "Carl Painter Jr"})
    marker = SimpleNamespace(value={"kind": "image_marker"})
    claim = SimpleNamespace(value={"kind": "paragraph"})
    assert _is_provenance_record(hdr)
    assert _is_provenance_record(marker)
    assert not _is_provenance_record(claim)
    assert not _is_provenance_record(SimpleNamespace(value=None))


def test_multi_sentence_colon_paragraph_is_not_promoted_to_heading():
    """Scanned PSOW: the seven-line scope paragraph ending '...The equipment in
    scope for this project is:' was lifted wholesale into a sub-section heading
    and vanished from the atom stream. Only a one-line intro is a heading."""
    from app.parsers.orbitbrief_pdf import _text_rich_sections

    txt = (
        "SCOPE OF SERVICES\n"
        "Customer has approximately 169 locations listed in Exhibit A where Provider will be asked to schedule and dispatch a\n"
        "technician to a location. Technician will then box up the old phone. The equipment in scope for this project is:\n"
        "\n"
        "* MP58-WH-E2-TEAMS\n"
        "* MP56-E2-TEAMS\n"
        "\n"
        "Milestone billing schedule:\n"
        "* 50% on signature\n"
        "* 50% on completion\n"
    )
    secs = _text_rich_sections(txt)
    paras = [b["text"] for s in secs for b in s.get("blocks", []) if b.get("kind") == "paragraph"]
    assert any(p.startswith("Customer has approximately 169 locations") and p.endswith("project is:") for p in paras), paras
    heads = [sub["heading"] for s in secs for sub in s.get("subsections", [])]
    assert "Milestone billing schedule" in heads
    assert not any("169 locations" in h for h in heads)


def test_blank_line_before_wrapped_tail_does_not_break_the_paragraph():
    """Tesseract opens a blank line before a hanging-indent tail
    ('...infrastructure for' / '' / 'installation.'); the blank is layout, not
    a paragraph break, when the line above ran full-width unpunctuated."""
    from app.parsers.orbitbrief_pdf import _text_rich_sections

    txt = (
        "ASSUMPTIONS\n"
        "Provider assumes the new Teams IP Phones are plug and play and require no new cabling or infrastructure for\n"
        "\n"
        "installation.\n"
        "\n"
        "Cabling is not included in this quote.\n"
    )
    paras = [b["text"] for s in _text_rich_sections(txt) for b in s.get("blocks", []) if b.get("kind") == "paragraph"]
    assert paras == [
        "Provider assumes the new Teams IP Phones are plug and play and require no new cabling or infrastructure for installation.",
        "Cabling is not included in this quote.",
    ], paras


def test_header_display_names_become_stakeholders(tmp_path):
    """Charlie Magee's own message carried a bare From address; three quoted
    'Cc:' lines wrote 'Charlie Magee <charlie.magee@cdw.com>'. The shape is
    the evidence."""
    from app.parsers.email_parser import EmailParser

    eml = tmp_path / "t.eml"
    eml.write_text(
        "From: charlie.magee@cdw.com\nTo: t@purtera-it.com\nSubject: Re: swap\nDate: Thu, 03 Sep 2026 17:37:05 +0000\n\n"
        "TT - clean up aisle Purtera.\n\n"
        "From: Trent Torrence <t@purtera-it.com>\nSent: Thursday, September 3, 2026 1:09 PM\n"
        "To: Patrick Kelly <patrick@purtera-it.com>; Carl Painter Jr <carlpai@cdw.com>\n"
        "Cc: Rhonda Sharp <rhonda.sharp@cdw.com>; Charlie Magee <charlie.magee@cdw.com>; rhonda.sharp@cdw.com <rhonda.sharp@cdw.com>\n"
        "Subject: Re: swap\n\nThank you for the opportunity!\n",
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact("p", "a", eml)
    people = {a.value.get("email"): a.value.get("name") for a in atoms if str(getattr(a.atom_type, "value", a.atom_type)) == "stakeholder"}
    assert people.get("charlie.magee@cdw.com") == "Charlie Magee"
    assert people.get("carlpai@cdw.com") == "Carl Painter Jr"
    assert people.get("rhonda.sharp@cdw.com") == "Rhonda Sharp"
    # one record per address
    assert sum(1 for a in atoms if str(getattr(a.atom_type, "value", a.atom_type)) == "stakeholder" and a.value.get("email") == "rhonda.sharp@cdw.com") == 1


def test_ocr_prefers_page_level_tesseract(monkeypatch):
    """Same engine, different layout: PyMuPDF's textpage re-orders words into
    its own blocks and shreds paragraphs; page-level tesseract keeps lines
    whole. The chain tries the page-level pass first and falls back."""
    import types, sys as _sys
    from app.parsers import _ocr_chain

    class _Pix:
        def tobytes(self, fmt):
            import io
            from PIL import Image
            buf = io.BytesIO(); Image.new("RGB", (20, 20), "white").save(buf, format="PNG"); return buf.getvalue()

    class _Page:
        def get_pixmap(self, dpi=200, alpha=False):
            return _Pix()
        def get_textpage_ocr(self, **kw):
            raise AssertionError("page-level tesseract must run first")

    fake = types.SimpleNamespace(image_to_string=lambda img, config="": "Customer has approximately 169 locations\nlisted in Exhibit A.\n\nNext paragraph.")
    monkeypatch.setitem(_sys.modules, "pytesseract", fake)
    out = _ocr_chain.ocr_pdf_page(_Page())
    assert out["backend"] == "tesseract_page"
    assert out["text"].startswith("Customer has approximately 169 locations")


def test_gateway_banner_carries_no_author(tmp_path):
    """Proofpoint inserts 'This Message Is From an External Sender' above the
    quoted reply on the recipient's copy; the block stamp made Trent its
    author. Chrome is unauthored; the real sentence keeps its author."""
    from app.parsers.email_parser import EmailParser

    eml = tmp_path / "b.eml"
    eml.write_text(
        "From: charlie.magee@cdw.com\nTo: t@purtera-it.com\nSubject: Re: swap\nDate: Thu, 03 Sep 2026 17:37:05 +0000\n\n"
        "TT - clean up aisle Purtera.\n\n"
        "From: Trent Torrence <t@purtera-it.com>\nDate: Thursday, September 3, 2026 at 1:09 PM\n"
        "To: Patrick Kelly <patrick@purtera-it.com>\nSubject: Re: swap\n"
        "This Message Is From an External Sender\n"
        "This message came from outside your organization.\n"
        "Report Suspicious<https://us-phishalarm-ewt.proofpoint.com/EWT/v1/abc>\n"
        "Thank you for the opportunity!\n",
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact("p", "a", eml)
    banners = [a for a in atoms if isinstance(a.value, dict) and a.value.get("kind") == "email_banner"]
    assert banners, [a.raw_text for a in atoms]
    assert all(not a.value.get("author") for a in banners)
    thanks = [a for a in atoms if "Thank you for the opportunity" in a.raw_text]
    assert thanks and all("t@purtera-it.com" in str(a.value.get("author") or "") for a in thanks)


def test_hyphenated_compound_is_not_debris():
    """'Customer-Designated' (19 letters) tripped the 18-letter-debris rule and
    the Services Fees clause was dropped as unreadable."""
    from app.core.text_quality import is_unreadable, readability

    clause = (
        "Services Fees will be calculated on a TIME AND MATERIALS basis. Any non-Hourly Units will be "
        "measured in one (1) unit increments when Services are performed remotely or at any "
        "Customer-Designated Location(s) (as defined below). Any Hourly Units will be measured in one (1) "
        "hour increments with a minimum of one (1) hour billed each day Services are performed remotely."
    )
    assert readability(clause) > 0.85
    assert not is_unreadable(clause)
    assert is_unreadable("Tes aks wilenur tht projetcompen xxqzvbnmklpoiuytrewq asdfghjklzxcvbnmq")


def test_nameless_stakeholder_sentence_is_retyped_not_dropped():
    from app.core.atom_substance_gate import drop_contextless_stakeholders

    clause = _atom(
        AtomType.stakeholder,
        "Each Party will appoint a person to act as that Party’s point of contact (“Contact Person”) as the time for performance nears and will communicate that person’s name and information to the other Party’s Contact Person.",
        kind="person", name=None, role="contact person",
    )
    clause.entity_keys = ["stakeholder:contact_person"]
    fragment = _atom(AtomType.stakeholder, "Step 5: Appoint a contact person", kind="person", name=None, role="contact person")
    kept, dropped = drop_contextless_stakeholders([clause, fragment])
    assert dropped == [fragment]
    assert kept == [clause]
    assert clause.atom_type == AtomType.contract_term
    assert "stakeholder_sentence_retyped" in clause.review_flags
    assert clause.entity_keys == []


def test_provenance_markers_leave_the_review_queue():
    from app.core.confidence_recalibration import accept_verified_high_confidence

    marker = _atom(AtomType.deal_metadata, "[binary region 3 of page 6 not recovered]", kind="binary_region_marker")
    marker.review_status = ReviewStatus.needs_review
    marker.confidence = 0.75
    hdr = _atom(AtomType.deal_metadata, "From: Carl Painter Jr | Sent: Thursday", kind="quoted_message_header")
    hdr.review_status = ReviewStatus.needs_review
    hdr.confidence = 0.45
    assert accept_verified_high_confidence([marker, hdr]) == 2
    assert marker.review_status == ReviewStatus.auto_accepted and hdr.review_status == ReviewStatus.auto_accepted


def test_bookkeeping_flags_are_not_doubt_but_doubt_flags_are():
    from app.core.confidence_recalibration import accept_verified_high_confidence
    from app.core.schemas import EvidenceReceipt

    def _verified(text, flags):
        a = _atom(AtomType.scope_item, text, kind="bullet")
        a.review_status = ReviewStatus.needs_review
        a.confidence = 0.88
        a.receipts = [EvidenceReceipt(atom_id=a.id, artifact_id="d1", filename="x.pdf", source_ref_id="s1", replay_status="verified", reason="", verifier_version="t")]
        a.review_flags = flags
        return a

    ok = _verified("Overseeing administration, financials, and team management.", ["unearned_contract_authority_demoted", "conversation_meta", "head_exclude", "unsupported_name_stripped", "task_tier_child", "quote_context:survey_design"])
    doubtful = _verified("Camera", ["calibration_abstain"])
    weak = _verified("Taxes: Excluded", ["weak_label"])
    assert accept_verified_high_confidence([ok, doubtful, weak]) == 1
    assert ok.review_status == ReviewStatus.auto_accepted
    assert doubtful.review_status == ReviewStatus.needs_review and weak.review_status == ReviewStatus.needs_review


def test_signature_cells_repeated_without_party_labels_are_swept():
    """The signature table reaches the page twice (labelled and unlabelled
    copies); the typer files the second copy as deal_metadata and it sits in
    the review queue as debris. Words already in the signer records = nothing new."""
    from app.core.atom_type_sanity import merge_signature_rows

    def _row(atom_type, text):
        a = _atom(atom_type, text, kind="table_row")
        a.source_refs = [SourceRef(id=f"s{abs(hash(text))}", artifact_id="d1", artifact_type="pdf", filename="x.pdf", locator={"page": 6}, extraction_method="t", parser_version="t")]
        return a

    rows = [
        _row(AtomType.signatory, "CDW Technologies LLC: By: Mike Murphy (Mar 26, 2025 10:24 EDT) | NewBold LLC: By: Shelly Lewis (Mar 25, 2025 11:49 EDT)"),
        _row(AtomType.signatory, "CDW Technologies LLC: Title: Professional Services Manager | NewBold LLC: Title: EVP & COO"),
        _row(AtomType.deal_metadata, "Name: Mike Murphy Name: | Shelly Lewis"),
        _row(AtomType.deal_metadata, "Mar 26, 2025 | Mar 25, 2025"),
        _row(AtomType.deal_metadata, "Shelly Lewis"),
        _row(AtomType.deal_metadata, "CDW Technologies LLC: Date: | NewBold LLC: Date:"),
        _row(AtomType.deal_metadata, "Provider will use the following subcontractor(s) to perform Services under this SOW: Not Applicable"),
    ]
    atoms = list(rows)
    merge_signature_rows(atoms)
    texts = [a.raw_text for a in atoms]
    assert len(atoms) == 2, texts
    sig = atoms[0]
    assert sig.value["name"] == "Mike Murphy" and sig.value["title"] == "Professional Services Manager"
    assert sig.value["role"] == "Professional Services Manager"
    assert [s["name"] for s in sig.value["signers"]] == ["Mike Murphy", "Shelly Lewis"]
    assert sig.value["parties"] == ["CDW Technologies LLC", "NewBold LLC"]
    assert texts[1].startswith("Provider will use the following subcontractor")


def test_a_sentence_of_necessity_is_not_banter():
    """'We will need to be able to address multiple sites daily and weekly.'
    (12 words, no digit, no proper noun) was dropped as conversational. A
    modal of necessity is grammar; the sentence is a demand on us."""
    from app.core.atom_substance_gate import drop_email_non_scope
    from app.core.schemas import AuthorityClass

    def _line(text):
        a = _atom(AtomType.scope_item, text, kind="email_body_line", quoted=True, author="Carl Painter Jr")
        a.authority_class = AuthorityClass.quoted_old_email
        return a

    need = _line("We will need to be able to address multiple sites daily and weekly.")
    banter = _line("It was good talking with you today.")
    kept, dropped = drop_email_non_scope([need, banter])
    assert kept == [need] and dropped == [banter]


# ───────────────────── round 23: fresh-envelope residuals ─────────────────────


def test_signature_page_furniture_is_not_a_signer():
    """Scanned PSOW: 'Vernon Hills, IL 6001', '200 N. Milwaukee Ave.', 'Mar 6, 2025'
    each shipped as a signatory with that text as its name."""
    from app.core.atom_type_sanity import demote_signatory_chrome

    rows = [
        _atom(AtomType.signatory, "Vernon Hills, IL 6001", kind="paragraph", name="Vernon Hills, IL 6001"),
        _atom(AtomType.signatory, "200 N. Milwaukee Ave.", kind="paragraph", name="200 N. Milwaukee Ave."),
        _atom(AtomType.signatory, "Mar 6, 2025", kind="paragraph", name="Mar 6, 2025"),
        _atom(AtomType.signatory, "Name: Samantha Ojeda", kind="paragraph", name="Samantha Ojeda", role="Authorized Representative"),
        _atom(AtomType.signatory, "CDW Technologies LLC: Mike Murphy, Professional Services Manager", kind="signature_block", name="Mike Murphy", signers=[{"party": "CDW Technologies LLC", "name": "Mike Murphy"}]),
    ]
    assert demote_signatory_chrome(rows) == 3
    assert [a.atom_type for a in rows[:3]] == [AtomType.deal_metadata] * 3
    assert all(a.review_status == ReviewStatus.auto_accepted and a.value["kind"] == "signature_chrome" for a in rows[:3])
    assert rows[3].atom_type == AtomType.signatory and rows[4].atom_type == AtomType.signatory


def test_product_codes_become_bom_lines():
    """Four Yealink models parsed as bullets and were filed as deal_metadata;
    the BOM said zero devices. Letters + digits + hyphens, no words: a model."""
    from app.core.atom_type_sanity import retype_product_codes

    rows = [
        _atom(AtomType.deal_metadata, "MP58-WH-E2-TEAMS", kind="bullet"),
        _atom(AtomType.scope_item, "MPS6-E2-TEAMS", kind="bullet"),
        _atom(AtomType.deliverable, "C9300-48P", kind="bullet"),
        _atom(AtomType.scope_item, "Switch", kind="bullet"),
        _atom(AtomType.deal_metadata, "Mar 26, 2025", kind="paragraph"),
        _atom(AtomType.deal_metadata, "SOW 158279", kind="paragraph"),
        _atom(AtomType.scope_item, "Install 24 access points", kind="bullet"),
    ]
    assert retype_product_codes(rows) == 3
    assert [a.atom_type for a in rows[:3]] == [AtomType.bom_line] * 3
    assert rows[0].value["model"] == "MP58-WH-E2-TEAMS" and "device:mp58_wh_e2_teams" in rows[0].entity_keys
    assert [a.atom_type for a in rows[3:]] == [AtomType.scope_item, AtomType.deal_metadata, AtomType.deal_metadata, AtomType.scope_item]


def test_all_caps_organisation_is_not_a_person():
    from app.core.atom_substance_gate import drop_contextless_stakeholders

    org = _atom(AtomType.stakeholder, "DENTISTRY FOR CHILDREN +1 (847) 9689740", kind="person", name="DENTISTRY FOR CHILDREN", role="Customer", phone="+1 (847) 9689740")
    person = _atom(AtomType.stakeholder, "Carl Painter | Sr. Account Manager | carlpai@cdw.com", kind="person", name="Carl Painter", role="Sr. Account Manager", email="carlpai@cdw.com")
    caps_person = _atom(AtomType.stakeholder, "JOHN SMITH | Project Manager | js@x.com", kind="person", name="JOHN SMITH", role="Project Manager", email="js@x.com")
    kept, dropped = drop_contextless_stakeholders([org, person, caps_person])
    assert dropped == [org]
    assert kept == [person, caps_person]


def test_same_person_two_records_one_typo_fold_and_keep_every_field():
    """'Nisha Ngyuen Project Manager Networking' (a colleague's typo, has the
    title) and 'Nisha Nguyen <nisha.nguyen@cdw.com>' (has the address) are one
    person; the survivor must carry both the title and the address."""
    from app.core.semantic_dedup import _fold_bare_name_variants, _near_surname

    assert _near_surname("ngyuen", "nguyen") and _near_surname("donnelly", "donnely") and not _near_surname("sharp", "shark") and not _near_surname("lee", "leo")
    a = _atom(AtomType.stakeholder, "Nisha Ngyuen Project Manager Networking", kind="person", name="Nisha Ngyuen", role="Project Manager Networking")
    b = _atom(AtomType.stakeholder, "Nisha Nguyen | nisha.nguyen@cdw.com", kind="person", name="Nisha Nguyen", email="nisha.nguyen@cdw.com")
    c = _atom(AtomType.stakeholder, "Rhonda Sharp Professional Services Manager", kind="person", name="Rhonda Sharp", role="Professional Services Manager")
    d = _atom(AtomType.stakeholder, "Rhonda Sharp | rhonda.sharp@cdw.com", kind="person", name="Rhonda Sharp", email="rhonda.sharp@cdw.com")
    out = _fold_bare_name_variants([a, b, c, d])
    people = {x.value["name"]: x.value for x in out}
    assert set(people) == {"Nisha Nguyen", "Rhonda Sharp"}, list(people)
    assert people["Nisha Nguyen"]["email"] == "nisha.nguyen@cdw.com" and people["Nisha Nguyen"]["role"] == "Project Manager Networking"
    assert people["Rhonda Sharp"]["email"] == "rhonda.sharp@cdw.com" and people["Rhonda Sharp"]["role"] == "Professional Services Manager"


def test_parties_from_defined_terms_in_the_opening_clause():
    from app.core.document_parties import parties_from_page_text

    text = (
        "STATEMENT OF WORK\n"
        "This statement of work is made and entered into by and between the undersigned, CDW Technologies LLC (“Buyer”) and NewBold\n"
        "LLC (“Provider”, “Seller” and “we”). Services performed by Provider hereunder may benefit DENTISTRY FOR\n"
        "CHILDREN (“Customer”), a customer of Buyer or of Buyer's Affiliate.\n"
    )
    roles = parties_from_page_text(text)
    assert roles == {"buyer": "CDW Technologies LLC", "provider": "NewBold LLC", "customer": "DENTISTRY FOR CHILDREN"}, roles
    # A header cell that runs into the next column loses the contact tail.
    hdr = "Customer Name:                 DENTISTRY FOR CHILDREN                           +1 (847) 9689740\nProvider Name:                 NewBold LLC FKA NewBold Corporation              carlpai@cdw.com\n"
    r2 = parties_from_page_text(hdr)
    assert r2["customer"] == "DENTISTRY FOR CHILDREN" and r2["provider"] == "NewBold LLC FKA NewBold Corporation"


def test_inline_labelled_note_becomes_one_atom_per_field(tmp_path):
    """Live 010297: the whole note ('Address: … Duration: … Scope of work: …')
    was one physical_site whose address was the entire text."""
    from app.parsers.hubspot_note_parser import HubspotNoteParser

    p = tmp_path / "010298-hs-note-1-Address_.txt"
    p.write_text(
        "HubSpot Note: Address:\nHubSpot Note ID: 1\nDate: 2026-09-03T12:19:53.957Z\nAuthor: Trent Torrence\nAuthor-Email: t@purtera-it.com\n\n"
        "Address: 5 6 7 &amp; 8 FLR RMZ Eco World SEZ Campus 4A/4B, Sarjapur Marathahalli, Outer Ring Road Shell Business Operations-Bangalore "
        "Date | Time: When you can accommodate Duration: Approximately 1 week Tech/Engineer: 1 Implementation Specialist "
        "Scope of work: Verification of equipment, rack positioning, leveling, and grounding, installation of PDUs and pre-configured devices, structured cabling, power-up, and daily reporting with photographic documentation.\n",
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("p", "a", p)
    by_type = {}
    for a in atoms:
        by_type.setdefault(a.atom_type, []).append(a)
    site = by_type[AtomType.physical_site][0]
    assert site.value["address"].startswith("5 6 7 & 8 FLR RMZ Eco World SEZ Campus")
    # The facility is a capitalised run inside the address, never the floor
    # numbers and never a neighbouring field.
    assert site.value["name"] in ("RMZ Eco World SEZ Campus", "Outer Ring Road Shell Business Operations-Bangalore")
    assert "Duration" not in site.value["address"]
    assert any(a.value.get("field_name") == "Duration" for a in by_type[AtomType.constraint])
    assert any(a.value.get("field_name") == "Tech/Engineer" for a in by_type[AtomType.requirement])
    items = [a.raw_text for a in by_type[AtomType.scope_item] if a.value.get("kind") == "note_field_item"]
    assert "rack positioning" in items and "structured cabling" in items and "daily reporting with photographic documentation" in items
    assert not any("&amp;" in a.raw_text for a in atoms)
