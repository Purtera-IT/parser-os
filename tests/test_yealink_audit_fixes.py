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
    ]
    assert merge_signature_rows(atoms) == 5
    assert len(atoms) == 1
    signers = {s["party"]: s for s in atoms[0].value["signers"]}
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
    env = {"atoms": [{"artifact_id": "note", "atom_type": "deal_metadata", "structured": {"field_name": "hubspot_note_meta", "title": "psow from current partner", "author_email": "patrick@purtera-it.com"}}]}
    assert _link_caption_notes(docs, env, timeline) == 1
    pdf = docs[1]
    assert pdf["delivered_by"] == "patrick@purtera-it.com" and pdf["caption"] == "psow from current partner"
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
    ])
    names = [(a.artifact_id, a.value.get("name")) for a in out]
    assert names.count(("e2", "Quinton James")) + names.count(("e1", "Quinton James")) == 1
    assert ("e3", "Quinton James") in names
    bern = [a for a in out if a.value.get("name") == "Bernard Donnelly"]
    assert len(bern) == 1 and bern[0].value.get("title") == "Sr. CSDA"
    assert not any(a.value.get("name") is None for a in out)


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
