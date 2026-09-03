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
        _atom(AtomType.stakeholder, "Carl Painter | Sr. Account Manager", kind="person", name="Carl Painter", role="Sr. Account Manager", email="carlpai@cdw.com"),
        _atom(AtomType.stakeholder, "Jacob Long Project Manager Phone and ITAD", kind="person", name="Jacob Long", role="Project Manager Phone and ITAD"),
    ]
    kept, dropped = drop_contextless_stakeholders(atoms)
    assert [a.value.get("name") for a in dropped] == [None, "The Buyer"]
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
