"""A deal should say when it is holding another deal's documents."""
from app.core.orbitbrief_envelope import _deal_number_from_crm, _foreign_artifacts

CRM = {"deal_name": "010129 - GHA Assa Abbloy network swap + cabling"}


def _docs(*names):
    return [{"filename": n, "artifact_id": f"art_{i}"} for i, n in enumerate(names)]


def test_deal_number_comes_from_the_crm_deal_name():
    assert _deal_number_from_crm(CRM) == "010129"
    assert _deal_number_from_crm({"deal_name": "no number here"}) is None
    assert _deal_number_from_crm(None) is None


def test_a_distant_deal_number_is_flagged():
    # The real case: a GHA deal holding a Marco's Pizza SOW, which OrbitBrief
    # then briefed as though it were the deal's own scope.
    out = _foreign_artifacts(
        crm=CRM, documents=_docs("010013 Marcos New Store Installs (6.12.26_v2).docx")
    )
    assert len(out) == 1
    assert out[0]["claims_deal"] == "010013"
    assert out[0]["deal_number"] == "010129"


def test_the_deals_own_artifacts_are_not_flagged():
    assert _foreign_artifacts(crm=CRM, documents=_docs("010129-hs-email-113272561734.eml")) == []


def test_an_adjacent_number_is_not_flagged():
    # Adjacent numbers are overwhelmingly the same customer and project — a
    # renumber, or a survey/install pair sharing documents. 60 of 79 corpus
    # mismatches were adjacent; flagging them makes the signal 76% false.
    assert _foreign_artifacts(crm=CRM, documents=_docs("010130 - same customer v2.docx")) == []
    assert _foreign_artifacts(crm=CRM, documents=_docs("010127 - same customer.docx")) == []


def test_digits_inside_uuids_and_screenshots_are_not_deal_numbers():
    # An unanchored six-digit search pulls "878374" out of a UUID and "150656"
    # out of a screenshot stamp. Both are noise, not identity.
    assert _foreign_artifacts(
        crm=CRM,
        documents=_docs(
            "Work-order-extra-photos-7de99fd3-7d3d-42dc-ac52-b878374bd7c1.jpeg",
            "Screenshot 2026-08-17 150656.png",
            "358758.jpeg",
        ),
    ) == []


def test_no_crm_means_no_claim():
    # Without an authoritative deal number, guessing one from the documents
    # being checked would let a contaminated deal validate itself.
    assert _foreign_artifacts(crm=None, documents=_docs("010013 Marcos.docx")) == []


CENTRICS = {
    "deal_name": "010128 - CentricsIT Marcos - MOMS POS Installation- Harrisburg, PA",
    "account_name": "CentricsIT",
}


def test_hint_reads_the_deal_name_not_just_the_account():
    # The account is often the reseller while the document names the end
    # customer. CentricsIT's deal is for Marco's, and the shared runbook is
    # "010013 Marcos New Store Installs" — recognisable from the deal name,
    # invisible from the account alone.
    out = _foreign_artifacts(
        crm=CENTRICS, documents=_docs("010013 Marcos New Store Installs (6.12.26_v2).docx")
    )
    assert len(out) == 1
    assert out[0]["same_account_hint"] is True


def test_a_genuinely_different_customer_still_reads_false():
    out = _foreign_artifacts(
        crm={"deal_name": "010129 - GHA Assa Abbloy network swap", "account_name": "GHA Technologies"},
        documents=_docs("010013 Marcos New Store Installs (6.12.26_v2).docx"),
    )
    assert out[0]["same_account_hint"] is False


def test_generic_words_alone_do_not_link_a_document():
    # "Installation" and "Services" appear everywhere; matching on them would
    # mark every misfile as a sibling.
    out = _foreign_artifacts(
        crm={"deal_name": "010200 - Acme Installation Services", "account_name": "Acme"},
        documents=_docs("010111 - Globex Installation Services.docx"),
    )
    assert out[0]["same_account_hint"] is False
