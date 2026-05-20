"""Site-extraction hardening tests.

A real deal that says "3 sites" must produce 3 site entity keys, NOT
30. The OPTBOT Atlanta deal package surfaced an 80% false-positive
rate because the proper-noun matcher tagged role titles, document
names, pipeline/system terms, and concept phrases as ``site:`` keys.

Each test below pins one class of regression:

  - Site codes (ATL-HQ / ATL-WEST / ATL-AIR) extracted as first-class entities
  - Role / person titles never extracted as sites
  - Document / artifact names never extracted as sites
  - Pipeline / system / cloud-resource terms never extracted as sites
  - Concept / process phrases never extracted as sites
  - Real sites with org-suffix tails ("Innovation Tower", "Atlanta Headquarters") survive
  - Known non-site hyphen codes (Wi-Fi, RJ-45) ignored
"""
from __future__ import annotations

import pytest

from app.core.entity_extraction import _emit_proper_nouns, _emit_sites


# ─── A. site codes ───


@pytest.mark.parametrize(
    "text,expected_key",
    [
        ("Site ATL-HQ contains the main MDF.", "site:atl_hq"),
        ("Allocation: ATL-HQ 52, ATL-WEST 27, ATL-AIR 15.", "site:atl_west"),
        ("Mobilization at ATL-AIR begins on 2026-05-20.", "site:atl_air"),
        ("Refresh sweeps NYC-DC1 in phase 2.", "site:nyc_dc1"),
        ("Survey team visits SFO-HQ next week.", "site:sfo_hq"),
        ("Phase 3 hits CHI-MAIN-EAST after the holiday.", "site:chi_main_east"),
    ],
)
def test_site_codes_are_extracted(text: str, expected_key: str) -> None:
    keys = _emit_sites(text)
    assert expected_key in keys, f"missing {expected_key}; got {sorted(keys)}"


@pytest.mark.parametrize(
    "text",
    [
        "Patch cables terminate in RJ-45 keystones at every drop.",
        "All runs use CAT-6 plenum cable.",
        "Wireless coverage is Wi-Fi 7 throughout the building.",
        "The PO-1234 purchase order ships next Friday.",
    ],
)
def test_non_site_hyphen_codes_are_ignored(text: str) -> None:
    keys = _emit_sites(text)
    site_keys = {k for k in keys if k.startswith("site:")}
    assert not site_keys, f"non-site hyphen code leaked as site: {sorted(site_keys)}"


# ─── B. role / person titles ───


@pytest.mark.parametrize(
    "phrase",
    [
        "Regional Facilities Manager Jane Doe",
        "Senior Procurement Manager",
        "Security Architecture Lead",
        "VP Workplace Operations",
        "Executive Sponsor Jordan Ames",
        "Director of Engineering",
        "Network Operations Engineer",
        "Senior Solutions Architect",
        "Chief Technology Officer",
        "Project Stakeholder List",
        "Vendor Project Manager",
        "Mock Vendor PM",
    ],
)
def test_role_titles_never_become_sites(phrase: str) -> None:
    keys = _emit_proper_nouns(phrase, vendor_keys=set())
    site_keys = {k for k in keys if k.startswith("site:")}
    assert not site_keys, (
        f"role title leaked as site: {sorted(site_keys)} from {phrase!r}"
    )


# ─── C. document / artifact names ───


@pytest.mark.parametrize(
    "phrase",
    [
        "Executive Deal Brief Mock Document",
        "Site Surveys DOCX",
        "Project Schedule XLSX",
        "Security Integration Notes Document",
        "Procurement Packet PDF",
        "OPTBOT Security and Integration Notes Mock Document",
        "Integration Notes Mock Document",
        "Procurement Packet Mock Document",
        "Hardware Bill of Materials Workbook",
        "Cutover Plan Spreadsheet",
        "Statement of Work Draft",
        "Revision History Memo",
    ],
)
def test_document_names_never_become_sites(phrase: str) -> None:
    keys = _emit_proper_nouns(phrase, vendor_keys=set())
    site_keys = {k for k in keys if k.startswith("site:")}
    assert not site_keys, (
        f"document name leaked as site: {sorted(site_keys)} from {phrase!r}"
    )


# ─── D. pipeline / system / cloud-resource terms ───


@pytest.mark.parametrize(
    "phrase",
    [
        "Azure Dev Storage",
        "HubSpot Dev Deal",
        "OrbitBrief Dev Workspace",
        "Global Gateway Connector",
        "Parser OS Dev Workers",
        "Azure Copy Job",
        "HubSpot Timeline Note",
        "Dev Environment Container",
        "Mock Confidential Pipeline",
        "Parser Extraction Baseline",
        "OrbitBrief Brief Generation",
    ],
)
def test_pipeline_terms_never_become_sites(phrase: str) -> None:
    keys = _emit_proper_nouns(phrase, vendor_keys=set())
    site_keys = {k for k in keys if k.startswith("site:")}
    assert not site_keys, (
        f"pipeline term leaked as site: {sorted(site_keys)} from {phrase!r}"
    )


# ─── E. concept / process phrases ───


@pytest.mark.parametrize(
    "phrase",
    [
        "Three Site Modernization",
        "Three Site Modernization HubSpot Deal",
        "Target Close Date",
        "Total Mock Amount",
        "Access and Security Controls Use",
        "Network Modernization Rollout",
        "Hardware Refresh Cutover",
        "Infrastructure Migration Plan",
        "Phase Two Implementation",
    ],
)
def test_concept_phrases_never_become_sites(phrase: str) -> None:
    keys = _emit_proper_nouns(phrase, vendor_keys=set())
    site_keys = {k for k in keys if k.startswith("site:")}
    assert not site_keys, (
        f"concept phrase leaked as site: {sorted(site_keys)} from {phrase!r}"
    )


# ─── F. real sites must survive ───


@pytest.mark.parametrize(
    "phrase,expected_substring",
    [
        ("Atlanta Headquarters Innovation Tower 1180 Peachtree Street NE", "site:"),
        ("Site visit: Airport Logistics Annex on May 20.", "site:"),
        ("Andrews Information Systems Building hosts the dispatch room.", "site:"),
        ("Perry Street Parking Deck has 1200 spaces.", "site:"),
        ("Carter G. Woodson Middle School completes the rollout.", "site:"),
    ],
)
def test_real_sites_are_still_captured(phrase: str, expected_substring: str) -> None:
    site_keys = _emit_proper_nouns(phrase, vendor_keys=set()) | _emit_sites(phrase)
    site_only = {k for k in site_keys if k.startswith("site:")}
    assert site_only, (
        f"real site missed; phrase={phrase!r}; got nothing site-shaped"
    )


def test_atl_west_no_longer_missed() -> None:
    """The OPTBOT regression: BOM allocation says
    ``ATL-HQ 52, ATL-WEST 27, ATL-AIR 15`` and ATL-WEST was the
    single missed real site in the prior run.
    """
    text = "Wi-Fi 7 APs: 94 units x $995 | allocated ATL-HQ 52, ATL-WEST 27, ATL-AIR 15"
    keys = _emit_sites(text)
    site_keys = {k for k in keys if k.startswith("site:")}
    assert "site:atl_hq" in site_keys
    assert "site:atl_west" in site_keys
    assert "site:atl_air" in site_keys


# ─── G. OPTBOT-shaped end-to-end regression ───


def test_optbot_paragraph_does_not_explode_sites() -> None:
    """A representative paragraph from the OPTBOT exec brief that
    previously emitted 8-10 false-positive site keys must now emit
    only the actual sites (ATL-HQ, ATL-WEST, ATL-AIR — none in this
    snippet so 0 is the right answer).
    """
    text = (
        "Jordan Ames, VP Workplace Operations, is the executive sponsor "
        "and approves business outcome and CFO escalations. Mock Vendor PM "
        "owns the project schedule XLSX and the integration notes mock "
        "document. The Azure Dev Storage container holds the parser-os dev "
        "batch ATL-047 OrbitBrief workspace artifacts. Three Site "
        "Modernization HubSpot Deal stage advances on cutover."
    )
    pn_keys = _emit_proper_nouns(text, vendor_keys=set())
    site_keys_pn = {k for k in pn_keys if k.startswith("site:")}
    # Nothing in this paragraph is a real site; all the capitalized
    # phrases are roles / documents / pipeline terms / concepts.
    assert not site_keys_pn, (
        f"OPTBOT-shaped paragraph leaked {len(site_keys_pn)} false sites: "
        f"{sorted(site_keys_pn)}"
    )


def test_real_optbot_atl_sites_resolve_from_full_text() -> None:
    """A representative passage that DOES carry the real sites must
    surface them, and nothing else.
    """
    text = (
        "Site ATL-HQ is the Atlanta Headquarters at the Innovation Tower, "
        "1180 Peachtree Street NE, Floors 12-15. ATL-WEST is the West "
        "Distribution Center. ATL-AIR is the Airport Logistics Annex."
    )
    keys = _emit_sites(text) | _emit_proper_nouns(text, vendor_keys=set())
    site_keys = {k for k in keys if k.startswith("site:")}
    # All three site codes captured
    assert "site:atl_hq" in site_keys
    assert "site:atl_west" in site_keys
    assert "site:atl_air" in site_keys
    # The named building survives the new filter
    assert "site:atlanta_headquarters" in site_keys or "site:innovation_tower" in site_keys
    # And nothing junk like "site:innovation_tower_1180" (a 4-word
    # mash) or "site:floors_12_15" (a range descriptor)
    junk_substrings = {"floors_12", "executive_sponsor"}
    for jk in junk_substrings:
        assert not any(jk in k for k in site_keys), (
            f"junk substring {jk!r} leaked into site_keys: {sorted(site_keys)}"
        )


# ─── H. site-code HEAD denylist (MOCK-/DEV-/MSA- never become sites) ───


@pytest.mark.parametrize(
    "text",
    [
        # The five exact OPTBOT leaks we just observed in /tmp/optbot_run/result.json
        "Use fictional Intune profile INTUNE-MOCK-OPTBOT-ATL-REFRESH.",
        "Quote: Q-DEV-ATL-047-R3 | Mock MSA: MOCK-MSA-2026-OPTBOT-001",
        "Reference ticket TKT-ATL-2026-001 for the change window.",
        "Workflow ID WF-ATL-FOO and project code PROJ-ATL-2026.",
        "Storage container azure-test/MOCK-ATL-REFRESH-DEV",
        # Generic family — anything matching MOCK-XXX / DEV-XXX / TEST-XXX
        "DEMO-ATL-001 is a sandbox project.",
        "FAKE-ATL artifacts must not appear in production.",
        "Run SAMPLE-ATL through the regression suite.",
        "Account HS-DEAL-ATL-2026 holds the test pipeline.",
        "Container ID INV-ATL-77421 is the procurement record.",
    ],
)
def test_site_code_head_denylist_blocks_id_prefixes(text: str) -> None:
    """Hyphenated codes that start with MOCK / DEV / MSA / INTUNE /
    PROJ / WF / TKT / HS / etc. must NOT emit site keys, even though
    their trailing segments look airport-shaped.
    """
    junk_substrings = {
        "mock_", "dev_atl", "msa_", "intune_", "demo_", "test_atl",
        "fake_", "sample_", "proj_", "wf_atl", "tkt_", "hs_deal",
        "inv_atl",
    }
    keys = _emit_sites(text)
    site_keys = {k for k in keys if k.startswith("site:")}
    for jk in junk_substrings:
        leaks = [k for k in site_keys if jk in k]
        assert not leaks, (
            f"head-denylist failure: {jk!r} leaked from {text!r} → {leaks}"
        )


# ─── I. structured-field pseudo-values (xlsx ALL / N/A / TBD) ───


@pytest.mark.parametrize(
    "value",
    ["ALL", "N/A", "n/a", "TBD", "TBA", "Various", "Multiple",
     "None", "—", "-", "Site", "Location", "Address", "See above",
     "see notes", "Unknown"],
)
def test_normalize_entity_key_drops_generic_site_values(value: str) -> None:
    """xlsx rows that put generic markers ("ALL", "N/A", "Various")
    in the site column must NOT produce site entities — those mean
    "applies everywhere", not "a place named ALL".
    """
    from app.core.normalizers import normalize_entity_key
    result = normalize_entity_key("site", value)
    assert result == "", (
        f"generic site value {value!r} should be filtered but got {result!r}"
    )


# ─── J. classification / mock / fictional words never become sites ───


@pytest.mark.parametrize(
    "phrase",
    [
        # The "site:mock_confidential_this" leak from OPTBOT
        "Mock Confidential This",
        "Mock Confidential. This",
        "Fictional Security Note",
        "Classification Mock Confidential",
        "Confidential Internal Document",
        "Restricted Test Sample",
        "Synthetic Dev Workspace",
    ],
)
def test_mock_classification_phrases_never_become_sites(phrase: str) -> None:
    keys = _emit_proper_nouns(phrase, vendor_keys=set())
    site_keys = {k for k in keys if k.startswith("site:")}
    assert not site_keys, (
        f"mock/classification phrase leaked: {sorted(site_keys)} from {phrase!r}"
    )
