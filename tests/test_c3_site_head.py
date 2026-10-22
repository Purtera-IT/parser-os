"""The site heads decide sites; rules only extract (SOWSMITH_C3_SITE_HEAD).

sites_audit.md section 4: the type head's physical_site answer was parked as
``c3_base_says`` and reached the atom only through an LLM fill behind two off
flags; two rule stages removed or retyped head-typed sites with no lock check;
the quote was sent a site count that ignored the head; and the site heads
(jdg:site, jdg:roster, jdg:sheet, jdg:facility) were never asked. All text is
made up; the reads are stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace as N

import pytest

from app.core import c3_type_base as C
from app.core.schemas import (ArtifactType, AtomType, AuthorityClass, EvidenceAtom,
                              ReviewStatus, SourceRef)


@pytest.fixture(autouse=True)
def _deal(monkeypatch):
    for name in ("SOWSMITH_C3_SITE_HEAD", "SOWSMITH_C3_SURE", "SOWSMITH_C3_SURE_BY_TYPE",
                 "SOWSMITH_C3_BRIEF", "SOWSMITH_C3_AUTHORITY"):
        monkeypatch.delenv(name, raising=False)
    C.set_deal([], {})
    yield
    C.set_deal(None)


def _on(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_SITE_HEAD", "on")


def _line(aid, text, type_str="deal_metadata", value=None, page=None, artifact="art_1"):
    return EvidenceAtom(
        id=aid, project_id="d1", artifact_id=artifact, atom_type=AtomType(type_str),
        raw_text=text, normalized_text=text.lower(), value=dict(value or {}), entity_keys=[],
        source_refs=[SourceRef(id=f"src_{aid}", artifact_id=artifact, artifact_type=ArtifactType.pdf,
                               filename="sow.pdf", locator={"page": page} if page is not None else {},
                               extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t")


# ------------------------------------------------- the head's site reaches the atom

def test_a_sure_site_reading_types_the_line_with_its_own_fields(monkeypatch):
    _on(monkeypatch)
    C._DEAL["types"] = {"s1": ["physical_site", 0.93]}
    a = _line("s1", "Install at 4100 Harbor View Rd, Tampa, FL 33602 before go-live")
    assert C.apply_site_head([a]) == 1
    assert a.atom_type == AtomType.physical_site
    v = a.value
    assert (v["street_address"], v["city"], v["state"], v["zip"]) == ("4100 Harbor View Rd", "Tampa", "FL", "33602")
    assert v["c3_label"]["act"] == "authority" and v["c3_label"]["by"] == "site_head"
    assert v["rule_type"] == "deal_metadata" and "deal_metadata" in v["alt_atom_types"]
    assert C.is_locked(a)


def test_no_llm_fill_and_no_authority_flag_needed(monkeypatch):
    """It was parked as c3_base_says for the fill behind SOWSMITH_C3_TYPE +
    SOWSMITH_C3_AUTHORITY; with the site flag alone it is applied."""
    _on(monkeypatch)
    assert not C.authority_enabled()
    C._DEAL["types"] = {"s1": ["physical_site", 0.9]}
    a = _line("s1", "Site: Fairfield, CA", value={"c3_base_says": {"type": "physical_site", "p": 0.9}})
    assert C.apply_site_head([a]) == 1
    assert "c3_base_says" not in a.value
    assert (a.value["city"], a.value["state"], a.value["name"]) == ("Fairfield", "CA", "Fairfield, CA")


def test_below_the_bar_or_flag_off_nothing_changes(monkeypatch):
    C._DEAL["types"] = {"s1": ["physical_site", 0.95], "s2": ["physical_site", 0.5]}
    a, b = _line("s1", "100 Main St, Austin, TX 78701"), _line("s2", "200 Main St, Austin, TX 78701")
    assert C.apply_site_head([a, b]) == 0                        # flag off
    _on(monkeypatch)
    assert C.apply_site_head([a, b]) == 1
    assert a.atom_type == AtomType.physical_site and b.atom_type == AtomType.deal_metadata


def test_a_town_only_site_carries_city_and_state():
    """The Deal Kit drops a street-less site without city + state once any site
    has an address; own_site_keys gave town-only sites an id and name only."""
    a = N(atom_type=AtomType("physical_site"), raw_text="Location is Fairfield, CA.", entity_keys=[], value={})
    assert C.own_site_keys([a]) == 1
    assert (a.value["city"], a.value["state"]) == ("Fairfield", "CA")


# ----------------------------------------------------- rules respect the head lock

def _canonical():
    return _line("atm_addr", "Address: 7 Penn Plaza West 31st Street, Floor 12", "physical_site",
                 {"id": "7_penn_plaza", "site_id": "7_penn_plaza",
                  "street_address": "7 Penn Plaza West 31st Street", "facility_name": "7 Penn Plaza"})


def test_semantic_dedup_does_not_remove_a_head_typed_name_only_site(monkeypatch):
    from app.core.semantic_dedup import _dedupe_physical_site_atoms

    alias = lambda: _line("atm_alias", "7 Penn Plaza", "physical_site", {"id": "7_penn", "site_id": "7_penn"})  # noqa: E731
    out = _dedupe_physical_site_atoms([_canonical(), alias()])
    assert "7 Penn Plaza" not in {a.raw_text for a in out}       # flag off: an alias, removed as before
    _on(monkeypatch)
    C._DEAL["types"] = {"atm_alias": ["physical_site", 0.97]}
    out = {a.raw_text: a for a in _dedupe_physical_site_atoms([_canonical(), alias()])}
    assert "7 Penn Plaza" in out and out["7 Penn Plaza"].atom_type == AtomType.physical_site


def test_semantic_dedup_never_demotes_a_head_typed_site(monkeypatch):
    from app.core.semantic_dedup import _demote_from_site

    _on(monkeypatch)
    C._DEAL["types"] = {"s1": ["physical_site", 0.97]}
    a = _line("s1", "Lease signed for the Dock 4 building", "physical_site", {"site_id": "dock_4"})
    _demote_from_site(a)
    assert a.atom_type == AtomType.physical_site and a.value["site_id"] == "dock_4"


def test_party_address_veto_leaves_a_head_typed_site(monkeypatch):
    from app.core.party_address_veto import veto_party_page_sites

    def atoms():
        site = _line("s1", "Customer site: 55 Dock Rd, Mobile, AL 36602", "physical_site", page=6)
        site.entity_keys = ["site:55_dock_rd_mobile_al_36602"]
        return site, _line("g1", "Acme Corp: By: J. Doe", "signatory", page=6)

    site, sig = atoms()
    assert veto_party_page_sites([site, sig]) == 1 and site.atom_type == AtomType.deal_metadata  # flag off
    _on(monkeypatch)
    C._DEAL["types"] = {"s1": ["physical_site", 0.96]}
    site, sig = atoms()
    assert veto_party_page_sites([site, sig]) == 0
    assert site.atom_type == AtomType.physical_site and site.entity_keys == ["site:55_dock_rd_mobile_al_36602"]


# --------------------------------------------------------- the quote's site count

def _sites(n):
    return [N(id=f"x{i}") for i in range(n)]


def test_quote_count_is_the_declared_count_when_the_head_is_sure(monkeypatch):
    from app.core import declared_scope

    monkeypatch.setattr(declared_scope, "_found_site_count", lambda atoms: 3)
    C._DEAL["deal"] = {"deal:declared_site_count": [12.4, 0.91]}
    assert C.quote_site_count([]) == (3, "site_list")            # flag off
    _on(monkeypatch)
    assert C.quote_site_count([]) == (12, "declared_site_count")
    C._DEAL["deal"] = {"deal:declared_site_count": [12.4, None]}  # the live head is a regression: no p
    assert C.quote_site_count([]) == (3, "site_list")
    C._DEAL["deal"] = {"deal:declared_site_count": [12.4, 0.4]}
    assert C.quote_site_count([]) == (3, "site_list")


def test_the_quote_request_carries_the_count(monkeypatch):
    seen = {}
    _on(monkeypatch)
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    C._DEAL["deal"] = {"deal:declared_site_count": [7.0, 0.95]}
    monkeypatch.setattr(C, "_request", lambda path, body, timeout: seen.update(body=body) or {"lines": []})
    C.quote([], {"deal": {"deal_name": "x"}})
    assert seen["body"]["deal"]["site_count"] == 7


# -------------------------------------------- the site heads are asked and carried

def test_the_read_asks_the_site_heads_only_when_on(monkeypatch):
    assert "site_pairs" not in C._brief_body({"document_heads": ["jdg:document_job"]})
    _on(monkeypatch)
    body = C._brief_body({"document_heads": ["jdg:document_job"]})
    assert body["site_pairs"] is True and body["full"] is True
    assert body["document_heads"] == ["jdg:document_job", "jdg:roster", "jdg:sheet"]
    assert set(C.SITE_LINE_HEADS) <= set(body["line_heads"])


def test_the_read_keeps_site_documents_and_pairs_apart_from_the_document_job(monkeypatch):
    _on(monkeypatch)
    got = {"documents": {"art_1": {"jdg:document_job": ["this_deal", 0.9], "jdg:roster": ["site_roster", 0.88],
                                   "jdg:sheet": ["scope", 0.7]}},
           "site_pairs": [["s1", "s2", "distinct_site", 0.81]]}
    C._keep_full(got)
    assert C._DEAL["site_docs"] == {"art_1": {"jdg:roster": ("site_roster", 0.88), "jdg:sheet": ("scope", 0.7)}}
    assert C._DEAL["site_pairs"] == [["s1", "s2", "distinct_site", 0.81]]


def test_site_heads_ride_on_each_site_for_the_deal_kit(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("SOWSMITH_C3_SURE", "0.7")
    C._DEAL["types"] = {"s1": ["physical_site", 0.95]}
    C._DEAL["reads"] = {"s1": {"jdg:facility": ["keep_facility", 0.8]}}
    C._DEAL["site_docs"] = {"art_1": {"jdg:roster": ("site_roster", 0.9)}}
    C._DEAL["site_pairs"] = [["s1", "s2", "distinct_site", 0.82], ["s1", "gone", "distinct_site", 0.9],
                             ["s2", "s3", "same_site", 0.5]]
    s1 = _line("s1", "Dock 4, 55 Dock Rd, Mobile, AL 36602", "physical_site")
    s2 = _line("s2", "Dock 5, 57 Dock Rd, Mobile, AL 36602", "physical_site")
    s3 = _line("s3", "Dock 5 annex", "physical_site")
    out = C.site_heads([s1, s2, s3])
    assert s1.value["c3_site"] == {"facility": ["keep_facility", 0.8], "roster": ["site_roster", 0.9],
                                   "distinct_from": ["s2"], "head_site": 0.95}
    assert s2.value["c3_site"]["distinct_from"] == ["s1"]
    assert "same_as" not in s3.value.get("c3_site", {})         # under the bar
    assert len(out["pairs"]) == 3 and out["site_count"]["source"] == "site_list"
    monkeypatch.delenv("SOWSMITH_C3_SITE_HEAD")
    assert C.site_heads([s1]) is None


# ------------------------------------------- site_geo_fallback: the vendor-address drop

_STORES_010194 = [
    ("DCA_Washington_529", "1110 U Street S.E."),
    ("PHL_Philadelphia_176", "69-73 W. Chelten Ave."),
    ("PHL_Philadelphia_132", "6161 Ridge Avenue, Suite A"),
    ("PHL_Philadelphia_290", "3625 Tulip Street, Store #7"),
    ("PHL_Philadelphia_RL_1567", "2715 S. Front Street, Suite 5"),
]


def _stores_010194():
    rows = [_line(f"st{i}", f"site_id: {sid} | address: {addr}", "physical_site",
                  {"kind": "physical_site", "site_id": sid, "street_address": addr}, artifact="art_xlsx")
            for i, (sid, addr) in enumerate(_STORES_010194)]
    letterhead = _line("lh", "Acme Networks LLC | 400 Commerce Blvd, Suite 210, Austin, TX 78701", "physical_site",
                       {"kind": "physical_site", "street_address": "400 Commerce Blvd, Suite 210"})
    return rows, letterhead


def _store_says_vendor(monkeypatch):
    """010194: a stored correction (not a teacher row) called every store row a
    letterhead address, and all 33 labeled job-site rows were dropped."""
    from app.core import decide as D

    store_rows = {addr for _, addr in _STORES_010194}

    def decide(kind, addr, *a, **k):
        verdict = "vendor_or_billing_address" if addr in store_rows else "job_site"
        return N(verdict=verdict, confidence=0.95, source="store", correction_id="corr_42")

    monkeypatch.setattr(D, "decide", decide)


def test_010194_store_rows_survive_the_vendor_drop_when_the_head_types_them(monkeypatch):
    from app.core.site_geo_fallback import suppress_vendor_sites

    _store_says_vendor(monkeypatch)
    rows, letterhead = _stores_010194()
    kept, dropped = suppress_vendor_sites(rows + [letterhead], project_id="010194")
    assert dropped == 5 and [a.id for a in kept] == ["lh"]   # flag off: the store drops every store row
    _on(monkeypatch)
    C._DEAL["types"] = {r.id: ["physical_site", 0.9] for r in rows}
    C._DEAL["types"]["lh"] = ["stakeholder", 0.92]           # the head reads the letterhead as a party's block
    rows, letterhead = _stores_010194()
    kept, dropped = suppress_vendor_sites(rows + [letterhead], project_id="010194")
    assert {a.id for a in kept} == {r.id for r in rows} and dropped == 1
    assert letterhead.value["_decision"]["source"] == "c3"


def test_the_vendor_call_is_the_heads_not_the_stores(monkeypatch):
    from app.core.site_geo_fallback import suppress_vendor_sites

    _store_says_vendor(monkeypatch)
    _on(monkeypatch)
    rows, letterhead = _stores_010194()
    # No head is sure about any line: nothing is dropped, whatever the store says.
    kept, dropped = suppress_vendor_sites(rows + [letterhead], project_id="010194")
    assert dropped == 0 and len(kept) == 6
    # col:supplier on our side, or jdg:site_role vendor, at the bar: the heads drop it.
    C._DEAL["reads"] = {"lh": {"col:supplier": ["us", 0.88]}}
    kept, dropped = suppress_vendor_sites(rows + [letterhead], project_id="010194")
    assert dropped == 1 and "lh" not in {a.id for a in kept}


def test_site_role_alone_only_proposes_and_a_roster_row_is_not_judged(monkeypatch):
    """Live jdg:site_role reads 26 of 010447's DC roster rows ("dc 5593 | Yes | No |
    PA-850 | Yes | 1") as vendor addresses at 0.70-0.97; col:supplier reads them customer."""
    from app.core.site_geo_fallback import VENDOR_SUSPECT_FLAG, suppress_vendor_sites

    _on(monkeypatch)
    rows, letterhead = _stores_010194()
    C._DEAL["reads"] = {r.id: {"jdg:site_role": ["vendor_or_billing_address", 0.95],
                               "col:supplier": ["customer", 0.9]} for r in rows}
    kept, dropped = suppress_vendor_sites(rows + [letterhead], project_id="010447")
    assert dropped == 0 and len(kept) == 6
    assert all(VENDOR_SUSPECT_FLAG in r.review_flags for r in rows)
    roster = _line("dc", "dc 5593 | Yes | No | PA-850 | Yes | 1", "physical_site", {"site_id": "dc 5593"})
    roster.source_refs[0].extraction_method = "xlsx_site_roster_v1"
    C._DEAL["types"] = {"dc": ["stakeholder", 0.99]}
    kept, dropped = suppress_vendor_sites([roster, letterhead], project_id="010447")
    assert dropped == 0
