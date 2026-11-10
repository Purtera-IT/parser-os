"""A dash record line stays whole, a flattened list splits, siblings agree.

A note listed sites one per line, each "City, ST – Facility – Equipment –
Task". The inline-dash splitter judged each line by the case of its words:
"... – just need pack/ship" stayed whole and "... – Need de-rack, pack and
ship" came apart into four atoms. A repeated facility name was then located
at the first body line holding it, and a repeat on a second line was dropped.

The decision is by shape: short pieces that are different attributes of one
thing (where / facility / equipment / task) are one record; a lead that
announces a list, a long topic lead with multi-word items, or parallel items
of one kind are a list. Letter case and place names play no part.

All text here is made up.
"""
from __future__ import annotations

from pathlib import Path

from app.core.sentences import split_inline_dash_lines, split_inline_dash_list
from app.parsers.email_parser import EmailParser
from app.parsers.hubspot_note_parser import HubspotNoteParser

SITE_A = "Tucson, AZ – Acme Vault – Boxed servers – just need pack/ship"
SITE_B = "Mesa, AZ – Acme Vault – Rack gear – Need de-rack, pack and ship"
STREET = "120 Example St, Tucson, AZ 85701 - Acme Vault - Boxed servers - Need pickup"

LIST_LEAD = "Ongoing support of the office network and servers"
LIST_ITEMS = [
    "Support for switches and access points",
    "Support for the storage array",
    "Monthly patching (Windows + Linux)",
    "24/7 on-call availability",
]
LIST_LINE = " - ".join([LIST_LEAD, *LIST_ITEMS])


def _note(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "999001-hs-note-555.txt"
    p.write_text(
        "HubSpot Note: Site list\nHubSpot Note ID: 555\nDate: 2026-06-02T10:00:00Z\n"
        "Author: Pat Example\n\n" + body + "\n",
        encoding="utf-8",
    )
    return p


def _content(atoms):
    return [a for a in atoms if (a.value or {}).get("kind") != "hubspot_note_meta"]


def test_record_line_is_not_split():
    assert split_inline_dash_list(SITE_A) == []
    assert split_inline_dash_list(SITE_B) == []
    assert split_inline_dash_list(STREET) == []


def test_two_record_lines_differing_only_in_case_stay_whole():
    a = "Mesa, AZ – Acme Vault – Rack gear – need de-rack, pack and ship"
    b = "Mesa, AZ – Acme Vault – Rack gear – Need De-Rack, Pack And Ship"
    assert split_inline_dash_list(a) == [] == split_inline_dash_list(b)
    assert split_inline_dash_list(a.upper()) == []
    assert split_inline_dash_lines([a, b]) == [[], []]


def test_record_line_with_no_place_name_stays_whole():
    assert split_inline_dash_list("Rack 4 – Cisco 9300 – needs reimage – ship back") == []
    assert split_inline_dash_list("Acme Vault - Boxed servers - Rack 4 - Need pack and ship") == []


def test_flattened_lists_split():
    # A lead that announces the list.
    assert split_inline_dash_list("Network refresh includes - switches - access points - firewalls") == [
        "Network refresh includes", "switches", "access points", "firewalls"]
    assert split_inline_dash_list("Deliverables: - Site survey - As-built drawings - Test results")[1:] == [
        "Site survey", "As-built drawings", "Test results"]
    # Parallel items of one kind, no lead-in.
    assert split_inline_dash_list("Cisco 9300 - Meraki MR46 - Palo Alto 440 - Fortinet 60F") == [
        "Cisco 9300", "Meraki MR46", "Palo Alto 440", "Fortinet 60F"]
    assert split_inline_dash_list(
        "Weekly tasks - Install the new APs - Install the new switches - Install the UPS"
    )[1:] == ["Install the new APs", "Install the new switches", "Install the UPS"]
    # Case does not change a list either.
    assert split_inline_dash_list(LIST_LINE.lower()) == [p.lower() for p in [LIST_LEAD, *LIST_ITEMS]]


def test_old_style_flattened_list_still_splits():
    assert split_inline_dash_list(LIST_LINE) == [LIST_LEAD, *LIST_ITEMS]
    assert split_inline_dash_lines([LIST_LINE, "Locations", "Tucson, AZ"])[0] == [LIST_LEAD, *LIST_ITEMS]


def test_mixed_block_siblings_agree():
    # Same dash shape, one block: one line's pieces look parallel, the
    # other's are a record. Both get one answer (whole).
    a = "Cisco 9300 - Meraki MR46 - Palo Alto 440 - Fortinet 60F"
    b = "Rack 4 - Cisco 9300 - needs reimage - ship back"
    assert split_inline_dash_list(a) and not split_inline_dash_list(b)
    assert split_inline_dash_lines([a, b]) == [[], []]
    assert split_inline_dash_lines([b, a]) == [[], []]
    # A blank line ends the block: apart, each is decided on its own.
    assert split_inline_dash_lines([a, "", b])[0] == split_inline_dash_list(a)
    # A lead that announces a list is its own shape: it is not pulled whole
    # by record siblings beside it.
    c = "Phase one includes - unbox - rack - cable"
    assert split_inline_dash_lines([c, b])[0] == ["Phase one includes", "unbox", "rack", "cable"]


def test_note_sibling_site_lines_both_stay_whole(tmp_path: Path):
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art_note", _note(tmp_path, f"{SITE_A}\n{SITE_B}")))
    texts = [a.raw_text for a in atoms]
    assert SITE_A in texts
    assert SITE_B in texts
    assert "Need de-rack, pack and ship" not in texts


def test_note_flattened_list_still_splits(tmp_path: Path):
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art_note", _note(tmp_path, LIST_LINE)))
    texts = {a.raw_text for a in atoms}
    assert LIST_LEAD in texts
    for item in LIST_ITEMS:
        assert item in texts, item


def test_repeated_facility_on_two_lines_keeps_both_with_own_line(tmp_path: Path):
    # Two flattened lists, each naming the same facility as an item.
    line1 = "Phase one scope includes - Acme Vault - Unbox and stage - Label every port"
    line2 = "Phase two scope includes - Acme Vault - Rack and cable - Test every link"
    path = _note(tmp_path, f"{line1}\n{line2}")
    raw = path.read_text(encoding="utf-8").splitlines()
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art_note", path))
    vaults = [a for a in atoms if a.raw_text == "Acme Vault"]
    assert len(vaults) == 2
    assert len({a.id for a in vaults}) == 2
    locs = sorted((a.source_refs[0].locator["line_start"], a.source_refs[0].locator["char_start"],
                   a.source_refs[0].locator["char_end"]) for a in vaults)
    l1 = raw.index(line1) + 1
    l2 = raw.index(line2) + 1
    assert [loc[0] for loc in locs] == [l1, l2]
    for (ln, c0, c1) in locs:
        assert raw[ln - 1][c0:c1] == "Acme Vault"


def test_same_words_twice_on_one_line_still_one_atom(tmp_path: Path):
    line = "Bring the ladder. Bring the ladder."
    atoms = _content(HubspotNoteParser().parse_artifact("p", "art_note", _note(tmp_path, line)))
    assert [a.raw_text for a in atoms].count("Bring the ladder.") == 1


def test_email_sibling_site_lines_both_stay_whole(tmp_path: Path):
    # A plain .eml with no charset: hyphens, so the dashes survive decoding.
    site_a, site_b = SITE_A.replace("–", "-"), SITE_B.replace("–", "-")
    p = tmp_path / "sites.eml"
    p.write_text(
        "From: Ops <ops@example.com>\nTo: t@purtera-it.com\nSubject: Sites\n"
        "Date: Mon, 1 Jun 2026 10:00:00 -0400\n\n"
        f"Hi,\n\n{site_a}\n{site_b}\n\nThanks,\nOps\n",
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact(project_id="p", artifact_id="art_mail", path=p)
    texts = [" ".join(a.raw_text.split()) for a in atoms]
    assert any(site_a in t for t in texts)
    assert any(site_b in t for t in texts)
    assert not any(t == "- Need de-rack, pack and ship" for t in texts)


def test_email_site_rows_under_a_location_label_are_all_kept(tmp_path: Path):
    # A "Site Address:" label read the next lines as its name / street /
    # city and swallowed them when no address came out: three whole site
    # rows vanished once they stopped being split into pieces.
    rows = [
        "ALPH        12 Example St, Tucson, AZ 85701   - 33 APs   - After Hours Install   - Clinic",
        "BETA        40 Sample Ave, Mesa, AZ 85201     - 4 APs    - After Hours Install   - Clinic",
        "GAMM        9 Test Rd, Tempe, AZ 85281        - 12 APs   - After Hours Install   - Clinic",
    ]
    p = tmp_path / "sites.eml"
    p.write_text(
        "From: Ops <ops@example.com>\nTo: t@purtera-it.com\nSubject: Sites\n"
        "Date: Mon, 1 Jun 2026 10:00:00 -0400\n\n"
        "Hi,\n\nSite Address:\n\n" + "\n\n".join(rows) + "\n\nThanks,\nOps\n",
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact(project_id="p", artifact_id="art_mail", path=p)
    texts = [" ".join(a.raw_text.split()) for a in atoms]
    for row in rows:
        assert any(t.startswith(row[:4]) and "Clinic" in t for t in texts), row[:4]
