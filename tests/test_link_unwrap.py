"""Live 010289: the ask's diagram arrived triple-wrapped (Outlook safelinks
around Proofpoint urldefense around the vendor's PNG). The PM needs the
picture, so the real URL has to come back out."""
from __future__ import annotations

from app.core.link_unwrap import is_image_url, unwrap_link

WRAPPED = (
    "https://nam13.safelinks.protection.outlook.com/?url=https%3A%2F%2Furldefense.com%2Fv3%2F__https%3A%2F"
    "huzzard.com%2Fwp-content%2Fuploads%2F2025%2F11%2FBPW061725-Rev-1-Barcode-Door-Access-Control-Diagram-"
    "PC-Controller-With-RS232-EXT-KIT-768x593.png__%3B!!HUqgN_M!ptE5tqP3HVLh3XO9QNUkAM0A%24&data=05%7C02%7C"
    "aj%40purtera-it.com%7C5dd04f5bc16c4843f45308df08ffd4a7"
)
DIRECT = (
    "https://huzzard.com/wp-content/uploads/2025/11/"
    "BPW061725-Rev-1-Barcode-Door-Access-Control-Diagram-PC-Controller-With-RS232-EXT-KIT-768x593.png"
)


def test_two_gateways_peel_down_to_the_vendors_png():
    assert unwrap_link(WRAPPED) == DIRECT
    assert is_image_url(DIRECT)


def test_unwrapping_is_idempotent_and_leaves_plain_links_alone():
    assert unwrap_link(DIRECT) == DIRECT
    assert unwrap_link(unwrap_link(WRAPPED)) == DIRECT
    assert unwrap_link("https://purtera-it.com/a page") == "https://purtera-it.com/a page"
    assert unwrap_link("") == ""
    assert not is_image_url("https://purtera-it.com/quote.pdf")
    assert not is_image_url("")


def test_the_note_diagram_field_becomes_an_image_atom(tmp_path):
    from app.parsers.hubspot_note_parser import HubspotNoteParser

    p = tmp_path / "010289-hs-note-1-The Ask.txt"
    p.write_text(
        "HubSpot Note: The Ask\nHubSpot Note ID: 1\nDate: 2026-09-08T14:39:55.273Z\nAuthor: AJ Evans\n"
        f"Author-Email: aj@purtera-it.com\n\nThe Ask Here are the details for the small job. "
        f"Provided by us: -Relay -PC with Access Control Software. Diagram: {WRAPPED}\n",
        encoding="utf-8",
    )
    img = [a for a in HubspotNoteParser().parse(p) if (a.value or {}).get("media_type") == "image"]
    assert len(img) == 1
    assert img[0].value["image_url"] == DIRECT
    assert img[0].value["wrapped_url"].startswith("https://nam13.safelinks")
