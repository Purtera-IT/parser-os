"""A list marker alone on its line (Outlook's plain text) and HTML list items keep their marker, and a Title Case
"Heading:" over a list is a heading, not a name (live 010482: "Delivery" became a site)."""
from pathlib import Path

from app.parsers.email_body import email_content_to_text, join_lone_list_markers
from app.parsers.email_parser import EmailParser


def test_a_lone_marker_joins_the_item_below_and_line_numbers_hold():
    text = "Services:\n\n  *\nDelivery\n  *\nUnpacking\n1.\nAre carts available?"
    out = join_lone_list_markers(text).split("\n")
    assert len(out) == len(text.split("\n"))
    assert out[3] == "  * Delivery" and out[5] == "  * Unpacking" and out[7] == "1. Are carts available?"
    assert out[2] == "" and out[4] == ""


def test_a_marker_with_no_item_after_it_stays():
    assert join_lone_list_markers("*\n*\nEnd") == "*\n\n* End"
    assert join_lone_list_markers("End\n*") == "End\n*"


def test_html_list_items_keep_their_markers():
    html = "<html><body><table><tr><td>sig</td></tr></table><ul><li>Delivery</li></ul><ol><li>One</li><li>Two</li></ol></body></html>"
    lines = email_content_to_text(html).splitlines()
    assert "• Delivery" in lines and "1. One" in lines and "2. Two" in lines


EML = """From: aj@example.com
To: david@example.com
Subject: Re: request
Date: Thu, 12 Nov 2026 15:19:45 +0000
Content-Type: text/plain; charset=utf-8

Hi David,
Initial ask below:

  *
Qty: 6 | Rackmount UPS
P/N: SRT8KRMXLT

White Glove Installation Services:

  *
Delivery
  *
Unpacking
  *
Power-up verification

Thank you,
AJ
"""


def test_the_items_keep_their_heading_and_marker(tmp_path: Path):
    p = tmp_path / "m.eml"
    p.write_text(EML, encoding="utf-8")
    by = {a.raw_text: a for a in EmailParser().parse(p)}
    for t in ("Delivery", "Unpacking", "Power-up verification"):
        v = by[t].value
        assert v.get("list_marker") == "*"
        assert v.get("lead_in") == ["White Glove Installation Services:"]
        assert getattr(by[t].atom_type, "value", by[t].atom_type) != "physical_site"
