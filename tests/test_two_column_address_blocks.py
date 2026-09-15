"""Two address blocks side by side are two addresses, not a table (010043's CDW order)."""
import fitz

from app.parsers.pdf.tables import _address_columns, _extract_column_tables

LEFT = ["LANE CONSTRUCTION", "90 FIELDSTONE CT", "CHESHIRE, CT 06410-1212", "Phone: (203) 235-3351"]
RIGHT = ["LANE CONSTRUCTION", "6125 TYVOLA CENTRE DR", "CHARLOTTE, NC 28217-6432", "Shipping Method: DROP SHIP-GROUND"]


def _pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i, (l, r) in enumerate(zip(LEFT, RIGHT)):
        y = 120 + i * 16
        page.insert_text((60, y), l, fontsize=10)
        page.insert_text((330, y), r, fontsize=10)
    p = tmp_path / "order.pdf"
    doc.save(str(p))
    return p


def test_bill_to_and_ship_to_columns_each_read_as_one_address(tmp_path):
    blocks, _ = _extract_column_tables(_pdf(tmp_path), 0)
    paras = [b for b in blocks if b.get("kind") == "paragraph" and b.get("extraction") == "address_column_v1"]
    assert len(paras) == 2, blocks
    assert "90 FIELDSTONE CT, CHESHIRE, CT 06410-1212" in paras[0]["text"]
    assert "6125 TYVOLA CENTRE DR, CHARLOTTE, NC 28217-6432" in paras[1]["text"]
    assert not any(b.get("kind") == "table" for b in blocks), "the pair is not a key: value table"


def test_a_label_column_beside_one_address_is_still_a_form():
    rows = [["Name", "Austin Coryell"], ["Address", "6125 Tyvola Centre Dr"], ["City", "Charlotte, NC 28217"], ["Date", "6/18/2026"]]
    assert _address_columns(rows) == []
