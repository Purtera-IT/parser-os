from app.parsers.pdf.site_roster import _tiny_table


class Page:
    def __init__(self, words):
        self.words = words

    def get_text(self, _kind):
        return self.words


def _row(y, cells):
    """Words at fixed x, 2.5 pt type, rows 3.3 pt apart: one (x0, y0, x1, y1, text) per word."""
    out = []
    for x, text in cells:
        at = x
        for word in text.split():
            out.append((at, y, at + len(word) * 1.0, y + 3.0, word))
            at += len(word) * 1.0 + 0.6
    return out


HEADER = [(52, "HC"), (66, "Store Name"), (147, "Region"), (237, "City"), (297, "Address"), (387, "Zip"), (454, "Email")]


def _rows(start):
    out = []
    for i in range(14):
        n = start + i
        out += _row(60 + i * 3.3, [(56, str(n)), (66, f"Store {n}"), (147, "TEN"), (237, "Knoxville"),
                                   (297, f"{n} Main St"), (392, "37912"), (454, f"hc{n}@x.com")])
    return out


def test_a_shrunk_spreadsheet_reads_as_rows_under_its_header():
    table, columns = _tiny_table(Page(_row(52, HEADER) + _rows(100)), None)
    assert table.rows[0] == ["HC", "Store Name", "Region", "City", "Address", "Zip", "Email"]
    assert table.rows[1] == ["100", "Store 100", "TEN", "Knoxville", "100 Main St", "37912", "hc100@x.com"]
    assert len(table.rows) == 15 and len(columns) == 7


def test_the_next_page_keeps_the_header_and_normal_type_is_left_alone():
    _, columns = _tiny_table(Page(_row(52, HEADER) + _rows(100)), None)
    table, _ = _tiny_table(Page(_rows(200)), columns)
    assert table.rows[1][0] == "200" and len(table.rows) == 15
    big = [(50 + i * 30, 100 + (i % 5) * 14, 75 + i * 30, 112 + (i % 5) * 14, "word") for i in range(50)]
    assert _tiny_table(Page(big), None)[0] is None
