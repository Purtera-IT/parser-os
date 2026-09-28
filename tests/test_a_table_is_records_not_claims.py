"""A row is a record. An atom is a claim. The parser must not confuse them.

Deal 010264 shipped two inventory workbooks. One of them produced 277,203
atoms and the deal never compiled -- not slowly, never. The whole of deal
010180, fully labelled, is 249 atoms.

These tests pin the reading that replaced the transcription, and every one
of them is a defect that shipped in the first version of it. The point of
naming them that way is that each is a specific wrong sentence a PM would
have read in their queue.
"""
from __future__ import annotations

from app.core.column_roles import classify, contact_supply, site_supply
from app.core.sheet_profile import (
    has_header,
    header_index,
    is_bulk_export,
    observe,
    profile_grid,
    subset_question,
    supplies,
)


def _grid(header, *rows):
    return [list(header)] + [list(r) for r in rows]


def _read(header, rows):
    g = _grid(header, *rows)
    hi = header_index(g)
    return observe(profile_grid("sheet", g, hi, headed=has_header(g, hi)))


def _keys(obs):
    return {o.key for o in obs}


# ── what a table is ──────────────────────────────────────────────────────


def test_an_export_is_read_and_a_written_sheet_is_not():
    """Two structural conditions, no vocabulary: many rows, one width."""
    export = _grid(["Serial", "Status"], *[[f"S{i}", "OK"] for i in range(400)])
    assert is_bulk_export(export, header_index(export))

    # A sheet somebody built by hand grows sections, spacers and side notes
    # of differing widths -- which is exactly what it must not be read as.
    written = _grid(
        ["Item", "Qty", "Note"],
        ["Cabling", "40", "per floor"],
        [],
        ["PHASE 2"],
        ["Switches", "6", ""],
    )
    assert not is_bulk_export(written, header_index(written))

    small = _grid(["A", "B"], *[[str(i), "x"] for i in range(30)])
    assert not is_bulk_export(small, header_index(small))


def test_a_heading_is_not_one_of_the_values_it_heads():
    """The 115,000-row cost-centre sheet has no header row.

    Read as though it did, its columns were called ``00``, ``KENNETH
    SULLIVAN`` and ``23231`` -- and the PM was asked where the 174 rows
    with no 23231 were.
    """
    body = [["OK" if i % 2 else "CRITICAL",
             "HENRICO" if i % 3 else "RICHMOND",
             "23231" if i % 5 else "23220"] for i in range(60)]
    assert has_header(_grid(["Status", "City", "Zip"], *body),
                      header_index(_grid(["Status", "City", "Zip"], *body)))

    # No header: the top row is simply the first record, and every one of
    # its values turns up again further down its own column.
    assert not has_header(body, header_index(body))


def test_a_headerless_sheet_says_so_instead_of_inventing_names():
    bare = [[f"{i:04d}", "CONVENIENCE SOLUTIONS", "5500 AUDUBON DR", "23231"]
            for i in range(300)]
    obs = observe(profile_grid("dump", bare, 0, headed=False))
    keys = _keys(obs)
    assert keys == {"table_shape", "no_header"}
    assert not any("23231" in o.headline for o in obs)


# ── what it says ─────────────────────────────────────────────────────────


def test_a_column_of_one_value_states_a_fact_about_every_row():
    """"Device Type: InTouch 9000" over all 1,563 rows is what makes the
    sheet the 9000s. Reported as "1 distinct value(s)" it reads as noise."""
    obs = _read(["Serial", "Device Type"],
                [[f"S{i}", "InTouch 9000"] for i in range(300)])
    const = next(o for o in obs if o.key == "constant:Device Type")
    assert const.headline == "Every row has Device Type = InTouch 9000"


def test_an_adverse_minority_is_a_question_not_a_statistic():
    rows = [[f"S{i}", "OK"] for i in range(270)] + \
           [[f"C{i}", "CRITICAL"] for i in range(30)]
    obs = _read(["Serial", "Status"], rows)
    q = next(o for o in obs if o.key.startswith("adverse:"))
    assert q.kind == "question"
    assert "30 of 300" in q.headline and "CRITICAL" in q.headline
    assert q.decides


def test_a_cost_column_holding_a_word_asks_who_owns_those():
    """ACQUISITON COST = Leased on 1,317 of 5,448 clocks is the ownership
    of the hardware being removed, and nobody wrote it in an email."""
    rows = [[f"S{i}", "1195"] for i in range(300)] + \
           [[f"L{i}", "Leased"] for i in range(200)]
    obs = _read(["Serial", "ACQUISITON COST"], rows)
    q = next(o for o in obs if o.key.startswith("ownership:"))
    assert "200 rows" in q.headline and "Leased" in q.headline


def test_a_european_figure_is_a_figure():
    """$1 520,91 read as a word said 1,346 clocks were leased. It is 1,317."""
    rows = [[f"S{i}", "$1 195,00"] for i in range(200)] + \
           [[f"E{i}", "$1 520,91"] for i in range(80)] + \
           [[f"L{i}", "Leased"] for i in range(60)]
    obs = _read(["Serial", "ACQUISITON COST"], rows)
    q = next(o for o in obs if o.key.startswith("ownership:"))
    assert "60 rows" in q.headline


# ── what it must not say ─────────────────────────────────────────────────


def test_an_account_code_is_never_summed_as_money():
    """"Cost Center" is named like money and holds a ten-digit code. Summed,
    it produced "$77,041,410,354" on a deal for 1,563 time clocks."""
    obs = _read(["Serial", "Cost Center"],
                [[f"S{i}", f"00{i:08d}"] for i in range(300)])
    assert not any(o.key.startswith("total:") for o in obs)


def test_a_status_column_is_not_a_place_and_not_a_price():
    """"Cost Center Status" contains both words and is neither."""
    rows = [[f"S{i}", "Open 2 Full Fiscal Years"] for i in range(250)] + \
           [[f"C{i}", "Closed Last Fiscal Year"] for i in range(50)]
    obs = _read(["Serial", "Cost Center Status"], rows)
    assert not any(o.key.startswith(("places:", "ownership:", "total:")) for o in obs)


def test_an_export_timestamp_is_not_a_way_the_fleet_divides():
    """476 clocks polled at 02:19 is a fact about when the report ran."""
    rows = [[f"S{i}", "2026-08-06 02:19:00"] for i in range(200)] + \
           [[f"T{i}", "2026-08-06 03:15:00"] for i in range(100)]
    obs = _read(["Serial", "Last Action Updated Time"], rows)
    assert not any(o.key.startswith("distribution:Last Action") for o in obs)


def test_a_failed_lookup_is_a_gap_not_a_value():
    """"Match Ser Num WFM: #N/A (634)" says only that a VLOOKUP missed."""
    rows = [[f"S{i}", "#N/A"] for i in range(295)] + \
           [[f"M{i}", f"00JC{i}"] for i in range(5)]
    obs = _read(["Serial", "Match Ser Num WFM"], rows)
    assert not any("#N/A" in o.headline for o in obs)


def test_a_column_three_people_filled_in_does_not_divide_the_population():
    rows = [[f"S{i}", ""] for i in range(297)] + \
           [[f"M{i}", f"note {i}"] for i in range(3)]
    obs = _read(["Serial", "Match in Not Registered"], rows)
    assert not any(o.key.startswith("distribution:Match") for o in obs)


def test_a_column_with_no_name_decides_nothing():
    obs = _read(["Serial", ""],
                [[f"S{i}", "Service Pack $109.00"] for i in range(300)])
    assert not any(o.key.endswith(("col_2", "Column2")) for o in obs)


def test_a_handful_of_blanks_is_a_typo_not_a_question():
    rows = [[f"S{i}", "x"] for i in range(299)] + [["", "x"]]
    obs = _read(["Serial Number", "Status"], rows)
    assert not any(o.key.startswith("unidentified:") for o in obs)


# ── what no single sheet can see ─────────────────────────────────────────


def test_a_filtered_view_beside_its_own_export_is_the_question():
    """1,563 devices on one sheet and 4,927 on another, and the SOW is
    titled for the smaller. Which one is the job decides the price."""
    small = _grid(["Serial", "Status", "Device Type", "Name", "Zip", "City"],
                  *[[f"S{i}", "OK", "InTouch 9000", "n", "23231", "HENRICO"]
                    for i in range(300)])
    large = _grid(["Serial", "Status", "Device Type", "Name"],
                  *[[f"S{i}", "OK", "InTouch 9100", "n"] for i in range(900)])
    ps = [
        profile_grid("DeviceList_9000", small, 0),
        profile_grid("DeviceList_ALL", large, 0),
    ]
    qs = subset_question(ps)
    assert qs, "a filtered sheet beside its export must be noticed"
    assert "300" in qs[0].headline and "900" in qs[0].headline

    # The filtered sheet carries extra pasted lookup columns, so the overlap
    # has to be measured against the SMALLER schema. Measured the other way
    # this pair scored 54% and the question was never asked.
    assert qs[0].decides


# ── what it can be asked for ─────────────────────────────────────────────


def test_a_sheet_declares_the_sites_it_can_supply():
    """The one thing this workbook was actually used for: the SOW's sites.

    A rollout is priced per place, and the row count is a device count --
    900 clocks standing at 3 doors is a 3-visit job.
    """
    rows = []
    for i in range(300):
        for street, city in (("5500 AUDUBON DR", "HENRICO"),
                             ("1499 LANEY WALKER BLVD", "AUGUSTA"),
                             ("480 NORTH BISBEE AVENUE", "WILLCOX")):
            rows.append([f"S{i}{city}", street, city, "VA", "23231"])
    g = _grid(["Serial", "Address", "City", "State", "Zip"], *rows)
    prof = profile_grid("devices", g, 0)
    sups, roles = supplies(prof, g, 0)
    site = next(s for s in sups if s.what == "site_address")
    assert site.distinct == 3
    assert site.rows == 900
    assert site.columns["street"] == "Address"
    assert site.columns["postal_code"] == "Zip"
    # A city is just a word; what makes it a city is sitting between a
    # street and a state. There is no gazetteer here and there must not be.
    assert site.columns["locality"] == "City"


def test_a_cost_centre_code_is_not_a_phone_number():
    """Ten digits is ten digits. "0010058001" was declared as somebody to
    call, which would have sent a crew's questions to an account code."""
    vals = [f"00{i:08d}" for i in range(50)]
    role = classify("Cost Center", vals, is_identifier=True)
    assert role is None or role.name != "phone"


def test_the_person_who_edited_the_row_is_not_the_site_contact():
    """The only email column in this workbook is "Modified By": nine Sodexo
    staff over 5,448 rows. Declared as contacts, a crew's access questions
    go to whoever last touched a cell."""
    roles = {"Modified By": classify(
        "Modified By", ["judy.palumbo@sodexo.com"] * 40 + ["greg.taper@sodexo.com"] * 10)}
    assert roles["Modified By"].name == "email"
    assert contact_supply(roles, {"Modified By": 9}, 5448) is None
    # A genuine per-site contact varies with the site.
    assert contact_supply(roles, {"Modified By": 4000}, 5448) is not None
