"""Nested TSC-style bullets (`` o ``, ``▪``) → ``bullet_tree``."""

from __future__ import annotations

from orbitbrief_page_os.segmentation.extract_overlay_text import (
    _attach_nested_bullet_tree_to_section,
    _maybe_tsc_continuation_and_followon_sections,
    _parse_tsc_o_only_bullet_run,
    _split_glued_callout_suffix,
    _structure_tsc_followon_sections,
)


def test_attach_nested_bullet_tree_squares_and_o() -> None:
    sec = {
        "paragraphs": ["Tractor Supply Stores (TSC)"],
        "bullet_items": [
            "Average of 10–13 APs per store o 5-7 indoor APs o 4-6 outdoor APs",
            "Includes: o Indoor sales floor o Outdoor areas including: ▪ A ▪ B",
        ],
        "closing_paragraphs": [],
    }
    _attach_nested_bullet_tree_to_section(sec)
    assert "bullet_tree" in sec
    tree = sec["bullet_tree"]
    assert len(tree) == 2
    assert tree[0]["text"].startswith("Average")
    assert len(tree[0]["children"]) == 2
    assert tree[1]["text"] == "Includes:"
    assert tree[1].get("role") == "list_intro"
    assert "bullet_items" not in sec
    outdoor = next(c for c in tree[1]["children"] if "Outdoor" in c["text"])
    assert outdoor.get("role") == "list_intro"
    assert len(outdoor["children"]) == 2
    assert outdoor["children"][0]["text"] == "A"
    body = sec["body"]
    assert "▪" not in body
    assert " o " not in body.lower()
    assert "Tractor Supply" in body
    assert "- Average" not in body
    assert "Includes:" not in body
    assert "subsections" not in sec


def test_glued_note_callout_splits_to_sibling_under_parent() -> None:
    """PDF text sometimes concatenates a bold NOTE onto the prior `` o `` line."""
    sec = {
        "paragraphs": [],
        "bullet_items": [
            "Site surveys o Number of indoor APs required "
            "o Number of outdoor APs required NOTE: some stores will have indoor APs "
            "replaced with outdoor APs",
        ],
    }
    _attach_nested_bullet_tree_to_section(sec)
    tree = sec["bullet_tree"]
    assert len(tree) == 1
    ch = tree[0]["children"]
    assert len(ch) == 3
    assert ch[0]["text"] == "Number of indoor APs required"
    assert ch[1]["text"] == "Number of outdoor APs required"
    assert ch[2].get("role") == "note_callout"
    assert ch[2]["text"].lower().startswith("note:")
    assert "children" in ch[2] and ch[2]["children"] == []


def test_split_glued_labeled_warning() -> None:
    h, tail = _split_glued_callout_suffix("Do the work WARNING: watch for hazards")
    assert h == "Do the work"
    assert tail.lower().startswith("warning:")


def test_split_glued_policy_after_smart_quoted_checklist() -> None:
    glued = (
        "Confirmed \u201cinstallation-ready\u201d Any missing components discovered onsite "
        "that were not identified during kitting will be the responsibility of the partner"
    )
    h, tail = _split_glued_callout_suffix(glued)
    assert tail is not None
    assert "installation-ready" in h
    assert tail.startswith("Any missing")


def test_quality_control_glued_warning_split_in_bullet_tree() -> None:
    glued = (
        "Confirmed \u201cinstallation-ready\u201d Any missing components discovered onsite "
        "that were not identified during kitting will be the responsibility of the partner"
    )
    sec = {
        "paragraphs": [],
        "bullet_items": [
            "QC o Validated for completeness prior to shipment o " + glued,
        ],
    }
    _attach_nested_bullet_tree_to_section(sec)
    kids = sec["bullet_tree"][0]["children"]
    assert len(kids) == 3
    assert "Validated" in kids[0]["text"]
    assert "installation-ready" in kids[1]["text"]
    assert "Any missing" not in kids[1]["text"]
    assert kids[2].get("role") == "note_callout"


def test_pts_stitch_then_parse() -> None:
    sec = {
        "paragraphs": [],
        "bullet_items": [
            "Lift usage for: o Outdoor o New runs Petsense Stores (PTS)",
            "Existing: o 1 AP per store",
        ],
    }
    _attach_nested_bullet_tree_to_section(sec)
    assert "subsections" not in sec
    texts = [n["text"] for n in sec["bullet_tree"]]
    assert any("Lift usage" in t for t in texts)
    pts = next(n for n in sec["bullet_tree"] if "Petsense" in n["text"])
    assert pts["children"][0]["text"] == "Existing:"
    assert pts["children"][0].get("role") == "list_intro"
    assert pts["children"][0]["children"][0]["text"] == "1 AP per store"


def test_followon_store_profiles_gets_tree() -> None:
    b, en, sq = "\u2022", "\u2013", "\u25aa"
    t = (
        "Store Profiles and Deployment Requirements Tractor Supply Stores (TSC) "
        f"{b} Average of 10{en}13 APs per store o 5-7 indoor APs o 4-6 outdoor APs "
        f"{b} Includes: o Indoor sales floor o Outdoor areas including: "
        f"{sq} Garden center {sq} Propane "
        f"{b} Lift usage required for: o A o B o C "
        f"{b} Petsense Stores (PTS) {b} Existing: o 1 AP per store "
        f"{b} Scope: o Replace o Install "
        f"{b} Lift usage not required under standard conditions "
        f"{b} Partner must provide: o Optional lift "
        f"Deployment Model Expectations {b} Each store must be standalone"
    )
    secs = _structure_tsc_followon_sections(t)
    assert len(secs) >= 2
    sp = secs[0]
    assert sp.get("title") == "Store Profiles and Deployment Requirements"
    assert "subsections" in sp
    assert "bullet_tree" not in sp
    subs = sp["subsections"]
    assert len(subs) == 2
    assert "TSC" in subs[0]["header"].upper()
    assert subs[1]["header"] == "Petsense Stores (PTS)"
    inc = next(n for n in subs[0]["bullet_tree"] if n.get("text") == "Includes:")
    assert inc.get("role") == "list_intro"
    pts_items = subs[1]["bullet_tree"]
    assert any(n.get("text", "").startswith("Scope") for n in pts_items)
    assert "▪" not in (sp.get("body") or "")
    assert "bullet_items" not in sp


def test_split_glued_partner_responsibility_after_tsc() -> None:
    glued = (
        "Escalated immediately to TSC Partner is responsible for ensuring "
        "all required materials are onsite prior to installation start"
    )
    h, tail = _split_glued_callout_suffix(glued)
    assert tail and "Partner is responsible" in tail
    assert h.endswith("TSC")


def test_parse_tsc_o_only_bullet_run_splits_hollow_items() -> None:
    nodes = _parse_tsc_o_only_bullet_run(
        "o Installation schedule o Store readiness"
    )
    assert [n["text"] for n in nodes] == ["Installation schedule", "Store readiness"]


def test_hollow_bullet_continuation_then_exception_handling_and_site_survey() -> None:
    """Page break after ``Align delivery timing with:`` — next page starts with `` o `` rows."""
    prev_txt = (
        "Operational Expectations • Support inbound "
        "Kitting & Asset Management All stores. "
        "Coordination • Align delivery timing with:"
    )
    page_txt = (
        "o Installation schedule o Store readiness "
        "Exception Handling • Delays or issues must be: o Escalated "
        "Site Survey • Submit forms"
    )

    class _P:
        def __init__(self, txt: str) -> None:
            self._txt = txt

        def get_text(self, mode: str) -> str:
            return self._txt

    class _Doc:
        def __init__(self, prev: str) -> None:
            self._prev = prev

        def __getitem__(self, idx: int) -> _P:
            if idx == 3:
                return _P(self._prev)
            return _P("")

    meta, secs = _maybe_tsc_continuation_and_followon_sections(
        page_txt, _Doc(prev_txt), 4, r"c:\fake.pdf"
    )
    assert meta is not None
    assert secs[0]["kind"] == "notes_continuation"
    assert secs[0]["continues_from"]["relation"] == "tsc_hollow_bullet_continuation"
    assert secs[0]["continues_from"]["section_title"] == "Coordination"
    bt = secs[0].get("bullet_tree") or []
    assert len(bt) == 2
    assert secs[1].get("title") == "Exception Handling"
    assert secs[2].get("title") == "Site Survey"


def test_site_survey_major_heading_splits_before_installation_execution() -> None:
    """Blue-band ``Site Survey`` must end before the next major ``Installation …`` block."""
    t = (
        "Exception Handling • Delays o Escalated "
        "Site Survey If necessary. Each visit one hour. Site surveys shall be conducted as follows: "
        "• First top bullet o nested "
        "Installation Execution General Requirements • Standard install duration: o One day"
    )
    secs = _structure_tsc_followon_sections(t)
    titles = [s.get("title") for s in secs]
    assert titles == [
        "Exception Handling",
        "Site Survey",
        "Installation Execution General Requirements",
    ]
    site = next(s for s in secs if s.get("title") == "Site Survey")
    assert site.get("heading_tier") == "major"
    bt = site.get("bullet_tree") or []
    assert len(bt) == 1
    assert bt[0].get("role") == "list_intro"
    assert "conducted as follows" in bt[0].get("text", "")
    assert any("First top" in (c.get("text") or "") for c in bt[0].get("children") or [])


def test_operational_expectations_page_splits() -> None:
    b = "\u2022"
    t = (
        f"Operational Expectations {b} One o two "
        f"Kitting & Asset Management Intro. {b} Kit o subkit "
        f"Kitting Requirements {b} Req o sub "
        f"Labeling & Asset Tracking {b} Lab "
        f"Quality Control {b} QC o ok "
        f"Shipping & Logistics Intro. {b} Log o row "
        f"Shipping Requirements {b} Ship o track "
        f"Coordination {b} Align o done"
    )
    secs = _structure_tsc_followon_sections(t)
    titles = [s.get("title") for s in secs]
    assert "Operational Expectations" in titles
    assert "Kitting & Asset Management" in titles
    assert "Coordination" in titles


def test_operational_expectations_parent_major_when_prior_warehousing() -> None:
    """Minor heading on page N continues prior page's Warehousing major section."""
    page2_txt = (
        "Store Profiles and Deployment Requirements • A "
        "Deployment Model Expectations • B "
        "Warehousing & Inventory Management Partner(s) must provide warehouse. "
        "Inventory Management • All hardware must be tracked"
    )

    class MockPage:
        def __init__(self, txt: str) -> None:
            self._txt = txt

        def get_text(self, mode: str) -> str:
            return self._txt

    class MockDoc:
        def __init__(self, by_idx: dict[int, str]) -> None:
            self._by = by_idx

        def __getitem__(self, i: int):
            return MockPage(self._by.get(i, ""))

        def __len__(self) -> int:
            return max(self._by.keys(), default=0) + 1

    from orbitbrief_page_os.segmentation import extract_overlay_text as ex

    doc = MockDoc({2: page2_txt})
    sections: list[dict] = [
        {
            "kind": "notes",
            "title": "Operational Expectations",
            "bullet_items": ["Support staggered inbound shipments"],
        }
    ]
    ex._annotate_tsc_subtitle_under_parent_major(sections, doc, 3, r"c:\fake.pdf")
    assert sections[0].get("parent_major_section") == "Warehousing & Inventory Management"
    assert sections[0].get("hierarchy") == "subtitle_continuation_under_major"
    assert sections[0].get("continues_from", {}).get("relation") == (
        "logical_subtitle_under_prior_page_major"
    )


def test_hoist_inventory_management_subtitle_under_warehousing() -> None:
    sec = {
        "kind": "notes",
        "title": "Warehousing & Inventory Management",
        "paragraphs": [
            "Partner(s) must provide warehouse and inventory capabilities.",
            "Inventory Management",
        ],
        "bullet_items": ["All hardware must be tracked at serial number level"],
    }
    from orbitbrief_page_os.segmentation import extract_overlay_text as ex

    ex._maybe_hoist_trailing_paragraph_as_subtitle(sec)
    assert sec.get("subtitle") == "Inventory Management"
    assert sec["paragraphs"] == ["Partner(s) must provide warehouse and inventory capabilities."]


def test_merge_kitting_major_absorbs_child_sections() -> None:
    from orbitbrief_page_os.segmentation import extract_overlay_text as ex

    sections = [
        {
            "kind": "notes",
            "title": "Kitting & Asset Management",
            "paragraphs": ["All stores must be fully staged."],
            "body": "All stores must be fully staged.",
        },
        {
            "kind": "notes",
            "title": "Kitting Requirements",
            "paragraphs": ["Partner(s) must:"],
            "bullet_items": ["Determine AP types"],
        },
        {
            "kind": "notes",
            "title": "Labeling & Asset Tracking",
            "bullet_tree": [{"text": "Each AP must be labeled", "children": []}],
        },
        {
            "kind": "notes",
            "title": "Quality Control",
            "bullet_tree": [{"text": "Each kit must be:", "children": []}],
        },
        {
            "kind": "notes",
            "title": "Shipping & Logistics",
            "paragraphs": ["Partner(s) are responsible for logistics."],
            "body": "Partner(s) are responsible for logistics.",
        },
        {
            "kind": "notes",
            "title": "Shipping Requirements",
            "bullet_tree": [{"text": "Kits must be delivered:", "children": []}],
        },
        {
            "kind": "notes",
            "title": "Coordination",
            "bullet_items": ["Align delivery timing with:"],
        },
    ]
    ex._merge_tsc_same_page_major_with_child_sections(sections)
    assert len(sections) == 1
    assert sections[0]["title"] == "Kitting & Asset Management"
    subs = sections[0]["subsections"]
    assert len(subs) == 5
    assert subs[0]["header"] == "Kitting Requirements"
    assert subs[4]["header"] == "Coordination"
    ship = next(s for s in subs if s.get("header") == "Shipping & Logistics")
    assert ship.get("hierarchy") == "intermediate_band_with_nested_subheads"
    assert isinstance(ship.get("subsections"), list)
    assert len(ship["subsections"]) == 1
    assert ship["subsections"][0]["header"] == "Shipping Requirements"
