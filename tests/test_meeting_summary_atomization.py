"""Meeting-summary header+bullet atomization (universal, not deal-specific)."""

from __future__ import annotations

from app.core.normalizers import detect_section, meeting_section_slug
from app.core.semantic_rules import is_meeting_section_header
from app.parsers.orbitbrief_pdf import (
    _checkbox_owner_action_body,
    _split_glued_meeting_summary_paragraph,
    _text_rich_sections,
    atoms_from_structured_doc,
)


def test_detect_section_covers_meeting_summary_headers():
    assert detect_section("Executive Summary") == "Executive Summary"
    assert detect_section("ACTION ITEMS") == "Action Items"
    assert detect_section("Action Items:") == "Action Items"
    assert detect_section("Key Decisions") == "Key Decisions"
    assert detect_section("Open Questions") == "Open Questions"
    assert detect_section("We discussed the executive summary") is None
    assert meeting_section_slug("Action Items") == "action_items"
    assert is_meeting_section_header("Key Decisions") is True
    assert is_meeting_section_header("Alex Rivera [00:04]") is False


def test_checkbox_owner_action_body():
    assert (
        _checkbox_owner_action_body("I Jacob to send the full equipment list.")
        == "Jacob to send the full equipment list."
    )
    assert (
        _checkbox_owner_action_body("I PurTera team to validate Okta integration.")
        == "PurTera team to validate Okta integration."
    )
    # Prose "I think…" must not become a bullet.
    assert _checkbox_owner_action_body("I think we should configure cameras.") is None


def test_text_rich_sections_splits_meeting_summary_headers_and_checkbox_bullets():
    page = """Meeting Summary and Full Transcript
Executive Summary
- Badge access is in scope.
- SSO is required.
Action Items
I Jacob to send the parts list.
I PurTera team to validate Okta integration.
I PurTera to prepare a quote.
I Daniel to coordinate paperwork.
I Customer to provide admin access.
Key Decisions
- Project should be remote.
- Network build-out is excluded.
"""
    secs = _text_rich_sections(page)
    by_heading = { (s.get("heading") or ""): s for s in secs }
    assert "Executive Summary" in by_heading
    assert "Action Items" in by_heading
    assert "Key Decisions" in by_heading

    action = by_heading["Action Items"]
    bullets = [b for b in (action.get("blocks") or []) if b.get("kind") == "bullet_list"]
    assert len(bullets) == 1
    items = [i["text"] for i in bullets[0]["items"]]
    assert len(items) == 5
    assert items[0].startswith("Jacob to send")
    assert "I Jacob" not in items[0]


def test_split_glued_action_items_paragraph_and_trailing_key_decisions():
    glued = (
        "Action Items I Jacob to send the full Ubiquiti equipment/parts list. "
        "I PurTera team to validate Okta integration capabilities through testing. "
        "I PurTera to prepare and deliver a quote/SOW. "
        "I Daniel to coordinate paperwork and signatures. "
        "I Customer to provide administrative access to the UDM environment if project proceeds. "
        "Key Decisions"
    )
    result = _split_glued_meeting_summary_paragraph(glued)
    assert result is not None
    blocks, trailing = result
    assert trailing == "Key Decisions"
    assert len(blocks) == 1
    assert blocks[0]["meeting_section"] == "Action Items"
    assert len(blocks[0]["items"]) == 5
    assert blocks[0]["items"][0]["text"].startswith("Jacob to send")


def test_atoms_from_structured_meeting_summary_carry_connective_tissue():
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "title": "Meeting Summary and Full Transcript",
                "sections": [
                    {
                        "heading": "Executive Summary",
                        "level": 2,
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [
                                    {"text": "Badge access is in scope."},
                                    {"text": "SSO is required."},
                                ],
                            }
                        ],
                        "subsections": [],
                    },
                    {
                        "heading": "Action Items",
                        "level": 2,
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [
                                    {"text": "Jacob to send the full equipment list."},
                                    {"text": "PurTera to prepare and deliver a quote/SOW."},
                                ],
                            }
                        ],
                        "subsections": [],
                    },
                    {
                        "heading": "Key Decisions",
                        "level": 2,
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [
                                    {"text": "Project should be performed remotely whenever possible."},
                                ],
                            }
                        ],
                        "subsections": [],
                    },
                ],
            }
        ],
    }
    atoms = list(
        atoms_from_structured_doc(
            structured_doc=structured,
            project_id="proj_demo",
            artifact_id="art_demo",
            filename="Meeting_Summary_and_Full_Transcript.pdf",
            parser_version="test",
        )
    )
    assert len(atoms) == 5
    # Document order: Exec → Action → Decisions (block_index survives id-sort).
    texts = [a.raw_text or "" for a in atoms]
    assert texts[0].startswith("Badge access")
    assert texts[2].startswith("Jacob to send")
    assert texts[-1].startswith("Project should be performed remotely")
    for i, a in enumerate(atoms):
        loc = a.source_refs[0].locator
        assert loc.get("block_index") == i
        assert loc.get("line_start") == i
        assert loc.get("section_path", [None])[0] == "Meeting Summary and Full Transcript"
        lead = loc.get("lead_in") or []
        assert "Meeting Summary and Full Transcript" in lead

    action = [a for a in atoms if "Jacob" in (a.raw_text or "")]
    assert len(action) == 1
    loc = action[0].source_refs[0].locator
    assert "Action Items" in (loc.get("section_path") or [])
    assert "Action Items" in (loc.get("lead_in") or [])
    val = action[0].value or {}
    assert val.get("list_section") == "action_items"
    assert val.get("section_header") == "Action Items"
    assert "Meeting Summary and Full Transcript" in (val.get("lead_in") or [])

    decision = [a for a in atoms if "remotely" in (a.raw_text or "")]
    assert len(decision) == 1
    dloc = decision[0].source_refs[0].locator
    assert "Key Decisions" in (dloc.get("section_path") or [])
    assert (decision[0].value or {}).get("list_section") == "key_decisions"


def test_meeting_summary_block_index_survives_id_sort():
    """Compiler sorts by atom.id; block_index must still restore reading order."""
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "sections": [
                    {
                        "heading": "Executive Summary",
                        "level": 2,
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [
                                    {"text": "Zebra comes first in document order."},
                                    {"text": "Apple comes second in document order."},
                                ],
                            }
                        ],
                        "subsections": [],
                    },
                    {
                        "heading": "Action Items",
                        "level": 2,
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [
                                    {"text": "Mango action item is third overall."},
                                ],
                            }
                        ],
                        "subsections": [],
                    },
                ],
            }
        ],
    }
    atoms = list(
        atoms_from_structured_doc(
            structured_doc=structured,
            project_id="proj_demo",
            artifact_id="art_demo",
            filename="Meeting_Summary_and_Full_Transcript.pdf",
            parser_version="test",
        )
    )
    by_id = sorted(atoms, key=lambda a: a.id)
    restored = sorted(
        by_id,
        key=lambda a: (a.source_refs[0].locator or {}).get("block_index", 10**9),
    )
    assert [a.raw_text for a in restored] == [
        "Zebra comes first in document order.",
        "Apple comes second in document order.",
        "Mango action item is third overall.",
    ]
    assert "Meeting Summary and Full Transcript" in (
        (restored[0].source_refs[0].locator or {}).get("lead_in") or []
    )
    assert "Executive Summary" in (
        (restored[0].source_refs[0].locator or {}).get("lead_in") or []
    )
    assert "Action Items" in (
        (restored[2].source_refs[0].locator or {}).get("lead_in") or []
    )


def test_glued_paragraph_above_bullet_list_stamps_trailing_header():
    """Action Items blob ends with 'Key Decisions'; next block is the decision list."""
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "sections": [
                    {
                        "heading": "",
                        "level": 2,
                        "blocks": [
                            {
                                "kind": "paragraph",
                                "text": (
                                    "Action Items I Jacob to send the parts list. "
                                    "I PurTera to prepare a quote. "
                                    "I Daniel to coordinate paperwork. "
                                    "I Customer to provide access. "
                                    "I PurTera team to validate Okta. "
                                    "Key Decisions"
                                ),
                            },
                            {
                                "kind": "bullet_list",
                                "items": [
                                    {"text": "Project should be performed remotely whenever possible."},
                                    {"text": "Network build-out is excluded from the primary scope."},
                                ],
                            },
                        ],
                        "subsections": [],
                    }
                ],
            }
        ],
    }
    atoms = list(
        atoms_from_structured_doc(
            structured_doc=structured,
            project_id="proj_demo",
            artifact_id="art_demo",
            filename="Meeting_Summary_and_Full_Transcript.pdf",
            parser_version="test",
        )
    )
    texts = [a.raw_text or "" for a in atoms]
    assert any("Jacob to send" in t for t in texts)
    assert not any(t.startswith("Action Items I Jacob") for t in texts)
    decisions = [
        a
        for a in atoms
        if "remotely" in (a.raw_text or "") or "Network build-out" in (a.raw_text or "")
    ]
    assert len(decisions) == 2
    for a in decisions:
        assert "Key Decisions" in (a.source_refs[0].locator.get("section_path") or [])
