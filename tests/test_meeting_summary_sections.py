"""Meeting-summary header+bullet atomization (universal, not deal-specific).

The ``meeting_section_header`` SemanticRule already existed but was never
wired into the PDF text-rich splitter — Title-Case headers (Action Items /
Key Decisions) were glued into prose, and checkbox glyphs extracted as
``I Owner to verb…`` never became bullets.
"""

from __future__ import annotations

from app.parsers.orbitbrief_pdf import (
    _atoms_for_block,
    _checkbox_owner_action_body,
    _is_meeting_section_heading_line,
    _split_glued_meeting_summary_paragraph,
    _text_rich_sections,
)


SUMMARY_PAGE = """\
Meeting Summary and Full Transcript
Executive Summary
- Badge access, camera configuration, and UID Enterprise onboarding are the primary focus areas.
- PurTera agreed to investigate Okta integration, validate feasibility, and prepare a statement of work and quote.
- Remote implementation was preferred over on-site work.
- Okta integration is considered a significant requirement by the customer.
- Scope discussed included badge readers, door access, cameras, UID Enterprise, and integration with Okta.
Action Items
I Jacob to send the full Ubiquiti equipment/parts list.
I PurTera team to validate Okta integration capabilities through testing.
I PurTera to prepare and deliver a quote/SOW.
I Daniel to coordinate paperwork and signatures.
I Customer to provide administrative access to the UDM/UniFi environment if project proceeds.
Key Decisions
- Project should be performed remotely whenever possible.
- Network build-out is excluded from the primary scope unless required later.
- SSO integration is a hard requirement.
- Remote white-glove is preferred over on-site.
"""


def test_meeting_section_header_rule_fires_offline():
    assert _is_meeting_section_heading_line("Action Items")
    assert _is_meeting_section_heading_line("EXECUTIVE SUMMARY")
    assert _is_meeting_section_heading_line("Key Decisions")
    assert not _is_meeting_section_heading_line("Jacob to send the full equipment list.")
    assert not _is_meeting_section_heading_line("I Jacob to send the list.")


def test_checkbox_owner_action_body_strips_i_marker():
    assert (
        _checkbox_owner_action_body("I Jacob to send the full equipment list.")
        == "Jacob to send the full equipment list."
    )
    assert (
        _checkbox_owner_action_body("I PurTera team to validate Okta integration.")
        == "PurTera team to validate Okta integration."
    )
    # Prose "I think…" must NOT become a bullet.
    assert _checkbox_owner_action_body("I think we should install cameras.") is None


def test_text_rich_sections_nests_bullets_under_meeting_headers():
    secs = _text_rich_sections(SUMMARY_PAGE)
    by_heading = { (s.get("heading") or ""): s for s in secs }
    assert "Executive Summary" in by_heading
    assert "Action Items" in by_heading
    assert "Key Decisions" in by_heading

    exec_items = [
        it.get("text")
        for b in (by_heading["Executive Summary"].get("blocks") or [])
        if b.get("kind") == "bullet_list"
        for it in (b.get("items") or [])
    ]
    assert len(exec_items) == 5
    assert any("Badge access" in (t or "") for t in exec_items)

    action_items = [
        it.get("text")
        for b in (by_heading["Action Items"].get("blocks") or [])
        if b.get("kind") == "bullet_list"
        for it in (b.get("items") or [])
    ]
    assert len(action_items) == 5
    assert any(t.startswith("Jacob to send") for t in action_items)
    assert not any(t.startswith("I Jacob") for t in action_items)

    decisions = [
        it.get("text")
        for b in (by_heading["Key Decisions"].get("blocks") or [])
        if b.get("kind") == "bullet_list"
        for it in (b.get("items") or [])
    ]
    assert len(decisions) == 4


def test_glued_action_items_paragraph_splits_into_atoms():
    glued = (
        "Action Items I Jacob to send the full Ubiquiti equipment/parts list. "
        "I PurTera team to validate Okta integration capabilities through testing. "
        "I PurTera to prepare and deliver a quote/SOW. "
        "I Daniel to coordinate paperwork and signatures. "
        "I Customer to provide administrative access to the UDM/UniFi environment "
        "if project proceeds. Key Decisions"
    )
    split = _split_glued_meeting_summary_paragraph(glued)
    assert split is not None
    blocks, trailing = split
    assert trailing == "Key Decisions"
    items = [
        it.get("text")
        for b in blocks
        if b.get("kind") == "bullet_list"
        for it in (b.get("items") or [])
    ]
    assert len(items) == 5
    assert all(not (t or "").startswith("I ") for t in items)

    atoms = list(
        _atoms_for_block(
            block={"kind": "paragraph", "text": glued},
            section_path=["Meeting Summary and Full Transcript"],
            page_index=0,
            project_id="proj",
            artifact_id="art",
            filename="Meeting_Summary_and_Full_Transcript.pdf",
            parser_version="test",
        )
    )
    assert len(atoms) == 5
    for a in atoms:
        loc = a.source_refs[0].locator or {}
        assert "Action Items" in (loc.get("section_path") or [])
        assert a.value.get("list_section") == "action_items"
        assert a.value.get("section_header") == "Action Items"
        assert "Action Items" in (a.value.get("lead_in") or [])
        assert "Action Items" not in (a.raw_text or "")
        assert not (a.raw_text or "").startswith("I ")
