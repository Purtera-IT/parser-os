"""A drawing is titled by the building, not by the job.

Live 010180, "CDW FlexTrade Cabling" — a Cat6A buildout on the 12th floor of
7 Penn Plaza. Its own floorplans arrived as

    07.21.26_FEIL ORGANIZATION_7 PENN PLAZA_12 FL_SP-6.pdf      (55 atoms)
    07.22.26_FEIL ORGANIZATION_7 PENN PLAZA_12 FL_SP-6-1.dwg    ( 1 atom)

Feil is the landlord. Every architectural sheet is titled with the building
and its owner, whoever the tenant is and whoever is doing the work. Against a
deal named "CDW FlexTrade Cabling" the judge saw a different name and returned
``other_job`` at 0.95, and the compile set aside the entire drawing set for
the floor the deal exists to cable:

    INFO: document_job_scope set aside 07.21.26_FEIL ORGANIZATION_7 PENN
          PLAZA_12 FL_SP-6.pdf (55 atoms, 1 document(s)); other_job 0.95 llm

It is also why no DWG atom had ever reached an envelope. That was chased
through the parser, the registry and ``ArtifactType`` — the drawings were
parsed correctly every time and deleted three stages later.

The stage's own premise says why this happens: it asks the model to compare
the WORK, "since customer, vendor, people and dates are shared by every job
the customer has". A drawing offers no work to compare — it is lines and a
title block — so the model has nothing left but the name, the one signal it
was told not to trust.

The trade is deliberate and asymmetric. Setting aside the deal's own floorplan
is silent and takes the scope with it; carrying one extra drawing is visible
and a PM can say so. A PM's taught verdict still removes a drawing: a person
looking at the sheet knows which job it is.
"""
from __future__ import annotations

import pytest

from app.core.document_job_scope import is_drawing

#: The two that started this, verbatim from 010180's manifest.
FEIL_PDF = "07.21.26_FEIL ORGANIZATION_7 PENN PLAZA_12 FL_SP-6.pdf"
FEIL_DWG = "07.22.26_FEIL ORGANIZATION_7 PENN PLAZA_12 FL_SP-6-1.dwg"


@pytest.mark.parametrize("names", [
    [FEIL_PDF],
    [FEIL_DWG],
    [FEIL_PDF, FEIL_DWG],
    ["floorplan.dwg", "riser diagram.dxf"],
    ["IDF closet.jpg"],
    ["A-101 First Floor Plan.pdf"],
    ["E1.2 Power Plan.pdf"],
])
def test_a_drawing_set_is_recognised(names):
    assert is_drawing(names) is True


@pytest.mark.parametrize("names", [
    # 010162's kiosk close-down — the case document_job_scope exists for, and
    # it must stay removable.
    ["Delta Close Down.pdf"],
    ["CDW Smart Hands SOW Delta Admin 70598001.pdf"],
    ["Budgetary Numbers 010180.xlsx"],
    ["010180-hs-email-117380900193.eml"],
    ["Signed SOW.pdf"],
    [],
])
def test_prose_documents_are_not_drawings(names):
    assert is_drawing(names) is False


def test_a_mixed_bundle_is_not_spared():
    """Sparing is for a bundle that is ONLY drawings. A drawing that arrived
    attached to another job's thread is judged with that thread, as it should
    be — the thread is what says which job it is."""
    assert is_drawing(["plan.dwg", "Delta Close Down.pdf"]) is False


def test_a_sheet_number_is_what_makes_a_pdf_a_drawing():
    """A .pdf is only a drawing when it is named like a sheet. Without that a
    PDF is prose and stays judgeable — otherwise this guard would swallow the
    very documents the stage was built to remove."""
    assert is_drawing(["SP-6.pdf"]) is True
    assert is_drawing(["Scope of Work.pdf"]) is False
