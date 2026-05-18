"""Tests for :mod:`app.takeoff.project_reference`.

Universal hardening: builds the reference layer correctly across
arbitrary firm sheet-numbering conventions, handles missing intro
pages gracefully, and exposes a clean readable view that LLMs / humans
can consume directly.
"""
from __future__ import annotations

from app.takeoff.project_reference import (
    project_reference_to_readable,
)


def test_readable_projection_drops_bbox_noise() -> None:
    full_doc = {
        "schema_version": "purtera.lowvoltage.project_reference.v1",
        "source_pdf": "ANY.pdf",
        "spec": None,
        "legend": {
            "tables": [
                {
                    "bbox_pt": [1, 2, 3, 4],
                    "sections": [
                        {
                            "title": "WIRELESS LEGEND",
                            "column_headers": [
                                {"text": "SYMBOL", "bbox_pt": [0, 0, 0, 0]},
                                {"text": "DESCRIPTION", "bbox_pt": [0, 0, 0, 0]},
                            ],
                            "rows": [
                                {
                                    "cells_by_header": {
                                        "SYMBOL": "WN",
                                        "DESCRIPTION": "1 PORT WAP",
                                    },
                                    "cells": [
                                        {"text": "WN", "bbox_pt": [0, 0, 0, 0]},
                                        {"text": "1 PORT WAP", "bbox_pt": [0, 0, 0, 0]},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        },
        "schedule": None,
        "intro_pages": {"legend": [1]},
        "summary": {"legend_rows_total": 1},
    }
    readable = project_reference_to_readable(full_doc)
    # bbox / cells noise is dropped from rows; column → value dict survives.
    legend = readable["legend"]
    table = legend["tables"][0]
    row = table["rows"][0]
    assert "bbox_pt" not in row
    assert "cells" not in row
    assert row["SYMBOL"] == "WN"
    assert row["DESCRIPTION"] == "1 PORT WAP"


def test_readable_handles_missing_sections() -> None:
    """When the reference has no spec / schedule, the readable view
    surfaces nulls rather than crashing."""
    doc = {
        "schema_version": "purtera.lowvoltage.project_reference.v1",
        "source_pdf": "X.pdf",
        "spec": None,
        "legend": None,
        "schedule": None,
        "intro_pages": {},
        "summary": {},
    }
    readable = project_reference_to_readable(doc)
    assert readable["spec"] is None
    assert readable["legend"] is None
    assert readable["schedule"] is None


def test_readable_handles_list_of_pages() -> None:
    """When the reference picked up multiple legend pages, the readable
    projection processes them all as a list."""
    doc = {
        "schema_version": "purtera.lowvoltage.project_reference.v1",
        "source_pdf": "X.pdf",
        "spec": None,
        "legend": [
            {"tables": [{"sections": [{"title": "P1", "column_headers": [], "rows": []}]}]},
            {"tables": [{"sections": [{"title": "P2", "column_headers": [], "rows": []}]}]},
        ],
        "schedule": None,
        "intro_pages": {"legend": [1, 2]},
        "summary": {},
    }
    readable = project_reference_to_readable(doc)
    assert isinstance(readable["legend"], list)
    assert len(readable["legend"]) == 2


def test_readable_preserves_intro_pages_bucket() -> None:
    doc = {
        "schema_version": "purtera.lowvoltage.project_reference.v1",
        "source_pdf": "X.pdf",
        "spec": None, "legend": None, "schedule": None,
        "intro_pages": {"spec": [0], "legend": [1], "component_schedule": [2]},
        "summary": {},
    }
    readable = project_reference_to_readable(doc)
    assert readable["intro_pages"] == {"spec": [0], "legend": [1], "component_schedule": [2]}
