"""Built-in TSC / PTS WAP follow-on titles (single source for template merge + parser)."""

from __future__ import annotations

# First heading on a typical follow-on page (split continuation prefix before this).
TSC_FOLLOWON_HEADINGS: tuple[str, ...] = (
    "Store Profiles and Deployment Requirements",
    "Deployment Model Expectations",
    "Warehousing & Inventory Management",
    "Inventory Management",
    "Operational Expectations",
    "Kitting & Asset Management",
    "Kitting Requirements",
    "Labeling & Asset Tracking",
    "Quality Control",
    "Shipping & Logistics",
    "Shipping Requirements",
    "Coordination",
    "Exception Handling",
    "Site Survey",
    "Installation Execution General Requirements",
    "Execution Scope",
)

# Visually distinct blue-band (major) section titles in this RFP family.
TSC_MAJOR_BAND_SECTION_TITLES: frozenset[str] = frozenset(
    {
        "Store Profiles and Deployment Requirements",
        "Deployment Model Expectations",
        "Warehousing & Inventory Management",
        "Operational Expectations",
        "Kitting & Asset Management",
        "Site Survey",
        "Installation Execution General Requirements",
    }
)
