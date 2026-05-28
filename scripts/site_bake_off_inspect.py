"""Inspect what fields the structural extractors actually emit on OPTBOT."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.parsers.orbitbrief_pdf import _fitz_site_roster_fallback, _text_based_site_roster_extract

OPTBOT_PDF = Path(r"C:\Users\lilli\test_deals\optbot\artifacts\08_site_roster_and_facilities_authoritative.pdf")
APS_PDF = Path(r"C:\Users\lilli\test_deals\aps_fiber\artifacts\APS_fiber_Attachment_A.pdf")


def dump_atom(a, idx):
    print(f"  atom[{idx}]")
    for attr in ("atom_type", "raw_text", "structured", "value", "entity_keys", "confidence"):
        v = getattr(a, attr, "<no attr>")
        s = str(v)
        if len(s) > 200:
            s = s[:200] + "..."
        print(f"    {attr}: {s}")


for label, pdf in [("OPTBOT", OPTBOT_PDF), ("APS", APS_PDF)]:
    for path_name, extractor in [
        ("fitz_table", _fitz_site_roster_fallback),
        ("text_based", _text_based_site_roster_extract),
    ]:
        print(f"\n=== {label} / {path_name} ===")
        atoms = extractor(
            pdf_path=pdf,
            project_id=label.lower(),
            artifact_id="art_test",
            parser_version="inspect_v1",
            already_emitted=set(),
        )
        print(f"  {len(atoms)} atoms emitted")
        for i, a in enumerate(atoms[:3]):
            dump_atom(a, i)
        if len(atoms) > 3:
            print(f"  ... +{len(atoms)-3} more")
