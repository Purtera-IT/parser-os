from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

from app.testing.mutators import (
    write_docx_fixture,
    write_email_fixture,
    write_quote_fixture,
    write_spreadsheet_fixture,
    write_transcript_fixture,
)

SPREADSHEET_MUTATIONS = [
    "header_qty_synonym",
    "header_count_synonym",
    "header_hash_synonym",
    "header_quantity_synonym",
    "shifted_header_row",
    "hidden_columns",
    "subtotal_rows",
    "blank_rows",
    "mixed_case",
    "row_order_shuffle",
    "extra_irrelevant_columns",
]
EMAIL_MUTATIONS = [
    "quoted_on_date",
    "original_message",
    "multiple_quoted_levels",
    "forwarding_prefix",
    "internal_only_note",
]
DOCX_MUTATIONS = [
    "scope_in_paragraph",
    "scope_in_table",
    "all_caps_exclusion",
]
TRANSCRIPT_MUTATIONS = [
    "speaker_labeled",
    "meeting_notes_sections",
    "unknown_speaker",
]
QUOTE_MUTATIONS = [
    "part_number_variation",
    "quantity_as_string",
    "unit_price_dollar",
    "lead_time_missing",
    "quote_total_row",
    "device_alias_ip_cam",
]


@dataclass(frozen=True)
class BaseScenario:
    scoped_device: str = "IP Camera"
    scoped_total: int = 91
    vendor_total: int = 72
    excluded_site: str = "West Wing"
    included_site: str = "Main Campus"
    access_constraint: str = "escort access after 5pm"
    open_question: str = "MDF badge access"


def _pick(options: list[str], seed: int, offset: int) -> str:
    return options[(seed + offset) % len(options)]


def default_mutations(seed: int) -> dict[str, str]:
    return {
        "spreadsheet": _pick(SPREADSHEET_MUTATIONS, seed, 0),
        "email": _pick(EMAIL_MUTATIONS, seed, 1),
        "docx": _pick(DOCX_MUTATIONS, seed, 2),
        "transcript": _pick(TRANSCRIPT_MUTATIONS, seed, 3),
        "quote": _pick(QUOTE_MUTATIONS, seed, 4),
    }


def generate_scenario(
    seed: int,
    mutations: dict[str, str] | None = None,
    output_root: Path | None = None,
) -> Path:
    rng = Random(seed)
    mutation_set = mutations or default_mutations(seed)
    base = BaseScenario()
    output_root = (output_root or Path.cwd() / "tmp" / "adversarial_fixtures").resolve()
    scenario_dir = output_root / f"scenario_{seed:04d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    main_qty = "50 EA" if seed % 2 == 0 else "50"
    west_qty = "41"
    if seed % 3 == 0:
        west_qty = "41 EA"

    write_spreadsheet_fixture(
        scenario_dir / "site_list.xlsx",
        included_site=base.included_site,
        excluded_site=base.excluded_site,
        scoped_device=base.scoped_device,
        main_qty=main_qty,
        west_qty=west_qty,
        access_constraint=base.access_constraint,
        mutation=mutation_set["spreadsheet"],
        rng=rng,
    )
    write_quote_fixture(
        scenario_dir / "vendor_quote.xlsx",
        scoped_device=base.scoped_device,
        vendor_total=str(base.vendor_total),
        mutation=mutation_set["quote"],
    )
    write_email_fixture(
        scenario_dir / "customer_email.txt",
        excluded_site=base.excluded_site,
        included_site=base.included_site,
        access_constraint=base.access_constraint,
        mutation=mutation_set["email"],
    )
    write_docx_fixture(
        scenario_dir / "sow_draft.docx",
        included_site=base.included_site,
        excluded_site=base.excluded_site,
        scoped_device=base.scoped_device,
        mutation=mutation_set["docx"],
    )
    write_transcript_fixture(
        scenario_dir / "kickoff_transcript.txt",
        included_site=base.included_site,
        excluded_site=base.excluded_site,
        scoped_device=base.scoped_device,
        access_constraint=base.access_constraint,
        open_question=base.open_question,
        mutation=mutation_set["transcript"],
    )

    metadata = {
        "seed": seed,
        "base_scenario": asdict(base),
        "mutations": mutation_set,
    }
    (scenario_dir / "scenario_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scenario_dir
