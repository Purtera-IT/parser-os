# LOWVOLT_002_MARRIOTT_ATLANTA_T

First hard regression case for the new `app/takeoff` low-voltage
construction takeoff pipeline. The source PDF is a 25-sheet 100% DD
package for the Marriott Atlanta hotel structured-cabling / security
scope (T-series sheets).

## Folder map

- `artifacts/2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf` — source.
- `labels/gold_takeoff.json` — expected counts, multipliers, zones,
  and warnings for the takeoff. The pipeline's output is compared
  against this file in `tests/test_takeoff_marriott_wn.py`.
- `project.yaml` — domain-pack pin (copper / low-voltage).
- `case_manifest.json` — high-level case description.

## What this case proves

- Sheet classifier correctly buckets 25 sheets across
  spec/legend/component_schedule/floor_plan/typical_plan/riser/
  equipment_room/detail.
- Adobe-style duplicate WN tokens at the same coordinate collapse to
  a single candidate via `dedupe_words(tolerance_pt=0.5)`.
- The `NOT IN SCOPE` flag on T1.00 propagates to `in_scope=false`
  and rejects the (otherwise legitimate) WN on that sheet.
- Legend page (T0.01) and detail page (T9.02) tokens are kept as
  rejected candidates so the audit trail is complete, but never
  count as devices.
- Floor multipliers (1, 2, 5, 9) hit exactly per the sheet titles.
- HOMERUN zone notes parse into IDF / MDF targets; multi-zone sheets
  flag `ambiguous_homerun_zone`; missing-level (T1.06 missing 12)
  and OCR-typo (T1.10 references level 10) warnings fire.
- Quote unitizer turns 174 base WN devices into 335 extended WN
  drops + matching cert tests + 1740 ft of service-loop allowance.

## Suggested command

```bash
python -c "
from pathlib import Path
from app.takeoff.pipeline import build_low_voltage_takeoff
p = Path('real_data_cases/LOWVOLT_002_MARRIOTT_ATLANTA_T/artifacts/2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf')
takeoff = build_low_voltage_takeoff(p)
print(takeoff.summary['wireless_node_outlet'])
"
```
