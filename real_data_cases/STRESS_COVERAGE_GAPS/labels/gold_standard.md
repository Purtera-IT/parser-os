# Gold standard — STRESS_COVERAGE_GAPS

**Bundle**: This is a documented case for **service lines where public pre-SOW data is thin or unavailable**. Per `SOURCE_NOTES.md`, four service lines have no rich public corpora and should be supplemented with synthetic artifacts.

This case is **metadata only** — no `artifacts/` directory exists, only `SOURCE_NOTES.md`. The gold standard documents what synthetic artifacts should be generated to round out the service-line coverage.

## Service lines documented as gaps

Per `SOURCE_NOTES.md`:

### 1. `fire_safety` — Fire device-schedule XLSX from NFPA-72 typical layouts

- **Why thin**: Vendor manuals are plentiful, real device schedules hide behind Bonfire/PlanetBids logins
- **Reference**: Duke Hospital site-specific fire plan exists publicly, but device counts are not in the document
- **Recommended synthetic artifact**:
  - XLSX with columns: `device_id`, `device_type`, `manufacturer`, `model`, `location`, `floor`, `addressable_loop`, `loop_address`, `notification_zone`, `acceptance_test_status`
  - 200+ devices across 4-floor hospital model
  - Mix of: smoke detectors (photoelectric/ionization), heat detectors (rate-of-rise/fixed-temp), pull stations, horns/strobes, speakers, sprinkler flow switches, sprinkler tamper switches, duct detectors, kitchen suppression
  - NFPA 72-style addressable layout: panel address + loop number + device number
  - Manufacturers: Notifier, SimplexGrinnell, Honeywell Fire, Edwards EST, Mircom, Siemens

### 2. `das` — Distributed Antenna System for in-building public safety / cellular

- **Why thin**: Almost no public real customer artifacts (mostly vendor whitepapers)
- **Reference**: City of Moore OK Public Safety System RFP #2025-006 is the closest public ref
- **Recommended synthetic artifact**:
  - 4-story building DAS RFP narrative referencing NFPA 1225, UL 2524, IFC §510
  - Specific scope items: BDA (Bidirectional Amplifier), donor antenna, indoor antennas (omnidirectional + sector), passive vs. active DAS, fiber distribution, signal strength survey, public-safety frequencies (700 MHz, 800 MHz, UHF)
  - Compliance: NFPA 1225 Standard for Emergency Services Communications, UL 2524 for In-Building Systems, IFC §510 Public Safety Coverage requirements
  - Acceptance criteria: 95% coverage at -95 dBm DAQ 3.0+

### 3. `electrical` — Panel schedule XLSX with mid-sheet totals

- **Why thin**: Embedded inside larger MEP packages, not standalone pre-SOW packets
- **Recommended synthetic artifact**:
  - Panel schedule XLSX with mid-sheet totals and merged-cell breaker rows
  - Columns: `circuit_no`, `breaker_size_a`, `phase_a_w`, `phase_b_w`, `phase_c_w`, `neutral`, `description`, `space_layout`
  - 84-circuit panel (3 phase, 4-wire) with mixed loads
  - Mid-sheet "Sub-Total per Phase" rows that demand structure-aware row interpretation
  - Column aggregates: connected load, demand factor, demand load, % loading per phase

### 4. `itad` — already partially in STRESS_ITAD_PAIR (see that case)

- **Recommended synthetic artifact**:
  - Asset-list XLSX with OEM/serial/condition columns
  - 50-500 mixed assets across laptops, desktops, servers, switches, displays
  - Realistic columns: asset_tag, type, OEM, model, serial, acquisition_date, condition (good/fair/EOL), data_class (PII/PHI/CUI/public)
  - See [STRESS_ITAD_PAIR/labels/gold_standard.md](../../STRESS_ITAD_PAIR/labels/gold_standard.md) for full ITAD ontology gap discussion

## Service line: meta (cross-cutting)

This case is not service-line-specific; it's a **meta-case documenting which packs lack real-data validation**.

**Recommended domain pack**: none directly — the case exists to inform pack-coverage planning.

## Expected parser behavior with empty case

Same as STRESS_ITAD_PAIR (see that gold standard). When parser-os encounters an empty case directory:

1. Detect empty `artifacts/` (or no `artifacts/` at all)
2. Read `SOURCE_NOTES.md` as case metadata
3. Produce a minimal envelope with `compile_status: no_artifacts_found`
4. Generate `coverage_gap:*` entries for each service line listed in SOURCE_NOTES.md

## Expected synthesis pipeline (recommended for downstream)

```
scripts/synthesize_gap_artifacts.py
  --service-line fire_safety \
  --output real_data_cases/SYNTH_FIRE_4_STORY_HOSPITAL/artifacts/

scripts/synthesize_gap_artifacts.py \
  --service-line das \
  --output real_data_cases/SYNTH_DAS_4_STORY_OFFICE/artifacts/

scripts/synthesize_gap_artifacts.py \
  --service-line electrical \
  --output real_data_cases/SYNTH_PANEL_SCHEDULE_84_CIRCUIT/artifacts/

scripts/synthesize_gap_artifacts.py \
  --service-line itad \
  --output real_data_cases/STRESS_ITAD_PAIR/artifacts/
```

When synthesized cases are added, the gap-detector should produce significantly fewer `coverage_gap:*` warnings.

## Gold standard for the gap-detector itself

When parser-os runs against the entire stress corpus:

1. **Detect that fire_safety, das, electrical have no real-data validation cases** — produce coverage_gap warnings.
2. **Detect that itad has only partial coverage (STRESS_ITAD_PAIR has empty artifacts)** — flag as `partial_coverage`.
3. **Quantify the gap** — for each gap service line, report:
   - Pack vocabulary terms count
   - Real-data atom count: 0
   - Validation status: `no_real_data`
4. **Recommend synthesis** — point to scripts/synthesize_gap_artifacts.py with the relevant `--service-line` argument.

This case effectively serves as the **gold reference for the gap-detector's own output schema**.
