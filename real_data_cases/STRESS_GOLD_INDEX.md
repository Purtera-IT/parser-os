# STRESS_* Gold Standard Index

This document indexes the gold standards for the 12 stress test cases in `real_data_cases/STRESS_*/`.

Each gold standard describes what parser-os should output when it runs against the case's artifacts: expected packet families, entity keys, exclusion patterns, ontology gaps, constraint patterns, cross-artifact edges, stress-test attributes, and known parser failure modes.

## Gold standard structure per case

```
real_data_cases/STRESS_<CASE>/
├── SOURCE_NOTES.md                    (case provenance + why-it's-a-stress-test)
├── artifacts/                         (the actual PDFs / XLSX files)
└── labels/
    ├── gold_standard.md               (composite gold — bundle-level expectations)
    ├── gold_standard.json             (machine-parseable summary)
    └── gold_standard_<artifact>.md    (per-artifact details, when bundle has 2+ artifacts)
```

## Cases

| Case ID | Service Line | Pack | Artifacts | Composite | Per-Artifact Files |
|---|---|---|---|---|---|
| `STRESS_VT_CAM` | security_camera | security_camera_pack | 1 PDF (16pp) | gold_standard.md/.json | (single artifact) |
| `STRESS_DOWNEY_CABLING` | copper_cabling | copper_cabling_pack | 2 PDFs | gold_standard.md/.json | (covered in composite) |
| `STRESS_NATOMAS_WIRELESS` | wireless | wireless_pack | 1 PDF (25pp) | gold_standard.md/.json | (single artifact) |
| `STRESS_MULTI_CAM` | security_camera | security_camera_pack | 3 PDFs | gold_standard.md/.json | (covered in composite) |
| `STRESS_ACS_USC_PIEDMONT` | access_control | access_control_pack | 2 PDFs | gold_standard.md/.json | (covered in composite) |
| `STRESS_PAGING_TRIO` | paging | paging_pack | 2 PDFs (1 missing) | gold_standard.md/.json | (covered in composite) |
| `STRESS_BMS_SPECS` | bms | bms_pack | 2 PDFs (1 missing) | gold_standard.md/.json | (covered in composite) |
| `STRESS_NET_MAINT` | networking + security_camera + access_control | multi-pack | 2 PDFs + 1 XLSX | gold_standard.md/.json | gold_standard_mobile.md, gold_standard_octa.md, gold_standard_ms_its.md |
| `STRESS_AV_TRIO` | av | av_pack | 3 PDFs | gold_standard.md/.json | gold_standard_icma.md, gold_standard_hayward.md, gold_standard_ambag.md |
| `STRESS_XLSX_RARE` | (default) | default_pack | 2 XLSX | gold_standard.md/.json | (covered in composite) |
| `STRESS_ITAD_PAIR` | itad (intended) | itad_pack | 0 (downloads 403/404'd) | gold_standard.md/.json | (documented gap) |
| `STRESS_COVERAGE_GAPS` | meta | none | 0 (intentional) | gold_standard.md/.json | (meta-case for gap detector) |

## Cross-cutting summaries

### Service-line coverage

| Service line | Cases providing real-data validation |
|---|---|
| `security_camera` | STRESS_VT_CAM, STRESS_MULTI_CAM, STRESS_NET_MAINT (Mobile + OCTA portion) |
| `access_control` | STRESS_ACS_USC_PIEDMONT, STRESS_NET_MAINT (OCTA portion) |
| `wireless` | STRESS_NATOMAS_WIRELESS |
| `copper_cabling` | STRESS_DOWNEY_CABLING |
| `av` | STRESS_AV_TRIO |
| `paging` | STRESS_PAGING_TRIO |
| `bms` | STRESS_BMS_SPECS |
| `networking` | STRESS_NET_MAINT (MS ITS Managed VPN portion) |
| `itad` | STRESS_ITAD_PAIR (gap — synth recommended) |
| `fire_safety` | (gap — STRESS_COVERAGE_GAPS) |
| `das` | (gap — STRESS_COVERAGE_GAPS) |
| `electrical` | (gap — STRESS_COVERAGE_GAPS) |

### Artifact count summary

- **18 PDFs** (across 8 cases with PDFs)
- **3 XLSX files** (1 in STRESS_NET_MAINT MS ITS, 2 in STRESS_XLSX_RARE)
- **0 in STRESS_ITAD_PAIR** (download failure)
- **0 in STRESS_COVERAGE_GAPS** (intentional meta-case)

### Parser-routing test coverage

- `orbitbrief_pdf` exercised 18 times across diverse documents (RFPs, master specs, addenda, board memos)
- `orbitbrief_xlsx` exercised 3 times with varying sheet counts and data shapes
- Multi-artifact bundles test the dispatcher's per-file routing

### Stress dimensions covered

- **Quantity contradictions** (STRESS_NATOMAS_WIRELESS — 500 vs 136 across same SKU)
- **Customer-authored BOM with proprietary parts** (STRESS_AV_TRIO Hayward — Bosch DICENTIS 16 SKUs; STRESS_DOWNEY_CABLING — Superior Essex + Leviton with color)
- **Master specs vs project RFPs** (STRESS_BMS_SPECS, STRESS_ACS_USC_PIEDMONT/USC)
- **Addendum-supersedes-original lattice** (STRESS_VT_CAM, STRESS_DOWNEY_CABLING, STRESS_AV_TRIO/AMBAG)
- **Multi-vendor existing infrastructure** (STRESS_NET_MAINT/OCTA Lenel + Milestone → Genetec)
- **Massive scope variance** (STRESS_MULTI_CAM — 50 vs 7,762 vs 0 new cameras across customers)
- **Federal/state procurement compliance overlay** (STRESS_AV_TRIO/AMBAG DBE, STRESS_NATOMAS E-rate, STRESS_NET_MAINT/OCTA Russia/Ukraine sanctions, STRESS_DOWNEY Iran Contracting Act)
- **Vendor evaluation tabulation in source** (STRESS_PAGING_TRIO/SJCD, STRESS_ACS_USC_PIEDMONT/Piedmont)
- **Embedded forms / blank rate cards** (STRESS_AV_TRIO/Hayward Attachment D, STRESS_NATOMAS cost proposal, STRESS_NET_MAINT/MS ITS, STRESS_XLSX_RARE NJEDA)
- **Brand-specific BOD with color** (STRESS_DOWNEY Superior Essex 66-240-5A *Green*)
- **Customer-authored Q&A as customer_current_authored** (STRESS_VT_CAM, STRESS_DOWNEY Addendum, STRESS_XLSX_RARE/CalSAWS 484 rows)
- **Tiered breakpoint pricing** (STRESS_XLSX_RARE/NJEDA AUM tiers)
- **Per-FTE pricing** (STRESS_PAGING_TRIO/UMaine 18,664 active users)
- **Hierarchical Ref# outline in XLSX** (STRESS_NET_MAINT/MS ITS, STRESS_XLSX_RARE/CalSAWS)

### Aggregate gold metrics (across all populated cases)

```
total_pdf_artifacts: 18
total_xlsx_artifacts: 3
total_distinct_customers: ~20+ (across all 9 populated cases)
total_distinct_sites: 350+
total_unique_vendors_referenced: 30+
total_expected_atom_count_min: 2,500+
total_expected_packet_count_min: 380+
total_expected_constraint_atoms: 280+
total_expected_compliance_atoms: 170+
total_csi_section_atoms: 100+ (Downey + USC + WSU + UH dominate)
```

## Suggested compile commands

To run parser-os against the entire stress corpus:

```bash
# Pull the whole corpus (idempotent)
bash scripts/fetch_stress_test_corpus.sh

# Compile each case + drop full review folders
mkdir -p /tmp/stress_review
for case in real_data_cases/STRESS_*; do
  case_id=$(basename "$case")
  python -m app.cli compile "$case" \
    --out "/tmp/${case_id}.json" \
    --review-out "/tmp/stress_review" --no-cache
done

# Compare actual output vs gold standards
python scripts/compare_compiles.py \
  --gold-dir real_data_cases/STRESS_*/labels/ \
  --compiled-dir /tmp/stress_review/
```

The gap-detector should fire for:
- itad pack vocabulary (no real validation data — STRESS_ITAD_PAIR has empty artifacts)
- fire_safety, das, electrical packs (no real data — see STRESS_COVERAGE_GAPS)
- E-rate / FCC / USAC / Secure Networks Act vocabulary (STRESS_NATOMAS_WIRELESS)
- BAS controller hierarchy (B-OWS/NAC/EAC/UC vs NAC/SNC/PEC/AUC) — STRESS_BMS_SPECS
- Bosch DICENTIS proprietary product family — STRESS_AV_TRIO/Hayward
- Talk-A-Phone emergency phone family — STRESS_ACS_USC_PIEDMONT/USC
- Object classification ontology (people/vehicles/attributes) — STRESS_MULTI_CAM/Santa Monica
- DAS public-safety frequencies (700/800 MHz, UHF) — gap, see STRESS_COVERAGE_GAPS
