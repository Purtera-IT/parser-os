#!/usr/bin/env bash
#
# Fetch the public pre-SOW stress-test corpus into real_data_cases/.
#
# Each bundle becomes a real_data_cases/<CASE_ID>/ directory with the original
# files in artifacts/ — matching the existing case convention so they plug
# straight into `python scripts/compile_real_data_case.py --case-id <CASE_ID>`.
#
# All sources are anonymous public HTTPS GETs (no cookies, no auth).
# Re-run is idempotent: existing files are skipped.
#
# Usage:
#   bash scripts/fetch_stress_test_corpus.sh                # fetch all bundles
#   bash scripts/fetch_stress_test_corpus.sh STRESS_VT_CAM  # fetch one bundle
#
# After fetching:
#   python scripts/compile_real_data_case.py --case-id STRESS_VT_CAM
#   # or for all of them:
#   for case in real_data_cases/STRESS_*; do
#     python scripts/compile_real_data_case.py --case-id "$(basename "$case")"
#   done

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASES_DIR="$ROOT/real_data_cases"
ONLY_CASE="${1:-}"

mkdir -p "$CASES_DIR"

fetch() {
  local case_id="$1"; shift
  local url="$1"; shift
  local filename="${1:-}"
  if [[ -n "$ONLY_CASE" && "$ONLY_CASE" != "$case_id" ]]; then
    return 0
  fi
  local case_dir="$CASES_DIR/$case_id/artifacts"
  mkdir -p "$case_dir"
  if [[ -z "$filename" ]]; then
    filename="$(basename "${url%%\?*}")"
  fi
  local out="$case_dir/$filename"
  if [[ -s "$out" ]]; then
    printf '  · skip (exists) %s\n' "$filename"
    return 0
  fi
  printf '  ↓ %s\n' "$filename"
  if ! curl -fsSL --retry 3 --connect-timeout 30 --max-time 600 -o "$out.partial" "$url"; then
    printf '  ✗ FAILED %s\n' "$url" >&2
    rm -f "$out.partial"
    return 1
  fi
  mv "$out.partial" "$out"
}

note() {
  local case_id="$1"; shift
  local body="$*"
  if [[ -n "$ONLY_CASE" && "$ONLY_CASE" != "$case_id" ]]; then
    return 0
  fi
  local case_dir="$CASES_DIR/$case_id"
  mkdir -p "$case_dir"
  printf '%s\n' "$body" > "$case_dir/SOURCE_NOTES.md"
}

# ─────────────────────────────────────────────────────────────────
# 1. Virginia Tech enterprise camera RFP (multi-doc, addenda)
# ─────────────────────────────────────────────────────────────────
echo "[1/10] STRESS_VT_CAM — Virginia Tech RFP 0016531 + Addenda"
note STRESS_VT_CAM "# Virginia Tech RFP 0016531 — enterprise video surveillance + addenda
- Service line: security_camera
- Why stress: addenda CHANGE scope vs original RFP — tests customer_current_authored vs quoted_old_email
- Source: https://hokieprivacy.org/files/cameras/"
fetch STRESS_VT_CAM "https://hokieprivacy.org/files/cameras/RFP_0016531_Addendum2.pdf"

# ─────────────────────────────────────────────────────────────────
# 2. Downey USD CAT6 cabling bid + addendum
# ─────────────────────────────────────────────────────────────────
echo "[2/10] STRESS_DOWNEY_CABLING — Downey USD Bid 23/24-20"
note STRESS_DOWNEY_CABLING "# Downey USD Bid 23/24-20 — multi-site CAT6 cabling for IP phone project
- Service line: copper_cabling
- Why stress: walkthrough notes that drop counts changed during walk
- Source: https://web.dusd.net"
fetch STRESS_DOWNEY_CABLING "https://web.dusd.net/wp-content/uploads/2023/12/Bid-23_24-20-Various-Site-CAT6-Cabling-for-IP-Phone-Project.pdf"
fetch STRESS_DOWNEY_CABLING "https://web.dusd.net/wp-content/uploads/2023/12/Addendum-1-Various-Sites-Cabling-1.pdf"

# ─────────────────────────────────────────────────────────────────
# 3. Natomas USD wireless RFP (eRate YR28)
# ─────────────────────────────────────────────────────────────────
echo "[3/10] STRESS_NATOMAS_WIRELESS — Natomas USD RFP 25-107"
note STRESS_NATOMAS_WIRELESS "# Natomas USD RFP 25-107 — Wireless Equipment, eRate YR28
- Service line: wireless
- Why stress: eligible-equipment matrix + pricing forms; classic K-12 format"
fetch STRESS_NATOMAS_WIRELESS "https://resources.finalsite.net/images/v1731965968/natomasunifiedorg/yj3wcurio1hkosiktbrx/NatomasYR28RFP25_107WirelessEquipment.pdf"

# ─────────────────────────────────────────────────────────────────
# 4. Multi-vendor camera bundle (Milwaukee + Santa Monica + CHA)
# ─────────────────────────────────────────────────────────────────
echo "[4/10] STRESS_MULTI_CAM — Milwaukee + Santa Monica + Chicago Housing"
note STRESS_MULTI_CAM "# Three-customer camera bundle (Milwaukee + Santa Monica + Chicago Housing Authority)
- Service line: security_camera
- Why stress: three different cities, different VMS vendors named (Genetec, Milestone, mixed)
- Multi-vendor BOM stress case for naming conflicts"
fetch STRESS_MULTI_CAM "https://wisconsinexaminer.com/wp-content/uploads/2024/06/B-RFP-17341-Scope-of-Work.pdf" "milwaukee_pole_cam_RFP17341.pdf"
fetch STRESS_MULTI_CAM "https://media.governmentnavigator.com/media/bid/1724275525_2024-08-21_418.pdf" "santa_monica_video_analytics_RFP.pdf"
fetch STRESS_MULTI_CAM "http://www.thecha.org/sites/default/files/2025-04/RFP-3276-Camera-System-Upgrade_04.25_Procurement.pdf" "chicago_housing_camera_RFP3276.pdf"

# ─────────────────────────────────────────────────────────────────
# 5. Access control: USC Master Spec + Piedmont Genetec RFP
# ─────────────────────────────────────────────────────────────────
echo "[5/10] STRESS_ACS_USC_PIEDMONT — USC Master Spec + Piedmont Genetec"
note STRESS_ACS_USC_PIEDMONT "# Access Control bundle — USC Master Spec (Lenel + Genetec hybrid) + Piedmont CA Genetec basis-of-design
- Service line: access_control
- Why stress: USC is narrative master spec (table-extraction hard mode); Piedmont names Genetec basis-of-design"
fetch STRESS_ACS_USC_PIEDMONT "https://fpm.usc.edu/wp-content/uploads/2024/07/Access-Control-Master-Specification-07.30.24.pdf" "usc_access_control_master_spec.pdf"
fetch STRESS_ACS_USC_PIEDMONT "https://piedmont.hosted.civiclive.com/common/pages/GetFile.ashx?key=viY8AX3Y" "piedmont_genetec_rfp.pdf"

# ─────────────────────────────────────────────────────────────────
# 6. Mass notification / paging trio
# ─────────────────────────────────────────────────────────────────
echo "[6/10] STRESS_PAGING_TRIO — UMaine + Manchester + San Jacinto"
note STRESS_PAGING_TRIO "# Mass notification / paging trio
- Service line: paging
- Why stress: three different licensing/maintenance pricing structures"
fetch STRESS_PAGING_TRIO "https://www.maine.edu/strategic-procurement/wp-content/uploads/sites/5/2018/01/RFP_24-18_Higher-Education-Emergency-Mass-Notification-Solution.pdf" "umaine_mass_notification_RFP_24-18.pdf"
fetch STRESS_PAGING_TRIO "https://www.manchesterschools.us/wp-content/uploads/2021/05/Overhead-Paging-RFP-Description-and-Specifications.pdf" "manchester_overhead_paging_RFP.pdf"
fetch STRESS_PAGING_TRIO "https://www.sanjac.edu/sites/default/files/inline-files/PR%205%20Mass%20Communication%20and%20Emergency%20Notification%20Services.pdf" "sanjac_mass_comm_PR5.pdf"

# ─────────────────────────────────────────────────────────────────
# 7. BMS / BAS spec trio
# ─────────────────────────────────────────────────────────────────
echo "[7/10] STRESS_BMS_SPECS — Wayne State + UH + Macquarie"
note STRESS_BMS_SPECS "# BAS / BMS spec trio (Tridium / Niagara / JACE explicit)
- Service line: bms
- Why stress: large narrative spec docs with required points lists, controller schedules, integration matrices.
- Tests 'spec without line-item BOM' extraction"
fetch STRESS_BMS_SPECS "https://facilities.wayne.edu/construction/bas-construction-standards-may-2025.pdf" "wayne_state_bas_construction_standards_2025.pdf"
fetch STRESS_BMS_SPECS "https://www.uh.edu/facilities-planning-construction/vendor-resources/owners-design-criteria/master-specs/jan-2017/division25jan2017.pdf" "uh_division_25_master_spec.pdf"
fetch STRESS_BMS_SPECS "https://www.property.mq.edu.au/__data/assets/pdf_file/0009/1261854/BMS-Specification-and-Standard-Macquarie-University-v2.8.pdf" "macquarie_bms_spec_v2.8.pdf"

# ─────────────────────────────────────────────────────────────────
# 8. Networking maintenance bundle (Mobile + OCTA + MS ITS XLSX)
# ─────────────────────────────────────────────────────────────────
echo "[8/10] STRESS_NET_MAINT — Mobile + OCTA + MS ITS (XLSX!)"
note STRESS_NET_MAINT "# Networking + camera maintenance bundle, includes a real XLSX attachment
- Service line: networking + security_camera maintenance
- Why stress: MS ITS attachment-A is a real XLSX (rare); also tests legacy/maintenance-contract vocab"
fetch STRESS_NET_MAINT "https://www.cityofmobile.org/bids_files/5954_2025RFP5954SecurityCamMaint.pdf" "mobile_camera_maint_RFP5954.pdf"
fetch STRESS_NET_MAINT "https://cammnet.octa.net/offlinepackages/42293_0.pdf" "octa_security_systems_repair_RFP_4-2293.pdf"
fetch STRESS_NET_MAINT "https://rfps.its.ms.gov/Procurement/rfps/4080/4080attach_a.xlsx" "ms_its_managed_vpn_RFP4080_attachA.xlsx"

# ─────────────────────────────────────────────────────────────────
# 9. ITAD pair (sparse public data)
# ─────────────────────────────────────────────────────────────────
echo "[9/10] STRESS_ITAD_PAIR — Natrona County + Michigan EGLE"
note STRESS_ITAD_PAIR "# ITAD pair — best two public RFPs found (corpus is thin for ITAD)
- Service line: itad
- Why stress: structured asset categories + certifications; recommend supplementing with synthetic asset-list XLSX"
fetch STRESS_ITAD_PAIR "https://go.boarddocs.com/wy/ncsd1/Board.nsf/files/DHDSAD71B831/\$file/NCSD%20IT%20device%20and%20e-waste%20disposal%20RFP%20-%20Requisition%20.pdf" "natrona_county_itad_rfp.pdf"
fetch STRESS_ITAD_PAIR "https://www.michigan.gov/egle/-/media/Project/Websites/egle/Documents/Programs/MMD/Electronic-Waste/2024-Ewaste-RFP.pdf" "michigan_egle_2024_ewaste_rfp.pdf"

# ─────────────────────────────────────────────────────────────────
# 10. AV bundle (AMBAG addendum + Hayward boardroom + ICMA)
# ─────────────────────────────────────────────────────────────────
echo "[10/10] STRESS_AV_TRIO — AMBAG + Hayward + ICMA"
note STRESS_AV_TRIO "# AV / UCC trio
- Service line: av
- Why stress: meeting/boardroom AV with addenda and pricing schedules — tests MTR/Zoom Room vocab"
fetch STRESS_AV_TRIO "https://ambag.org/sites/default/files/2023-07/MBARD%20Meeting%20Room%20AV%20Project_AddendumNo1_PDFA.pdf" "ambag_mbard_av_addendum1.pdf"
fetch STRESS_AV_TRIO "https://www.haywardrec.org/DocumentCenter/View/9622/RFP_Boardroom-AV-Systems-Upgrade-" "hayward_boardroom_av_rfp.pdf"
fetch STRESS_AV_TRIO "https://icma.org/sites/default/files/2023-12/RFP%20Audio%20Visual%202025-2028.pdf" "icma_av_2025_2028_rfp.pdf"

# ─────────────────────────────────────────────────────────────────
# Bonus: rare XLSX attachments (any service line) for spreadsheet parser stress
# ─────────────────────────────────────────────────────────────────
echo "[bonus] STRESS_XLSX_RARE — rare real XLSX attachments"
note STRESS_XLSX_RARE "# Rare public XLSX attachments (most procurement attachments are PDF — these are real spreadsheets)
- Service line: cross-cutting (good for testing xlsx_parser robustness on real-world layouts)
- Includes legacy .xls (Genetec pricelist) — tests legacy Excel handling"
fetch STRESS_XLSX_RARE "https://www.calsaws.org/wp-content/uploads/2022/09/CalSAWS-MO-RFP-01-2022-Question-and-Answer-Log-103122.xlsx" "calsaws_qa_log.xlsx"
fetch STRESS_XLSX_RARE "https://www.njeda.gov/wp-content/uploads/2022/01/2022-RFP-IPM-051-Fee-Schedule-1-21-22-Final.xlsx" "njeda_fee_schedule.xlsx"

# ─────────────────────────────────────────────────────────────────
# Coverage gaps (no automated fetch — listed in REVIEW)
# ─────────────────────────────────────────────────────────────────
note STRESS_COVERAGE_GAPS "# Service lines where public pre-SOW data is thin
The following service lines do not have rich public corpora.  We recommend
synthesizing realistic artifacts using the active domain pack vocab:

- **fire_safety** — synthesize device-schedule XLSX from NFPA-72 typical layouts;
  pair with public spec docs (Duke Hospital site-specific fire plan exists, but
  device counts aren't in it).
- **das** — synthesize an in-building DAS RFP for a 4-story building referencing
  NFPA 1225 + UL 2524 + IFC §510.  Closest public ref: City of Moore OK Public
  Safety System RFP #2025-006.
- **electrical** — synthesize a panel schedule XLSX with mid-sheet totals and
  merged-cell breaker rows.  Public docs are embedded in larger MEP packages.
- **itad** — already partially covered by STRESS_ITAD_PAIR; supplement with a
  synthetic asset-list XLSX (OEM/serial/condition columns)."

echo ""
echo "Done.  Stress-test corpus staged under real_data_cases/STRESS_*"
echo ""
echo "Next steps:"
echo "  1. Inspect the SOURCE_NOTES.md in each case dir to confirm contents"
echo "  2. Run a compile per case to see how the parser handles it:"
echo "       python -m app.cli compile real_data_cases/STRESS_VT_CAM \\"
echo "         --out /tmp/stress_vt_cam.json --review-out /tmp/stress_review --no-cache"
echo "  3. Open /tmp/stress_review/<compile_id>/REVIEW.md and walk the dossier."
echo "  4. ontology_gaps.md will surface real-world vocab the packs don't know yet."
