# Solution Architect Review Packet — 2dbc02c9-9d49-40ad-8048-aae78836b0eb

**PM readiness:** 🔴 Not SOW-ready: 9 blocker question(s) remain

> 2dbc02c9-9d49-40ad-8048-aae78836b0eb: Wireless / WLAN, Structured cabling, Security camera / VMS, Security access control at atl 047 04, atl 2026 047; 9 blocker and 21 warning SOW question(s) need PM/SA review.

## Solution architect review lane

Technical checks the SA should validate before design/SOW sign-off:

- Validate AP model/count, per-AP PoE class, cable certification level, mounting heights, and survey/post-validation expectations.
- Confirm SSID/VLAN/auth matrix, DFS/WIPS policy, device onboarding workflow, and E-rate/owner-furnished boundaries if applicable.
- Validate cable category, jacket rating, termination scheme, labeling standard, and test report requirement.
- Confirm pathway ownership, firestopping, rough-in / trim-out split, and MDF/IDF cable-management standard.
- Validate patch panels, faceplates, jacks, service loops, grounding/bonding, and closeout package requirements.
- Validate camera count/model/type, VMS platform, retention, recording mode, storage/NVR sizing, privacy masks, and acceptance testing.
- Validate panel/circuit/receptacle/UPS/generator/grounding details and electrical exclusion boundaries.

### SA-owned open items

- **Camera count and model:** How many cameras and what model(s) are in scope?
- **VMS platform:** What VMS platform is in scope?
- **Termination scheme (T568A vs T568B):** Will jacks be terminated to T568A or T568B, and is one scheme used uniformly site-wide?
- **Testing / certification standard:** What test standard and report format are required: Fluke Versiv permanent-link, TIA-568.2-D, etc.?
- **Per-AP cable certification level:** What cable category and certification level are required per AP drop: Cat6 vs Cat6A, shielded vs UTP, 4-pair permanent-link, mGig/6 GHz ready, OEM-specific requirement?
- **PoE class per AP (802.3af / at / bt):** What PoE class do the APs require (802.3af vs at vs bt, or Class 4 vs 6 vs 8) and is the source switch power budget reserved?
- **SSID / VLAN / auth matrix:** What is the full SSID/VLAN/auth matrix: SSID name, VLAN, 802.1X vs PSK vs WPA2/3 vs Open, RADIUS server, guest captive portal?
- **Privacy / compliance / masking:** Are privacy masks, compliance rules, or approval workflows required?
- **Recording mode:** Is recording continuous, motion, event-based, or hybrid?
- **Cable family / jacket / environment:** Is the cable UTP/STP, plenum/riser/OSP, shielded, armored, or otherwise environment-specific?
- **Core drilling / patching / paint boundary:** If new sleeves/core drilling, patching, paint matching, ceiling repair, or wall restoration are required, who performs and pays for that work?
- **Faceplate / jack color and count per location:** What faceplate/jack color, count per location, and port-icon scheme apply (e.g., blue=data, white=voice)?
- **Faceplates / jacks / biscuits:** Are faceplates, jacks, biscuits/surface boxes included? What counts/colors?
- **Grounding / bonding / backboard responsibility:** Who provides and installs the telecom backboard, ladder rack bonding, ground bar, and bonding conductor, and what TIA-607 acceptance evidence is required?
- **Rack strategy / anchoring / seismic:** What rack strategy applies: wall-mount, 2-post, 4-post, anchoring/anti-tip/seismic hardware, and rack grounding/bonding acceptance evidence?
- **Rough-in vs trim-out scope split:** How is the work split between rough-in (pulls, sleeves, fire-stop) and trim-out (terminate, dress, test, label) and who owns each phase?
- **Service loop length / cable management:** What service loop length is required at jack and patch ends, and which vertical+horizontal managers will be used?
- **Termination standard:** What termination standard and connector/jack class apply?
- **DFS channel policy:** Are DFS channels allowed, avoided, or conditionally used, and what happens when DFS events force channel changes?
- **Heatmap / RF deliverables:** What RF deliverables are required: heatmaps, SNR, channel reuse, coverage maps?

## What OrbitBrief found in the intake package

### Sites, access, and facilities

- **Sites, access, and facilities:** Wi-Fi 7 APs: 94 units x $995 | allocated ATL-HQ 52, ATL-WEST 27, ATL-AIR 15 | validates quantity and site-allocation parsing. PoE++ switches: 18 units x $6,125 | access layer refresh and PoE budget expansion for rooms and wireless. Video bars: 31 units x $2,895 | medium-room standardization with cam  
  _Source: artifacts/01_deal_overview_executive_brief.pdf — page 1; HARDWARE AND COMMERCIAL HIGHLIGHTS_
- **Sites, access, and facilities:** Restricted work windows: Before 07:00, after 18:00 weekdays, and all weekends at ATL-HQ-01 and ATL-WEST-02 require 48-hour notice to OPTBOT Facilities. Escort & badge: OPTBOT Facilities provides escorts, badge sponsorship, and lift access at no charge to PurTera. PurTera bills only labor for after-h  
  _Source: artifacts/08_site_roster_and_facilities_authoritative.pdf — page 0; ATL-CP-05_
- **Sites, access, and facilities:** Recommended next actions should include upload validation, Azure metadata check, parser extraction baseline, OrbitBrief summary review, and procurement approval. ACCESS AND SECURITY CONTROLS Use least privilege group OPTBOT-ATL-Refresh-Dev-Readers for any vendor portal test. Use fictional Intune pro  
  _Source: artifacts/06_security_it_integration_notes.pdf — page 1_
- **Sites, access, and facilities:** Access window Escort owner  
  _Source: artifacts/08_site_roster_and_facilities_authoritative.pdf — page 0; MDF / IDF_
- **Sites, access, and facilities:** OPTBOT College Park S 1850 Sullivan Rd, College Pa MDF-CP / stagin Mon-Fri 07:00-15: OPTBOT Logis  
  _Source: artifacts/08_site_roster_and_facilities_authoritative.pdf — page 0; ATL-CP-05_
- **Sites, access, and facilities:** Expedite +12% fee if <30 days Wireless access points  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0; 45 ARO_
- **Sites, access, and facilities:** OPTBOT provides WAN/MPLS handoff ports ready at each MDF on cutover day.  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 1_
- **Sites, access, and facilities:** OPTBOT is refreshing three Atlanta-area offices to create a common collaboration, wireless, and desk-accessory standard. The project is intentionally represented across PDF, DOCX, and XLSX formats so parser-os can reconcile facts from narrative paragraphs, tables, workbooks, and metadata-like labels  
  _Source: artifacts/01_deal_overview_executive_brief.pdf — page 0; EXECUTIVE NARRATIVE_
- **Sites, access, and facilities:** Current rooms use inconsistent camera, microphone, calendar, and scheduling-panel setups, causing meeting delays and uneven user experience. Desk-accessory standards vary by floor and site, creating support complexity and asset tracking gaps. The refresh standardizes collaboration spaces, improves w  
  _Source: artifacts/07_contracting_procurement_packet.pdf — page 0; BUSINESS JUSTIFICATION_
- **Sites, access, and facilities:** Standard business-hour access per site roster doc 08; after-hours per Section 2 of doc 08.  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 1_
- **Sites, access, and facilities:** Step: 1 | Timing: T-5 business days | Owner: PM | Checklist Item: Confirm site access and escort roster | Evidence Required: Approved access list  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Cutover Checklist; row 2_
- **Sites, access, and facilities:** The table below is the customer-approved site_roster. Each row is a physical_site with verified street address, primary MDF/IDF, and facility contact. kind=physical_site for all rows. Site ID Facility name Street address  
  _Source: artifacts/08_site_roster_and_facilities_authoritative.pdf — page 0_

### Scope and deliverables

- **Open question from source:** PDF page 3 appears to contain visual / table / diagram evidence that was not fully extracted.  
  _Source: artifacts/01_deal_overview_executive_brief.pdf — page 3_
- **Scope and deliverables:** Change order pricing (Time & Materials):  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_
- **Scope and deliverables:** Spare parts: 2% of switch/AP count held at ATL-CP-05 for 12 months post-ATP.  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 1_
- **Scope and deliverables:** Total mock deal amount: $1,847,250.00. CFO approval required over $1,500,000. Budget owner approval required over $250,000. Jordan Ames approves workplace outcome and business case. Priya Narang approves technical design. Camila Brooks approves security and data handling. Elliot Tran approves procur  
  _Source: artifacts/07_contracting_procurement_packet.pdf — page 0; BUDGET AND APPROVAL MATRIX_
- **Scope and deliverables:** dealname = OPTBOT Atlanta Office Refresh - Three Site Modernization amount = 1847250 dealstage = contractsent_mock_dev closedate = 2026-07-31 project_sites = ATL-HQ; ATL-WEST; ATL-AIR implementation_window = 2026-05-20 through 2026-08-14 parser_batch_id = parser-os-dev-batch-ATL-047 orbitbrief_works  
  _Source: artifacts/06_security_it_integration_notes.pdf — page 0; HUBSPOT FIELD MAPPING_
- **Scope and deliverables:** T&M not-to-exceed without signed CO: $250,000 cumulative  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_
- **Scope and deliverables:** 000087 - OPTBOT Atlanta Office Refresh | HubSpot 60355665326 08 - Site Roster & Facilities (Authoritative) Customer-supplied final supplement - closes site_roster & after-hours gaps This document is the authoritative site roster for the OPTBOT Atlanta Office Refresh program. It supersedes informal  
  _Source: artifacts/08_site_roster_and_facilities_authoritative.pdf — page 0_
- **Scope and deliverables:** XLSX. 5. Project Schedule XLSX. 6. Security and Integration Notes PDF. 7. Contracting and Procurement Packet PDF. Expected HubSpot attachment count: seven. Expected Azure blob count: seven. Expected parser  
  _Source: artifacts/01_deal_overview_executive_brief.pdf — page 1; RECOMMENDED UPLOAD ORDER_
- **Scope and deliverables:** Delay due to permit or union labor rules at ATL-AIR-03 is force majeure with schedule day-for-day extension.  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 1_
- **Scope and deliverables:** Materials: cost + 15% handling  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_
- **Scope and deliverables:** PurTera carries GL and workers comp; OPTBOT added as additional insured.  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 1_
- **Scope and deliverables:** After-hours / weekend labor: $248/hr (1.5x)  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_

### BOM, procurement, and pricing

- **Quantity evidence:** Quantity 1  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Services; row 3_
- **Quantity evidence:** Quantity 2  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Services; row 7_
- **Quantity evidence:** Quantity 1680  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Services; row 5_
- **Quantity evidence:** Quantity 18  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Services; row 6_
- **BOM / vendor line item:** Line item CoreEdge CX-48P 48-port PoE++ access switch  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Hardware BOM; row 3_
- **BOM / vendor line item:** Line item After-hours installation labor  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Services; row 5_
- **BOM / vendor line item:** Line item Hypercare support  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Services; row 7_
- **BOM / vendor line item:** Line item DockFlex 180 Docking station USB-C 180W  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Hardware BOM; row 7_
- **BOM / vendor line item:** Line item Project management and weekly governance  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Services; row 3_
- **BOM / vendor line item:** Line item Discovery workshops and technical design  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Services; row 2_
- **BOM / vendor line item:** Line item PowerKeep 1500 Line-interactive UPS 1500VA  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Hardware BOM; row 10_
- **BOM / vendor line item:** Line item FieldTab R12 Rugged logistics tablet  
  _Source: artifacts/04_hardware_bill_of_materials.xlsx — sheet Hardware BOM; row 9_

### Asset inventory

- **Asset inventory:** Asset inventory | Serial number, site code, room or user area, and deployment status captured for all tracked hardware.  
  _Source: artifacts/02_statement_of_work.docx — row 3_

### Network, ports, VLANs, and circuits

- **Network, ports, VLANs, and circuits:** Brief title should mention OPTBOT Atlanta Office Refresh. Summary should mention all three sites and the mock-only classification. Key risks should include ATL-WEST circuit timing, ATL-HQ blackout, ATL-AIR RF reliability, and CFO approval threshold.  
  _Source: artifacts/06_security_it_integration_notes.pdf — page 0; ORBITBRIEF EXPECTATIONS_
- **Network, ports, VLANs, and circuits:** Power circuits are 20A minimum for new AV racks; customer electrician provides drops.  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 1_
- **Network, ports, VLANs, and circuits:** PurTera will perform and document the following before energizing new circuits: Megger (insulation resistance): Minimum 1.0 MOhm at 500 V DC on feeders >25 A; 0.5 MOhm minimum on branch circuits. Ground resistance: Less than 5.0 ohms measured at each new rack PDU ground point (fall-of-potential or a  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_
- **Network, ports, VLANs, and circuits:** Step: 2 | Timing: T-3 business days | Owner: IT | Checklist Item: Validate VLANs, DHCP scopes, DNS, firewall rules | Evidence Required: Network readiness email  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Cutover Checklist; row 3_
- **Network, ports, VLANs, and circuits:** R-01 | Circuit upgrade at ATL-WEST may miss procurement deadline | Medium | High | Track weekly with carrier; stage temporary 5G failover kit.  
  _Source: artifacts/02_statement_of_work.docx — row 1_

### Managed-services operations

- **Managed-services operations:** Step: 7 | Timing: Next business day | Owner: Help Desk | Checklist Item: Monitor tickets and floor-walker feedback | Evidence Required: Hypercare log  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Cutover Checklist; row 8_

### Acceptance, validation, cutover, and runbooks

- **Acceptance, validation, cutover, and runbooks:** 20% - Production cutover complete at ATL-HQ-01 and ATL-WEST-02  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_
- **Acceptance, validation, cutover, and runbooks:** 40% - Factory Acceptance Test (FAT) sign-off for core network kit  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_
- **Acceptance, validation, cutover, and runbooks:** 10% - Final acceptance (ATP) across all five sites in site roster doc 08  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_
- **Acceptance, validation, cutover, and runbooks:** Hardware subtotal target: $1,015,626.00. Services subtotal target: $536,030.00. Logistics, freight, contingency, taxes, and fees target: $295,594.00. Grand total target: $1,847,250.00. Payment schedule: 30% at order acceptance, 40% on equipment receipt, 20% at site acceptance, 10% after hypercare cl  
  _Source: artifacts/01_deal_overview_executive_brief.pdf — page 1; COMMERCIAL SUMMARY_
- **Acceptance, validation, cutover, and runbooks:** Cutover blackout dates  
  _Source: artifacts/08_site_roster_and_facilities_authoritative.pdf — page 0; ATL-CP-05_
- **Acceptance, validation, cutover, and runbooks:** 30% - Upon Purchase Order acceptance and project kickoff  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_
- **Acceptance, validation, cutover, and runbooks:** Electrical acceptance tests (electrical.acceptance)  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_
- **Acceptance, validation, cutover, and runbooks:** No work permitted: 2026-11-26 through 2026-11-28 (Thanksgiving), 2026-12-24 through 2027-01-02 (year-end freeze). ATL-AIR-03: no cutover during peak travel weeks without 14-day written waiver from OPTBOT Security. Page 1/1 | PurPulse 841ea7e0-0e2f-412a-aebc-5794c199b85c  
  _Source: artifacts/08_site_roster_and_facilities_authoritative.pdf — page 0; ATL-CP-05_
- **Acceptance, validation, cutover, and runbooks:** Source artifact for this roster: 08_site_roster_and_facilities_authoritative.pdf (this file). Cross-reference discovery: 03_site_surveys_and_requirements.docx, 05_project_schedule_and_cutover_plan.xlsx. Each site_id above must publish as kind=physical_site with member evidence from this roster table  
  _Source: artifacts/08_site_roster_and_facilities_authoritative.pdf — page 0; ATL-CP-05_
- **Acceptance, validation, cutover, and runbooks:** palletize, and stage deployment kits. Phase 3 Site implementation | 2026-07-06 to 2026-07-24 | install site waves, commission rooms, validate Wi-Fi, reconcile assets. Phase 4 Cutover and adoption | 2026-07-27 to 2026-07-31 | run cutover checklist, floor support, signoff, final punch list. Phase 5 Po  
  _Source: artifacts/01_deal_overview_executive_brief.pdf — page 1_
- **Acceptance, validation, cutover, and runbooks:** Discovery files 01-07 remain supporting evidence; this document 09 controls pricing and acceptance where conflicts  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 1_
- **Acceptance, validation, cutover, and runbooks:** Step: 5 | Timing: Cutover evening | Owner: Install Lead | Checklist Item: Replace room devices and APs by wave | Evidence Required: Install checklist  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Cutover Checklist; row 6_

### Risks, assumptions, and constraints

- **Risk or constraint:** Risk ID: R-02 | Description: Executive blackout window at ATL-HQ compresses floor 15 install | Probability: High | Impact: Medium | Mitigation: Pre-stage materials and schedule second crew for post-blackout evening. | Owner: Renee Watkins | Review Cadence: Weekly governance  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Risk Register; row 3_
- **Risk or constraint:** Risk ID: R-05 | Description: Procurement approval matrix requires CFO signoff over $1.5M | Probability: Low | Impact: High | Mitigation: Include CFO packet and budget holder memo in contracting docs. | Owner: Renee Watkins | Review Cadence: Weekly governance  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Risk Register; row 6_
- **Risk or constraint:** Risk ID: R-04 | Description: Legacy conference room cabling may not support new camera placement | Probability: Medium | Impact: Medium | Mitigation: Use survey photos, pull test cables, keep alternate wall-mount kits. | Owner: Renee Watkins | Review Cadence: Weekly governance  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Risk Register; row 5_
- **Risk or constraint:** Risk ID: R-03 | Description: Warehouse RF interference at ATL-AIR may reduce scan reliability | Probability: Medium | Impact: Medium | Mitigation: Complete post-install RF validation and adjust AP channel plan. | Owner: Renee Watkins | Review Cadence: Weekly governance  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Risk Register; row 4_
- **Risk or constraint:** Risk ID: R-01 | Description: Circuit upgrade at ATL-WEST may miss procurement deadline | Probability: Medium | Impact: High | Mitigation: Track weekly with carrier; stage temporary 5G failover kit. | Owner: Renee Watkins | Review Cadence: Weekly governance  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Risk Register; row 2_
- **Risks, assumptions, and constraints:** Explicit program assumptions (global.explicit_assumptions)  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 1_
- **Risks, assumptions, and constraints:** Jordan Ames: Approved business case pending final cutover calendar. Priya Narang: Approved technical design pending ATL-WEST carrier confirmation. Camila Brooks: Approved dev-only handling with production-blocking controls. Elliot Tran: Procurement can issue PO-MOCK-77421 after CFO threshold approva  
  _Source: artifacts/07_contracting_procurement_packet.pdf — page 0; MOCK APPROVAL NOTES_
- **Risks, assumptions, and constraints:** PDF extraction should capture contact names, sites, addresses, deal amount, payment schedule, risks, milestones, and approval thresholds. DOCX extraction should preserve SOW sections, site survey tables, assumptions, exclusions, acceptance criteria, and requirements. XLSX extraction should preserve  
  _Source: artifacts/06_security_it_integration_notes.pdf — page 0; PARSER-OS EXPECTATIONS_
- **Risks, assumptions, and constraints:** OPTBOT provides site access, escorts, loading dock windows, and after-hours approvals at least five business days before each installation wave.  
  _Source: artifacts/02_statement_of_work.docx — row None_
- **Risks, assumptions, and constraints:** Site access constraint  
  _Source: artifacts/05_project_schedule_and_cutover_plan.xlsx — sheet Detailed Tasks; row 13_
- **Risks, assumptions, and constraints:** ATL-WEST circuit upgrade may miss procurement deadline; temporary 5G failover kit should be staged. ATL-HQ executive blackout window compresses floor 15 installation; second crew should be ready after blackout. ATL-AIR warehouse RF interference may affect scanner reliability; post-install RF validat  
  _Source: artifacts/01_deal_overview_executive_brief.pdf — page 1; RISKS AND WATCH ITEMS_
- **Risks, assumptions, and constraints:** Classification: Mock Confidential. This fictional security note is for dev-environment ingestion, extraction, and summarization only. Allowed destinations: HubSpot dev, Azure dev blob storage, parser-os dev workers, OrbitBrief dev workspace. Blocked destinations: production CRM, production Azure ten  
  _Source: artifacts/06_security_it_integration_notes.pdf — page 0; PURPOSE AND CLASSIFICATION_

### Exclusions and commercial boundaries

- **Exclusion / boundary:** 7. Out of Scope  
  _Source: artifacts/02_statement_of_work.docx — row None_
- **Exclusion / boundary:** Production tenant access, real OPTBOT credentials, and real payment processing are explicitly excluded.  
  _Source: artifacts/02_statement_of_work.docx — row None_
- **Exclusion / boundary:** New construction, electrical trenching, permanent conduit installation, furniture procurement, production billing, legal review, and real customer communications are out of scope.  
  _Source: artifacts/02_statement_of_work.docx — row None_
- **Exclusions and commercial boundaries:** Contract type: Fixed Price Agreement. Grand Total (not-to-exceed for defined scope): USD $1,847,250.00. Currency: United States Dollars (USD). Taxes: Excluded; OPTBOT responsible for sales/use tax unless otherwise stated in MSA. Milestone billing schedule:  
  _Source: artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf — page 0_

## Source inventory read

| File | Type | Parser | Evidence items | Status |
|---|---|---|---:|:--|
| `artifacts/08_site_roster_and_facilities_authoritative.pdf` | pdf | orbitbrief_pdf | 29 | ✅ |
| `artifacts/02_statement_of_work.docx` | docx | docx | 43 | ✅ |
| `artifacts/06_security_it_integration_notes.pdf` | pdf | orbitbrief_pdf | 10 | ✅ |
| `artifacts/09_commercial_pricing_acceptance_assumptions_final.pdf` | pdf | orbitbrief_pdf | 40 | ✅ |
| `artifacts/01_deal_overview_executive_brief.pdf` | pdf | orbitbrief_pdf | 18 | ✅ |
| `artifacts/03_site_surveys_and_requirements.docx` | docx | docx | 49 | ✅ |
| `artifacts/04_hardware_bill_of_materials.xlsx` | xlsx | quote | 55 | ✅ |
| `artifacts/05_project_schedule_and_cutover_plan.xlsx` | xlsx | xlsx | 40 | ✅ |
| `artifacts/07_contracting_procurement_packet.pdf` | pdf | orbitbrief_pdf | 11 | ✅ |
