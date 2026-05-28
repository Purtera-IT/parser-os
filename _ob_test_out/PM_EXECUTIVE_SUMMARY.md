# PM Intake Readiness — 2dbc02c9-9d49-40ad-8048-aae78836b0eb

**Status:** 🔴 **Not SOW-ready: 9 blocker question(s) remain**

> 2dbc02c9-9d49-40ad-8048-aae78836b0eb: Wireless / WLAN, Structured cabling, Security camera / VMS, Security access control at atl 047 04, atl 2026 047; 9 blocker and 21 warning SOW question(s) need PM/SA review.

## What the PM does next

1. Do **not** publish the SOW yet.
2. Send the blocker questions below to the customer / vendor / internal owner.
3. Assign the solution-architect review items before draft lock.

## PM scorecard

| Metric | Value |
|---|---:|
| Source files read | 9 |
| Evidence items extracted | 295 |
| PM-visible evidence cards | 71 |
| Confirmed physical sites | 19 |
| SOW blocker questions | 9 |
| SOW warning questions | 21 |
| Top workstream | Security camera / VMS |

## Confirmed sites

| Site | Kind | Confirmed | Evidence items | Source files |
|---|---|:-:|---:|---:|
| atl 047 04 | physical_site | ✓ | 3 | 1 |
| atl 2026 047 | physical_site | ✓ | 9 | 5 |
| atl air 03 | physical_site | ✓ | 6 | 2 |
| atl air asset type warehouse | physical_site | ✓ | 7 | 5 |
| atl cp 05 | physical_site | ✓ | 12 | 2 |
| atl hq 01 | physical_site | ✓ | 6 | 3 |
| atl hq asset type room | physical_site | ✓ | 6 | 4 |
| college pa | physical_site | ✓ | 6 | 4 |
| dev atl 047 | physical_site | ✓ | 4 | 4 |
| hs deal | physical_site | ✓ | 9 | 5 |
| mdf 3a | physical_site | ✓ | 3 | 1 |
| mon fri | physical_site | ✓ | 12 | 1 |
| mon sat | physical_site | ✓ | 3 | 1 |
| optbot college | physical_site | ✓ | 3 | 1 |
| optbot facil | physical_site | ✓ | 16 | 2 |
| optbot logis | physical_site | ✓ | 7 | 2 |
| optbot secur | physical_site | ✓ | 8 | 3 |
| optbot west campus | physical_site | ✓ | 3 | 1 |
| po mock | physical_site | ✓ | 7 | 4 |

## Detected workstreams

| Workstream | Routed? | SOW checks active? | Blockers | Warnings |
|---|:-:|:-:|---:|---:|
| Wireless / WLAN | ✓ | ✓ | 3 | 5 |
| Structured cabling | ✓ | ✓ | 2 | 9 |
| Security camera / VMS | ✓ | ✓ | 2 | 2 |
| Security access control | ✓ | ✓ | 1 | 3 |
| Sites / facilities |  | ✓ | 1 | 2 |
| Commercial terms |  | ✓ | 0 | 0 |
| Delivery / execution planning | ✓ | ✓ | 0 | 0 |
| Electrical / power | ✓ | ✓ | 0 | 0 |
| Hardware / equipment |  | ✓ | 0 | 0 |
| Procurement / finance | ✓ | ✓ | 0 | 0 |

## Questions to resolve before SOW

### Must resolve before SOW

- **Security access control — Door count / door type:** How many doors/openings/readers and what door types are in scope?
- **Security camera / VMS — Camera count and model:** How many cameras and what model(s) are in scope?
- **Security camera / VMS — VMS platform:** What VMS platform is in scope?
- **Sites / facilities — site_roster atom must produce publishable site cluster:** Why did the customer-supplied site roster row not publish as a physical-site cluster? (Verify Site Reality v5 site_roster promotion path.)
- **Structured cabling — Termination scheme (T568A vs T568B):** Will jacks be terminated to T568A or T568B, and is one scheme used uniformly site-wide?
- **Structured cabling — Testing / certification standard:** What test standard and report format are required: Fluke Versiv permanent-link, TIA-568.2-D, etc.?
- **Wireless / WLAN — Per-AP cable certification level:** What cable category and certification level are required per AP drop: Cat6 vs Cat6A, shielded vs UTP, 4-pair permanent-link, mGig/6 GHz ready, OEM-specific requirement?
- **Wireless / WLAN — PoE class per AP (802.3af / at / bt):** What PoE class do the APs require (802.3af vs at vs bt, or Class 4 vs 6 vs 8) and is the source switch power budget reserved?
- **Wireless / WLAN — SSID / VLAN / auth matrix:** What is the full SSID/VLAN/auth matrix: SSID name, VLAN, 802.1X vs PSK vs WPA2/3 vs Open, RADIUS server, guest captive portal?

### PM review / clarification

- **Security access control — Access platform / integration:** What software platform and cardholder/visitor integrations are in scope?
- **Security access control — Locking hardware:** What lock hardware is included/excluded: electric strike, maglock, electrified mortise?
- **Security access control — REX / DPS / door monitoring:** Are REX, DPS, bond sensors, and fault monitoring included?
- **Security camera / VMS — Privacy / compliance / masking:** Are privacy masks, compliance rules, or approval workflows required?
- **Security camera / VMS — Recording mode:** Is recording continuous, motion, event-based, or hybrid?
- **Sites / facilities — After-hours escort / site-staff billing:** Who provides and pays for after-hours / weekend escorts, custodial coverage, lift access, and building supervision during restricted work windows?
- **Sites / facilities — Site cluster kind + evidence provenance:** Verify each published site cluster carries kind=physical_site and that member_atom_ids / artifact_ids are populated; if not, the synthesis rendering or model is broken.
- **Structured cabling — Cable family / jacket / environment:** Is the cable UTP/STP, plenum/riser/OSP, shielded, armored, or otherwise environment-specific?
- **Structured cabling — Core drilling / patching / paint boundary:** If new sleeves/core drilling, patching, paint matching, ceiling repair, or wall restoration are required, who performs and pays for that work?
- **Structured cabling — Faceplate / jack color and count per location:** What faceplate/jack color, count per location, and port-icon scheme apply (e.g., blue=data, white=voice)?
- **Structured cabling — Faceplates / jacks / biscuits:** Are faceplates, jacks, biscuits/surface boxes included? What counts/colors?
- **Structured cabling — Grounding / bonding / backboard responsibility:** Who provides and installs the telecom backboard, ladder rack bonding, ground bar, and bonding conductor, and what TIA-607 acceptance evidence is required?
- **Structured cabling — Rack strategy / anchoring / seismic:** What rack strategy applies: wall-mount, 2-post, 4-post, anchoring/anti-tip/seismic hardware, and rack grounding/bonding acceptance evidence?
- **Structured cabling — Rough-in vs trim-out scope split:** How is the work split between rough-in (pulls, sleeves, fire-stop) and trim-out (terminate, dress, test, label) and who owns each phase?
- **Structured cabling — Service loop length / cable management:** What service loop length is required at jack and patch ends, and which vertical+horizontal managers will be used?
- **Structured cabling — Termination standard:** What termination standard and connector/jack class apply?
- **Wireless / WLAN — DFS channel policy:** Are DFS channels allowed, avoided, or conditionally used, and what happens when DFS events force channel changes?
- **Wireless / WLAN — Heatmap / RF deliverables:** What RF deliverables are required: heatmaps, SNR, channel reuse, coverage maps?
- **Wireless / WLAN — Owner-furnished AP/cable boundary:** Are APs, mounts, and AP cabling owner-furnished or integrator-furnished, and where is the demarcation?
- **Wireless / WLAN — Survey type:** Is this predictive, passive, AP-on-a-stick, active, or post-validation survey work?
- **Wireless / WLAN — WIPS / rogue-AP policy:** Should WIPS/rogue-AP events be monitored, who receives alerts, and who owns containment / remediation?

### Nice-to-have / polish

- **Structured cabling — Attic stock + warranty registration in acceptance package:** What attic stock (extra cables, jacks, faceplates) is required and is OEM warranty registration (Panduit, Belden, CommScope, Leviton, etc.) part of closeout?
- **Structured cabling — Demolition / abandoned cable removal:** Is demolition or abandoned cable removal included or excluded?

## Customer clarification email starter

```text
Subject: Clarifications needed before SOW draft

Hi team,

We reviewed the intake package and need the following clarifications before we can finalize the SOW:

MUST-ANSWER before we can draft scope:
  1. How many doors/openings/readers and what door types are in scope?
  2. How many cameras and what model(s) are in scope?
  3. What VMS platform is in scope?
  4. Will jacks be terminated to T568A or T568B, and is one scheme used uniformly site-wide?
  5. What test standard and report format are required: Fluke Versiv permanent-link, TIA-568.2-D, etc.?
  6. What cable category and certification level are required per AP drop: Cat6 vs Cat6A, shielded vs UTP, 4-pair permanent-link, mGig/6 GHz ready, OEM-specific requirement?
  7. What PoE class do the APs require (802.3af vs at vs bt, or Class 4 vs 6 vs 8) and is the source switch power budget reserved?
  8. What is the full SSID/VLAN/auth matrix: SSID name, VLAN, 802.1X vs PSK vs WPA2/3 vs Open, RADIUS server, guest captive portal?

CONFIRMATIONS that will shape commercial terms and assumptions:
  9. What software platform and cardholder/visitor integrations are in scope?
  10. What lock hardware is included/excluded: electric strike, maglock, electrified mortise?
  11. Are REX, DPS, bond sensors, and fault monitoring included?
  12. Are privacy masks, compliance rules, or approval workflows required?
  13. Is recording continuous, motion, event-based, or hybrid?
  14. Who provides and pays for after-hours / weekend escorts, custodial coverage, lift access, and building supervision during restricted work windows?
  15. Is the cable UTP/STP, plenum/riser/OSP, shielded, armored, or otherwise environment-specific?
  16. If new sleeves/core drilling, patching, paint matching, ceiling repair, or wall restoration are required, who performs and pays for that work?
  17. What faceplate/jack color, count per location, and port-icon scheme apply (e.g., blue=data, white=voice)?
  18. Are faceplates, jacks, biscuits/surface boxes included? What counts/colors?

Once we have these answers, we can finalize the scope, assumptions, exclusions, acceptance criteria, and commercial terms.

Thanks,
Project team
```
