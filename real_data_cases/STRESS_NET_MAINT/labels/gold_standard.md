# Gold standard — STRESS_NET_MAINT (composite)

**Bundle**: 3 networking + maintenance artifacts spanning municipal security camera maintenance → multi-vendor security systems repair → state managed VPN service.

| File | Pages | Customer | Shape |
|---|---|---|---|
| `mobile_camera_maint_RFP5954.pdf` | 10 | City of Mobile, AL | Security Camera Maintenance Services with 80+ site Camera Inventory Exhibit A (1,523 cameras total) |
| `octa_security_systems_repair_RFP_4-2293.pdf` | ~30+ | Orange County Transportation Authority | Multi-vendor security systems maintenance (Lenel + Milestone → Genetec transition); $480,454 / 3 yr; 5-tier SLA |
| `ms_its_managed_vpn_RFP4080_attachA.xlsx` | 3 sheets | State of Mississippi ITS | Standalone XLSX attachment; Managed VPN technical spec + cost + scoring; 105-pt scale |

**Service line**: `networking` + `security_camera` + `access_control` (multi-pack)
**Recommended domain pack**: `networking_pack` for MS ITS; `security_camera_pack` for Mobile + OCTA's VMS portion; `access_control_pack` for OCTA's ACS portion.

This case is the gold reference for **rare real-world XLSX attachments + multi-pack routing**. The MS ITS XLSX is the only standalone spreadsheet in the entire corpus.

See per-artifact gold standards:
- [`gold_standard_mobile.md`](gold_standard_mobile.md)
- [`gold_standard_octa.md`](gold_standard_octa.md)
- [`gold_standard_ms_its.md`](gold_standard_ms_its.md)

## Cross-artifact bundle expectations

### Expected cross-artifact edges

- **0 cross-customer `quantity_conflict` edges** — Mobile (1,523 cameras) vs OCTA (mixed Axis/Pelco fleet) vs MS ITS (no quantities) are different customers. No conflict.
- **Service-line variety**: `security_camera` (Mobile), `security_camera` + `access_control` (OCTA), `networking` (MS ITS). Tests parser's ability to route different artifacts in the same case to different domain packs.
- **`vendor:genetec`** appears in Mobile (Genetec Security Center mentioned as VMS option) and OCTA (Lenel→Genetec transition). MS ITS doesn't mention Genetec. Cross-artifact edge possible but weak.
- **`vendor:milestone`** appears in OCTA only. No cross-artifact edge.
- **`vendor:lenel`** appears in OCTA only.
- **No vendor anchors span all 3 artifacts** — they're 3 different procurement contexts.
- **Mixed parser routing**:
  - Mobile → `orbitbrief_pdf`
  - OCTA → `orbitbrief_pdf`
  - MS ITS → `orbitbrief_xlsx` (the rare XLSX-direct case)

### Expected aggregate metrics

```
expected_min_atom_count: 380
expected_min_packet_count: 50
expected_min_distinct_customers: 3   # Mobile, OCTA, MS ITS
expected_min_distinct_sites: 90+      # 80 Mobile + 8 OCTA + State of MS counts
expected_min_unique_vendors_referenced: 15  # Genetec, Lenel, Milestone, Nedap, Zenitel, Grandstream, Aiphone, Axis, Pelco, Sony, Microsoft, Cisco (implied), Streamvault, etc.
expected_min_constraint_atoms: 40
expected_min_compliance_atoms: 20
expected_pdf_artifacts: 2
expected_xlsx_artifacts: 1
expected_min_template_unsupported_atoms: 8
```

## Cross-bundle stress-test attributes

1. **3 different parser pipelines exercised** — `orbitbrief_pdf` (Mobile + OCTA) and `orbitbrief_xlsx` (MS ITS). Tests parser routing.
2. **3 different SLA models**:
   - Mobile: Red/Yellow/Green/White (4 colors)
   - OCTA: Critical/Urgent/Minimal/Normal (4 named tiers)
   - MS ITS: SLA per service in cost table
   The parser should detect each as a 4-tier SLA system but preserve the customer-specific naming.
3. **3 different vendor-credential models**:
   - Mobile: A.M. Best A-VII insurance carrier rating + AL Electrical Contractor License + CompTIA Network+ + MPD background check
   - OCTA: Lenel 3-tier certification + Milestone Elite Partner + Master Technician + 50 miles office + C-10/C-7 California licenses
   - MS ITS: Resume requirements + state-approval-of-individuals + clearance/certification per resume
4. **3 different scope shapes**:
   - Mobile: maintenance with 1,523-camera roster (existing fleet)
   - OCTA: maintenance + transition (multi-vendor → single-vendor)
   - MS ITS: greenfield managed-service procurement (no existing fleet referenced)
5. **3 different document formats** (1 PDF with table, 1 PDF with narrative+attachments, 1 XLSX). Tests universal projection.
6. **OCTA's "transitioning to Genetec"** is a critical temporal-state element — the parser should produce a `decision:vendor_transition_in_progress` packet that overrides any "scope = Lenel" implication.
7. **Mobile's Exhibit A 80-row table** + **MS ITS's hierarchical Ref# outline** + **OCTA's 5-tier SLA matrix** — three different small-table structures that all need universal projection.
8. **No ground-truth quantities in MS ITS** — pure technical-spec form. Tests parser restraint (don't hallucinate quantities from a spec sheet).

## Known difficulties & where the parser will likely fail

1. **Different parser routing per artifact** — the case dispatcher must correctly route each file. If MS ITS goes to PDF parser, structure will be lost.
2. **OCTA's "currently using Lenel + transitioning to Genetec"** — if the parser produces both `vendor:lenel` and `vendor:genetec` as `current` scope, it'll over-extract. Should produce `vendor:lenel` (current, transitioning out) + `vendor:genetec` (transition target).
3. **Mobile's GIS map (page 7) + 80-row roster (pages 7–10)** — single PDF mixing graphic and tabular content. The parser must handle the page-mode transition.
4. **MS ITS Sheet 3 scoring weights** — easy to miss because it's separate from the Sheet 1 technical specs. Should produce procurement-evaluation entities.
5. **No vendor names in MS ITS** — it's a vendor-asks form. Parser should produce zero `vendor:*` entities for the active project (vs. potentially many for OCTA).
6. **Mobile's "Genetec Security Center or a similar product" wording** — "may be" introduces uncertainty. Parser should produce `missing_info:vms_software_decision` rather than `scope_inclusion:genetec`.
7. **Combining 3 customers' compliance stacks** is high-volume. The parser must classify atoms correctly:
   - Mobile: City of Mobile business license, AL Electrical, MPD background check
   - OCTA: California Form 700, Political Reform Act, EO 13660-14065 sanctions
   - MS ITS: State of Mississippi RFP framework
   Parser must NOT cross-apply state-specific regulations to other customers' atoms.
