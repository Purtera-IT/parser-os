# Gold standard — STRESS_AV_TRIO (composite)

**Bundle**: 3 AV RFPs spanning conference-services contract → boardroom remodel → multi-agency conference-room install
- `icma_av_2025_2028_rfp.pdf` — service contract for ICMA's annual conference, 4-year, 4-cities (Tampa→Long Beach→Toronto→TBD), 5–6K attendees
- `hayward_boardroom_av_rfp.pdf` — small boardroom remodel, customer-authored Bosch DICENTIS BOD, 38 line-items
- `ambag_mbard_av_addendum1.pdf` — multi-agency joint conference room, narrative scope, federal funding (Caltrans/FHWA/DBE)

**Service line**: `av` (audio_visual)
**Recommended domain pack**: `av_pack`

The three artifacts together stress different shapes of AV work:
1. ICMA — narrative service-only, no devices counts → tests `service-line` ontology
2. Hayward — quantified BOD with proprietary parts → tests **device extraction + manufacturer/model/qty matrix**
3. AMBAG — narrative-with-tasks, federal compliance overlay → tests **scope-inclusion + DBE/regulatory** classification

See per-artifact gold standards:
- [`gold_standard_icma.md`](gold_standard_icma.md)
- [`gold_standard_hayward.md`](gold_standard_hayward.md)
- [`gold_standard_ambag.md`](gold_standard_ambag.md)

## Cross-artifact expectations (this is the bundle gold)

When all 3 artifacts are compiled together as a single case (which is unusual — they are *different customers* — but still the corpus reality):

### Expected cross-artifact edges

- **Zero `quantity_conflict` edges** between artifacts — they are 3 different customers. Cross-customer quantity comparisons are nonsense.
- **Zero `vendor_overlap` edges** — no vendor names appear in two artifacts.
- **Multiple `service_line_overlap` edges** — all three are AV service line. The graph builder should produce a `pack:av` cluster with 3 customer nodes.
- **`pack_routing_consistency` edge** — all three should resolve to `av_pack` (or a near-equivalent). Inconsistent pack routing across the bundle would be a routing bug.

### Expected entity_keys (must include from at least one artifact)

- `customer:icma`, `customer:hayward_area_recreation_and_park_district`, `customer:ambag`, `customer:mbard`
- 3+ distinct site addresses
- 13+ device kinds across the three (microphones, cameras, monitors, projectors, displays, speakers, mixers, switchers, racks, voting devices)

### Expected packet families (cross-bundle)

| Family | Why this bundle generates them |
|---|---|
| `customer_override` | AMBAG: deadline change. Hayward: 5 vs 7 voting licenses. ICMA: pricing-flexibility rule. |
| `scope_inclusion` | All three: device + service inclusions. |
| `scope_exclusion` | Hayward: electrical/blocking exclusions. ICMA: affiliate event scope-out. |
| `missing_info` | All three: vendor-questions / blank rate cards / unspecified counts. |
| `site_access` | ICMA: pre-conference inspections. AMBAG: mandatory walkthrough. |
| `meeting_decision` | All three. |
| `action_item` | All three. |

### Expected aggregate metrics

```
expected_min_atom_count: 250
expected_min_packet_count: 45
expected_min_distinct_customers: 4   # ICMA, HARD, AMBAG, MBARD
expected_min_distinct_sites: 5       # Tampa, Long Beach, Toronto, Hayward boardroom, AMBAG/MBARD conf room
expected_min_unique_vendors_referenced: 14  # Bosch, LG, Chief, Middle Atlantic, Furman, Evolution, Legrand, Bose, Atlona, Planar, Comprehensive, MuxLab, Rave (no), etc.
expected_min_constraint_atoms: 25
expected_min_compliance_atoms: 15    # DBE, Title VI, EEO, ADA, prevailing wage, insurance
expected_min_unsupported_receipts: 10  # blank tables + draft agreement placeholders
```

## Why this case is a stress test for the AV pack

1. **Customer-authored BOD vs. vendor-asked equipment list** — Hayward provides one (Bosch DICENTIS, 38 line items); the other two do not. The parser must detect the BOD as `customer_current_authored` while ICMA's equipment matrix (which has empty rate cells) is a *vendor proposal solicitation*.
2. **Schedule-embedded scope** — ICMA's room×time matrix; AMBAG's task list with deliverables; Hayward's procurement-then-install model. Three different scope-shapes.
3. **Federal-funded vs. local-funded** — AMBAG triggers DBE/FHWA/Caltrans compliance ontology that the av_pack alone won't have.
4. **Bosch DICENTIS proprietary product family** — 16 SKUs sharing a `DCNM-*` prefix. Tests product-family resolution.
5. **3 different procurement timelines** — ICMA: 4-year contract, multi-event. Hayward: 2-month install window. AMBAG: complete by June 30, 2024 + 5 years of support. The parser should produce 3 different `timeline:*` entity keys.
6. **Single-year, multi-year, and event-based pricing structures** — Hayward (fixed price), AMBAG (firm-fixed-price + 5yr hourly support), ICMA (transaction/fixed/LOE flexible). All three are valid AV-vendor pricing patterns; the parser should classify each correctly.
