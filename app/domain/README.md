# `app/domain/` — domain packs + project config

A **domain pack** is a YAML file that teaches the entity extractor + the
classifier + the packetizer how a specific vertical (security cameras,
wireless, AV, BMS, …) talks. Packs are pure data — no Python — so you
can ship a new vertical without touching parser code.

## Built-in packs

| Pack | File | Service lines |
|---|---|---|
| security_camera | `security_camera_pack.yaml` | security_camera, video_surveillance, vms |
| wireless | `wireless_pack.yaml` | wireless, wifi, wlan |
| networking | `networking_pack.yaml` | networking, switching, routing |
| copper_cabling | `copper_cabling.yaml` | copper, structured_cabling, low_voltage |
| av | `av_pack.yaml` | av, audio_visual, conferencing |
| bms | `bms_pack.yaml` | bms, building_automation, hvac_controls |
| paging | `paging_pack.yaml` | paging, mass_notification, intercom |
| fire_safety | `fire_safety_pack.yaml` | fire_alarm, fire_safety, life_safety |
| das | `das_pack.yaml` | das, distributed_antenna, in_building_wireless |
| electrical | `electrical_pack.yaml` | electrical, power, grounding |
| access_control | `access_control_pack.yaml` | access_control, intrusion |
| itad | `itad_pack.yaml` | itad, asset_disposition |
| default_pack | `default_pack.yaml` | default (catch-all reference pack) |

## Pack schema (`schemas.py::DomainPack`)

```yaml
pack_id: my_pack            # canonical key (matches the filename minus _pack.yaml)
name: "My Vertical Pack"
version: "1.0.0"

service_lines:              # keywords that auto_route_pack matches against
  - my_vertical
  - some_synonym

entity_types:               # generic alias index — emits "<name>:<canonical>"
  - name: device
    aliases: [device_alias_1, device_alias_2]
    examples: ["Example 1", "Example 2"]
  - name: site
    aliases: [campus, building, floor, hospital]

device_aliases:             # device:<canonical_key> entity-key emitter
  ip_camera:                # canonical key (becomes "device:ip_camera")
    - IP Camera             # surface forms (case-insensitive, word-boundary matched)
    - dome camera
    - PTZ
  nvr:
    - NVR
    - network video recorder

site_alias_patterns:        # regex patterns that match site shapes
  - "store\\s*#?\\d+"
  - "main[-_ ]campus"

action_aliases:             # action_alias keys for action_item / customer_instruction atoms
  cabling:
    - pull cable
    - run cable

constraint_patterns:        # phrase substrings that promote text → constraint atom
  access:
    - escort required
    - badge required

exclusion_patterns:         # phrase substrings that promote text → exclusion atom
  - out of scope
  - by others

customer_instruction_patterns:
  - owner preferred
  - district has selected

quantity_units:             # unit → canonical conversion table
  count: [each, ea, qty, no.]
  length: [ft, foot, feet, meter]

artifact_role_patterns:     # filename / sample-text patterns for pack auto-routing
  high_confidence:
    - "wireless"
    - "wlan"

risk_defaults:              # per-packet-family risk floor (0.0–1.0)
  scope_exclusion: 0.7
  quantity_conflict: 0.8

packet_family_hints:        # optional family-specific packetizer hints
```

## How packs reach the pipeline

`enrich_entities`, `graph_builder`, `packetizer`, and `orbitbrief_pdf`'s
classifier all consume the **active pack**. The active pack is set
once per compile by `app.domain.set_active_domain_pack()`; the
orchestrator picks it via this priority order (highest wins):

1. `--domain-pack <pack_id>` CLI flag
2. `project.yaml::domain_pack` field
3. `project.yaml::service_line` (looked up against the pack synonym table)
4. `SOURCE_NOTES.md` content scoring (each pack's `service_lines` keywords)
5. Filename keyword matching across artifacts
6. Per-pack `artifact_role_patterns` content scoring on a sample of the artifacts
7. `default_pack` (catch-all)

`auto_route_pack()` in `pack_router.py` implements this. The chosen
pack and the routing source land in
`compile_result.manifest.parser_routing` for traceability.

## Adding a new pack

1. **Copy a similar pack** as a starting point. `wireless_pack.yaml` is
   small and well-organized; `default_pack.yaml` is the kitchen-sink
   reference.

2. **Pick a `pack_id`** — must match `<pack_id>_pack.yaml` filename
   convention (or just `<pack_id>.yaml` for legacy compatibility).
   Keep it short and lowercase.

3. **Fill in `device_aliases`** — the highest-leverage section. Each
   `canonical: [surface, surface, ...]` mapping becomes a
   `device:<canonical>` entity key whenever any surface form word-boundary
   matches. Avoid bare common-noun surfaces (`light`, `panel`) — they
   produce noise. The longer / more distinctive the surface form, the
   safer.

4. **Add `exclusion_patterns` + `constraint_patterns`** — these are
   substring matches against the normalized atom text, used by both the
   classifier (atom_type promotion) and `graph_builder` (`excludes` /
   `requires` edge eligibility for `customer_instruction` atoms).

5. **Add `service_lines`** — the keywords `auto_route_pack` matches
   against `project.yaml::service_line` and SOURCE_NOTES.md content.
   List every synonym you can think of (the wireless pack has `wireless`,
   `wifi`, `wlan`, `802.11`, …).

6. **Validate it loads** — `python -c "from app.domain import load_domain_pack; load_domain_pack('my_pack')"`. Then run
   `parser-os compile real_data_cases/STRESS_X --domain-pack my_pack`
   on a representative case and inspect the resulting `entity_keys` and
   `atom_type` distribution.

7. **Certify it** — run `python scripts/certify_domain_pack.py
   --pack my_pack --cases real_data_cases/STRESS_*` to score how well
   it covers a sample of cases. The certification prints a per-case
   coverage table.

## Cross-pack vendor catalog

Some vendors (Cisco, Bosch, ThyssenKrupp, T2 Systems, Crestron, …) appear
across multiple verticals. Rather than duplicating them in every pack,
they live in one place: `app/core/entity_extraction.py::_CROSS_PACK_VENDORS`.
Adding a vendor there makes it visible to *every* pack's compile.

## Project config (`project.yaml`)

```yaml
# parser-os project configuration — all keys optional

domain_pack: security_camera_pack    # pin (overrides auto-routing)
service_line: video_surveillance     # alternative to domain_pack

context_notes: |
  Free-text project context. Shown in the review-folder header.

customer: virginia_tech              # mirrored into manifest
project_name: VT Video Surveillance Addendum 2

parserignore_extra:                  # extra globs on top of built-in skips
  - "Attachment D - Sample Agreement.*"
  - "*.draft.pdf"
```

`load_project_config()` in `project_config.py` reads it. Unknown keys are
silently dropped so future schema additions don't break old configs.
