# Measurement Report: OPTBOT + APS Fiber

All measurements below were run locally with deterministic settings:

```bash
SOWSMITH_MULTI_ENTITY_DISABLE=1 \
SOWSMITH_SITE_LLM_DISABLE=1 \
SOWSMITH_VISION_DISABLE=1 \
SOWSMITH_TYPED_CLASSIFIER_DISABLE=1 \
python -m app.cli compile <deal> --out <json> --skip-orbitbrief --no-cache
```

## OPTBOT

| Metric | v52 baseline | v53.12 current-best | v54 patched deterministic |
|---|---:|---:|---:|
| Total atoms | 567 | 448 | 486 |
| `physical_site` | 0 | 3 | 5 |
| `risk` | 62 | 31 | 10 |
| `acceptance_criterion` | 65 | 36 | 5 |
| `requirement` | 8 | 56 | 6 |
| `milestone_phase` | 8 | 9 | 6 |
| `bom_line` | 10 | 10 | 10 |
| `site_allocation` | 24 | 24 | 30 |
| `stakeholder` | 5 | 5 | 10 |
| generic `entity` atoms | 15 | 4 | 0 |
| atoms with `value.entity_type == "site"` | 0 | 0 | 0 |

Patched physical-site IDs are exactly:

```text
ATL-HQ-01
ATL-WEST-02
ATL-AIR-03
ATL-047-04
ATL-CP-05
```

The OPTBOT compile completed with `errors=0` without `--allow-errors`.

## APS Fiber

The narrative brief says Attachment B has 132 sites. The bundled PDF text currently contains contiguous `site_no` values 1 through 159. The patched parser extracts all 159 authoritative rows rather than truncating to the stale narrative count.

| Metric | v54 patched deterministic |
|---|---:|
| Total atoms | 534 |
| `physical_site` | 159 |
| `stakeholder` | 8 |
| generic `entity` atoms | 0 |
| missing required site fields (`site_no`, `administrative_site_name`, `street`, `city`, `zip`, `lat_long`) | 0 |
| address-as-id | 0 |
| PO-box-as-site-id | 0 |
| customer-name-as-site-id | 0 |

The APS compile completed with `errors=0` without `--allow-errors`.

## Not completed in this patch

- Learned confidence calibrator and ECE `< 0.05`; no labeled gold file was provided in the bundle.
- Full structured-output LLM extractor replacement for all categories.
- Full root-cause rewrite of all 20 failure modes. This patch focuses on the failures blocking pack accuracy: roster detection, physical-site contamination, post-replay receipt validity, deterministic stakeholder recall, and roster-scale performance.
