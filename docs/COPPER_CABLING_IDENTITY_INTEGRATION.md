
# Copper Cabling Identity Integration Plan

## Target architecture

Move all item matching out of parser-specific code and into:

```python
from app.core.item_identity import canonical_item_identity, canonical_material_key, enrich_value_with_identity
```

## Wire points

### 1. Quote parser
After creating each vendor_line_item and quantity value:

```python
value = enrich_value_with_identity(value, raw_text=raw_text)
```

Use the enriched fields:

```text
normalized_item
item_kind
material_family
comparison_group
identity_confidence
identity_matched_by
is_scope_pollution_candidate
inclusion_status
```

### 2. XLSX parser
When emitting wide quantity atoms, keep existing explicit normalized_item. Then optionally enrich if missing:

```python
value = enrich_value_with_identity(value, raw_text=raw_text)
```

Do not override explicit Cat6 UTP/STP/RJ45 values unless identity confidence is higher and compatible.

### 3. Graph builder
Replace local `_canonical_material_key` with:

```python
from app.core.item_identity import canonical_material_key, is_primary_vendor_quantity
```

Use `is_primary_vendor_quantity(atom.value)` before summing vendor lines.

### 4. Packetizer
Packet anchors should use canonical identity:

```text
rj45 -> cabling:rj45 or connector:rj45
cat6_utp -> material:cat6_utp
cat6_stp -> material:cat6_stp
power -> scope:power
raceway -> pathway:raceway
certification_testing -> requirement:certification
```

### 5. Gold comparison
Use `canonical_material_key` so gold checks tolerate name drift.

## Universal rules

- Cat6A must not match Cat6.
- UTP and STP must not merge.
- Power must not match PoE.
- Data drop can infer RJ45 only with lower confidence.
- Vendor quote can conflict with scope but cannot govern scope.
- Optional/excluded/alternate vendor lines should not be counted in primary vendor total.

## Next rerun after install

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_item_identity.py tests/test_graph_builder.py tests/test_authority.py tests/test_packetizer.py tests/test_copper_low_voltage_adversarial.py -q
python scripts/compile_real_data_case.py --case-id COPPER_001_SPRING_LAKE_AUDITORIUM --root "c:\Users\lilli\Downloads\purtera_copper_low_voltage_public_validation_packs\purtera_copper_low_voltage_validation_packs\real_data_cases"
```

## Gold target after integration

- rj45 72 vs 68 -> quantity_conflict
- cat6_utp 66 vs 60 -> vendor_mismatch
- cat6_stp 6 vs 8 -> vendor_mismatch
- power included in quote but excluded from scope -> scope_exclusion
- raceway/conduit excluded/unknown -> missing_info
- certification excluded/unknown -> missing_info
- lift/catwalk/after-hours -> site_access
