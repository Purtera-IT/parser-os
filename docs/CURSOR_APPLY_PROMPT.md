
# Cursor Apply Prompt

Copy the files from this ZIP into the repo, preserving paths.

Then implement this safely:

1. Run `tests/test_item_identity.py`.
2. Wire `app.core.item_identity.canonical_material_key` into `app/core/graph_builder.py` in place of local material-key matching.
3. Wire `is_primary_vendor_quantity` so excluded/optional/alternate/allowance vendor lines do not count in primary vendor totals.
4. Wire `enrich_value_with_identity` into quote parser and xlsx parser only when fields are missing or compatible.
5. Do not break existing COPPER_001 behavior: RJ45 72 vs 68, Cat6 UTP 66 vs 60, Cat6 STP 6 vs 8 must still create contradiction edges.
6. If `app/domain/copper_cabling.yaml` does not load because schema is strict, leave it as reference data and do not block the run. Add a TODO adapter instead.

Run:

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_item_identity.py tests/test_quote_parser.py tests/test_xlsx_parser.py tests/test_graph_builder.py tests/test_authority.py tests/test_packetizer.py tests/test_copper_low_voltage_adversarial.py -q

Then rerun COPPER_001 and report packet gold results.
