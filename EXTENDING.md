# Extending Parser-OS

## Adding a new atom type

1. Add the enum value to `app/core/schemas.py::AtomType`.
2. Define the strict value shape expected for that atom type. Prefer explicit fields over catch-all `entity` values.
3. Add semantic-key handling in `app/core/semantic_dedup.py` so duplicate facts collapse by stable business identity rather than raw text.
4. Add confidence/replay expectations if the atom is emitted after source replay. Post-replay atoms must still carry an `EvidenceReceipt` with `replay_status="unsupported"` and a reason explaining why replay could not run.
5. Add pack-level regression tests using real artifact fixtures, not synthetic one-line strings only.

## Adding a table schema

Known schemas live in `app/core/table_schema_registry.py`. Add a schema when the table family is common and stable enough to maintain deterministically. Do not add a customer-specific header list; use header roles and semantic fields.

For unknown table families, the intended next step is schema induction:

```text
headers + sample rows + source context
  -> structured-output LLM: table_family, field_roles, row_schema
  -> validate against AtomType value schema
  -> cache induced schema by normalized header signature
  -> emit typed atoms with receipts
```

## Adding a domain pack

A domain pack should supply vocabulary and aliases, not override evidence contracts. The extraction pipeline should still produce the same atom types for IT refreshes, fiber WANs, retail rollouts, medical campuses, military bases, AV installations, and datacenter moves.

Avoid:

- customer-specific reject sets,
- long keyword waterfalls,
- regexes enumerating business values,
- schemas that only work for one file name.

Prefer:

- typed schemas,
- embedding retrieval for candidate selection,
- structured-output extraction,
- pack-level gold tests.
