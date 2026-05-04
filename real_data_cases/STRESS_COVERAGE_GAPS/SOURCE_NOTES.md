# Service lines where public pre-SOW data is thin
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
  synthetic asset-list XLSX (OEM/serial/condition columns).
