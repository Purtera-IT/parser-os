# Universal keep-vs-typed rubric (all managed services)

Purpose: make the single highest-error decision — *is this atom a real deal item or
boilerplate?* — **reproducible**. Two strong independent LLMs agree only ~59% on this
without a rule; that caps model accuracy at ~59% no matter the GPU. This rubric is the
fix. It is **role-based, never domain-based** — it never names a trade (TV/AP/fiber/
camera), so it applies identically to AV, cabling, electrical, fire-alarm, and managed-IT.

## The one principle
Every managed-services deal commits a provider to do specific things, for specific
parties, at specific places/times/prices/conditions. So one question decides it:

> **Does this atom state a discrete piece of WHAT we are committing to for THIS deal —
> work, a party, a place, a price, a time, or a condition?**
> - YES -> **TYPED**
> - NO  -> **KEEP** (scaffolding: document structure, schema/field definitions,
>   reference legends, form labels, or identity boilerplate carrying no commitment)

## Decision test (apply in order)
1. **Is it document navigation?** (index, table of contents, "Section 3.2", sheet/page
   refs) -> **KEEP**. It points to content; it is not content.
2. **Is it a legend / notation key / definition table?** -> **KEEP**.
   - Exception: if it states a **quantity** of equipment/material ("48-PORT PATCH PANEL x 12")
     -> that quantity is a **BOM line -> TYPED**.
3. **Is it schema / field-list metadata?** ("ServiceNow Fields:", "Available Fields:",
   "Index: N | <field name>") -> **KEEP**. It describes the data model/integration, not the deal.
4. **Is it a form label/scaffold?** ("Project Name:", "Provider Name:", bare "Key |")
   -> the **label is KEEP**; type the **value** if it is a real fact. Type the fact, never the empty field.
5. **Is it a contact / signature block?** ("Name | Title | Email")
   -> **TYPED (stakeholder) ONLY if the person acts in THIS deal's execution** (PM, site
   contact, approver). A sales rep / quote signatory -> **KEEP / metadata**.
6. **Does it name an action/service the provider will perform, or a requirement,
   site fact, price, milestone, or condition?** -> **TYPED** — even if terse
   ("OS patching support", "test the HA pair"). Brevity is not boilerplate.
7. Still genuinely 50/50 under all of the above -> **DROP** (guess-free; never invent a label).

## Anchored exemplars (from real disagreements)
KEEP:
- `DRAWING INDEX: T1.11 | col_1: LEVEL 24 FLOOR PLAN`            (navigation)
- `STRUCTURED CABLING SYMBOL LEGEND: D | 1 PORT GUESTROOM OUTLET` (legend)
- `Index: 83 | ServiceNow Fields: Telco Address`                 (schema metadata)
- `Provider Name: | PurTera`                                     (form scaffold)
- `Trent Torrence | VP, Sales | <EMAIL>`                          (sales signatory, not an actor)

TYPED:
- `Operating System Patching and Upgrade Support`               (service/work)
- `Test and confirm status of the HA pair during the onsite`    (task)
- `48 PORT CATEGORY 6 PATCH PANEL x 12`                          (BOM quantity)
- `Provide access to all 23 dwellings and installation locations` (site)
- `Shane Hoff | CEO | <EMAIL>` IF named as deal approver         (stakeholder-as-actor)

## Why universal (the test)
If a rule can't be written without naming a specific service, it isn't universal.
Every rule above is a predicate over **role** (commitment / scaffolding / schema /
actor), so it transfers to any trade unchanged.

---

# Fine-type facet rubric (the 7 facets = the PM dashboard sections)

Once an atom is TYPED, route it to ONE facet. These 7 facets ARE the dashboard
sections (sites / financials / scope / compliance / people / timeline / meta), so
collapsing the 41 micro-types to facets loses no product value and is ~85%
reproducible (measured: 0.846 two-model agreement) vs the un-learnable 41-way.

## The 7 facets
- **SITE** — a place, site access, or physical-site attribute.
- **COMMERCIAL** — a price, rate, quantity/BOM line, payment term, or total.
- **WORK** — an action/service/task/requirement/deliverable/milestone/acceptance.
- **COMPLIANCE** — a rule, certification, approval, insurance, or regulatory obligation.
- **PARTY** — a person/org that ACTS in the deal (PM, site contact, approver).
- **TIMING** — a deadline, blackout window, or lead-time/sequencing constraint.
- **META** — deal-level metadata (project name, provider, document fields).

## Collapse map (clean — these agreed 3–4/4, route directly)
SITE ← physical_site, site_attribute, site_access_restriction, site_room_mix, site_infrastructure
COMMERCIAL ← service_line, bom_line, payment_term, commercial_total, pricing_assumption, site_budget
WORK ← requirement, task, deliverable, acceptance_criterion, milestone_phase, cutover_step,
       electrical_acceptance_test, site_implementation_note, site_access_window
COMPLIANCE ← compliance_rule, compliance_classification, approval_authority, submission_req
TIMING ← blackout_date_range, lead_time_constraint
META ← deal_metadata, eval_criterion, approval_decision

## Rulings for the 7 FUZZY types (split 2/4 — decide by PRIMARY function)
- **stakeholder → PARTY** if the person acts in execution (PM/contact/approver); **META** if a
  bare name/contact on a header; never split a person across facets.
- **deadline → TIMING** always (the facet is about *when*, even if tied to a milestone).
- **dependency → TIMING** (it's a sequencing/precondition that gates *when* work happens).
- **change_order_rule → COMPLIANCE** (it governs the contract process, not the work itself).
- **exclusion → WORK** (a negative scope statement still defines scope).
- **bonding_insurance → COMPLIANCE** (a legal/regulatory obligation, not a price).
- **integration_checkpoint → WORK** (a technical validation step the provider performs).

Tie-break principle: route to the facet whose **dashboard section the PM needs it in**.
Still genuinely multi-facet after this → DROP (guess-free), or surface for PM ruling.
