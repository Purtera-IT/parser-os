# Label task (one deal)

Goal: write training labels for the C3 base heads on lines of ONE deal, so a refit fixes the head errors the SOW graders proved. Read-only except your one output file.

Inputs:
- Rubric (the only rules you may apply): /Users/purtera/c3-local/RUBRIC_UNIVERSAL_SOW_HEADS.md
- Grade reports for this deal (rounds 7-11): /private/tmp/claude-501/-Users-purtera/a02898c4-8e61-4640-a057-e8aa8aa45e75/scratchpad/sowq/{DEAL}/grade*.md
- The deal's envelope (atoms with structured.c3_label = what the head said, c3_brief sections): /private/tmp/claude-501/-Users-purtera/a02898c4-8e61-4640-a057-e8aa8aa45e75/scratchpad/final_{DEAL}_env.json
- The live base request (the exact lines the base reads; atom key = "key"): /Users/purtera/c3-serve/dump_{UUID}.json
- Sources if you need to check meaning: /private/tmp/claude-501/-Users-purtera/a02898c4-8e61-4640-a057-e8aa8aa45e75/scratchpad/sowq/{DEAL}/sources/

Do this:
1. Collect every line the grade reports name as a head/label/wiring problem (atom ids atm_...), plus lines of the same kind in the dump that the rubric clearly covers (e.g. EVERY row of a location table, EVERY sibling in a ship-out list, EVERY meeting Start:/End: header). Only lines whose key exists in the dump file.
2. For each, decide the label strictly by the rubric. Only the fields the rubric speaks to: label_type, supplier, co_action (keep/reject), sow_section. Leave a field out when the rubric does not settle it.
3. Never label from wording alone when the source says otherwise; check the source line's context when the grade report is unsure.
4. Never copy credentials into the file.

Output: write JSON to /Users/purtera/c3-local/sowfix/labels_{DEAL}.json as
{"deal": "{DEAL}", "uuid": "{UUID}", "labels": [{"atom_id": "atm_...", "label_type": "...", "supplier": "...", "co_action": "...", "sow_section": "...", "rule": "label_type 1", "why": "one short sentence"}]}
Valid label_type values are the existing atom types (physical_site, scope_item, task, bom_line, deal_metadata, stakeholder, small_talk, dependency, exclusion, deliverable, assumption, open_question, answered_question, milestone_phase, constraint, ...). supplier in {us, partner, customer, third_party}. co_action in {keep, reject}. sow_section in {overview, scope_of_work, customer_responsibilities, assumptions, out_of_scope, deliverables, fees, contacts, other}.
Reply with one line: the count of labels by field.
