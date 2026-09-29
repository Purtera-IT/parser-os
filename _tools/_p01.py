# -*- coding: utf-8 -*-
"""Pass 01 -- NMC-hs-email-115241880664: the supplier quote the deal is costed from."""
from _mk import build

DOC = "010237-hs-email-115241880664.eml"
W = lambda t: {"hint": "own_words", "kind": "text", "text": t}
D = {"hint": "doc_type", "kind": "text",
     "text": "NMC budgetary quote to PurTera, sent by Suresh Nalla (snalla@nmcms.com)"}
WHO = {"hint": "who_said_it", "kind": "text",
       "text": "Suresh Nalla, NMC -- our subcontractor, not the customer"}

SPEC = {
 0: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Thank you for the opportunity.")], "entity_keys": [],
     "note": "Pleasantry opening a supplier quote. No price, no scope, no commitment -- the quote "
             "begins on the next line. Rejected rather than dropped so the admission head gets a "
             "negative example from a document whose every other line is load-bearing."},

 1: {"label_type": "deal_metadata", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("Please find below the NMC budgetary quote"), D, WHO],
     "entity_keys": ["party:nmc", "party:purtera", "artifact:nmc_quote", "platform:workato"],
     "reads_set": {"opens_block": {"value": True, "source": "human"}},
     "note": "The line that makes every number below it a COST and not a price. NMC is quoting US; "
             "PurTera then resells to SHI. Without this attribution the $16,000.00 per month reads "
             "as what the deal earns, and the 21% margin becomes invisible -- the same BOM-sized "
             "question the doctrine raises about who Provided by us means, one level up. It also "
             "opens the block: both options and all six lines beneath them are governed by it."},

 2: {"label_type": "service_line", "weight_tier": "load_bearing", "about": "deal",
     "wants": "decide-internally", "supplier": "partner",
     "hint_refs": [W("Managed Services"), D],
     "entity_keys": ["party:nmc", "service:managed_services", "option:1"],
     "reads_set": {"opens_block": {"value": True, "source": "human"}},
     "note": "Option 1 of two, and the one the Deal Kit actually bought: its 16,000 a month became "
             "Unit Cost Rate on the Gantt sheet. Which option we take is a decision only we can "
             "make and it changes the cost basis by tens of thousands, so decide-internally rather "
             "than nothing. Load-bearing because the two options carry different coverage windows "
             "and different billing shapes, and this deal quoted one while contracting the other."},

 3: {"label_type": "service_line", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("Fixed Cost: $16,000.00 per month"), D, WHO],
     "entity_keys": ["party:nmc", "price:16000", "service:managed_services", "option:1",
                     "commercial:unit_cost"],
     "note": "The single most consequential number in the pre-quote record, and the only typed "
             "input behind the whole price. The Deal Kit Gantt sheet carries O2=16000 hard-typed "
             "and K2 = O2/0.79, so the 121,518.99 the Work Order commits to is this figure divided "
             "by a 21% margin target and multiplied by six. Every other number on the revenue line "
             "is derived from it. Typed service_line rather than commercial_total because it is a "
             "unit price with a unit (per month), not an aggregate over component rows -- there are "
             "no component rows."},

 4: {"label_type": "constraint", "weight_tier": "load_bearing", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "partner",
     "hint_refs": [W("40 hours per week"), W("9:00 AM"), D],
     "entity_keys": ["party:nmc", "quantity:40", "service:managed_services", "option:1",
                     "coverage:weekday_business_hours"],
     "note": "What the 16,000 actually buys, and it does not match what we promised. This is Option "
             "1 coverage and it starts at 9:00; the customer asked for 8am-5pm and the SLA we "
             "attached commits to 8:00 AM to 5:00 PM. We bought nine-to-five and sold eight-to-five "
             "-- an uncovered hour every weekday for six months. Typed constraint, not "
             "site_access_window: the registry defines that type as when a SITE is accessible, and "
             "this engagement is 100% remote, so a site-access head would be trained on a deal that "
             "has no site."},

 5: {"label_type": "rate_card", "weight_tier": "load_bearing", "about": "deal",
     "wants": "decide-internally", "supplier": "partner",
     "hint_refs": [W("Weekend Support: $125.00/hour, based on actual consumption"), D],
     "entity_keys": ["party:nmc", "price:125", "service:weekend_support", "option:1",
                     "commercial:variable_cost"],
     "note": "The uncosted exposure, stated plainly by the supplier and carried into no financial "
             "line anywhere. The Deal Kit shows Materials, Lift/Rental and Miscellaneous all at "
             "zero, so there is no allowance for a single weekend hour. It matters because the "
             "customer RFI answer asked for six days a week: if the sixth day is consumed as "
             "weekend, roughly 8 hours x 26 weeks at 125 is about 26,000 of unbudgeted cost against "
             "a total deal margin of 25,323.99. decide-internally because nobody has to ask the "
             "customer anything to notice it -- we have to price it or cap it."},

 6: {"label_type": "service_line", "weight_tier": "ordinary", "about": "deal",
     "wants": "decide-internally", "supplier": "partner",
     "hint_refs": [W("Staff Augmentation"), D],
     "entity_keys": ["party:nmc", "service:staff_augmentation", "option:2"],
     "reads_set": {"opens_block": {"value": True, "source": "human"}},
     "note": "The option we did NOT buy on cost, and the one the signed Work Order describes on "
             "scope: the SOW executive summary sells a staff augmentation for Middleware AMS at "
             "1,008 estimated hours, which is this option shape, while the Deal Kit costs Option 1 "
             "fixed monthly. Ordinary rather than load-bearing because no number of ours derives "
             "from it -- its weight is that it exists and got mixed with Option 1."},

 7: {"label_type": "rate_card", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("Hourly Rate: $130.00/hour"), D],
     "entity_keys": ["party:nmc", "price:130", "service:staff_augmentation", "option:2"],
     "note": "Option 2 price, kept because it is the arithmetic showing the two options were never "
             "interchangeable: the SOW 1,008 estimated hours at this rate is 131,040, more than the "
             "121,518.99 the Work Order caps us at. Quoting Option 2 hours while buying Option 1 "
             "fixed monthly is the only reason that is not a loss."},

 8: {"label_type": "constraint", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("8:00 AM"), D],
     "entity_keys": ["party:nmc", "service:staff_augmentation", "option:2",
                     "coverage:weekday_business_hours"],
     "note": "Option 2 covers 8:00, Option 1 covers 9:00. The hour we promised the customer exists "
             "in the option we did not buy, which is the cleanest statement of the mix-up on this "
             "deal. Kept as its own atom rather than folded into Option 1 coverage because the "
             "difference between the two lines IS the finding."},

 9: {"label_type": "contract_term", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("Project Duration: 6 months"), D],
     "entity_keys": ["quantity:6", "party:nmc", "term:6_months"],
     "note": "The six that multiplies everything. Deal Kit Unit Sell Quantity and Unit Cost Quantity "
             "are both 6, the Work Order runs 14 Sep 2026 to 14 Mar 2027, and the SLA states the "
             "same term independently -- three sources agreeing, which is why duration is the one "
             "commercial fact on this deal nobody has to chase."},

10: {"label_type": "contract_term", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("Service Type: 100% Remote"), D],
     "entity_keys": ["party:nmc", "delivery:remote", "service:managed_services"],
     "note": "Stated by the supplier, repeated by the SLA (Support is provided remotely) and by the "
             "Work Order location table (Remote / Remote), and contradicted by the rate code the "
             "deal was priced under, PS-L3-ENG-LABOR-ONSITE. Nothing in the entire pre-quote record "
             "contains the word onsite. Load-bearing because remote delivery is also why the Work "
             "Order can say no travel expenses are expected, which is what keeps Misc at zero."},

11: {"label_type": "payment_term", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Invoicing will be done on a monthly basis"), W("actual weekend consumption"), D],
     "entity_keys": ["party:nmc", "service:weekend_support", "commercial:monthly_billing"],
     "note": "The billing shape, and the second place the weekend leak is stated. Based on the "
             "applicable service option AND actual weekend consumption says the monthly invoice is "
             "a fixed part plus a variable tail, so our cost is not 16,000 a month -- it is 16,000 "
             "a month plus whatever weekend hours happen. The Deal Kit models only the fixed half."},

12: {"label_type": "source_caveat", "weight_tier": "load_bearing", "about": "partner",
     "wants": "confirm-with-customer", "supplier": "partner",
     "hint_refs": [W("for the Staff Augmentation option"), W("the SLA"), D],
     "entity_keys": ["party:nmc", "artifact:sla", "artifact:engineer_profile",
                     "service:staff_augmentation", "option:2"],
     "reads_set": {"points_at_artifact": {"value": "sla+cv", "source": "human"}},
     "note": "A fact about two source documents that changes how both must be read, which is what "
             "source_caveat is for. NMC says the attached SLA and engineer profile are for the "
             "Staff Augmentation option -- Option 2 -- yet the SLA Service Package says Engagement "
             "Model: Recurring monthly managed service, which is Option 1. The English is genuinely "
             "ambiguous about whether for the Staff Augmentation option governs both attachments or "
             "only the profile, and that ambiguity decides which option the 8x5 coverage and the "
             "125 weekend rate actually bind. about:partner because it is a fact about how our "
             "supplier packages its paperwork, and it must be asked rather than guessed."},

13: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Please let us know if you have any questions")], "entity_keys": [],
     "note": "Standard sign-off. No claim and no consequence -- and unlike the opening pleasantry it "
             "is not even a courtesy specific to this quote."},
}

build(DOC, SPEC, "pass_01.json")
