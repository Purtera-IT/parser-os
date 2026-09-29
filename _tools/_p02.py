# -*- coding: utf-8 -*-
"""Pass 02 -- NMC SLA v1.1: the document that defines what we committed to."""
from _mk import build

DOC = "NMC-SLA-Purtera IT-Workato_Middleware-v.1"
W = lambda t: {"hint": "own_words", "kind": "text", "text": t}
S = lambda t: {"hint": "section", "kind": "text", "text": t}
D = {"hint": "doc_type", "kind": "text",
     "text": "NMC Service Level Agreement v1.1, 19 Aug 2026 -- the supplier SLA attached to the quote"}

# The four severity rows, and the lossy second read of each that the table also produced.
SEV = {
 13: ("P1", "1 hour", "4 hours", "Production integration down"),
 20: ("P2", "4 hours", "8 hours", "Major function impaired"),
 41: ("P3", "8 hours", "2 business days", "Partial or intermittent issue"),
 24: ("P4", "24 hours", "5 business days", "Minor issue"),
}
SPEC = {}

for idx, (p, resp, res, desc) in SEV.items():
    SPEC[idx] = {
        "label_type": "acceptance_criterion", "weight_tier": "load_bearing", "about": "deal",
        "wants": "nothing", "supplier": "partner",
        "hint_refs": [W(p), W(resp), S("Service Levels (SLA)"), D],
        "entity_keys": [f"severity:{p.lower()}", "service:incident_support", "party:nmc",
                        "system:servicenow"],
        "note": f"The {p} row of the service-level table, complete: severity, its definition, the "
                f"response target ({resp}) and the resolution target ({res}). Load-bearing because "
                "these four rows are the only measurable commitment in the engagement -- a managed "
                "service with no deliverable is judged entirely on whether it hit these. They also "
                "do not match what the customer asked for in the RFI, which answered P1-1, P2-2-4, "
                "P3-4-8, P4-8-16 hours; the SLA moved P3 and P4 onto business days, which is a "
                "materially weaker commitment nobody appears to have flagged back.",
    }

# The same table read a second time, lossily: description plus one number, no severity code.
for idx, full, num in ((7, "P2", "4 hours"), (11, "P4", "24 hours"),
                       (36, "P1", "1 hour"), (43, "P3", "8 hours")):
    SPEC[idx] = {
        "label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
        "hint_refs": [W(num), D], "entity_keys": [],
        "note": f"A second, lossier read of the {full} row that the complete row already carries. "
                "It keeps the severity description and one number but drops the severity code and "
                "the second number, so the surviving figure cannot be told from a response target "
                "or a resolution target -- the one thing the row exists to distinguish. Rejected as "
                "table scaffolding rather than kept as a fact, and typed _keep rather than left as "
                "the parser bom_line: a bill-of-materials type on an SLA response target would "
                "teach a BOM head that service levels are hardware lines.",
    }

SPEC.update({
 0: {"label_type": "dependency", "weight_tier": "load_bearing", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "customer",
     "hint_refs": [W("The Customer provides timely administrative access to the Workato workspace"),
                   S("Assumptions"), D],
     "entity_keys": ["party:customer", "platform:workato", "access:admin", "risk:onboarding_delay"],
     "note": "The precondition the whole engagement sits on: without admin access to the workspace "
             "and the integrated systems there is no service to deliver, only a resource we are "
             "paying 16,000 a month for. It is a dependency rather than an assumption because it "
             "names who must act, and the RFI answer on access was only Admin- tech support, which "
             "does not say the access is arranged. On a fixed monthly cost an access delay is pure "
             "margin loss -- we pay NMC whether or not the customer has let them in."},

 1: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Document Version"), W("v1.1"), D],
     "entity_keys": ["artifact:sla", "version:v1_1", "party:nmc"],
     "note": "Version 1.1, not 1.0 -- there was an earlier draft of this SLA that the deal does not "
             "hold. Worth keeping because the thread shows the SLA was still being finalised on 10 "
             "August (Could you please help us with the below points so we can finalize the SLA) "
             "and the answers given then are what changed between the versions."},

 2: {"label_type": "constraint", "weight_tier": "load_bearing", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "partner",
     "hint_refs": [W("8 hours per day, 5 days per week (8x5)"), W("8:00 AM to 5:00 PM EST"),
                   S("Service Package"), D],
     "entity_keys": ["quantity:40", "quantity:8", "quantity:5", "coverage:weekday_business_hours",
                     "party:nmc", "service:managed_services"],
     "note": "The coverage we actually committed to, and it is not the coverage we bought. This "
             "says 8:00 AM to 5:00 PM; NMC Option 1, the option the Deal Kit costs at 16,000 a "
             "month, covers 9:00 AM to 5:00 PM. It is also five days where the customer RFI answer "
             "asked for six. Both gaps point the same way -- we promised more hours than we "
             "purchased -- and the with additional weekend coverage tail is what makes the shortfall "
             "billable at 125 an hour rather than merely uncovered. Typed constraint, not "
             "site_access_window: there is no site on a 100% remote engagement."},

 3: {"label_type": "dependency", "weight_tier": "ordinary", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "customer",
     "hint_refs": [W("Provisioning of administrative access to the Workato workspace"),
                   S("Transition and Onboarding"), D],
     "entity_keys": ["party:customer", "platform:workato", "access:admin", "phase:onboarding"],
     "note": "The onboarding half of the access dependency: the same requirement stated as a task "
             "with a deadline rather than as a standing assumption. Kept separately because this one "
             "sits inside the onboarding checklist, and the SLA says the service-level clock only "
             "starts once onboarding completes -- so this line, not the assumption, is what gates "
             "the commencement date the six-month term is measured from."},

 4: {"label_type": "assumption", "weight_tier": "load_bearing", "about": "deal",
     "wants": "decide-internally", "supplier": "partner",
     "hint_refs": [W("Resolution targets assume the necessary monitoring and alerting is in place"),
                   S("Assumptions"), D],
     "entity_keys": ["party:nmc", "scope:monitoring", "phase:onboarding", "risk:scope_creep"],
     "note": "A conditional that quietly adds unpriced work. The customer answered the monitoring "
             "RFI question with ?? -- literally two question marks, the only non-answer in the "
             "questionnaire -- so the where it is not branch is the likely one, and this line then "
             "commits us to standing up monitoring and alerting during onboarding at no stated cost. "
             "decide-internally because we can size it before commencement; nobody needs to ask the "
             "customer to know the RFI never answered it."},

 5: {"label_type": "change_order_rule", "weight_tier": "load_bearing", "about": "deal",
     "wants": "nothing", "supplier": "partner",
     "hint_refs": [W("Any change to the scope, coverage, service levels"),
                   S("Project Change Procedure"), D],
     "entity_keys": ["party:nmc", "process:change_request", "scope:governance"],
     "note": "The clause that makes every coverage mismatch on this deal recoverable instead of "
             "absorbed. It names coverage explicitly, so the 9:00-versus-8:00 gap and the fifth "
             "versus sixth day are change-request territory rather than free. It is the single most "
             "useful sentence in the SLA for protecting the 21% margin, and it is worth nothing "
             "unless somebody notices the gaps before they are consumed."},

 6: {"label_type": "contract_term", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("Support is provided remotely."), S("Assumptions"), D],
     "entity_keys": ["delivery:remote", "party:nmc", "service:managed_services"],
     "note": "The second independent statement that this engagement is remote, after the quote Service "
             "Type: 100% Remote and before the Work Order Remote / Remote location table. Three "
             "sources, no dissent -- and the deal was still priced on PS-L3-ENG-LABOR-ONSITE. Kept "
             "as contract_term rather than the parser scope_item because being remote is a term of "
             "how the service is delivered, not an item of work performed."},

 8: {"label_type": "service_line", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("ongoing Level 2 (L2) application management and support services"),
                   S("Overview"), D],
     "entity_keys": ["party:nmc", "support_level:l2", "platform:workato", "service:ams"],
     "note": "The scope sentence, and the strongest of the four places the record says L2. The deal "
             "was coded PS-L3-ENG-LABOR-ONSITE in the Deal Kit. The honest counter-reading is that "
             "the code may describe the RESOURCE grade rather than the service tier -- the engineer "
             "NMC proposed has 17 years and an architect title, which is plausibly L3 -- but the "
             "service being sold is L2 here, in the customer own RFI answer, and in the Service "
             "Package, and no reading of any document supports ONSITE."},

 9: {"label_type": "dependency", "weight_tier": "load_bearing", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "customer",
     "hint_refs": [W("Knowledge transfer from the Customer or the outgoing team"),
                   S("Transition and Onboarding"), D],
     "entity_keys": ["party:customer", "phase:onboarding", "scope:knowledge_transfer",
                     "risk:incumbent_handover"],
     "note": "Names an outgoing team, which is the only trace in the whole deal that this is a "
             "takeover from an incumbent rather than a greenfield engagement. That matters for "
             "pricing: a handover that does not happen lands as discovery effort on a fixed monthly "
             "cost. The customer answered yes to knowledge transfer in the RFI without saying from "
             "whom or for how long, so the duration remains unknown."},

10: {"label_type": "dependency", "weight_tier": "ordinary", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "customer",
     "hint_refs": [W("Review of existing documentation"), S("Transition and Onboarding"), D],
     "entity_keys": ["party:customer", "phase:onboarding", "artifact:runbook",
                     "artifact:recipe_inventory"],
     "note": "The documentation the onboarding depends on. The customer answered the corresponding "
             "RFI question with a bare yes, which asserts the documents exist but says nothing about "
             "whether they are current -- and this line requires a review plus confirmation of "
             "monitoring, so a yes that turns out to be stale converts directly into onboarding days "
             "we did not price."},

12: {"label_type": "dependency", "weight_tier": "ordinary", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "customer",
     "hint_refs": [W("Establishment of the ServiceNow queues"), S("Transition and Onboarding"), D],
     "entity_keys": ["party:customer", "system:servicenow", "phase:onboarding",
                     "scope:escalation_path"],
     "note": "ServiceNow is the customer tool, so the queues, severity definitions and escalation "
             "contacts are all theirs to create. Until they exist the SLA cannot be measured at all "
             "-- every target in the table is defined as measured on incidents logged through "
             "ServiceNow -- which makes this the one onboarding item that gates the service levels "
             "rather than merely the service."},

14: {"label_type": "payment_term", "weight_tier": "load_bearing", "about": "deal",
     "wants": "decide-internally", "supplier": "partner",
     "hint_refs": [W("The weekly 40 hours will be fixed/static"),
                   W("variable based on actual consumption"), S("Service Package"), D],
     "entity_keys": ["party:nmc", "quantity:40", "price:125", "service:weekend_support",
                     "commercial:variable_cost"],
     "note": "The clearest statement of the deal financial shape and the reason its margin is not "
             "what the Deal Kit says. Fixed plus variable: 40 hours static, weekends at 125 an hour "
             "on consumption. The Deal Kit models only the static half -- Materials, Lift and "
             "Miscellaneous are all zero -- so every weekend hour is margin erosion against a total "
             "margin of 25,323.99. Typed payment_term rather than the parser change_order_rule "
             "because it describes how the invoice is computed, not when a change request is needed."},

15: {"label_type": "exclusion", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("outside the Workato platform and the named integration"), S("Out of Scope"), D],
     "entity_keys": ["scope:out_of_scope", "platform:workato", "system:sap"],
     "note": "The boundary that makes the SAP integration the named integration and everything "
             "around it somebody else problem. Load-bearing because the RFI established the "
             "integration touches ERP and procurement across three environments, so a failure will "
             "routinely originate outside Workato -- and this line plus the best-effort assumption "
             "is what stops those becoming our SLA breaches."},

16: {"label_type": "assumption", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("best-effort in nature for issues whose root cause lies outside"),
                   S("Assumptions"), D],
     "entity_keys": ["party:nmc", "scope:best_effort", "platform:workato", "risk:sla_breach"],
     "note": "The qualifier that converts a firm-looking resolution table into a conditional one. "
             "Read with the P1 row, the four-hour resolution target is only firm when the fault is "
             "inside Workato -- and middleware faults usually are not. This is the sentence that "
             "would decide an SLA credit dispute, and it is the reason the Response Target is "
             "described separately as a firm commitment while resolution is not."},

17: {"label_type": "service_line", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Incident management and break-fix support"), S("Scope of Services"), D],
     "entity_keys": ["service:break_fix", "service:incident_support", "platform:workato",
                     "system:sap", "party:nmc"],
     "note": "The core service line -- what the 16,000 a month is for. Names the SAP integration "
             "specifically, which ties the money to the scope boundary two lines down and matches "
             "the customer RFI answer Workato- SAP integration. Everything else in Scope of Services "
             "is an activity around this one."},

18: {"label_type": "contract_term", "weight_tier": "load_bearing", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "partner",
     "hint_refs": [W("Recurring monthly managed service"), S("Service Package"), D],
     "entity_keys": ["party:nmc", "service:managed_services", "option:1",
                     "commercial:monthly_billing"],
     "note": "The SLA declares itself Option 1 -- a recurring monthly managed service -- which "
             "directly contradicts the covering email describing this same attachment as being for "
             "the Staff Augmentation option, and contradicts the signed Work Order, which sells a "
             "staff augmentation priced on 1,008 estimated hours. Three documents, two engagement "
             "models, and the difference decides whether we owe a service level or a person. It has "
             "to be asked rather than inferred, which is why confirm-with-customer."},

19: {"label_type": "acceptance_criterion", "weight_tier": "ordinary", "about": "deal",
     "wants": "nothing", "supplier": "partner",
     "hint_refs": [W("Resolution Target is the time to restore service"), S("Service Levels (SLA)"), D],
     "entity_keys": ["service:incident_support", "party:nmc", "scope:best_effort"],
     "note": "Defines the second column of the severity table and immediately weakens it: restore "
             "service, best-effort where the root cause is elsewhere. Kept as its own atom because "
             "the four severity rows are unreadable without it -- a bare 4 hours means nothing until "
             "you know it measures restoration and not a fix, and is a target and not a guarantee."},

21: {"label_type": "change_order_rule", "weight_tier": "load_bearing", "about": "deal",
     "wants": "nothing", "supplier": "partner",
     "hint_refs": [W("Development of new integrations, recipes, or connectors"), S("Out of Scope"), D],
     "entity_keys": ["scope:out_of_scope", "scope:development", "process:change_request",
                     "platform:workato"],
     "note": "Excludes new development and routes it to the change procedure, which is the correct "
             "commercial answer to the customer RFI reply Break fix initially, but new -- an answer "
             "that says development is coming without saying when or how much. This line is what "
             "makes that future work billable rather than expected, so it protects the margin on a "
             "fixed monthly deal whose scope the customer has already signalled will grow."},

22: {"label_type": "exclusion", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Physical activities such as racking, stacking, and cabling."),
                   S("Out of Scope"), D],
     "entity_keys": ["scope:out_of_scope", "scope:physical_work"],
     "note": "Boilerplate for a 100% remote software engagement -- nobody was going to rack anything "
             "for a Workato support contract. Kept rather than rejected because it is a real "
             "exclusion in a real Out of Scope list, but slight: it removes no work anyone expected, "
             "and on this deal it is the one exclusion that decides nothing."},

23: {"label_type": "service_line", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Monitoring and triage of recipe failures"), S("Scope of Services"), D],
     "entity_keys": ["service:monitoring", "service:triage", "platform:workato", "party:nmc"],
     "note": "Monitoring as a service activity, which sits awkwardly beside the assumption that "
             "monitoring and alerting is already in place and will be set up during onboarding if "
             "not. This line says we watch; that one says the watching infrastructure may not exist. "
             "The customer answered the monitoring RFI question with ??, so the gap is real."},

25: {"label_type": "exclusion", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("The service scope is break-fix and incident support"),
                   S("Enhancement and Development Work"), D],
     "entity_keys": ["scope:break_fix", "scope:out_of_scope", "scope:development"],
     "note": "The scope stated as a negative, and the cleanest sentence to put in front of a customer "
             "who answered Break fix initially, but new. It is load-bearing because on a fixed "
             "monthly managed service the only way to lose money on scope is to absorb enhancement "
             "work as if it were support, and this is the line that says we do not."},

26: {"label_type": "dependency", "weight_tier": "ordinary", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "customer",
     "hint_refs": [W("single point of contact"), S("Assumptions"), D],
     "entity_keys": ["party:customer", "role:spoc", "risk:resolution_delay"],
     "note": "A named single point of contact, which the deal does not have: the RFI never asked for "
             "one and no customer-side individual appears anywhere in the pre-quote record. Every "
             "person named on this deal works for NMC or PurTera. On an engagement whose resolution "
             "targets depend on customer input, an unnamed SPOC is a live gap rather than a "
             "formality."},

27: {"label_type": "contract_term", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("does not compete with, the incident support commitment"),
                   S("Enhancement and Development Work"), D],
     "entity_keys": ["scope:development", "service:incident_support", "party:nmc"],
     "note": "Ring-fences the support commitment from enhancement work, so a busy development month "
             "cannot be offered as an excuse for a missed severity target. It is the protective twin "
             "of the exclusion above -- that one keeps enhancement out of the price, this one keeps "
             "it out of the capacity."},

28: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "hint_refs": [W("August 19, 2026"), D],
     "entity_keys": ["date:2026_08_19", "artifact:sla", "party:nmc"],
     "note": "Dates the SLA to 19 August, the day before the quote. That ordering is what makes this "
             "deal an as-of compile that can see the SLA at all: the manifest cut is 20 August "
             "15:09:40, and every document authored after it -- the Deal Kit, the Work Order -- is "
             "held out. This line is how a reader can tell the SLA is inside the cut."},

29: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Workato Middleware Application Management Services (AMS)"), D],
     "entity_keys": ["platform:workato", "service:ams", "artifact:sla"],
     "note": "The project name, and the only place the acronym AMS is expanded in a document rather "
             "than an email. Typed deal_metadata rather than the parser task: a title names the "
             "engagement, it is not labour anyone performs, and typing it task would put a document "
             "header into the work a Deal Kit prices."},

30: {"label_type": "service_line", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("two applications: ERP and procurement"),
                   W("development, test, and production"), S("Scope of Services"), D],
     "entity_keys": ["quantity:2", "system:erp", "system:procurement", "platform:workato",
                     "environment:dev_test_prod"],
     "note": "The only sizing statement in the engagement: two applications, three environments. It "
             "matches the customer RFI answers exactly -- 2 integrated applications, ERP and "
             "procuremnt, all 3 environments -- which makes it the one place the SLA demonstrably "
             "consumed the questionnaire rather than restating a template. Load-bearing because two "
             "applications is the closest thing this deal has to a quantity, and the 15 incidents a "
             "week the customer estimated has to be serviced across all three environments."},

31: {"label_type": "contract_term", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Support Level: Level 2 (L2)"), S("Service Package"), D],
     "entity_keys": ["support_level:l2", "party:nmc", "service:managed_services"],
     "note": "L2 stated as a package term rather than prose, which makes it the most citable of the "
             "four L2 statements and the one that most directly contradicts the PS-L3 rate code the "
             "Deal Kit priced under. Typed contract_term rather than the parser deal_metadata: the "
             "support tier is what we are contractually obliged to staff, not a header fact about "
             "the deal."},

32: {"label_type": "dependency", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "customer",
     "hint_refs": [W("third-party vendors where a fix depends on a system outside"),
                   S("Scope of Services"), D],
     "entity_keys": ["party:customer", "party:third_party", "scope:coordination",
                     "risk:resolution_delay"],
     "note": "Commits us to coordinate with parties we have no contract with and cannot compel. Read "
             "beside the best-effort assumption this is the operational half of the same idea: the "
             "assumption protects the target, this line describes the work of chasing someone else "
             "vendor while the clock runs. Unbudgeted on a fixed monthly service."},

33: {"label_type": "change_order_rule", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("handled as a separate request outside the service levels"),
                   S("Enhancement and Development Work"), D],
     "entity_keys": ["scope:development", "process:change_request", "party:nmc"],
     "note": "The procedural mechanism behind the enhancement exclusion: separate request, outside "
             "the service levels. Kept distinct from the exclusion itself because an exclusion says "
             "what is not included and this says what happens instead -- and on a deal where the "
             "customer has already said new work is coming, the what-happens-instead is the part "
             "that has to be quoted."},

34: {"label_type": "milestone_phase", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("a short onboarding period will be completed"),
                   S("Transition and Onboarding"), D],
     "entity_keys": ["phase:onboarding", "party:nmc", "risk:commencement_slip"],
     "note": "Establishes that the engagement has two phases and that the service levels do not "
             "apply in the first one. Short is not a duration, and the Work Order six-month term "
             "runs from a commencement date at the end of it -- so an onboarding that drifts either "
             "delays revenue or eats into the six months we sold. The deal contains no estimate of "
             "how long it takes and nobody asked."},

35: {"label_type": "dependency", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "customer",
     "hint_refs": [W("Incident Logging: ServiceNow (Customer-provided)"), S("Service Package"), D],
     "entity_keys": ["system:servicenow", "party:customer", "scope:tooling"],
     "note": "The customer supplies the ticketing system, which the RFI confirms independently. That "
             "is convenient commercially -- no tooling cost falls on us -- and it is also a "
             "dependency, because every SLA target is measured on tickets in their system, so we "
             "cannot evidence our own performance without access they control. Typed dependency "
             "rather than the parser site_implementation_note: there is no site."},

37: {"label_type": "exclusion", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Project management or major upgrade"), S("Out of Scope"), D],
     "entity_keys": ["scope:out_of_scope", "scope:project_management"],
     "note": "Excludes project management, which is quietly consistent with the Deal Kit: its PMO "
             "line carries 0 revenue against 195 of cost, so PM on this engagement is unpriced on "
             "our side and out of scope on the supplier side. Whoever ends up running it is doing "
             "it for nothing, and this is the line that says NMC will not."},

38: {"label_type": "contract_term", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Service Term: 6 months from the service commencement date"),
                   S("Service Package"), D],
     "entity_keys": ["quantity:6", "term:6_months", "party:nmc", "phase:onboarding"],
     "note": "Six months, but measured from the commencement date rather than signature -- which the "
             "quote Project Duration: 6 months does not say and the Work Order fixed dates of 14 Sep "
             "to 14 Mar do not allow for. If onboarding runs long the supplier six months and the "
             "customer six months stop being the same six months, and we are the ones holding both "
             "ends. That gap between a relative term and two absolute dates is the finding."},

39: {"label_type": "acceptance_criterion", "weight_tier": "load_bearing", "about": "deal",
     "wants": "nothing", "supplier": "partner",
     "hint_refs": [W("apply from the agreed service commencement date"),
                   S("Transition and Onboarding"), D],
     "entity_keys": ["phase:onboarding", "service:incident_support", "risk:commencement_slip"],
     "note": "Says explicitly that the severity table is dormant until onboarding ends, which is what "
             "makes the undefined length of onboarding a commercial fact rather than a scheduling "
             "detail. It is also the line that defines agreed service commencement date -- a date "
             "nobody has agreed anywhere in this deal."},

40: {"label_type": "acceptance_criterion", "weight_tier": "ordinary", "about": "deal",
     "wants": "nothing", "supplier": "partner",
     "hint_refs": [W("incidents logged through ServiceNow"), S("Service Levels (SLA)"), D],
     "entity_keys": ["system:servicenow", "service:incident_support", "party:customer"],
     "note": "Scopes the SLA to tickets in the customer system: an incident raised by phone, email or "
             "a hallway conversation is outside the measured set. Combined with the customer owning "
             "ServiceNow, this means neither party can measure the service without the other -- and "
             "it is the reason establishing the queues is an onboarding gate rather than a "
             "convenience."},

42: {"label_type": "deliverable", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Regular service review"), S("Scope of Services"), D],
     "entity_keys": ["service:service_review", "party:nmc", "scope:governance"],
     "note": "The only recurring artefact the engagement produces, and the only place service-level "
             "attainment gets reported. Typed deliverable rather than service_line because it names "
             "a thing handed over on a cadence rather than an activity performed continuously. "
             "Regular is undefined -- monthly and quarterly are both regular, and they are different "
             "amounts of work on a six-month deal."},

44: {"label_type": "service_line", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Root-cause analysis for recurring issues"), S("Scope of Services"), D],
     "entity_keys": ["service:rca", "platform:workato", "party:nmc"],
     "note": "RCA with recommendations, which is the boundary between support and consulting and is "
             "included here at no separate charge. Worth keeping because recommendations to improve "
             "integration stability is exactly the work that turns into the new development the "
             "exclusions push out to a change request -- this service line is how that pipeline gets "
             "generated."},

45: {"label_type": "exclusion", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "customer",
     "hint_refs": [W("Workato platform licensing"), S("Out of Scope"), D],
     "entity_keys": ["scope:out_of_scope", "platform:workato", "scope:licensing",
                     "party:customer"],
     "note": "Licensing stays with the customer, which is why the Deal Kit Materials line is legitimately "
             "zero on a platform engagement -- there is no software to resell. Load-bearing because "
             "on a middleware deal the licence is usually the largest number, and its absence from "
             "our quote is correct rather than an omission."},

46: {"label_type": "acceptance_criterion", "weight_tier": "load_bearing", "about": "deal",
     "wants": "nothing", "supplier": "partner",
     "hint_refs": [W("Where automated failure alerting is not yet in place"),
                   S("Service Levels (SLA)"), D],
     "entity_keys": ["scope:monitoring", "service:incident_support", "party:nmc"],
     "note": "Moves the start of the clock from when the failure happened to when somebody noticed, "
             "wherever alerting is missing -- and the customer answered the alerting question with "
             "??. So on the most likely reading of this deal, the resolution targets are measured "
             "from human detection for at least the first part of the term. That is a large, quiet "
             "softening of the only commitment the engagement makes."},

47: {"label_type": "acceptance_criterion", "weight_tier": "load_bearing", "about": "deal",
     "wants": "nothing", "supplier": "partner",
     "hint_refs": [W("This is a firm commitment."), S("Service Levels (SLA)"), D],
     "entity_keys": ["service:incident_support", "party:nmc", "severity:all"],
     "note": "The one unqualified promise in the document. Response is firm; resolution is "
             "best-effort and, where alerting is missing, measured from detection. So the whole "
             "enforceable surface of this managed service is the response column -- 1, 4, 8 and 24 "
             "hours. Anyone reading the severity table as four firm targets is reading it wrong, and "
             "this sentence is why."},

48: {"label_type": "service_line", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Logging, tracking, and updating of all incidents"), S("Scope of Services"), D],
     "entity_keys": ["system:servicenow", "service:incident_support", "party:nmc"],
     "note": "The administrative half of incident management, in the customer tool. Kept separate "
             "from break-fix because it is the activity that produces the evidence the SLA is "
             "measured on -- if this is done badly the service can be delivered well and still fail "
             "its targets on the record."},

49: {"label_type": "dependency", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "customer",
     "hint_refs": [W("maintains valid Workato licensing and vendor support"), S("Assumptions"), D],
     "entity_keys": ["party:customer", "platform:workato", "scope:licensing", "risk:platform_lapse"],
     "note": "The operational twin of the licensing exclusion: that one says we do not sell it, this "
             "says the service fails without it. On a six-month term a licence or vendor-support "
             "lapse would stop us delivering while we still owe NMC 16,000 a month, which is the "
             "asymmetry worth carrying into the brief."},
})

build(DOC, SPEC, "pass_02.json")
