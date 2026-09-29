# -*- coding: utf-8 -*-
"""Pass 03 -- the main thread (inbound from nmcms.com), PurTera <-> NMC, Jul 21 to Aug 20."""
from _mk import build

DOC = "010237-hs-email-115199549032.eml"
W = lambda t: {"hint": "own_words", "kind": "text", "text": t}
D = {"hint": "doc_type", "kind": "text",
     "text": "Flattened email thread, inbound from nmcms.com -- an NMC message carrying a quoted "
             "chain of Trent Torrence (PurTera) and Suresh Nalla (NMC) messages"}
TRENT = {"hint": "who_said_it", "kind": "text",
         "text": "Trent Torrence, PurTera EVP of Sales -- quoted inside an NMC-sent message"}
SURESH = {"hint": "who_said_it", "kind": "text", "text": "Suresh Nalla, NMC"}

SPEC = {}

# --- the eight message boundaries the substance gate removed -------------------
GATED = {44: "August 13, 2026", 45: "August 4, 2026", 46: "August 6, 2026",
         47: "July 21, 2026", 48: "July 24, 2026", 49: "August 7, 2026",
         50: "August 12, 2026", 51: "August 19, 2026"}
for idx, when in GATED.items():
    SPEC[idx] = {
        "label_type": "deal_metadata", "weight_tier": "load_bearing", "about": "deal",
        "wants": "nothing", "hint_refs": [W(when), D],
        "entity_keys": ["artifact:email_thread", "party:nmc", "party:purtera"],
        "note": "The substance gate dropped this, and it is wrong to drop it. A From/Sent header is "
                "not prose, it is the boundary that tells a reader which of three parties said the "
                "sentences underneath -- the doctrine rule that the speaker comes from the original "
                "and that the person who filed a message is not the person who spoke. The gate kept "
                "seven of these headers and removed eight, which is the worst of the three "
                "available answers: a thread with all its boundaries can be attributed, a thread "
                "with none is obviously unattributable, and a thread with half of them looks "
                "attributable and is not. Overturning the drop rather than confirming it, because "
                "these eight carry the only evidence that this conversation started on 21 July -- a "
                "month before the quote and before any document the deal holds.",
    }

# --- the three message boundaries the gate kept, beside the eight it removed ---
for idx, when, who in ((15, "06 August 2026", "party:purtera"),
                       (18, "07 August 2026", "party:purtera"),
                       (20, "August 10, 2026", "party:nmc")):
    SPEC[idx] = {
        "label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal",
        "wants": "nothing", "hint_refs": [W(when), D],
        "entity_keys": [who, "artifact:email_thread"],
        "note": "A message boundary the substance gate kept. It is labelled identically to the eight "
                "it dropped, on purpose: the eleven headers in this thread are the same kind of line "
                "doing the same job, and the only difference between the kept and the dropped is "
                "which side of a threshold their wording fell on. Labelling them alike is what lets "
                "the admission head learn that the class is uniform rather than inheriting the "
                "pattern that split it.",
    }

SPEC.update({
 0: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Good afternoon.")], "entity_keys": [],
     "note": "Greeting. Rejected -- but note the gate kept this and removed eight message headers "
             "from the same document, which is the inversion worth having in the training set."},
 1: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("I hope")], "entity_keys": [],
     "note": "Pleasantry. No claim about the work."},

 2: {"label_type": "open_question", "weight_tier": "ordinary", "about": "partner",
     "wants": "nothing", "hint_refs": [W("a good fit for your team"), D, TRENT],
     "entity_keys": ["party:nmc", "party:purtera", "service:ams"],
     "note": "PurTera sounding NMC out before committing -- the first move of the deal, a month "
             "before any pricing. about:partner because it is a fact about how we work with NMC "
             "rather than about what gets built. It is answered by the quote that eventually "
             "arrives, so it is not left open."},

 3: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "partner",
     "wants": "nothing", "hint_refs": [W("please don"), W("spend time searching for a resource"), D, TRENT],
     "entity_keys": ["party:nmc", "party:purtera", "relationship:low_pressure_ask"],
     "note": "Tells the supplier not to go hunting, which sets the whole engagement expectation: we "
             "wanted a bench resource, not a recruitment exercise. It matters commercially because "
             "the resource NMC eventually proposed is a 17-year integration architect -- if that "
             "person was sourced rather than benched, the six-month availability behind the fixed "
             "16,000 a month is a thinner promise than it looks."},

 4: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "partner",
     "wants": "nothing", "hint_refs": [W("readily available"), D, TRENT],
     "entity_keys": ["party:nmc", "party:purtera", "relationship:low_pressure_ask"],
     "note": "The same expectation stated as a condition on us rather than on them. Kept as its own "
             "atom because together these two lines are the only account of why this deal went to "
             "NMC at all, and the brief has nothing else to say about partner selection."},

 5: {"label_type": "task", "weight_tier": "load_bearing", "about": "deal",
     "wants": "nothing", "hint_refs": [W("Need a AMS(Application management services) resource"),
                                       W("next 6 months"), D],
     "entity_keys": ["platform:workato", "service:ams", "quantity:6", "term:6_months",
                     "quantity:1"],
     "note": "The original ask, and the deal in one sentence: one resource, Workato middleware, AMS, "
             "six months. Every later document is an elaboration of it, and the six months here is "
             "the earliest statement of the number the Deal Kit multiplies by. It also says a "
             "RESOURCE -- singular, a person -- which is the staff-augmentation reading, not the "
             "managed-service reading, and the deal never resolves which of the two it is."},

 6: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "internal",
     "wants": "nothing", "hint_refs": [W("drafted a few RFI questions"), D, SURESH],
     "entity_keys": ["party:nmc", "artifact:rfi", "process:discovery"],
     "note": "NMC generating the questionnaire, which is why the RFI in this deal is NMC-authored "
             "rather than ours. That provenance matters when reading the answers: the questions were "
             "written by the party who would price the work, so the support-hours question offering "
             "24/7, 16/5, 8/6 is NMC asking in its own shorthand -- and that shorthand is what the "
             "customer answered, and what Excel then destroyed."},

 7: {"label_type": "commitment", "weight_tier": "ordinary", "about": "deal",
     "wants": "nothing", "hint_refs": [W("Statement of Work (SOW) and a budgetary quote"), D, SURESH],
     "entity_keys": ["party:nmc", "artifact:sow", "artifact:nmc_quote"],
     "note": "NMC promising the two documents the deal is eventually priced from, conditional on the "
             "customer responses arriving. It is a commitment rather than metadata because it puts "
             "the next move on them and makes the RFI answers the gating artefact -- which they were: "
             "the quote arrived the day after the RFI came back."},

 8: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal",
     "wants": "nothing", "hint_refs": [W("attached RFI questionnaire"), D, SURESH],
     "entity_keys": ["artifact:rfi", "party:nmc", "party:customer"],
     "reads_set": {"points_at_artifact": {"value": "rfi", "source": "human"}},
     "note": "Points at the RFI and, in to share with the customer, records that it travels through "
             "us to the end client -- three hops, which is how a shorthand answer like 8/6 ends up "
             "being read by a party who never saw the question asked."},

 9: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal",
     "wants": "nothing", "hint_refs": [W("attached responses to the questions"), D, TRENT],
     "entity_keys": ["artifact:rfi", "party:purtera", "party:customer"],
     "reads_set": {"points_at_artifact": {"value": "rfi_responses", "source": "human"}},
     "note": "The answers coming back up the chain. This is the moment the corrupted support-hours "
             "cell entered the deal: the spreadsheet the customer filled in is the one the walk "
             "holds, and its hours answer is already the date 6-Aug by the time anyone at NMC opens "
             "it."},

10: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("let me know what questions you have")], "entity_keys": [],
     "note": "Routine handoff courtesy. The real questions arrive later as their own atoms."},

11: {"label_type": "open_question", "weight_tier": "slight", "about": "internal",
     "wants": "nothing", "hint_refs": [W("Following up to see if you have any questions"), D],
     "entity_keys": ["party:nmc", "party:purtera"],
     "note": "A chase with no subject -- it asks whether there are questions, not a question. Typed "
             "open_question because that is literally its shape, but slight: nothing in the deal "
             "turns on it, and treating every follow-up as an open item is how a chase list becomes "
             "noise."},

12: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Thank you for sharing the responses.")], "entity_keys": [],
     "note": "Acknowledgement of receipt. The substantive reply is the next line."},

13: {"label_type": "deal_state", "weight_tier": "ordinary", "about": "partner",
     "wants": "nothing", "hint_refs": [W("engineering team is currently reviewing"), D, SURESH],
     "entity_keys": ["party:nmc", "artifact:rfi", "process:discovery"],
     "note": "Where the deal stood on that day: with NMC engineering, pre-quote. Typed deal_state "
             "because it reports the position of the work rather than a fact about the deal itself, "
             "and a brief that leads with what we are waiting on needs exactly these."},

14: {"label_type": "deal_state", "weight_tier": "slight", "about": "partner", "wants": "nothing",
     "hint_refs": [W("if any additional information or clarification is required"), D, SURESH],
     "entity_keys": ["party:nmc"],
     "note": "A conditional promise to come back, which they did -- the 10 August message asking "
             "three specific questions is this line being kept. Slight on its own; its value is that "
             "it makes the later questions expected rather than a surprise."},

16: {"label_type": "deal_state", "weight_tier": "slight", "about": "partner", "wants": "nothing",
     "hint_refs": [W("a quick update"), D, SURESH],
     "entity_keys": ["party:nmc"], "note": "Announces an update; the update is the next line."},

17: {"label_type": "commitment", "weight_tier": "ordinary", "about": "partner",
     "wants": "nothing", "hint_refs": [W("drafting the proposal for the Workato resource"),
                                       W("by Monday"), D, SURESH],
     "entity_keys": ["party:nmc", "artifact:nmc_quote", "date:by_monday"],
     "reads_set": {"blocked_on": {"value": "partner", "source": "human"}},
     "note": "A dated promise from the supplier, and it slipped: this was sent 6 August saying "
             "Monday, and the quote did not arrive until 19 August after two chases. That slip is "
             "the only performance signal the deal carries about NMC, and it belongs in the brief "
             "why-you-should-care rather than the SOW -- which is why about:partner."},

19: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Have a great weekend.")], "entity_keys": [],
     "note": "Sign-off."},

21: {"label_type": "open_question", "weight_tier": "load_bearing", "about": "deal",
     "wants": "confirm-with-customer", "hint_refs": [W("so we can finalize the SLA"), D, SURESH],
     "entity_keys": ["party:nmc", "artifact:sla", "process:discovery"],
     "reads_set": {"opens_block": {"value": True, "source": "human"},
                   "blocked_on": {"value": "customer", "source": "human"}},
     "note": "Opens the block of three questions that produced SLA v1.1, and states what they gate. "
             "Load-bearing because the answers given underneath are what the SLA was written from -- "
             "so every gap between a question and its answer here becomes a gap in the contract."},

22: {"label_type": "open_question", "weight_tier": "load_bearing", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "partner",
     "hint_refs": [W("8 hours a day, 6 days a week"), W("does not specify the actual"), D, SURESH],
     "entity_keys": ["quantity:8", "quantity:6", "coverage:six_day_week", "party:nmc",
                     "artifact:rfi", "risk:uncosted_weekend"],
     "note": "The most valuable sentence in the pre-quote record. The customer RFI answer to the "
             "support-hours question is stored in the spreadsheet as the date 6-Aug, because they "
             "typed 8/6 -- eight hours a day, six days a week -- into a cell Excel had formatted as "
             "d-mmm. The fact was destroyed at the moment it was entered, upstream of the parser. "
             "This line is the only place in the entire deal where it survives, recovered by a human "
             "reading the questionnaire in context. It matters because six days is the uncosted "
             "weekend: NMC prices 40 hours fixed and bills weekends at 125 an hour on consumption, "
             "the SLA lands at 8x5, and the Deal Kit carries no weekend allowance at all against a "
             "25,323.99 margin. Everything downstream answers the hours and silently drops the days."},

23: {"label_type": "open_question", "weight_tier": "ordinary", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "partner",
     "hint_refs": [W("access to the non-production environments"), D, SURESH],
     "entity_keys": ["party:nmc", "environment:dev_test", "access:admin", "platform:workato"],
     "note": "Asked because the RFI answer all 3 said which environments exist, not which ones we "
             "get into. The distinction is real: the SLA scope covers development, test and "
             "production, so an answer of production-only would have put the SLA out of step with "
             "the access on day one."},

24: {"label_type": "open_question", "weight_tier": "ordinary", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "partner",
     "hint_refs": [W("client email accounts"), W("partner company email accounts"), D, SURESH],
     "entity_keys": ["party:nmc", "party:customer", "scope:tooling"],
     "note": "A question about identity rather than tooling: whose email the NMC engineer uses "
             "decides whether the customer experiences this as their own staff or as a "
             "subcontractor. On a deal sold through SHI as a channel, with NMC behind us, that is "
             "three layers between the engineer and the end client."},

25: {"label_type": "deal_metadata", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "hint_refs": [W("11 August 2026"), D],
     "entity_keys": ["party:purtera", "artifact:email_thread", "date:2026_08_11"],
     "note": "The boundary that attributes the three one-line answers below it to Trent rather than "
             "to NMC. Without it the answers read as the questioner answering himself. Kept live by "
             "the gate where eight of its siblings were not, which is the inconsistency the drops "
             "above record."},

26: {"label_type": "customer_instruction", "weight_tier": "load_bearing", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "customer",
     "hint_refs": [W("Worktime - 8am- 5pm"), D, TRENT],
     "entity_keys": ["coverage:weekday_business_hours", "party:customer", "party:purtera",
                     "risk:uncosted_weekend"],
     "note": "The answer that lost the sixth day. NMC asked two things -- what hours, given the RFI "
             "says eight a day over six days -- and this answers only the hours. Nobody ever "
             "confirms or denies the six days, and the SLA comes back 8x5. This single line is where "
             "a 48-hour week became a 40-hour week with no one deciding it, and it is still "
             "unresolved: the customer wrote six days down and has never been told they are getting "
             "five. Typed customer_instruction because Trent is relaying the client requirement, not "
             "setting it."},

27: {"label_type": "answered_question", "weight_tier": "ordinary", "about": "deal",
     "wants": "nothing", "supplier": "customer", "hint_refs": [W("Prod and Test"), D, TRENT],
     "entity_keys": ["environment:dev_test", "environment:prod", "access:admin",
                     "platform:workato"],
     "note": "Answers the non-production access question -- and answers it narrower than the RFI. "
             "The customer had said all 3 environments exist and the SLA scope covers development, "
             "test and production; this grants Prod and Test. Development is missing from the "
             "answer and present in the contract, which is a small, precise mismatch of exactly the "
             "kind that surfaces on day one of onboarding."},

28: {"label_type": "answered_question", "weight_tier": "load_bearing", "about": "deal",
     "wants": "nothing", "supplier": "customer",
     "hint_refs": [W("SKLSI email and communication channel."), D, TRENT],
     "entity_keys": ["party:sklsi", "party:customer", "scope:tooling"],
     "note": "The only place the end client is named anywhere in the pre-quote record: SKLSI, which "
             "is SK Life Science, the SK Life of the deal title. The Deal Kit End User field says "
             "TBD. So the parse holds a fact the priced document left blank, and it arrives "
             "sideways -- as the answer to a question about email accounts, not as a customer "
             "declaration. It also answers the identity question: the engineer works under the "
             "client domain, so the end client sees SKLSI staff, not a subcontractor."},

29: {"label_type": "open_question", "weight_tier": "ordinary", "about": "deal",
     "wants": "confirm-with-customer", "supplier": "partner",
     "hint_refs": [W("customer is on EST or any other timezone"), D, SURESH],
     "entity_keys": ["party:nmc", "party:customer", "coverage:timezone"],
     "note": "Asked because the coverage window is meaningless without a zone -- 8am-5pm is a "
             "different service in Seoul than in New Jersey, and SK Life is a Korean parent. A "
             "reasonable question that the RFI, which offered 24/7, 16/5, 8/6 and asked for the "
             "applicable time zone in the same breath, should have settled and did not."},

30: {"label_type": "answered_question", "weight_tier": "load_bearing", "about": "deal",
     "wants": "nothing", "supplier": "customer", "hint_refs": [W("They are EST"), D, TRENT],
     "entity_keys": ["coverage:timezone", "coverage:est", "party:customer"],
     "note": "Settles the zone, and it is load-bearing because every hours statement in the deal "
             "inherits it: NMC 9:00 to 5:00 EST, the SLA 8:00 AM to 5:00 PM EST, the customer "
             "8am-5pm. Without this line those three are not even comparable, and the one-hour gap "
             "between what we bought and what we promised could not be stated as a gap."},

31: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "hint_refs": [W("19 August 2026"), D],
     "entity_keys": ["party:purtera", "artifact:email_thread", "date:2026_08_19"],
     "note": "Message boundary. Attributes the chase below it, eight days after the Monday that was "
             "promised on 6 August."},

32: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Good morning.")], "entity_keys": [], "note": "Greeting."},

33: {"label_type": "deal_state", "weight_tier": "ordinary", "about": "partner", "wants": "nothing",
     "hint_refs": [W("Checking in on this opportunity."), D, TRENT],
     "entity_keys": ["party:purtera", "party:nmc"],
     "reads_set": {"blocked_on": {"value": "partner", "source": "human"}},
     "note": "The second chase. Two chases across thirteen days against a promise of Monday is the "
             "deal only supplier-performance record, and blocked_on says whose inbox it sat in -- "
             "theirs, not ours. That is the distinction a brief should lead with."},

34: {"label_type": "open_question", "weight_tier": "ordinary", "about": "partner",
     "wants": "nothing", "hint_refs": [W("when you"), W("have it over to us"), D, TRENT],
     "entity_keys": ["party:nmc", "artifact:nmc_quote"],
     "reads_set": {"blocked_on": {"value": "partner", "source": "human"}},
     "note": "Asks for a date and gets one the same hour. Kept because the pairing with the reply "
             "below is a clean answers edge across a message boundary, which is the relation the "
             "edge head has no gold for."},

35: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "hint_refs": [W("19 August 2026 17:52"), D],
     "entity_keys": ["party:nmc", "artifact:email_thread", "date:2026_08_19"],
     "note": "Message boundary, and the one that shows the reply came 28 minutes after the chase. "
             "Note it lacks an email address where its siblings carry one, so the attribution here "
             "rests on the display name alone."},

36: {"label_type": "commitment", "weight_tier": "load_bearing", "about": "partner",
     "wants": "nothing", "supplier": "partner",
     "hint_refs": [W("Will have the SLA and pricing back to you today."), D, SURESH],
     "entity_keys": ["party:nmc", "artifact:sla", "artifact:nmc_quote", "date:2026_08_19"],
     "reads_set": {"commitment": {"value": True, "source": "human"},
                   "blocked_on": {"value": "partner", "source": "human"}},
     "note": "The promise that was kept: the SLA is dated 19 August and the quote followed. It is "
             "load-bearing because it dates the arrival of both priced documents to the day before "
             "the manifest as-of cut of 20 August 15:09:40 -- which is precisely why this deal can "
             "see its own cost basis at all while the Deal Kit and Work Order that came hours later "
             "are held out."},

37: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "hint_refs": [W("20 August 2026 01:04"), D],
     "entity_keys": ["party:purtera", "artifact:email_thread", "date:2026_08_20"],
     "note": "The last boundary in the thread, hours before the quote cut. Everything after this "
             "point in the deal history is outside what this compile is allowed to know."},

38: {"label_type": "stakeholder", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("Suresh Nalla"), W("snalla@nmcms.com"), D],
     "entity_keys": ["stakeholder:suresh_nalla", "email:snalla_nmcms_com", "party:nmc",
                     "role:supplier_contact"],
     "note": "The counterparty who quoted the deal, authored the SLA questions and sent both priced "
             "documents. He is the only person on either side whose messages set a price, which "
             "makes him the single name a PM needs. Load-bearing for that reason, not because a "
             "signature block is inherently important."},

39: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("further assistance in closing this opportunity")], "entity_keys": [],
     "note": "Closing courtesy from a signature block."},

40: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Have a good day.")], "entity_keys": [], "note": "Sign-off."},

41: {"label_type": "stakeholder", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("Fred Syed"), W("fsyed@nmcms.com"), D],
     "entity_keys": ["stakeholder:fred_syed", "email:fsyed_nmcms_com", "party:nmc"],
     "note": "On copy throughout and never speaks. Kept because a silent recipient is still a party "
             "to the agreement and would otherwise vanish, but slight: nothing in the deal turns on "
             "him and inventing a role for him from a Cc line would be exactly the resolver mistake "
             "the doctrine warns about."},

42: {"label_type": "stakeholder", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("Syed Shakil"), W("sshakil@nmcms.com"), D],
     "entity_keys": ["stakeholder:syed_shakil", "email:sshakil_nmcms_com", "party:nmc"],
     "note": "The second silent Cc. Same reasoning as Fred Syed: recorded, unresolved, not promoted "
             "to a role the documents never give him."},

43: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "hint_refs": [W("NMC Budgetary Quote"), W("snalla@nmcms.com"), D],
     "entity_keys": ["party:nmc", "party:purtera", "artifact:nmc_quote", "artifact:email_thread"],
     "note": "The outermost envelope of the thread, naming sender, both recipients and the subject. "
             "It is the header that makes this whole document inbound from NMC, which is what "
             "settles that the quoted content below is a chain and not one voice -- and therefore "
             "why the eight dropped boundaries mattered."},
})

build(DOC, SPEC, "pass_03.json")
