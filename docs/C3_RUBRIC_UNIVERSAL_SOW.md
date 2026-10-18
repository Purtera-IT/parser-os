# Universal rubric: base-head labels the SOW depends on (2026-10-18)

One principle: **label what a line does in the job, judged by who acts and what is decided, never by the trade, the customer, or the wording.** Every rule below applies to every managed-service job (cabling, AV, POS, phones, time clocks, moves, smart hands, UPS) and every reseller (CDW, SHI, direct).

Each rule cites one example the graders proved wrong on a real deal (deal, line, what the head said → the label).

## col:label_type (what the line is)

1. **A row of a location table is `physical_site`.** A row that names one place (a store/site number or name together with a street, city, state or ZIP) is a site, whatever other columns it carries (region, terminal counts, SCO).
   - 010279 "Store: 3 | Region: Southern | City: Charlotte | State: NC | Terminals: 9": physical_site 0.42 / stakeholder / deal_metadata → **physical_site**.
   - 000132 "Tupelo, MS" in a Locations list: deal_metadata 0.30 → **physical_site**.
2. **A request for added work is `scope_item`, even phrased as a question or a wish.** "I would also like to add X — is that something you can help with?" asks for X.
   - 010198 "add a Ubiquiti router with cellular backup": answered_question 0.53 → **scope_item**.
3. **An item in a list the customer ships or supplies is `bom_line`.** Every sibling of a "we'd ship out / we will provide" list gets the same type.
   - 000123 "Printer" (_keep), "Credit card reader (pin pad)" (task 0.78) beside "Samsung Tablet" (bom_line) → **bom_line**.
4. **A stated condition of how long or how much is `scope_item`, not chatter.** "Onsite for 2-3 hours to make sure the POS runs on the network" states the work.
   - 010198: answered_question 0.94 → **scope_item**.
5. **A program goal written by the customer is `deal_metadata`, not a provider task**, unless a later line assigns it to the provider.
   - 010264 calendar agenda "retire and dispose of the 9000 clocks": scope_item 0.73 → **deal_metadata**.
6. **A calendar or meeting header (Start:, End:, Recorded-At:, a timestamp) is `deal_metadata`.**
   - 010284 "Start: 2026-07-24T13:00:00Z": kept as scope → **deal_metadata**.
7. **A team roster ("delivery by Chase, Kelly and Octavian") is `stakeholder`.**
   - 010082: scope_item 0.56 → **stakeholder**.
8. **An email signature's out-of-office or calendar line is `small_talk`.**
   - 010147 "Planned OOO 7/24, 8/7, 8/21": _keep 0.55 → **small_talk**.
9. **"X support including Y" is `scope_item`.** It names a service in scope.
   - 000132 "End-user support including O365/M365": deal_metadata 0.84 → **scope_item**.
10. **A line that names a party's own work and rules out the other's is `scope_item`** (a scope boundary).
   - 010279 "We are pricing physical installation labor only": site_implementation_note 0.54 → **scope_item**.

## col:supplier (who furnishes or does it, relative to PurTera)

- **us**: PurTera, or the reseller doing field work through PurTera ("CDW installs phones" on a CDW subcontract).
- **partner**: the reseller acting as itself: its receiving, configuring, shipping or procurement ("CDW will acquire, config and ship").
- **customer**: the end customer: "we will unrack", "they will handle the gateways", "the customer will provide HDMI cabling".
- **third_party**: a named vendor that is neither (recycler, transport carrier, other contractor): "ThinkTLS will provide a box truck with liftgate".

Proved wrong:
- 010083 "central receiving, scheduled shipping, dispatch/control": us 0.92 → **partner**.
- 010376 "box truck with liftgate" (ThinkTLS): us 0.89 → **third_party**.
- 010147 "We will also unrack and ready any cables" (customer's questionnaire): us 0.93 → **customer**.
- 010326 "They will handle the gateways and the R&S" (PurTera note about MPU): us → **customer**.
- 010279 "Provide proof of delivery", "inventory recap on demand" (RFP vendor duties): customer → **us**.

## read:co_action (does the line belong in the job record: keep / reject / ignore)

- **keep** a short line that decides who does what, what is in or out, a count, a date, or a deliverable, however brief: "They will handle the gateways and the R&S, we would be handling phones on the desks, ATAs, wall mounting" (010326, kept 0.41 → **keep**); "Deliverables - just notate the phones/desk/and SN" (0.27 → **keep**); "We would like to have the Logitech Sight installed in this specific room" (010171, 0.39 → **keep**); "Troubleshooting and incident response" (000132, 0.19 → **keep**).
- **reject** chatter, signatures, banners, meeting headers, and lines about another job.

## read:sow_section (where the line goes in a SOW)

- A provider task never goes to `customer_responsibilities` because the customer wrote the list it sits in (010279 "Service and resolve hardware failures": customer_responsibilities → **scope_of_work**).
- A requested service never goes to `assumptions` (000132 end-user support → **scope_of_work**).
