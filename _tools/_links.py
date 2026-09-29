# -*- coding: utf-8 -*-
"""Evidence links for 010237, written as atom_label_links rows.

Each link is (doc, index) -> (doc, index) in the same walk-order coordinates the
pass files use, so a link can be read next to the dump it came from. The note on
each is the argument for THESE two atoms: the edge head has no gold anywhere, and
a row saying contradicts with no reason is the thinnest version of the richest
reasoning on the deal.
"""
import json, os, sys, uuid
from pathlib import Path
import psycopg2

DEAL = os.environ["DEAL"]
MINE = "developer@purtera-it.com"
HERE = Path(__file__).parent
WALK = json.loads((HERE / "walk_180.json").read_text(encoding="utf-8"))["atoms"]
LIVE = [a for a in WALK if "dedup" not in (a.get("suppressedBy") or "")]

QUOTE = "010237-hs-email-115241880664.eml"
THREAD = "010237-hs-email-115199549032.eml"
SLA = "NMC-SLA-Purtera IT-Workato_Middleware-v.1"
RFI = "Copy of SHI Workato questions RFI_RV updated.xlsx"
CV = "PurteraIT-Abdul Rehman -Workato.pdf"
NOTE = "010237-hs-note-115206809191-Have another oppty for you all. This one lengthier one (6months.).txt"
OUTB = "010237-hs-email-115208628622.eml"


def A(doc, idx):
    return [a for a in LIVE if a["filename"] == doc][idx]


LINKS = [
 # ---------------------------------------------------------------- contradicts
 ((QUOTE, 4), (SLA, 2), "contradicts",
  "The coverage we bought against the coverage we committed to. NMC Option 1, the option the Deal "
  "Kit costs at 16,000 a month, covers 9:00 AM to 5:00 PM; the SLA attached to that same quote "
  "commits to 8:00 AM to 5:00 PM. One hour every weekday for six months, on the side where we owe "
  "it. Neither document acknowledges the other."),
 ((THREAD, 22), (SLA, 2), "contradicts",
  "The sixth day. NMC read the customer RFI answer as eight hours a day over six days and said so "
  "in writing; the SLA came back 8x5, forty hours. Nobody ever told the customer their sixth day "
  "was gone, and the SLA weekend tail is billable at 125 an hour, so the difference is not merely "
  "uncovered time -- it is the deal single largest uncosted exposure."),
 ((THREAD, 22), (QUOTE, 4), "contradicts",
  "The same sixth day against the priced line rather than the contract line. The customer asked for "
  "48 hours a week; Option 1 prices 40 and puts everything beyond it on consumption at 125 an hour. "
  "Eight hours a week over roughly 26 weeks is about 26,000, against a total deal margin of "
  "25,323.99 -- so if the customer consumes what they asked for, the margin is gone."),
 ((QUOTE, 12), (SLA, 18), "contradicts",
  "NMC covering note says the attached SLA is for the Staff Augmentation option; the SLA own "
  "Service Package says Engagement Model: Recurring monthly managed service, which is Option 1. The "
  "supplier contradicts itself about which of its two quotes the contract governs, and the answer "
  "decides whether we owe a service level or a person."),
 ((RFI, 6), (SLA, 41), "contradicts",
  "The customer asked for P3 resolution in 4-8 hours and marked the whole set Tentative. The SLA "
  "answers P3 with an 8-hour response and a 2-business-day resolution. Moving from hours to "
  "business days is a materially weaker commitment, and it was never flagged back to the party who "
  "wrote Tentative beside their numbers."),
 ((RFI, 6), (SLA, 24), "contradicts",
  "The same substitution at P4, and larger: the customer asked for 8-16 hours, the SLA returns 24 "
  "hours to respond and 5 business days to resolve. On a low-severity queue that is the difference "
  "between same-week and next-week, and it is the tier most likely to carry the fifteen incidents a "
  "week the customer estimated."),
 ((RFI, 4), (THREAD, 27), "contradicts",
  "The customer confirmed all three environments exist; the access actually granted is Prod and "
  "Test. Development is missing from the grant and present in the contract, which makes it a "
  "day-one onboarding failure rather than a discovery item."),
 ((SLA, 30), (THREAD, 27), "contradicts",
  "The SLA commits to supporting the integration across development, test and production. The "
  "access answer gives Prod and Test. We contracted for an environment nobody has let us into."),
 ((THREAD, 5), (SLA, 18), "contradicts",
  "The original ask is for a resource -- one person, staff augmentation in shape -- and the SLA "
  "sells a recurring monthly managed service governed by severity targets. The deal never resolves "
  "which of the two it is, and the two are priced, staffed and measured differently."),

 # -------------------------------------------------------------------- answers
 ((THREAD, 22), (THREAD, 26), "answers",
  "The partial answer that lost the sixth day. NMC asked two things in one sentence -- what are the "
  "actual working hours, given the RFI says eight a day over six days -- and this reply gives the "
  "hours only. The days half is never answered by anyone, and the SLA then assumed five."),
 ((THREAD, 23), (THREAD, 27), "answers",
  "Non-production access, answered narrower than it was asked: the question named Dev/Test and the "
  "answer returns Prod and Test, quietly substituting production for development."),
 ((THREAD, 24), (THREAD, 28), "answers",
  "The communication-channel question, answered with SKLSI email -- which settles that the engineer "
  "works under the client domain, and incidentally names the end client for the only time in the "
  "deal."),
 ((THREAD, 29), (THREAD, 30), "answers",
  "Timezone, asked and answered cleanly in the same exchange. It is load-bearing because every "
  "coverage window in the deal is stated in EST and none of them is comparable without it."),
 ((THREAD, 34), (THREAD, 36), "answers",
  "A chase for a date, answered 28 minutes later with a same-day promise that was kept: the SLA is "
  "dated 19 August and the quote followed. Crossing a message boundary, which is the kind of edge "
  "the relation head has no gold for."),
 ((RFI, 19), (THREAD, 22), "answers",
  "The recovery. The RFI cell holds the date 6-Aug because Excel converted the customer 8/6 on "
  "entry; this is the only place in the deal where the intended answer survives, reconstructed by a "
  "human reading the questionnaire in context. A machine reading the spreadsheet alone cannot get "
  "back what this sentence carries."),
 ((THREAD, 2), (QUOTE, 3), "answers",
  "The deal opening question -- is this a good fit for your team -- closed a month later by a price. "
  "Kept as an edge because it spans two documents and the whole procurement arc between them."),
 ((THREAD, 21), (THREAD, 26), "answers",
  "The first of the three answers that produced SLA v1.1. The request was to help finalise the SLA; "
  "these replies are what it was finalised from, which makes every gap between them and the "
  "document a gap in the contract."),
 ((THREAD, 21), (THREAD, 27), "answers",
  "The second of the three SLA-finalising answers, on environment access."),
 ((THREAD, 21), (THREAD, 28), "answers",
  "The third, on communication channel -- and the one that named SKLSI."),
 ((THREAD, 11), (THREAD, 13), "answers",
  "A follow-up chase answered with a status rather than a substantive reply: the engineering team "
  "is still reviewing. Useful as gold precisely because it is a weak answer to a weak question."),
 ((THREAD, 17), (THREAD, 36), "answers",
  "A promise of Monday on 6 August, finally answered on 19 August with today. The gap between them "
  "is the only supplier-performance signal the deal carries."),

 # ------------------------------------------------------------------- supports
 ((RFI, 9), (SLA, 31), "supports",
  "The customer selected L2 from a list that explicitly offered L3; the SLA Service Package states "
  "Support Level: Level 2 (L2). Client choice and contract agree, which is what makes the Deal Kit "
  "PS-L3-ENG-LABOR-ONSITE rate code the outlier rather than these."),
 ((RFI, 9), (SLA, 8), "supports",
  "The same selection supporting the SLA scope sentence rather than its package block. Four "
  "independent L2 statements exist in this deal and no document anywhere says L3 of the service."),
 ((RFI, 3), (SLA, 17), "supports",
  "The customer named Workato-SAP integration; the SLA adopts it as the named integration that "
  "bounds scope. This is the clearest evidence the SLA was written from the questionnaire rather "
  "than from a template."),
 ((RFI, 17), (SLA, 30), "supports",
  "Two integrated applications, answered in the RFI and carried into SLA scope as two applications: "
  "ERP and procurement. It is also the count this deal has instead of sites or devices, so it is "
  "the only sizing any estimate can rest on."),
 ((RFI, 16), (SLA, 30), "supports",
  "The same support from the applications question rather than the count: ERP and procuremnt, "
  "misspelled in the source, corrected in the contract."),
 ((RFI, 12), (SLA, 35), "supports",
  "ServiceNow confirmed by the customer and adopted by the SLA as the sole channel the service "
  "levels are measured through -- which also makes our own performance evidence live in a system we "
  "do not control."),
 ((QUOTE, 10), (SLA, 6), "supports",
  "Remote delivery stated by the quote and repeated by the SLA. Two of the three statements that "
  "make ONSITE in the priced rate code unsupported by anything in the record."),
 ((QUOTE, 9), (SLA, 38), "supports",
  "Six months, stated independently by the quote and the SLA. The one commercial fact on this deal "
  "that three sources agree on and nobody has to chase."),
 ((QUOTE, 5), (SLA, 14), "supports",
  "The weekend rate quoted at 125 an hour, and the SLA billing basis restating it as the variable "
  "half of a fixed-plus-variable invoice. Two documents describe the exposure and no financial line "
  "anywhere carries it."),
 ((RFI, 5), (SLA, 4), "supports",
  "The customer answered the monitoring question with two question marks, and this SLA assumption "
  "is conditional on exactly that: resolution targets assume monitoring is in place, and where it "
  "is not we build it during onboarding. The non-answer makes the unpriced branch the likely one."),
 ((RFI, 5), (SLA, 46), "supports",
  "The same non-answer against the clause that matters more: where automated alerting is missing, "
  "resolution timing runs from human detection. So the blank in the questionnaire quietly softens "
  "the only commitment the engagement makes."),
 ((RFI, 15), (SLA, 25), "supports",
  "The customer said break fix initially, but new; the SLA answers it by fixing scope at break-fix "
  "and incident support. The correct commercial response to a client who has already signalled the "
  "scope will grow."),
 ((RFI, 15), (SLA, 21), "supports",
  "The same answer against the exclusion that routes the coming development work to a change "
  "request. On a fixed monthly fee this is the clause that keeps growth billable rather than "
  "absorbed."),
 ((RFI, 2), (SLA, 10), "supports",
  "A bare yes on documentation, against an SLA onboarding gate that requires reviewing four named "
  "artefact types and confirming monitoring. The answer asserts existence and cannot carry the "
  "weight the gate puts on it."),
 ((RFI, 11), (SLA, 9), "supports",
  "Knowledge transfer confirmed as yes, with neither format nor duration -- and the SLA gate it "
  "supports involves an outgoing team, so duration is precisely the number that sets the "
  "commencement date the six-month term runs from."),
 ((RFI, 8), (SLA, 0), "supports",
  "Admin- tech support, three words, standing under the SLA first assumption that the customer "
  "provides timely administrative access. NMC restated the same requirement twice in the SLA, which "
  "is what a supplier does when an answer is thinner than the dependency it carries."),
 ((RFI, 0), (SLA, 40), "supports",
  "Fifteen incidents a week, guessed, against an SLA that measures every target through ServiceNow "
  "and caps incident volume nowhere. On a fixed monthly managed service the volume risk is entirely "
  "ours and this estimate is the only bound on it."),
 ((CV, 8), (SLA, 23), "supports",
  "The engineer own history says Workato monitoring, troubleshooting and optimizing integration "
  "flows; the SLA scope line commits to monitoring and triage of recipe failures. Near word-for-word "
  "-- this is the atom that makes the CV relevant rather than decorative."),
 ((CV, 5), (CV, 0), "supports",
  "Workato named in the engineer project history, supporting the Integration Expert headline. Two "
  "of the three clean atoms on a document whose other 33 rows were destroyed by column bleed."),
 ((QUOTE, 3), (QUOTE, 2), "supports",
  "The price under its option header. The 16,000 means nothing without knowing it is Option 1 "
  "Managed Services and not Option 2 Staff Augmentation, and the Deal Kit consumed one of them."),
 ((OUTB, 1), (THREAD, 26), "supports",
  "Trent title -- EVP of Sales -- read against the answer where he relays the coverage requirement "
  "and drops the six days. The person who lost the sixth day was carrying the commercial "
  "relationship, not reading the SLA, which is the most charitable and most useful explanation of "
  "how it happened."),

 # -------------------------------------------------------------------- governs
 ((QUOTE, 1), (QUOTE, 2), "governs",
  "The line announcing the NMC budgetary quote frames everything beneath it as OUR COST rather than "
  "our price. Without this level the option headers float free and the 16,000 reads as revenue."),
 ((QUOTE, 1), (QUOTE, 6), "governs",
  "The same announcement over the second option, which is what makes the two options siblings under "
  "one quote rather than two unrelated prices."),
 ((QUOTE, 2), (QUOTE, 3), "governs",
  "Option 1 over its price. Structure, not restatement: the parser may assert this about other "
  "atoms because it mints no sentence nobody said."),
 ((QUOTE, 2), (QUOTE, 4), "governs",
  "Option 1 over its coverage window, which is the link that makes 9:00 AM a fact about the option "
  "we bought rather than a floating time."),
 ((QUOTE, 2), (QUOTE, 5), "governs",
  "Option 1 over the weekend rate. The 125 an hour belongs to the managed-service option, which is "
  "the one the Deal Kit costed -- so the exposure attaches to the option actually purchased."),
 ((QUOTE, 6), (QUOTE, 7), "governs",
  "Option 2 over its hourly rate, the 130 that makes the SOW 1,008 estimated hours arithmetic "
  "work out to more than the deal sells for."),
 ((QUOTE, 6), (QUOTE, 8), "governs",
  "Option 2 over its coverage window. The 8:00 start the customer wanted lives under the option we "
  "did not buy, and this link is what makes that legible."),
 ((THREAD, 21), (THREAD, 22), "governs",
  "The request to help finalise the SLA opening the block of three questions. The block framing is "
  "what makes these three a set whose answers became a contract, rather than three loose questions."),
 ((THREAD, 21), (THREAD, 23), "governs",
  "The same block over the environment-access question."),
 ((THREAD, 21), (THREAD, 24), "governs",
  "The same block over the communication-channel question."),
 ((SLA, 34), (SLA, 3), "governs",
  "The onboarding section over the access-provisioning task. It matters because the SLA says the "
  "service levels do not start until onboarding completes, so everything under this header gates "
  "the commencement date the six-month term is measured from."),
 ((SLA, 34), (SLA, 9), "governs",
  "Onboarding over knowledge transfer -- the item whose undefined duration is what actually sets "
  "the commencement date."),
 ((SLA, 34), (SLA, 10), "governs",
  "Onboarding over the documentation review."),
 ((SLA, 34), (SLA, 12), "governs",
  "Onboarding over establishing the ServiceNow queues, which is the one item that gates the service "
  "levels rather than merely the service: every target is defined as measured on tickets in that "
  "system."),

 # -------------------------------------------------------------------- same_as
 ((THREAD, 5), (NOTE, 1), "same_as",
  "The same sentence in two documents: the original ask, as an email atom and as the HubSpot note "
  "Trent filed. This is the doctrine copy-of case at sentence level -- one message, one record -- "
  "and it is worth an explicit edge because the note is internal and the email is inbound, so a "
  "reader cannot otherwise tell they are one statement rather than two independent confirmations."),
 ((SLA, 13), (SLA, 36), "same_as",
  "The P1 severity row and the lossier second read of it the same table produced. The full row "
  "carries severity, definition, response and resolution; the fragment keeps the definition and one "
  "number and drops which target it is."),
 ((SLA, 20), (SLA, 7), "same_as",
  "P2, full row against its degraded twin. Four of these pairs exist on this document."),
 ((SLA, 41), (SLA, 43), "same_as",
  "P3, full row against its degraded twin."),
 ((SLA, 24), (SLA, 11), "same_as",
  "P4, full row against its degraded twin -- and the one where the loss is largest, because the "
  "fragment keeps 24 hours while the full row shows that is the response and the resolution is five "
  "business days."),
 ((RFI, 6), (RFI, 10), "same_as",
  "The complete severity answer against one tier torn out of it. The fragment cannot say it is "
  "Tentative, which the full answer does, and Tentative is what makes the SLA substitution a live "
  "issue rather than a settled one."),
 ((RFI, 6), (RFI, 18), "same_as",
  "The same complete answer against the P1 fragment."),
 ((RFI, 13), (RFI, 20), "same_as",
  "Two fragments that are byte-identical to each other: the same torn P4 tier emitted twice. Neither "
  "is labellable and the pair is a straight duplicate, which is the cheapest possible gold for a "
  "document-level duplicate head."),
 ((RFI, 14), (RFI, 21), "same_as",
  "The second identical fragment pair, at P2."),
 ((CV, 22), (CV, 45), "same_as",
  "The orphaned fragment system monitoring. emitted twice from the same column bleed. A duplicate "
  "of wreckage, which is worth recording because it shows the CV was read across the gutter twice "
  "rather than once."),

 # -------------------------------------------------------------------- context
 ((THREAD, 38), (QUOTE, 1), "context",
  "Suresh Nalla signature block against the quote he sent. He is the only person on either side "
  "whose messages set a price, so this is the identity behind every number in the deal cost basis."),
 ((OUTB, 1), (NOTE, 2), "context",
  "Trent identity against the note he filed. The note text addresses PurTera in the second person "
  "-- for you all -- so the filer is demonstrably not the speaker, which is the doctrine rule that "
  "the person who pasted is not the person who spoke."),
 ((RFI, 1), (RFI, 0), "context",
  "Five production recipes against an estimate of fifteen incidents a week. The two answers sit in "
  "the same questionnaire and together they say either the estimate is pessimistic or the platform "
  "is unstable -- and the deal is priced on a fixed monthly fee without anyone asking which."),
 ((CV, 26), (SLA, 2), "context",
  "Two past employers in Lahore against a commitment to cover 8:00 AM to 5:00 PM EST, which is a "
  "night shift from Pakistan. It is context rather than a contradiction on purpose: the engineer "
  "current employer is listed as Multiple Locations and no atom states where he is now, so this "
  "raises the question and must not answer it."),
 ((THREAD, 3), (CV, 0), "context",
  "Do not spend time searching for a resource, read against the resource that was proposed. If a "
  "17-year integration architect was sourced rather than benched, the six-month availability behind "
  "a fixed monthly cost is a thinner promise than the quote implies."),
]


def main():
    apply = "--apply" in sys.argv
    rows = []
    for (fd, fi), (td, ti), rel, note in LINKS:
        a, b = A(fd, fi), A(td, ti)
        rows.append((str(uuid.uuid4()), DEAL, MINE, "atom", a["labelKey"], a["text"],
                     "atom", b.get("atomId"), b["labelKey"], b.get("artifactId"),
                     b["filename"], None if b.get("page") is None else str(b["page"]),
                     b["text"], rel, note, None, "train"))
    from collections import Counter
    print(f"{len(rows)} links: {dict(Counter(r[13] for r in rows))}")
    short = [r for r in rows if len(r[14]) < 24]
    if short:
        raise SystemExit(f"{len(short)} link(s) with a note too thin to train on")
    for r in rows[:6]:
        print(f"   {r[13]:12} {r[5][:46]:48} -> {r[12][:46]}")
    if not apply:
        print("\ndry run -- pass --apply to write")
        return
    cn = psycopg2.connect(os.environ["PG"], connect_timeout=30)
    cur = cn.cursor()
    cur.execute("DELETE FROM public.atom_label_links WHERE deal_id=%s AND labeler=%s", (DEAL, MINE))
    cur.executemany(
        "INSERT INTO public.atom_label_links (id, deal_id, labeler, from_head, from_key, from_text,"
        " to_kind, to_atom_id, to_label_key, to_artifact_id, to_filename, to_page, to_text,"
        " relation, note, compile_id, purpose)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
    cn.commit(); cur.close(); cn.close()
    print(f"\nwrote {len(rows)} links as {MINE}")


if __name__ == "__main__":
    main()
