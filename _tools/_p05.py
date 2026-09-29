# -*- coding: utf-8 -*-
"""Pass 05 -- the outbound thank-you and the three HubSpot notes."""
from _mk import build

W = lambda t: {"hint": "own_words", "kind": "text", "text": t}

OUT = {"hint": "doc_type", "kind": "text",
       "text": "Outbound email, purtera-it.com -> NMC, 19 Aug 2026"}
NOTE = {"hint": "doc_type", "kind": "text",
        "text": "HubSpot note filed by Trent Torrence (internal) -- the filer is not necessarily "
                "the speaker"}

# --- 010237-hs-email-115208628622.eml ----------------------------------------
build("010237-hs-email-115208628622.eml", {
 0: {"label_type": "_keep", "weight_tier": "slight", "about": "partner", "wants": "nothing",
     "hint_refs": [W("Thank you, Suresh.")], "entity_keys": [],
     "note": "Acknowledgement by name. No claim about the work -- the whole document is a courtesy "
             "reply to the quote, which is why it produced three atoms and no facts."},

 1: {"label_type": "stakeholder", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "us",
     "hint_refs": [W("Trent Torrence"), W("Executive Vice President of Sales"),
                   W("t@purtera-it.com"), OUT],
     "entity_keys": ["stakeholder:trent_torrence", "email:t_purtera_it_com", "party:purtera",
                     "role:evp_sales"],
     "note": "The only place in the deal that gives Trent a title, and it changes how his messages "
             "read: an EVP of Sales relaying customer requirements is speaking commercially, not "
             "technically. That matters for the atom where he answers the coverage question with "
             "Worktime - 8am- 5pm and drops the six days -- the person who lost the sixth day was "
             "the one carrying the commercial relationship, not an engineer reading the SLA. He is "
             "also the author of all three HubSpot notes, so this is the identity behind the filer "
             "on every internal record in the deal."},

 2: {"label_type": "deal_metadata", "weight_tier": "slight", "about": "partner", "wants": "nothing",
     "hint_refs": [W("working with us on this one"), OUT],
     "entity_keys": ["party:nmc", "party:purtera", "relationship:goodwill"],
     "note": "Relationship talk, kept rather than rejected because of what it sits next to: this is "
             "the only outbound message in the deal and it is a thank-you, sent the day the pricing "
             "arrived after two chases. It says nothing about scope and belongs nowhere near the "
             "SOW, but it is a small true fact about how the partnership is being handled, which is "
             "what about:partner is for."},
}, "pass_05a.json")

# --- 010237-hs-note-115206809191 ---------------------------------------------
build("010237-hs-note-115206809191-Have another oppty for you all. This one lengthier one (6months.).txt", {
 0: {"label_type": "open_question", "weight_tier": "load_bearing", "about": "deal",
     "wants": "chase-conversation", "hint_refs": [W("Preliminary ask"), W("what other high level and specifics needed"), NOTE],
     "entity_keys": ["party:purtera", "service:ams", "process:discovery"],
     "note": "Labels the entire ask as preliminary, which is the frame everything downstream "
             "inherits: the six months, the one resource and the AMS scope all arrive inside a "
             "message that says up front it is not the full picture. chase-conversation because it "
             "explicitly invites a further exchange about what else is needed, and the deal contains "
             "no record of that exchange happening -- the next thing that arrives is an RFI."},

 1: {"label_type": "task", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Need a AMS(Application management services) resource"), W("next 6 months"), NOTE],
     "entity_keys": ["platform:workato", "service:ams", "quantity:6", "term:6_months",
                     "quantity:1"],
     "note": "The earliest statement of the deal, and word for word the same sentence that appears "
             "in the email thread -- the same ask travelling by two routes, which is why the fold "
             "between note and email matters on this deal. Kept as the note copy because the note is "
             "dated 19:25 and is the record HubSpot holds. One resource, six months, Workato AMS: "
             "the three facts the whole 121,518.99 rests on were fixed in this line before any "
             "document existed."},

 2: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "account", "wants": "nothing",
     "hint_refs": [W("Have another oppty for you all."), NOTE],
     "entity_keys": ["party:purtera", "relationship:repeat_business"],
     "note": "Another says there is a prior stream of work with the same counterparty, and for you "
             "all says the sender is not us. This is the account signal on the deal -- the reason it "
             "exists at all -- and by the doctrine it must reach the brief and never the SOW. It is "
             "also the clearest sign the note is filed content rather than authored content: an "
             "internal note whose text addresses PurTera in the second person was written by "
             "somebody else and pasted in."},

 3: {"label_type": "deal_metadata", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "hint_refs": [W("This one lengthier one (6months.)"), NOTE],
     "entity_keys": ["term:6_months", "quantity:6", "relationship:repeat_business"],
     "note": "Sizes this deal against the unnamed previous one -- lengthier, six months. It is the "
             "job_scale reading of the doctrine worked example in its natural habitat: a throwaway "
             "clause carrying a fact that changes how many months of resource get planned. The "
             "comparison also implies the earlier engagement was shorter, which is a fact about the "
             "account that exists nowhere else."},

 4: {"label_type": "commitment", "weight_tier": "load_bearing", "about": "deal",
     "wants": "chase-conversation", "hint_refs": [W("provide some time slots next week"),
                                                  W("Lorena and I"), NOTE],
     "entity_keys": ["party:customer", "stakeholder:lorena", "process:discovery",
                     "role:unresolved"],
     "note": "Offers a discovery call with the end customer and names Lorena, a person who appears "
             "nowhere else in the deal and whose organisation is never stated. By the doctrine a "
             "role word is not a party and a name without an affiliation must stay unresolved -- the "
             "and I places her alongside the sender, who is not us, so she is most likely on the "
             "counterparty side, and most likely is not a label. What is learnable is the gap: a "
             "discovery call with the customer was offered and the deal holds no record of it ever "
             "happening, so the requirements were fixed entirely through a questionnaire that came "
             "back with two non-answers and a corrupted cell."},
}, "pass_05b.json")

# --- 010237-hs-note-115206706200-Note.txt ------------------------------------
build("010237-hs-note-115206706200-Note.txt", {
 0: {"label_type": "_keep", "weight_tier": "slight", "about": "deal", "wants": "nothing",
     "hint_refs": [W("Note")], "entity_keys": [],
     "note": "The entire content of this document is the word Note. A 139-byte HubSpot note whose "
             "body is its own type name says nothing about anything. Rejected, and worth keeping "
             "visible in red rather than dropped so the record shows the deal holds a document with "
             "no content -- which is a different thing from a document the parser failed on."},
}, "pass_05c.json")

# --- 010237-hs-note-115206652190-Here you go.txt -----------------------------
build("010237-hs-note-115206652190-Here you go.txt", {
 0: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("author=Trent Torrence"), W("author_affiliation=internal")],
     "entity_keys": [],
     "note": "This is not a statement anybody made -- it is the parser own provenance banner "
             "(note_id, author, affiliation, timestamp) emitted as if it were content. The doctrine "
             "is explicit: nothing the parser writes is an atom, and if the parser has something to "
             "say it says it about a sentence someone said. The document is called Here you go.txt "
             "and the phrase Here you go appears nowhere in the atom, so the one thing a human "
             "actually typed has been replaced by metadata about it. Rejected as the type, but the "
             "real fix is upstream: these fields belong on the atom as attribution, not instead of "
             "it. This is the same class of defect as the eight message headers the gate dropped -- "
             "attribution and content have been swapped rather than joined."},
}, "pass_05d.json")
