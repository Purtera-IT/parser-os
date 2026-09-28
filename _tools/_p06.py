# -*- coding: utf-8 -*-
"""Pass 06 -- PurteraIT-Abdul Rehman-Workato.pdf, the proposed resource CV."""
import json
from pathlib import Path
from _mk import build, LIVE

DOC = "PurteraIT-Abdul Rehman -Workato.pdf"
W = lambda t: {"hint": "own_words", "kind": "text", "text": t}
D = {"hint": "doc_type", "kind": "text",
     "text": "Two-column CV PDF for the engineer NMC proposed, attached to the quote"}

SEL = [a for a in LIVE if a["filename"] == DOC]

GLUED_NOTE = (
    "Two columns of a CV glued into one atom. This PDF sets PROFESSIONAL SUMMARY beside "
    "PROFESSIONAL EXPERIENCE and the text layer was read straight across the gutter, so every one "
    "of these rows is the tail of a sentence from the left column joined to an unrelated sentence "
    "from the right. Neither half is a statement anybody made, and the pair is not a statement "
    "either -- there is nothing here a type could be true of. Rejected as wreckage rather than "
    "typed, because the parser gave 33 of this document 51 atoms the type scope_item, and a "
    "scope head trained on a CV fragment learns that a candidate career history is deal scope. "
    "The defect is a reading-order problem, not a content problem: the same PDF read column by "
    "column would yield a clean resume.")

SPEC = {}
for i, a in enumerate(SEL):
    t = a["text"]
    if "PROFESSIONAL SUMMARY:" in t and "PROFESSIONAL EXPERIENCE:" in t:
        SPEC[i] = {"label_type": "_keep", "weight_tier": "slight", "about": "internal",
                   "wants": "nothing", "entity_keys": [],
                   "hint_refs": [W("PROFESSIONAL SUMMARY:"), D], "note": GLUED_NOTE}

# The seniority claim is inside one of the glued rows, and it is the only one in the deal.
SPEC[9] = {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
           "entity_keys": [], "hint_refs": [W("17+ years"), D],
           "note": GLUED_NOTE + " This row is the costliest of them: its left half carries 17+ "
                   "years, the only seniority claim anywhere in the deal, and seniority is the one "
                   "reading under which the Deal Kit PS-L3-ENG-LABOR-ONSITE rate code could be "
                   "defended -- the service is L2 in four places, but the person proposed for it "
                   "may well be an L3-grade engineer. So the fact that would settle the most "
                   "valuable open question on this deal is sitting in an atom that cannot be "
                   "labelled, and it is lost not because nobody read the CV but because of a "
                   "column-order bug. Recorded here so the miss is visible rather than silent."}

SPEC.update({
 0: {"label_type": "stakeholder", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Integration Expert"), W("Enterprise Integration Architecture"), D],
     "entity_keys": ["role:integration_expert", "party:nmc", "scope:integration_architecture"],
     "note": "The resource headline: the title and specialisms NMC is putting forward for the "
             "16,000 a month. Load-bearing because this document is the entire justification that "
             "the supplier can staff the engagement, and it is the only atom on it that states what "
             "the person is. The individual name appears only in the filename, never in any atom -- "
             "so the deal can say what was proposed and cannot say who, which is a gap worth having "
             "on the record for a staff-augmentation engagement."},

 1: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Increased project execution efficiency by 25%")], "entity_keys": [],
     "note": "A CV achievement claim about unrelated past work. It is readable, unlike most of this "
             "document, and it still decides nothing here: a 25% efficiency figure from a previous "
             "employer is not a fact about supporting Workato recipes for six months. Rejected on "
             "relevance rather than on wreckage, which is a distinction worth having in the "
             "training set for a document where both kinds sit side by side."},

 2: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Multiple Companies")], "entity_keys": [],
     "note": "Column bleed of the simpler kind: an employment header with the tail of the previous "
             "bullet welded onto it, so the row ends mid-thought with methodology integration. Same "
             "reading-order defect as the PROFESSIONAL SUMMARY rows, without the two labels that "
             "make those ones obvious."},

 3: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Led enterprise programs for Mitel Communications")], "entity_keys": [],
     "note": "Past-client namedropping, and truncated: the sentence continues into the next atom "
             "with the country list. Past employers of a subcontractor engineer are not facts about "
             "this deal and must not reach the SOW."},

 4: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Dealer Track (USA)")], "entity_keys": [],
     "note": "The continuation of the previous line, orphaned. Read alone it is a list of three "
             "companies with no verb -- the clearest small example on this document of why a line "
             "torn off its sentence cannot be labelled."},

 5: {"label_type": "deal_metadata", "weight_tier": "load_bearing", "about": "deal",
     "wants": "nothing", "supplier": "partner",
     "hint_refs": [W("using Workato and Pentaho"), D],
     "entity_keys": ["platform:workato", "role:integration_expert", "party:nmc"],
     "note": "Names Workato in the engineer own history, which is the single thing this CV exists to "
             "prove and one of only two atoms on the document that actually proves it. It is "
             "load-bearing on a deal where the supplier was explicitly asked not to go searching for "
             "a resource -- the ask was for somebody already available, and this is the evidence "
             "that the person offered has the platform. about:deal rather than internal because it "
             "bears on whether this engagement can be delivered, not on how we work."},

 6: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Led integration/migration of data based on API")], "entity_keys": [],
     "note": "Half a bullet, ending on a comma. Its other half is the next atom."},

 7: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Webhooks and scripting")], "entity_keys": [],
     "note": "The second half of the previous bullet, beginning mid-clause. The pair would be one "
             "readable sentence; split, neither is."},

 8: {"label_type": "service_line", "weight_tier": "load_bearing", "about": "deal", "wants": "nothing",
     "supplier": "partner",
     "hint_refs": [W("Workato monitoring, troubleshooting and optimizing integration flows"), D],
     "entity_keys": ["platform:workato", "service:monitoring", "service:triage",
                     "role:integration_expert", "party:nmc"],
     "note": "The closest match in the whole document to the work actually sold: Workato monitoring "
             "and troubleshooting of integration flows is almost word for word the SLA scope line "
             "covering monitoring and triage of recipe failures. This is the atom that makes the CV "
             "relevant rather than decorative, and it is one of only two clean Workato statements on "
             "a 51-atom document -- the rest of the evidence for this engineer suitability was "
             "destroyed by the column bleed."},

17: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("stability.")], "entity_keys": [],
     "note": "A single orphaned word with a full stop: the tail of a right-column sentence whose "
             "body is in another atom. Wreckage."},

22: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("system monitoring.")], "entity_keys": [],
     "note": "Two words orphaned from the right column, and a duplicate of the same fragment lower "
             "in the document. The CV was read across the gutter twice over."},

26: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Lahore, Pakistan"), D], "entity_keys": [],
     "note": "Glued like its neighbours, but its right half carries the one fact on this document "
             "with a commercial consequence: an employer in Lahore, Pakistan, from 2018 to 2021. "
             "The deal sells 8:00 AM to 5:00 PM EST coverage, which is a night shift from Pakistan, "
             "and the sibling opportunity in the same account is titled Off Shore Remote India -- so "
             "offshore delivery is the house pattern, not a leap. It must stay unresolved, though: "
             "the engineer current employer is listed as Multiple Locations and no atom states where "
             "he is now. By the doctrine that is the labelling rather than a gap in it -- two past "
             "Lahore employers cannot establish present residence, and the honest output is a "
             "question for NMC about which timezone the resource actually sits in, not a conclusion."},

42: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("project lifecycles using Agile")], "entity_keys": [],
     "note": "The same column bleed a second time, now without the PROFESSIONAL SUMMARY labels to "
             "mark the seam -- this row and the five after it repeat text already glued higher up. "
             "A second lossy pass over the same region, which is worse than the first because "
             "nothing signals that it is a repeat."},

43: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Evaluated and implemented API-led integration")], "entity_keys": [],
     "note": "Column bleed, second pass. Duplicate of content already captured and already broken."},

44: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("reducing technical debt")], "entity_keys": [],
     "note": "Column bleed, second pass. Two half-sentences from opposite columns."},

45: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("system monitoring.")], "entity_keys": [],
     "note": "The orphaned fragment from atom 22, emitted a second time. A duplicate of wreckage."},

46: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("CORE EXPERTISE")], "entity_keys": [],
     "note": "A section heading welded to a right-column bullet. The heading alone would be "
             "scaffolding; joined to unrelated prose it is scaffolding and wreckage at once."},

47: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Integration architecture and design")], "entity_keys": [],
     "note": "A skills bullet glued to an achievement bullet from the other column. The left half "
             "duplicates the specialisms already carried cleanly by the document first atom."},

48: {"label_type": "certification", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("M.Sc. Project Management"), W("Manchester"), D],
     "entity_keys": ["role:integration_expert", "party:nmc", "credential:msc_project_management"],
     "note": "A clean atom on a broken document -- the education block escaped the column bleed "
             "because it is single-column. Kept as a real credential of the proposed resource. It is "
             "ordinary rather than load-bearing: a project management masters is not what this "
             "engagement buys, and the SLA explicitly puts project management out of scope."},

49: {"label_type": "certification", "weight_tier": "ordinary", "about": "deal", "wants": "nothing",
     "supplier": "partner", "hint_refs": [W("BS Hons. Information Technology"), D],
     "entity_keys": ["role:integration_expert", "party:nmc", "credential:bs_information_technology"],
     "note": "The second clean credential, from University of the Punjab. Worth keeping beside the "
             "Manchester masters because together they are the only verifiable statements about the "
             "engineer that survived the parse intact, and the Punjab degree is a second, "
             "independent pointer at a Pakistan-based career that the coverage question turns on."},

50: {"label_type": "_keep", "weight_tier": "slight", "about": "internal", "wants": "nothing",
     "hint_refs": [W("Rolustech"), W("Lahore, Pakistan")], "entity_keys": [],
     "note": "An employment header carrying the label of the column it came from. It repeats the "
             "Lahore signal already noted, and like every other row on this document it arrives "
             "welded to its own section heading rather than standing as a statement."},
})

build(DOC, SPEC, "pass_06.json")
print(f"(document has {len(SEL)} live atoms; spec covers {len(SPEC)})")
