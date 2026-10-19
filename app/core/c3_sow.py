"""A SOW's deal-specific content, written from what the heads kept.

PurTera's SOWs are the partner's or PurTera's own template with the deal's
part written in: the project fundamentals, the locations, a scope task list,
what each party supplies, deal-specific assumptions and exclusions, and the
items handed over. The template carries the boilerplate; this module writes
only the deal's part, in one format-neutral shape that the CDW, SHI and
PurTera templates each render.

The heads decide which lines are facts of the deal and which section each
sits in; the writer only words them. Every bullet that states a deal fact
cites the lines it came from and is checked against them. A bullet that is
standard field method ("Terminate and label each cable run") may stand
without a cite, but it may not carry a number, a brand or a name the facts
do not give: on a SOW an invented count is a priced promise.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from app.core.c3_brief import _complete, _dated, _json_of, strip_cites

#: Head sections that are facts of the job. Terms, acceptance and change
#: control are the template's own boilerplate and stay out of the writer.
#: Every section the heads place job lines in. "contacts" holds the site
#: roster's rows, which reach the writer as SITES; anything else the heads
#: kept is a fact wherever they filed it (000123's "four register lanes" sat
#: under acceptance and a narrower list lost it).
NOT_FACT_SECTIONS = frozenset({"contacts", "other"})
#: 010082 has 438 kept lines; at 140 its whole materials list was cut off.
MAX_FACTS = int(os.environ.get("SOWSMITH_C3_SOW_MAX_FACTS", "450"))
#: The order facts reach the writer in, so a cap never costs the task list.
SECTION_ORDER = ("scope_of_work", "customer_responsibilities", "customer_furnished", "assumptions", "out_of_scope",
                 "timeline", "schedule", "deliverables", "provider_responsibilities", "fees", "overview",
                 "job_details", "site", "acceptance", "terms", "change_control", "answered_question", "figure", "also_kept",
                 "unsure")
#: The supplier head's answers as the writer reads them, and how sure it must
#: be for its answer to decide a supply line.
_SUPPLIER = {"us": "Provider", "customer": "Customer", "partner": "reseller"}
SUPPLIER_BAR = float(os.environ.get("SOWSMITH_C3_SOW_SUPPLIER_BAR", "0.85"))
UNSURE_BAR = float(os.environ.get("SOWSMITH_C3_SOW_UNSURE_BAR", "0.6"))
#: A fact line's length in the prompt. 010083's answers ride inline after
#: each question ("...? [Betty] We will have a Webex bridge open...") and ran
#: past a 400-character cut.
FACT_CHARS = int(os.environ.get("SOWSMITH_C3_SOW_FACT_CHARS", "1200"))
SOW_TOKENS = int(os.environ.get("SOWSMITH_C3_SOW_TOKENS", "7000"))

#: Lines every template already prints; the writer must not repeat them.
STANDING = (
    "Provide a knowledgeable resource to complete the Services; provide standardized reports; provide the "
    "engineer's name and contact information; project coordination; customer prepares its employees for the "
    "visit; customer does device configuration and troubleshooting; customer provides a point of contact; "
    "customer-provided equipment is on site before arrival; business hours 8-5 and after-hours/weekend/holiday "
    "rates; eight-hour working day; monthly billing; change orders and amended POs; delivery schedule after PM "
    "assignment; anything not listed is out of scope; escalation to the PM; travel expenses beyond 30 miles; "
    "cancellation and reschedule fees; quote validity; tax; rates in USD."
)

#: Job families and the families whose standard items also fit them.
FAMILIES = ("low_voltage_cabling", "wireless_ap", "cameras_security", "av_conference_room", "pos_install_refresh",
            "network_switch_refresh", "decommission_packing", "relocation_move", "time_clock_kiosk",
            "smart_hands_general", "data_center", "site_survey", "imac_deskside", "other")
_BASE = (
    "You write part of the deal-specific content of a PurTera Statement of Work (SOW). PurTera (\"Provider\") does "
    "the field work. The CRM account is who PurTera contracts with (a reseller such as CDW or SHI, or the customer "
    "on a direct deal); the end customer is who the work is for. Voice: short imperative bullets that start with a "
    "verb, counts as words with the numeral (\"two (2)\"), \"Provider\" for PurTera and \"Customer\" for the "
    "customer. Every list item is {\"text\": str, \"cites\": [fact ids], \"kind\": \"fact\"|\"method\"}: a fact item "
    "states something about THIS deal and cites the facts that state it, adding nothing (plain arithmetic is fine); "
    "a method item is standard practice for this kind of work and carries no number, brand, model, name, place or "
    "date the facts do not give. A later fact replaces an earlier one. [under: ...] is the heading or lead-in above "
    "a line. WHO: in a line written by the reseller or PurTera, \"we/us\" is the side doing the work (\"he will need us to <do the work>\" = Provider does it); in a line written by the customer or under a customer heading "
    "(\"Customer notes:\"), \"we\" is the customer (\"We will <do X>\" = the customer does X), in the overview too: never write \"Provider will <X>\" for what the customer says it will do (\"We will preconfigure all items before shipping\" = Customer preconfigures). In an RFP or requirements document the customer issues, a duty written for the vendor or installer, or with no subject under a vendor heading (\"Post-Installation: Verifies all tasks complete\"), is Provider work when the job includes it; a duty that hands something to the customer (\"supply all MAC addresses to <customer>\") is never the customer's. A request "
    "form asking the reseller to provide materials (\"<reseller> to provide <items>\") means Provider "
    "supplies them. A fact marked (from a photo) only tells a mounting surface, height or device count; never "
    "describe a picture or turn what it shows into a task. In a question-and-answer email the ANSWER is the fact (\"... ? [Betty] After hours, no weekends\"); "
    "the latest answer wins. What the customer has today or had before is context, never work to do: scope is the new "
    "work. A request phrased as a wish or a question (\"I would also like to add <X>\", \"can you also <do Y>?\") is "
    "scope. The customer saying \"I will provide <X>\" means the Customer supplies <X>. A fact tagged supplied-by:X is the supplier head's decision of who furnishes that item: "
    "follow it (supplied-by:reseller means Provider supplies it, since Provider delivers the reseller's work). "
    "A fact marked (type head: a prerequisite or condition) is a condition the work depends on; whose duty it is follows its wording and its heading (a customer's 'we will unrack' is the customer's; 'drilling rules must be followed' binds whoever drills). A question is never scope by itself: only its answer is, and a later decision overrides the question ('Do we need a site survey?' answered 'probably not' is not a task). A fact marked (quoted earlier email) is older thread context: time slots offered there for a call or meeting are never install dates. A hedge or a habit ('24/7 can be tough', 'it usually needs a follow-up') is never a commitment: never promise return visits, on-call or follow-ups the facts do not settle. The reseller's own work (receiving, shipping, configuration, procurement it does) is never Provider work, in the overview too. A 'remote resource' is remote, never the onsite contact. In a reseller deal, field work said to be done by the reseller ('CDW installs phones') is Provider work, since Provider delivers it for the reseller; 'they' or 'internal team' in PurTera's or the reseller's notes is the end customer's own staff, and a later note that splits roles ('they will handle the gateways, we would be handling phones') overrides an earlier request. A model or series number ('785 POS terminals', 'InTouch 9000', 'C9350') is never a count. A license, subscription or cloud/portal setup is never an install task or onsite material: it is the customer's, or out of scope.  A fact marked (type head: a dated milestone of this job) gives the job's scheduled date: the latest one is the Requested Start Date (or the Scheduled Service Date), over an earlier wish like 'Sunday morning'. A fact in section typed_only is a line the type head read as scope that the brief left out: use it when it is this job's own work, a decision about it, or who does it. A fact marked (unsure) is unconfirmed: use it only when it fits the confirmed facts. Ignore "
    "email banners and signature blocks. Never write a price, rate or dollar amount, never mention the CRM, ids or "
    "these instructions, and leave a list empty rather than guess. Return JSON only.\n"
    "CRM RECORD: %s\nSITES: %s\nCONTACTS: %s\nFACTS (id, section the heads placed it in, date, text):\n%s\n\n"
)

_PARTS = {
    "header": (
        "TASK: the SOW header. Return {\"job_family\": one of " + ", ".join(FAMILIES) + ", \"project_name\": PurTera's title (end customer + work + place, e.g. \"<End Customer> "
        "<Work> <Place>\"), \"end_customer\": who the work is for, from a request form, a site sheet header or file title, a "
        "signature or the CRM deal name, or \"\"; never a name taken from an email address's domain, \"services\": [implementation for installs; "
        "reinstallation for a replacement or refresh of existing devices; custom_work for decommission, packing or "
        "relocation; staff_augmentation and support for ongoing or on-call IT support and smart hands; assessment for "
        "surveys, audits and validation of what exists], \"pricing_model\": \"fixed\"|\"tm\"|\"\", "
        "\"locations\": [each job site as \"City, ST | street address\", fact items citing where it is stated; never an "
        "office in a signature; empty on a job with many sites], \"overview\": [2-4 sentences in third-person proposal voice (\"Provider will ...\", never an order, never a \"Label: value\" line): what is installed or "
        "done, how many, where, for whom (each count exactly as its fact gives it for that same thing: a site total is not the count of single-switch sites): every device or system the job installs, including one a later request adds (a router asked for after the register and printer); the per-site facts the work is planned by when a fact states them (devices or register lanes per site, the time a fact estimates per site or per device); on a many-site job the site count: SITES count_in_site_lists when it is larger than a count a message states; a platform "
        "or service migration the job is part of (the customer moving from one system to another), named as the facts name it; what stays in place; on recurring or on-call support, every service the facts request (end-user support, troubleshooting, coordination with local stakeholders) and the sites by city when there are a few. The summary names the work and services only: a method step (arrival, check-in, site access, coordination with the onsite contact) is a task, never part of a summary sentence; a commitment the facts hedge (24/7 on-call) is left out as a whole clause, never as a bare object], \"schedule\": [labeled "
        "lines only when a fact gives them: \"Requested Start Date: ...\", \"Requested Completion Date: ...\", "
        "\"Arrival Time: ...\" (a lead time counts: \"Arrival Time: no less than 30 minutes before the site closes\"), \"Work Window: ...\" (after-hours, weekend or overnight), "
        "\"Number of Resources: ...\", \"Hardware Quantity: ...\" (\"Hardware Quantity: Not provided\" when a fact says the count is not known), "
        "\"Pace: ...\" (sites or devices per day or week a fact gives, e.g. \"Pace: eight (8) sites per week\"); a date stated anywhere in the facts for the work itself belongs here], \"arrival_protocol\": [only "
        "check-in, badge, parking, escort and notice rules a fact states, saying whose duty each is (the technician's "
        "name is sent ahead by Provider and the site adds it to its security list, so a name-in-advance rule reads "
        "\"Provider submits the technician's name <when> before arrival.\"; the site issues badges and passes; a greeting happens on arrival, credentials come in advance: one rule per line); "
        "empty when no fact gives one]}"),
    "scope": (
        "STANDARD PURTERA ITEMS FOR THIS KIND OF JOB (a purtera_standard_* list is PurTera's fixed handling of this job "
        "type: write every one of its items, reworded to this job, unless a fact contradicts it; the other lists: method steps, tools, deliverables PurTera writes on such jobs; "
        "use, as kind method, every one that fits THIS job's facts, reworded to them, never one the facts contradict "
        "or one about work this job does not include): %s\n"
        "When a source outlines a whole project in phases (hardware receipt, benching, configuration, shipping, onsite "
        "install, removal), only the onsite phase is this SOW's: the other phases' steps (validate receipt, label equipment "
        "before shipment, record serials of benched equipment) are another party's and are neither tasks nor customer "
        "duties. A commercial term or emphasis a fact puts on a task (rework of Provider's own errors at no charge, "
        "\"critical\", \"assume all\", a requirement that holds even where it seems not to apply) stays with that task or "
        "becomes its own assumption; it is never dropped from a line that cites it.\n"
        "TASK: the task list. Return {\"scope_lead\": one line introducing the tasks (\"Provider will dispatch <N> (<n>) <trade> "
        "technicians to perform the following:\"; a count only when a fact states the staffing, never "
        "from meeting attendees), \"scope\": [group the tasks under short headings, one per area, system or room (an item "
        "{\"text\": \"<Heading>\", \"kind\": \"heading\", \"cites\": []} of at most six words, then that group's tasks; a device heading holds only the tasks that act on that device, its install task first, and steps common to every device go under a general heading); "
        "the tasks in the order the work is done: arrive and check in; each "
        "piece of work with its counts, lengths and locations kept together (\"<location>: approximately <N> ft\"); "
        "every device to be installed gets an install or mount task with its count, model and location; every count, "
        "length and model the job lines give appears in a task; routing, termination, mounting and testing; when runs end at an IDF or rack, terminate and connect them "
        "there; cleanup; completion confirmed with the onsite contact. When the request says assist the customer's "
        "staff, the tasks assist as directed. A manufacturer's install or setup guide tells how, not what: write the "
        "SOW-level work it supports (mount, connect, power, enroll or test the device, as the job needs), never its "
        "numbered steps, menus, settings or values. On a replacement job, removing the old unit is a task and where it "
        "goes (left onsite, boxed, returned) follows the facts; a replacement that is not one for one (two or more old units "
        "replaced by one new one, a consolidation) is its own task with both models, and a sequence a fact sets (check rack "
        "space before removing the old unit) is kept. A time the technician posts or records, a photo or a serial number the "
        "customer asks for is a deliverable, never a customer duty. A request the customer made to the reseller or to its own team (receiving, "
        "staging, shipping, procurement, configuration the reseller does) is not Provider work. Every written instruction about "
        "where or how a device is mounted, a cable is routed, hidden or covered is its own task with its location; a removal "
        "task names what its lines keep (\"except the HDMI adapter\"): never cite a line and leave out its exception. Never claim a route or a device-to-location pairing no fact gives], "
        "\"deliverables\": [{\"item\": \"Work Completion Report\", \"description\": [the finished outcomes with their "
        "counts, one for every device or system installed], \"format\": \"PDF\"}]}"),
    "supply": (
        "STANDARD PURTERA ITEMS FOR THIS KIND OF JOB (a purtera_standard_* list is PurTera's fixed handling of this job "
        "type: write every one of its items, reworded to this job, unless a fact contradicts it; the other lists: tools Provider brings; customer duties PurTera writes on such "
        "jobs; use those that fit THIS job's facts, never one the facts contradict): %s\n"
        "TASK: who supplies what. Tools must agree with the job's facts: no lift or scissor lift in the tools when a "
        "ladder is confirmed. A line that splits who supplies what (\"Provider will supply the backing board; the customer "
        "orders the rack\") gives each party its own item: write both halves. Return {\"provider_materials\": [materials Provider supplies, one item each, each "
        "citing the fact that assigns it], \"provider_tools\": [tools Provider brings; standard hand tools may be "
        "uncited], \"customer_materials\": [equipment and materials the customer or end customer supplies, each "
        "citing the fact that assigns it; an item on the customer's purchase order is customer-supplied], "
        "\"customer_responsibilities\": [duties a fact gives the customer or end customer, e.g. providing an "
        "escort, a staging area, packing materials, directions; never power or PoE, which is an assumption. A site "
        "rule the customer sets for Provider's technician (send the technician's name ahead, check in with security) "
        "is Provider's duty and belongs in the arrival protocol, never here; what the customer hands over for it "
        "(visitor badges, parking passes, an escort) IS a customer responsibility]}"),
    "bounds": (
        "STANDARD PURTERA ITEMS FOR THIS KIND OF JOB (a purtera_standard_* list is PurTera's fixed handling of this job "
        "type: write every one of its items, reworded to this job, unless a fact contradicts it; the other lists: assumptions and exclusions PurTera writes on such jobs; use "
        "every one that fits THIS job, reworded to it, never one that excludes work the draft includes or contradicts "
        "a fact): %s\n"
        "TASK: assumptions and exclusions, written against the DRAFT below so they never contradict it. An assumption is "
        "a plain statement of a condition (\"All cable runs are under 200 feet.\"), never an order to anyone; a duty "
        "of the customer belongs in customer responsibilities, not here. Every line keeps its cited fact's meaning: a fact that "
        "names a service or task (\"End-user support including O365 clients\") is requested work, never an exclusion or a "
        "\"not included\" assumption; exclude only what a fact itself excludes or limits. A fact from a "
        "figure or photo is a site condition only when it bears on the work (a mounting surface, a height); never "
        "describe what a photo shows. The template "
        "already prices after-hours, weekend and holiday work and travel, and already states business hours, "
        "billing and cancellation terms: never repeat or exclude those. Return {\"assumptions\": [5-10 conditions "
        "that bound the price: each one a fact settles (a ladder confirmed -> \"No lift is required.\" and "
        "\"Standard ladders reach every work location.\"; the longest run -> \"All cable runs are under N feet.\" with "
        "N the longest rounded up to 50; a PurTera person's own stated assumption, reworded as a SOW assumption; a replacement "
        "the facts say is not one for one at some sites -> \"Some sites may consolidate <old> into <new>.\") and "
        "the standard ones for THIS trade only, and a lift, ladder or pathway line only on ceiling, height or cabling work, never on desk or countertop work (cabling or devices: pathways accessible, PoE ports available on the "
        "IDF switch, equipment on site before arrival; decommission or packing: equipment powered down and released, "
        "the customer directs what is packed; any job: permits by the customer)], \"out_of_scope\": [5-9 exclusions as "
        "noun phrases: those a fact states and the standard ones for THIS trade (cabling or devices: lift rental only on ceiling or height work, "
        "electrical work, core drilling or structural work, patching or painting, switch or firewall work, licensing "
        "or cloud setup; decommission or packing: transport, shipping, disposal, data destruction or asset "
        "disposition, configuration, and never a cabling or construction exclusion on such a job); never anything the draft's scope includes, any equipment it lists as in scope, or what the customer is said to supply]}\n"
        "DRAFT: %s"),
}

_PARTY_PROMPT = (
    "Each numbered item is assigned to a party (Provider = PurTera, the field-services firm; Customer = the customer "
    "or end customer). Say which assignments the cited lines STATE: the line must say that party supplies, ships, "
    "orders, brings or already has the item (a heading such as \"we'd ship out:\" written by the customer; \"<reseller> "
    "to provide\" for Provider, since Provider delivers the reseller's work; \"tech to provide\"; an item ordered by "
    "the customer; a reseller's order or quote line to the customer, with ITEM / QTY / part numbers, is the customer's "
    "purchase, so Customer; a supplier-head tag). A line that only names the item does not. Return JSON only: "
    "{\"stated\": [item numbers]}.\n%s"
)

_COVER_PROMPT = (
    "Below is a PurTera SOW draft (JSON) and deal facts it does not use yet. For EVERY unused fact that a PM would "
    "put in the SOW (each device or item to install with its size, model and count; each material or tool someone "
    "brings; each site condition; each access rule (a count, length, location, model, access condition, date, staffing, who supplies what, a "
    "stated assumption or exclusion, a requested task), add ONE bullet to the list it belongs in, citing it. Skip "
    "facts that are chatter, already covered, about another job, or pricing. Every line in which someone states an "
    "assumption (\"assume ...\") becomes an assumption. A line from an answered question is scope when it asks for "
    "work (\"I would also like to add <X> - can you help?\"). Same voice and WHO rules as the draft "
    "(\"we\" under a customer heading is the customer; a reseller asked to provide materials means Provider "
    "supplies them). Return JSON only: {\"add\": [{\"list\": one of %s, \"text\": str, \"cites\": [ids], "
    "\"kind\": \"fact\"}]}. Return {\"add\": []} if nothing is missing.\nDRAFT: %s\nUNUSED FACTS:\n%s"
)

_MUST_PROMPT = (
    "These deal facts carry specifics (counts, sizes, models, products, materials, who supplies what) that the PurTera "
    "SOW draft below does not mention. Place EACH one where it belongs in the SOW, as one bullet citing it, unless it "
    "is about existing equipment that is not part of the new work or another job. A line saying what stays in place "
    "(displays that remain, an item kept during the removal) is an assumption or an exclusion; an instruction about how or "
    "where the work is done (move a cable behind the wall, no visible cables, mount the switch behind a display, cover "
    "the floor cable) is a scope task that keeps its location. A fact in section answered_question "
    "was answered: a request in it (\"I would also like to add <X> - can you help?\") is scope to write. Same "
    "WHO rules: \"we\" under a customer heading is the customer; a reseller asked to provide materials means Provider "
    "supplies them; \"tech to provide <X>\" means Provider supplies <X>. A commercial term or an emphasis the customer "
    "puts on the work is placed whole, where it binds, even when another line already cites the fact for its task: rework "
    "of Provider's own errors done at no charge is an assumption; work offered later at extra cost is out of scope; "
    "\"critical\" or \"assume all\" goes on the task it qualifies (\"Label every active cable; cable identification is a "
    "critical requirement\"); a requirement that holds where it seems not to apply (\"not all stores have a rack but "
    "still need rack mount hardware\") keeps that case; a step the fact gives inside a longer answer (cut and dress very "
    "long cable) is its own task. Return JSON only: {\"add\": [{\"list\": one "
    "of %s, \"text\": str, \"cites\": [ids], \"kind\": \"fact\"}]}.\nDRAFT: %s\nFACTS:\n%s"
)

#: A commercial term or emphasis a fact puts on the work: placed even when its
#: atom is cited for something else (010209 pass U: "this extra time must be
#: spent at no charge to RL" cited on "Label every cable", the term lost).
_TERM = re.compile(r"(?i)\b(at|for) no (charge|cost)\b|\bno[- ]charge\b|\b(at|for) (a|an additional|extra) (cost|charge|fee)\b|"
                   r"\bcritical\b|\bassume all\b|\bnot all\b[^.]{0,60}\bstill\b|\bcut\b[^.]{0,30}\bcable")
_TERM_WORDS = {"charge": r"no[- ]charge|no cost|at no", "cost": r"no cost|extra cost|additional cost|at a cost|separately",
               "critical": r"critical", "assume all": r"\ball\b", "still": r"no rack|without a rack|even where|still",
               "cut": r"\bcut\b|excess|long cable"}

_VERIFY_PROMPT = (
    "Each numbered SOW bullet cites deal lines. Say which bullets the cited lines support: every count, length, "
    "place, model and date in the bullet must be stated by a cited line ABOUT THE SAME THING (a bare \"34\" "
    "does not make a cable run 34 ft). A faithful rewording, "
    "a plain arithmetic total, and a standard way of doing exactly the work the lines describe count as "
    "supported; a count, length, place, model, date, party or condition the lines do not give, or the wrong "
    "party doing or supplying something, does not (\"he will need us to run the cabling\", written by the reseller, "
    "means Provider runs it: it does NOT support the customer supplying cable, nor cabling being excluded). Return JSON only: {\"supported\": [bullet numbers]}.\n%s"
)

LISTS = ("locations", "overview", "schedule", "arrival_protocol", "scope", "equipment_in_scope", "provider_materials",
         "provider_tools", "customer_materials", "customer_responsibilities", "assumptions", "out_of_scope")

_NUM_WORDS = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|"
              "hundred")
_NUM = re.compile(r"\b(\d[\d,./]*|(?:%s))\b" % _NUM_WORDS, re.I)
#: A staffing count ("two (2) cabling technicians"): only a fact may set it.
_STAFF = re.compile(r"\b(\d+|%s)\b(\s*\(\d+\))?\s+(\w+\s+)?(technicians?|techs?|resources?|installers?|engineers?|"
                    r"people|crew)\b" % _NUM_WORDS, re.I)
#: Words a method bullet may capitalise without a fact naming them.
_STOCK = frozenset("""provider customer the a an all any each this these those cat5 cat5e cat6 cat6a rj45 idf mdf
poe ups av it pos sow pm pmo onsite on-site usb hdmi vesa wi-fi wifi tv tvs ap aps lan wan ip coi""".split())


_NEIGHBOURS = {
    "cameras_security": ("low_voltage_cabling", "wireless_ap"),
    "pos_install_refresh": ("smart_hands_general", "imac_deskside"),
    "time_clock_kiosk": ("smart_hands_general", "low_voltage_cabling"),
    "network_switch_refresh": ("data_center", "smart_hands_general"),
    "decommission_packing": ("smart_hands_general", "relocation_move"),
    "relocation_move": ("data_center", "smart_hands_general"),
    "site_survey": ("wireless_ap",),
    "imac_deskside": ("smart_hands_general",),
    "wireless_ap": ("low_voltage_cabling",),
    "data_center": ("smart_hands_general",),
    "other": ("smart_hands_general",),
}
_BOILERPLATE = re.compile(r"(?i)business hours|cancel|reschedul|change order|out of scope\b.*not specified|not expressly|"
                          r"billed|invoice|travel|purchase order|\bPMO\b|standardized report|point of contact for the project")
LIBRARY_ITEMS = int(os.environ.get("SOWSMITH_C3_SOW_LIBRARY_ITEMS", "18"))


def _library() -> dict[str, Any]:
    try:
        return json.loads((Path(__file__).parent / "sow_method_library.json").read_text())
    except Exception:
        return {}


def purtera_standards(family: str) -> dict[str, list[str]]:
    """PurTera's fixed handling of a job type (purtera_standards.json): items a
    PM writes on every job of the type whether or not the deal states them.
    Template content, owner-reviewed, so not subject to the two-deal rule."""
    try:
        data = json.loads((Path(__file__).parent / "purtera_standards.json").read_text())
    except Exception:
        return {}
    # A family with no block of its own takes the generic smart-hands block
    # when that is its neighbour, as the method library does (010419 pass V:
    # a generic "other" call wrote an IT transition with no standards at
    # all). Never a specific trade's block: wireless_ap's neighbour is
    # cabling, whose standards wrote cable runs into 010344, an AP job that
    # excludes new structured cabling.
    fam = data.get(family) or (data.get("smart_hands_general") or {} if "smart_hands_general" in _NEIGHBOURS.get(family, ()) else {})
    return {k: [str(t) for t in v] for k, v in fam.items() if not k.startswith("_") and isinstance(v, list)}


_WHEN = re.compile(r"^\{when:(.*?)\}\s+(?=[A-Z])")  # the regex may hold {n,m}: end at "} " before the item


def _standard_applies(t: str, facts_text: list[str], unless_text: list[str] | None = None) -> tuple[bool, str]:
    """A standard item may carry "{when:<regex>}": written only when a deal fact matches."""
    # "{unless:<regex>}": skipped when a deal fact matches (010198: config
    # exclusions beside the router setup the job does).
    u = re.match(r"^\{unless:(.*?)\}\s+(?=[A-Z{])", t)
    if u:
        if any(re.search(u.group(1), f, re.I) for f in (facts_text if unless_text is None else unless_text)):
            return False, t[u.end():]
        t = t[u.end():]
    m = _WHEN.match(t)
    if not m:
        return True, t
    pat = re.compile(m.group(1), re.I)
    # "existing" is not a keep word: "reuse the existing mounts" is exactly
    # what the backing and inspect-mount items are for (010217, 000123).
    keep_words = re.compile(r"(?i)\b(stay|stays|remain|remains|no change|already|left in place)\b")
    return any(pat.search(f) and not keep_words.search(f) for f in facts_text), t[m.end():]


LIBRARY_PER_SECTION = {"tools": 6}
LIBRARY_DEFAULT = int(os.environ.get("SOWSMITH_C3_SOW_LIBRARY_PICK", "8"))


def _relevant(items: dict[str, list[str]], query: list[str]) -> dict[str, list[str]]:
    """The standard items closest to this deal's own job lines: a camera job
    took "standard key door locks are precut" from a cabling SOW when every
    item of its family was offered."""
    if not query:
        return {k: v[:LIBRARY_PER_SECTION.get(k, LIBRARY_DEFAULT)] for k, v in items.items()}
    try:
        import numpy as np

        from app.core.embedding_retrieval import _embed_azure

        flat = [(k, t) for k, v in items.items() for t in v]
        vecs = _embed_azure([t for _, t in flat] + query[:40])
        q = [v for v in vecs[len(flat):] if v]
        if not q:
            raise ValueError("no query vectors")
        Q = np.array(q, dtype=np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
        scored: dict[str, list[tuple[float, str]]] = {}
        for (k, t), v in zip(flat, vecs[:len(flat)]):
            if not v:
                continue
            x = np.array(v, dtype=np.float32)
            x /= np.linalg.norm(x) + 1e-9
            scored.setdefault(k, []).append((float((Q @ x).max()), t))
        return {k: [t for _, t in sorted(v, reverse=True)[:LIBRARY_PER_SECTION.get(k, LIBRARY_DEFAULT)]]
                for k, v in scored.items()}
    except Exception:
        return {k: v[:LIBRARY_PER_SECTION.get(k, LIBRARY_DEFAULT)] for k, v in items.items()}


def standard_items(family: str, deal_no: str = "", query: list[str] | None = None) -> dict[str, list[str]]:
    """PurTera's standard items for a job family, from its own past SOWs (each
    item written on two or more deals): the method steps, tools and terms a PM
    writes from experience, which no deal line states. An item only this deal's
    own SOW wrote is left out, so a deal is never written from its own answer."""
    lib = _library()
    out: dict[str, list[str]] = {}
    for fam in (family,) + _NEIGHBOURS.get(family, ()):
        for sec, items in (lib.get(fam) or {}).items():
            for it in items:
                deals = [d for d in it.get("deals") or [] if d != deal_no]
                if not deals or _BOILERPLATE.search(it["text"]):
                    continue
                lst = out.setdefault(sec, [])
                if it["text"] not in lst and len(lst) < LIBRARY_ITEMS * 2:
                    lst.append(it["text"])
    return _relevant(out, list(query or []))


# A SOW is sent to customers and resellers; a credential never belongs in one.
# 010264's Wi-Fi setup doc gave the writer a clock maintenance password and an
# SSID, and both were printed under Testing and Closeout. The value is withheld
# before the writer sees the line, and any item still carrying one is dropped.
_SECRET = re.compile(r"(?i)\b(pass(?:word|code|phrase)?|pwd|pin|ssid|wpa2?(?:[- ]?psk)?|psk|network key|"
                     r"security key|api[- ]?key|secret|token|username|user ?id|login)\b(\s*(?:is|=|:|-)?\s*)"
                     r"([^\s,;]+)")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")
_WORD = re.compile(r"(?i)[a-z]+[.,;:]?")


def _is_secret(m: re.Match) -> bool:
    """A value follows the keyword: after a colon/equals, or one with a digit
    or symbol in it ("password 8812ab"); "password reset" is a task, not one."""
    sep, val = m.group(2).strip(), m.group(3)
    if val.startswith("["):
        return False
    if sep in (":", "=") or sep.lower() == "is" and not _WORD.fullmatch(val):
        return True
    return bool(re.search(r"\d|[^A-Za-z.,;:'\")-]", val))


def _withhold(text: str) -> str:
    text = _SECRET.sub(lambda m: m.group(1) + m.group(2) + "[withheld]" if _is_secret(m) else m.group(0), text)
    return _IP.sub("[address withheld]", text)


def _scrub(content: dict[str, Any]) -> int:
    """Drop every item that still carries a credential or an address."""
    n = 0
    if isinstance(content.get("writer_draft"), dict):  # the writer's raw lists carry what it was given
        _scrub(content["writer_draft"])  # not counted: the SOW never shows it
    for k, v in list(content.items()):
        if not isinstance(v, list):
            continue
        keep = []
        for x in v:
            t = str(x.get("text") if isinstance(x, dict) else x)
            if "[withheld]" in t or "address withheld" in t or _IP.search(t) or \
                    any(_is_secret(m) for m in _SECRET.finditer(t)):
                n += 1
                continue
            keep.append(x)
        content[k] = keep
    return n


_SCOPE_TYPES = {"scope_item", "task", "dependency", "deliverable", "exclusion", "site_implementation_note",
                "requirement", "decision", "acceptance_criterion", "work_scope_item"}
# Email chatter the type head can still read as work ("We'll get back to you",
# "Please advise any questions by Wednesday").
_CHATTER = re.compile(r"(?i)^(we('| a)re (on|good)|we('|wi)ll (get|send|be|let|have|usually)|i('|wi)ll |please |let me know|"
                      r"once (we|i) |thanks|thank you|reminder|i am free|we can do a call)|\?\s*$")
TYPED_BAR = float(os.environ.get("SOWSMITH_C3_SOW_TYPED_BAR", "0.5"))
#: 25 cut 010326's ATA split ("we would be handling phones on the desks, ATAs,
#: wall mounting phones", _keep 0.42) behind sheet cells and table rows.
TYPED_MAX = int(os.environ.get("SOWSMITH_C3_SOW_TYPED_MAX", "40"))


#: A form field or note that records that something was not said: "Photo
#: requirements mentioned: None mentioned", "Paying contact: Unknown". A count
#: the request says is not known stays ("Hardware Quantity: Not provided").
_ABSENCE = re.compile(r"(?i)^(?!.*\b(quantity|count|qty)\b)[^|:]{2,60}:\s*(none( mentioned| specified| provided| stated| given)?|unknown|n/?a|"
                      r"not (mentioned|specified|provided|stated|given|known))\s*\.?\s*$")
#: The same said as a SOW line: "No photo documentation requirements were
#: specified." contradicted the photo deliverable beside it (010082).
_ABSENCE_LINE = re.compile(r"(?i)^\s*(no\b.{3,80}\b(were|was|are|is|have been|has been)\s+(specified|mentioned|provided|stated|given|identified)|"
                           r".{3,60}\b(is|are) (unknown|not known|not specified))(\s+(for|in|on|at|from) [^.]{0,40})?\s*\.?\s*$")
_CHATTER_TYPES = ("small_talk", "signature")


def _rejected_atoms(envelope: dict[str, Any]) -> set[str]:
    """Atoms that are not facts of the job whatever section the brief put them
    in: typed small talk or signature (the type head's label, else the
    parser's type), a parser admission reject, and the items of a list whose
    other items are mostly chatter (010082's signature menu "Need help with:
    Firewall Request / ISP Upgrade / Network Consultation/New Installation /
    New Unit Opening": the one item typed scope_item became a Provider task)."""
    out: set[str] = set()
    lines: dict[tuple, dict[int, list[bool]]] = {}
    items: list[tuple[str, tuple, int]] = []
    for a in envelope.get("atoms") or []:
        st = a.get("structured") or {}
        lab = str((st.get("c3_label") or {}).get("type") or "")
        chat = lab in _CHATTER_TYPES or (not lab and str(a.get("atom_type") or "") in _CHATTER_TYPES) \
            or bool(st.get("admission_reject"))
        if chat:
            out.add(str(a.get("id")))
        loc = a.get("locator") or {}
        ln = loc.get("line_start")
        if not isinstance(ln, int):
            continue
        key = (a.get("artifact_id"), loc.get("message_index"))
        lines.setdefault(key, {}).setdefault(ln, []).append(chat)
        if not chat and (loc.get("lead_in") or st.get("lead_in") or st.get("list_item")):
            items.append((str(a.get("id")), key, ln))
    # A list item among chatter is chatter: its neighbours within three lines
    # are mostly small talk or signature.
    for i, key, ln in items:
        near = [c for l2, cs in lines[key].items() if l2 != ln and abs(l2 - ln) <= 3 for c in cs]
        if len(near) >= 2 and sum(near) >= 0.6 * len(near):
            out.add(i)
    return out


#: A written instruction about the work or what stays: "should be moved behind
#: wall", "No cables should be visible after mount", "These 4 TVs to Stay in Place".
_DIRECTIVE = re.compile(r"(?i)\b(should|will need to|needs? to|must|has to|have to)\s+(also\s+)?(be\s+)?(moved|mounted|installed|run|placed|hidden|"
                        r"routed|relocated|replaced|removed|visible|concealed|reused|covered)\b|\bno cables? (should|will) be visible|"
                        r"\b(stay|stays|remain|remains) (in place|as is|mounted)\b|\bto (stay|remain) in place\b")


def facts(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    """The deal's facts: every line a head placed in a job section, plus the
    lines kept with less certainty and the figures read off its images."""
    brief = envelope.get("c3_brief") or {}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    secs = brief.get("sections") or {}
    for sec in [k for k in secs if k not in NOT_FACT_SECTIONS]:
        for x in secs.get(sec) or []:
            if x["id"] in seen:
                continue
            seen.add(x["id"])
            out.append({"id": x["id"], "section": sec, "when": x.get("when") or "", "under": x.get("under") or "",
                        "text": str(x["text"]), "supplier": x.get("supplier"), "type": x.get("type") or ""})
    # A line the keep head rejected at a coin flip has not been decided: the
    # SOW is where a dropped cable run becomes an unpriced one (010043's
    # "IDF to Front door 146ft", reject 0.53), so the writer sees it, marked.
    unsure = [{"id": x["id"], "text": x["text"]} for x in brief.get("label_next") or []
              if x.get("head") == "read:co_action" and x.get("base_says") == "reject"
              and float(x.get("p") or 1) < UNSURE_BAR]
    # And a line the keep head chose to keep below the brief's bar (010171's
    # Logitech Sight request, keep 0.39): the head said keep; the SOW sees it.
    unsure += [{"id": x["id"], "text": x["text"]} for x in brief.get("kept_unsure") or []]
    # A question the deal already answered is not an open question, so the
    # brief leaves it out; for a SOW it is often the request itself: 010198's
    # "I would also like to add a Ubiquiti router with cellular backup - is
    # that something you can help with?" (the type head: answered_question).
    answered = [{"id": a["id"], "text": a.get("text") or ""} for a in envelope.get("atoms") or []
                if ((a.get("structured") or {}).get("c3_label") or {}).get("type") == "answered_question"
                and a.get("id") not in seen and len(str(a.get("text") or "")) > 25]
    # The type head's own scope reading, for a line the brief's keep head left
    # out: that head picks what a short brief says, not what a SOW must carry.
    # 010279's "We are pricing physical installation labor only"
    # (site_implementation_note 0.54), 010376's "get them onto a pallet and
    # assist the driver" (scope_item 0.66). Documents only, not call
    # transcripts, where scope words are mostly talk.
    docs = {str(d.get("artifact_id") or d.get("id") or ""): str(d.get("filename") or "")
            for d in envelope.get("documents") or []}
    typed = []
    for a in envelope.get("atoms") or []:
        lab = (a.get("structured") or {}).get("c3_label") or {}
        txt = str(a.get("text") or "")
        # When the type head abstains (_keep), the parser's own scope type
        # stands at a looser bar: 010171's "We would like to have the Logitech
        # Sight installed in this specific room" (_keep 0.39, parsed scope_item).
        abstained = lab.get("type") == "_keep" and str(a.get("atom_type") or "") in ("scope_item", "task", "requirement")
        if (a.get("id") in seen or (lab.get("type") not in _SCOPE_TYPES and not abstained)
                or float(lab.get("p") or 0) < (0.35 if abstained else TYPED_BAR)
                or not 15 < len(txt) < 600 or _CHATTER.search(txt.strip()) or txt.count("|") >= 2
                or re.search(r"(?i)unsubscribe|book time|my calendar|confidential|don.t often get email|"
                             r"external sender|contents of this email|intended only for|^(start|end|recorded-at|url|"
                             r"join-url|location|title|organizer|attendees|subject|from|to|sent)\s*:|https?://|"
                             r"^(hey|hi|hello|thanks|happy to|u the best|looping in)\b", txt.strip())
                or re.search(r"(?i)^\[ocr|^enter .* here$|\|\s*$|rolled up\)|^\d+(\.\d+)*\s+[A-Z]", txt.strip())
                # A table row ending in a count cell ("Zoo (1215 N 8th St ...) | 1")
                # is a site-list row, which reaches the SOW as the site table; as
                # typed-only lines 010326's phone-count rows took the slots of the
                # RedSky migration decision and the ATA split.
                or re.search(r"\|\s*\d{1,5}\s*$", txt) or re.match(r"(?i)\s*transcript[- ]id\s*:", txt)
                or (abstained and len(txt) < 25) or "fireflies" in docs.get(str(a.get("artifact_id") or ""), "")
                or "transcript" in docs.get(str(a.get("artifact_id") or ""), "").lower()):
            continue
        typed.append((float(lab.get("p") or 0), {"id": a["id"], "text": txt}))
    typed = [x for _, x in sorted(typed, key=lambda t: -t[0])[:TYPED_MAX]]
    # A written instruction about the work is the job whatever its type label:
    # 010120's survey slides ("this should be moved behind wall to not be
    # visible", "No cables should be visible after mount") never reached the
    # writer (types _keep / assumption, below every bar).
    directive = _DIRECTIVE
    got = {x["id"] for x in typed}
    for a in envelope.get("atoms") or []:
        txt = str(a.get("text") or "")
        name = docs.get(str(a.get("artifact_id") or ""), "").lower()
        if (a.get("id") in seen or a.get("id") in got or not 15 < len(txt) < 600 or not directive.search(txt)
                or "fireflies" in name or "transcript" in name or (a.get("structured") or {}).get("quoted")
                or txt.rstrip().endswith("?")):
            continue
        typed.append({"id": a["id"], "text": txt})
        got.add(a["id"])
    for sec, items in (("also_kept", brief.get("also_mentioned") or []), ("figure", brief.get("figures") or []),
                       ("unsure", unsure), ("answered_question", answered[:40]), ("typed_only", typed)):
        for x in items:
            if x["id"] in seen:
                continue
            seen.add(x["id"])
            out.append({"id": x["id"], "section": sec, "when": x.get("when") or "", "under": "",
                        "text": str(x["text"]), "supplier": x.get("supplier")})
    # A line the type head or the parser rejected is never a fact, wherever
    # the brief filed it; nor is a form's absence note (010082: a signature's
    # "New Unit Opening" (small_talk, confidence 0) reached the summary, and
    # "Photo requirements mentioned: None mentioned" became an assumption).
    rejected = _rejected_atoms(envelope)
    out = [f for f in out if f["id"] not in rejected and not _ABSENCE.search(f["text"])]
    # The brief keeps a short copy of each line; the writer reads the whole one.
    full = {a.get("id"): str(a.get("text") or "") for a in envelope.get("atoms") or []}
    # A line a vision pass read off a photo or plan describes a picture; it is
    # context, never something the coverage passes must place (010043's
    # "mat with a red lane logo" became a scope task).
    pictured = {a.get("id") for a in envelope.get("atoms") or []
                if (a.get("structured") or {}).get("kind") in ("image_fact", "image_ocr_text")}
    quoted = {a.get("id") for a in envelope.get("atoms") or [] if (a.get("structured") or {}).get("quoted")}
    asked = {a.get("id") for a in envelope.get("atoms") or []
             if ((a.get("structured") or {}).get("c3_label") or {}).get("type") in ("open_question", "answered_question")
             or str(a.get("text") or "").rstrip().endswith("?")}
    for f in out:
        f["pictured"] = f["id"] in pictured or f["section"] == "figure"
        f["quoted"] = f["id"] in quoted
        f["asked"] = f["id"] in asked
    # The heading a line was written under says whose voice "we" is: 000123's
    # "We will preconfigure all items" sat under "Customer notes:" and became a
    # Provider task because the brief kept no heading for it.
    label = {a.get("id"): str((a.get("structured") or {}).get("list_label")
                              or (a.get("structured") or {}).get("parent_field") or "").strip()
             for a in envelope.get("atoms") or []}
    for f in out:
        if not f["under"] and label.get(f["id"]):
            f["under"] = label[f["id"]]
    for f in out:
        if len(full.get(f["id"], "")) > len(f["text"]):
            f["text"] = full[f["id"]]
        f["text"] = _withhold(f["text"])
    _lead_ins(out, envelope.get("atoms") or [])
    _list_siblings(out, envelope.get("atoms") or [])
    rank = {k: i for i, k in enumerate(SECTION_ORDER)}
    seen_text: set[str] = set()
    uniq = []
    for f in sorted(out, key=lambda f: rank.get(f["section"], len(rank))):
        key = re.sub(r"\W+", " ", f["text"].lower()).strip()
        if key in seen_text:
            continue
        seen_text.add(key)
        uniq.append(f)
    return uniq[:MAX_FACTS]


_LEAD_IN = re.compile(r":\s*$")


def _lead_ins(fs: list[dict[str, Any]], atoms: list[dict[str, Any]]) -> None:
    """A line with no heading gets the list lead-in above it in its document
    ("...overall we'd ship out:"), read for context and never cited: without
    it 000123's "Associated mounting hardware and cables" read as Provider's
    to supply, when the customer ships it."""
    by_art: dict[str, list[tuple[int, str]]] = {}
    where: dict[str, tuple[str, int]] = {}
    for a in atoms:
        loc = a.get("locator") or {}
        ln = loc.get("line_start")
        if not isinstance(ln, int) or not a.get("artifact_id"):
            continue
        by_art.setdefault(a["artifact_id"], []).append((ln, str(a.get("text") or "")))
        where[a["id"]] = (a["artifact_id"], ln)
    for v in by_art.values():
        v.sort()
    for f in fs:
        if f["id"] not in where:
            continue
        art, ln = where[f["id"]]
        # The two nearest lead-ins: "Customer notes:" above "...we'd ship
        # out:" says whose list it is.
        heads = [t.strip() for _, t in by_art[art] if ln - 12 <= _ < ln and _LEAD_IN.search(t.strip())
                 and len(t.strip()) < 200]
        if not heads:
            continue
        if not f["under"]:
            f["under"] = " > ".join(h[:120] for h in heads[-2:])
        elif len(f["text"]) <= 90 and heads[-1][:120] not in f["under"] and len(heads[-1]) > len(f["under"]):
            # A list label ("Customer notes:") hides the sentence that opens
            # the list (000123: "Cords for the above" under "...overall we'd
            # ship out:" read as anyone's item).
            f["under"] = f["under"][:120] + " > " + heads[-1][:120]


# The type head's label, where it settles whose duty a line is.
_TYPE_ROLE = {"dependency": "type head: a prerequisite or condition",
              "exclusion": "type head: excluded", "deliverable": "type head: a deliverable",
              "acceptance_criterion": "type head: an acceptance criterion",
              "pricing_assumption": "type head: a pricing assumption",
              "milestone_phase": "type head: a dated milestone of this job"}


def _fact_line(f: dict[str, Any]) -> str:
    sup = f.get("supplier")
    who = (f" supplied-by:{_SUPPLIER.get(sup[0], sup[0])}" if sup and float(sup[1] or 0) >= SUPPLIER_BAR
           and sup[0] in _SUPPLIER and f.get("type") in _ITEM_TYPES else "")
    role = _TYPE_ROLE.get(f.get("type") or "")
    return (f"[{f['id']}] ({f['section']}" + (f", {role}" if role else "") + (", from a photo" if f.get("pictured") else "")
            + (", quoted earlier email" if f.get("quoted") else "") + (", a question" if f.get("asked") else "")
            + (f", {f['when']}" if f["when"] else "") + who + ") "
            + (f"[under: {f['under']}] " if f["under"] else "") + _dated(f["text"], f["when"] or None)[:FACT_CHARS])


_MEASURED = re.compile(r"(?i)^\s*(.{2,40}?)\s*[:\-–]?\s*(?:approx(?:imately)?\.?\s*)?(\d[\d,.]*)\s*(ft|feet|foot|'|m|meters?|units?|x)\b")


def _list_siblings(out: list[dict[str, Any]], atoms: list[dict[str, Any]]) -> int:
    """A line of the same measured list as a kept line shares its fate. Of
    010043's "IDF to side door 154ft / IDF to back door 88ft / IDF to Front
    door 146ft", the keep head kept one and rejected two at a coin flip, so the
    SOW priced one cable run of three. Same document, same opening words, a
    measurement: the sibling joins its kept line's section."""
    by_id = {a.get("id"): a for a in atoms}
    weak = ("unsure", "also_kept", "typed_only")
    # A BOM row ("1: QM75C Samsung 75IN ...", "2 | TS525TU | Chief ...") is a
    # list member too: 010116's matrix switcher and serial cable were rejected
    # beside identical kept rows of the same block.
    bom = re.compile(r"^\s*\d{1,4}\s*[:|x]\s*(?:\d{1,4}\s*\|\s*)?\|?\s*[A-Z0-9][A-Z0-9./_-]{3,30}\b")
    def shape(t: str):
        m = _MEASURED.match(t)
        if m:
            return " ".join(m.group(1).lower().split()[:2])
        return "bom-row" if bom.match(t) else None
    _MEASURED_OR_BOM = type("_M", (), {"match": staticmethod(lambda t: shape(str(t)))})
    key_of = lambda f: (str((by_id.get(f["id"]) or {}).get("artifact_id") or ""), shape(str(f["text"])))
    kept = {}
    for f in out:
        if shape(str(f["text"])) and f["section"] not in weak:
            kept.setdefault(key_of(f), f)
    if not kept:
        return 0
    n = 0
    for f in out:
        if f["section"] in weak and shape(str(f["text"])) and key_of(f) in kept:
            f["section"] = kept[key_of(f)]["section"]  # an unsure sibling is decided by its kept one
            n += 1
    have = {f["id"] for f in out}
    for a in atoms:
        if a.get("id") in have:
            continue
        t = str(a.get("text") or "")
        if not shape(t):
            continue
        sib = kept.get((str(a.get("artifact_id") or ""), shape(t)))
        if sib:
            out.append({"id": a.get("id"), "section": sib["section"], "when": sib.get("when") or "",
                        "under": sib.get("under") or "", "text": t, "supplier": sib.get("supplier"),
                        "type": sib.get("type") or ""})
            have.add(a.get("id"))
            n += 1
    return n


_RESELLER_SHIPS = re.compile(r"(?i)\b(cdw|shi|insight|presidio|connection|zones|the reseller|distributor)\b[^.]{0,60}?"
                             r"\b(acquire|ship|provid|order|procur|configur|supply|supplies|deliver)")
_HARDWARE = re.compile(r"(?i)\b(switch(es)?|router|firewall|access points?|aps?\b|power suppl|power cords?|stack(ing)? cables?|"
                       r"sfps?|optics?|rack ears?|transceivers?|licen[cs]|cameras?|phones?|displays?|tvs?|ups\b|batter(y|ies)|"
                       r"mounts?|monitors?|screens?|soundbars?|speakers?|printers?|tablets?)")
_NO_RACK = re.compile(r"(?i)\b(no (dedicated )?rack|without a rack|not enough rack|rack space (may|might) not|may need .{0,20}rack)")


def _reseller_supplies(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """When a fact says the reseller acquires/ships/configures the equipment,
    the hardware is the customer's to furnish, never Provider's materials
    (010083: CDW ships 194 switches, the SOW listed switches, power supplies
    and SFPs as technician-provided). And an assumption a fact contradicts
    goes ("sufficient rack space" where sites have no rack)."""
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    # One item, one party: what the customer side furnishes is not also
    # Provider's (010217 listed the mounts under both).
    cm = content.get("customer_materials") or []
    content["provider_materials"] = [x for x in content.get("provider_materials") or [] if not _said(tx(x), cm, 0.6)]
    we_provide = any(re.search(r"(?i)\bwe (are|will|'ll|will be) (be )?(providing|provide|supplying|ship)", f["text"])
                     and _HARDWARE.search(f["text"] + " display mount") for f in fs)
    # The customer ships the gear ("we'd ship out:", "provides all hardware"):
    # its cables and mounting hardware come with it (000123 put "associated
    # mounting hardware / cables" on Provider with no owner stated).
    if any(re.search(r"(?i)\bwe'?d ship( out)?\b|we (will|'ll) ship( out)?\b|provides? all (the )?(hardware|equipment)", f["text"]) for f in fs):
        pm = content.get("provider_materials") or []
        assoc = [x for x in pm if re.search(r"(?i)associated|mounting hardware|\bcables?\b|\bcords?\b", tx(x)) and not (isinstance(x, dict) and x.get("cites"))]
        if assoc:
            content["provider_materials"] = [x for x in pm if x not in assoc]
            content.setdefault("customer_materials", []).extend(assoc)
    if we_provide or any(_RESELLER_SHIPS.search(f["text"]) for f in fs):
        pm = content.get("provider_materials") or []
        hw = [x for x in pm if _HARDWARE.search(tx(x))]
        if hw:
            content["provider_materials"] = [x for x in pm if x not in hw]
            content.setdefault("customer_materials", []).extend(hw)
    if any(_RESELLER_SHIPS.search(f["text"]) for f in fs):
        # A customer's request to the reseller ("receive the switches at central
        # receiving and schedule shipment to each site") is not Provider work
        # once the reseller says it ships everything (010083).
        for k in ("scope", "overview", "deliverables", "assumptions"):
            content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and not x.get("standard")
                          and re.search(r"(?i)central(ly)? receiv|schedule(d)? (the )?ship|receive,? schedule,? (and )?ship|receive the .{0,30}(switch|equipment|hardware)",
                                        str(x.get("text") or x.get("item") or "")))]
    if any(_NO_RACK.search(f["text"]) for f in fs):
        content["assumptions"] = [a for a in content.get("assumptions") or []
                                  if not re.search(r"(?i)sufficient rack|rack space is available|no rack modification", tx(a))]


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_NUMW = "(?:" + "|".join(sorted(set(_ONES + _TENS[2:] + ["hundred"]), key=len, reverse=True)) + ")"
#: A spelled count, compound or not ("forty-eight", "one hundred sixty").
_NUM_RUN = re.compile(r"\b%s(?:[\s-]+%s)*\b" % (_NUMW, _NUMW), re.I)


def _digits(t: str) -> str:
    """Every spelled count as its digits, once: "forty-eight (48)" -> "48"."""
    t = re.sub(r"(?i)(%s)\s*\((\d+)\)" % _NUM_RUN.pattern,
               lambda m: m.group(2) if str(_words_value(m.group(1))) == m.group(2) else m.group(0), t)
    return _NUM_RUN.sub(lambda m: str(_words_value(m.group(0))), t)


def _words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _words(n % 100))
    return str(n)


def _words_value(t: str) -> int | None:
    """The number a run of number words spells ("one hundred sixty" -> 160), else None."""
    words = re.findall(r"[a-z]+", t.lower())
    total, cur, seen = 0, 0, False
    for w in words:
        if w in _ONES:
            cur += _ONES.index(w); seen = True
        elif w in _TENS:
            cur += 10 * _TENS.index(w); seen = True
        elif w == "hundred":
            cur = max(cur, 1) * 100; seen = True
        elif seen:
            total, cur, seen = 0, 0, False  # number words ended before; restart on the last run
    return total + cur if seen else None


def _sites(envelope: dict[str, Any], pm_handoff: dict[str, Any] | None) -> list[dict[str, str]]:
    rows = (pm_handoff or {}).get("sites") or []
    out = []
    for s in rows:
        street, city, st, zp = (str(s.get(k) or "").strip() for k in ("address", "city", "state", "postal_code"))
        tail = ", ".join(x for x in (city, " ".join(x for x in (st, zp) if x)) if x)
        full = ", ".join(x for x in (street, tail) if x) if street and street.lower() not in tail.lower() else tail
        out.append({"name": str(s.get("name") or ""), "city": city, "address": full or str(s.get("name") or ""),
                    "street": street, "state": st, "postal_code": zp})
    return out


def sow_model() -> str | None:
    """``SOWSMITH_C3_SOW_MODEL``: the deployment the SOW writer and its checks
    use (dev: gpt-5.4-mini); unset, the teacher model."""
    return os.environ.get("SOWSMITH_C3_SOW_MODEL", "").strip() or None


#: Writer list -> the read:sow_section answer a line placed there stands for.
_LIST_SECTION = {"overview": "overview", "scope": "scope_of_work", "arrival_protocol": "scope_of_work",
                 "equipment_in_scope": "scope_of_work", "customer_responsibilities": "customer_responsibilities",
                 "customer_materials": "customer_responsibilities", "assumptions": "assumptions",
                 "out_of_scope": "out_of_scope", "provider_materials": "provider_responsibilities",
                 "provider_tools": "provider_responsibilities", "schedule": "timeline"}


def _snap(content: dict[str, Any]) -> dict[tuple[str, ...], tuple[str, str]]:
    """Each cited line, keyed by its cites, with the list it sits in and its text."""
    out: dict[tuple[str, ...], tuple[str, str]] = {}
    # The summary last: a task citing what a summary sentence cites is the
    # line that matters (it was invisible to the drop guard behind it).
    for k in sorted(_LIST_SECTION, key=lambda k: k == "overview"):
        for x in content.get(k) or []:
            if isinstance(x, dict) and x.get("cites") and x.get("kind") != "heading" and not x.get("standard"):
                out.setdefault(tuple(x["cites"]), (k, str(x.get("text") or "")))
    return out


#: Every kind of decision a rule (or a PM in SowSmith) makes, and the super head
#: it trains. One row per kind: SowSmith's Teach queue, the PM feedback route
#: (app/core/pm_feedback.py) and ml.c3.labels_from_sow all read this table, so a
#: new kind of fix is one row here plus its mirror in headCorrections.ts.
TEACHABLE: dict[str, tuple[str, dict[str, str] | None]] = {
    "sow_section": ("read:sow_section", None),                      # which SOW section
    "sow_keep": ("read:co_action", {"keep": "keep", "drop": "reject"}),  # belongs in the SOW at all
    "sow_party": ("read:task_owner_role", None),                    # technician vs customer does it
    "sow_supplier": ("col:supplier", None),                         # us / customer / partner supplies it
}
_PARTY = {"scope": "technician", "arrival_protocol": "technician", "customer_responsibilities": "customer"}
_SUPPLIER_LIST = {"provider_materials": "us", "provider_tools": "us", "customer_materials": "customer"}


def _decisions(rule: str, before: dict, after: dict) -> list[dict[str, Any]]:
    """What a rule decided, as head answers a PM can confirm or reject: a cited
    line moved to another list is a read:sow_section call on its atoms, a cited
    line removed is a sow_keep drop. These are the labels the heads must learn
    before the rule can be retired (owner rule: heads decide)."""
    out = []
    for cites, (lst, text) in before.items():
        now = after.get(cites)
        if now is None:
            out.append({"rule": rule, "head": "sow_keep", "atom_ids": list(cites), "value": "drop",
                        "was": _LIST_SECTION[lst], "text": text[:300]})
        else:
            if _LIST_SECTION[now[0]] != _LIST_SECTION[lst]:
                out.append({"rule": rule, "head": "sow_section", "atom_ids": list(cites), "value": _LIST_SECTION[now[0]],
                            "was": _LIST_SECTION[lst], "text": now[1][:300]})
            if lst in _PARTY and now[0] in _PARTY and _PARTY[lst] != _PARTY[now[0]]:
                out.append({"rule": rule, "head": "sow_party", "atom_ids": list(cites), "value": _PARTY[now[0]],
                            "was": _PARTY[lst], "text": now[1][:300]})
            if lst in _SUPPLIER_LIST and now[0] in _SUPPLIER_LIST and _SUPPLIER_LIST[lst] != _SUPPLIER_LIST[now[0]]:
                out.append({"rule": rule, "head": "sow_supplier", "atom_ids": list(cites), "value": _SUPPLIER_LIST[now[0]],
                            "was": _SUPPLIER_LIST[lst], "text": now[1][:300]})
    return out


#: Rules whose drops are guarded: they may not remove the only line that
#: cites a deal fact (010171 lost its network drops and floor outlets to
#: "standards" and its passthrough routing to "tidy"; 010198 its register
#: install and 010279 its RFP task line to "grounding").
GUARDED_RULES = frozenset({"standards", "tidy", "grounding"})
#: A possible follow-up visit (010419: "some follow-up needed the next day"
#: on a one-day price; "A second day may be needed").
_FOLLOW_UP = re.compile(r"(?i)follow-?up (visit|day)|follow[- ]?up\b.{0,30}\bnext day|next[- ]day (visit|follow)|return (visit|trip)|"
                        r"second (visit|day)\b|come back|additional (visit|day) (may|might|could) be")
#: Lines that are noise whatever they cite: a greeting, PurTera's pitch, a
#: leasing brochure, a picture, presales paperwork, lease terms, a hedged
#: follow-up or a priced maybe. A rule's drop of such a line stands.
_NOISE_TEXT = re.compile(
    r"(?i)^\s*(hi|hello|hey|good (morning|afternoon)|hope (you|all|everyone)|thanks|thank you|happy)\b|hope you'?re|both doing well|"
    r"\bvetted\b|\d+\+? countries|engineers across|our capabilities|global (network|footprint)|handholding|hand-holding|"
    r"deep familiarity|nationwide coverage|we pride|white[- ]glove|leasing (team|agent|office)|for lease|\brsf\b|rentable|"
    r"\battached (floor ?plan|drawing|diagram|layout|map)|complete the sow|for the sow|to quote|for (the )?quot(e|ing)|"
    r"pricing purposes|residual value|lease term|financing (terms|agreement)|depending on (cost|price|budget)|"
    r"for an additional (cost|fee)|follow-?up (visit|day)|return (visit|trip)|come back|\bbench(ed|ing)\b|"
    r"^\s*(the )?[\w ()#-]{0,40}\b(contains|features|is equipped with|is furnished with|is shown with)\b")


def _guard_drops(content: dict[str, Any], rule: str, before: dict, lists_before: dict[str, list]) -> list[dict[str, Any]]:  # noqa: C901
    """Put back a cited line a rule removed when one of its facts is cited by
    no surviving line, unless the line is noise (see _NOISE_TEXT, and the
    atoms in content["_noise_atoms"]: captions, small talk, brochures, open
    questions, existing gear, labor-only logistics). When a surviving line
    says the same thing (a standard step), the fact moves onto that line's
    cites instead. Each guarded drop is a sow_keep "keep" label."""
    after = _snap(content)
    noise = set(content.get("_noise_atoms") or [])
    # The summary only names the work: a fact the overview cites is not in the
    # task list (010217: the 98" install's atom lived on in the summary, so
    # its only task line went).
    live = lambda own: [x for k in _LIST_SECTION if k != "overview" or own == "overview"
                        for x in content.get(k) or [] if isinstance(x, dict) and x.get("kind") != "heading"]
    out = []
    for cites, (lst, text) in before.items():
        if cites in after and not (after[cites][0] == "overview" and lst != "overview"):
            continue
        alive = {c for x in live(lst) for c in x.get("cites") or []}
        # A noise cite (a quoted copy, a picture, existing gear) does not make
        # the line noise: only a fact that is not noise needs a home (010217:
        # the email thread's quoted copy of "reuse the existing mounts" let
        # tidy drop the 86" install that also cited the note's own lines).
        # A summary sentence leaning on a quoted or pictured line stays a
        # paraphrase of it (010209: the RFQ's quoted "manage the entire
        # project" summary the grounding rule rightly dropped).
        orphan = [c for c in cites if c not in alive and c not in noise]
        extra = str(content.get("_noise_re") or "")
        if not orphan or (lst == "overview" and any(c in noise for c in cites)) or _NOISE_TEXT.search(text) or _FOLLOW_UP.search(text) \
                or (extra and re.search(extra, text, re.I)):
            continue
        if lst == "equipment_in_scope" and re.search(r"(?i)\bexisting\b", text):
            continue  # gear the job does not touch
        # A line the SOW elsewhere calls optional is not put back as a firm
        # one (010082 pass T: "Assume a project survey is required" restored
        # beside "Determine whether a site survey is required, as survey is optional").
        if any(re.search(r"(?i)\boptional\b|\bdecide whether\b|\bdetermine whether\b", str(y.get("text")))
               and len(_object_toks(text) & _object_toks(str(y.get("text")))) >= 1
               and re.search(r"(?i)\b(survey|walk-?through|assessment|site visit)\b", text)
               for y in live(lst)):
            continue
        if lst == "schedule" or (lst == "overview" and re.match(r"^[A-Z][\w /&-]{1,30}:\s", text)):
            continue  # label hygiene: the schedule's own labels, form residue in the summary
        if _LEDGER.search(text) or re.search(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}|point of contact|\b(main|primary|tech|backup) contact\b|"
                                            r"escalation contact|customer-side project manager|hours of operation|"
                                            r"monday.{0,15}friday.{0,30}8:?0?0?\s*(am|a\.m\.).{0,10}5:?0?0?\s*(pm|p\.m\.)", text, re.I):
            continue  # accounting fields, contacts and 8-5 site hours have their own template blocks
        if any(_said(text, [str(t)], 0.6) for d in content.get("deliverables") or [] if isinstance(d, dict)
               for t in d.get("description") or []):
            continue  # moved into the Work Completion Report (photos, serials)
        # Back where its party puts it: a customer's duty written as a task or
        # an assumption is a customer responsibility (010300 "Customer will
        # guide technician ...", 010264 "Provide site access ...").
        dest = lst
        if (lst == "scope" and re.match(r"(?i)\s*(the )?customer\b", text)) or \
                (lst == "assumptions" and re.match(r"(?i)\s*(provide|ensure|supply|furnish)\b", text)):
            dest = "customer_responsibilities"
        # Never across rooms or sites (010217 pass U: the Room 247 98-inch
        # install "merged" into another room's line and vanished).
        dup = next((x for x in content.get(dest) or [] if isinstance(x, dict) and x.get("kind") != "heading"
                    and _said(text, [x], 0.6 if dest == "scope" else 0.55)
                    and not (_places(text) and _places(str(x.get("text"))) and _places(text) != _places(str(x.get("text"))))), None)
        if dup is not None:
            dup["cites"] = list(dict.fromkeys(list(dup.get("cites") or []) + orphan))
            # The fact's own name survives the merge (010373: "Humanscale"
            # left the install line when the standard step absorbed it).
            for mm in re.finditer(r"(?<!^)(?<![.:]\s)\b([A-Z][a-zA-Z0-9-]{2,})\s+((?:[a-z][\w-]*\s+)?[a-z][\w-]*)", text):
                brand, head = mm.group(1), mm.group(2)
                dt = str(dup.get("text") or "")
                if brand.lower() not in dt.lower() and brand.lower() not in _STOCK and head in dt:
                    dup["text"] = dt.replace(head, f"{brand} {head}", 1)
            how = "merged"
        else:
            orig = next((x for x in lists_before.get(lst) or [] if isinstance(x, dict) and tuple(x.get("cites") or []) == cites), None)
            if orig is None:
                continue
            items = content.setdefault(dest, [])
            close = next((i for i, y in enumerate(items) if isinstance(y, dict) and y.get("kind") == "heading"
                          and re.search(r"(?i)clos|clean|wrap|complet", str(y.get("text")))), len(items)) if dest == "scope" else len(items)
            # Back where it stood: after the nearest line that stood before it
            # and is still there (010217: the Training Room 250 install came
            # back under Room 247, the last heading before Closeout).
            if dest == lst:
                was = lists_before.get(lst) or []
                at = next((i for i, y in enumerate(was) if y is orig), -1)
                for prev in reversed(was[:max(0, at)]):
                    j = next((i for i, y in enumerate(items) if y is prev), None)
                    if j is not None:
                        close = j + 1
                        break
                else:
                    if at == 0:
                        close = 0
            items.insert(close, orig)
            how = "restored" if dest == lst else "restored_to_" + dest
        out.append({"rule": rule, "head": "sow_keep", "atom_ids": list(cites), "value": "keep", "was": _LIST_SECTION[lst],
                    "text": text[:300], "guard": how})
    return out


def _std_gone(content: dict[str, Any], where: str) -> None:
    """Record the standard items no longer in the SOW, and what took them."""
    objs = content.get("_std_objs")
    if not objs:
        return
    live = [x for k in LISTS for x in content.get(k) or [] if isinstance(x, dict) and x.get("standard")]
    ids, texts = {id(x) for x in live}, {str(x.get("text")) for x in live}
    keep = []
    for i, t in objs:
        if i in ids or t in texts:
            keep.append([i, t])
        else:
            content.setdefault("standards_suppressed", []).append({"text": t, "reason": where})
    content["_std_objs"] = keep


def _places(t: str) -> frozenset[str]:
    """The rooms, areas or sites a line names ("room 247", "Training Room 250", "the NOC")."""
    return frozenset(m.group(0).lower() for m in re.finditer(
        r"(?i)\b(?:room|office|suite|floor|building|store|site|closet|lobby|idf|mdf)\s+#?[\w-]*\d[\w-]*\b|\b(?:the\s+)?NOC\b|\b[A-Z][a-z]+ (?:Room|Office)\b", t))


def _slot_back(content: dict[str, Any], before: list[Any], rule: str) -> None:
    """A cited step that took a standard step's slot and is then removed or
    moved out of the task list gives the slot back: the standard step returns
    where the step stood (010085: the counted "Install approximately twenty
    (20) network switches" took "Install the new switches" and grounding then
    dropped it, leaving no install step)."""
    slots = content.get("_slots") or {}
    if not slots:
        return
    sc = content.setdefault("scope", [])
    for i, x in enumerate(before):
        std = slots.get(id(x))
        if std is None or any(x is y for y in sc):
            continue
        slots.pop(id(x))
        if not _said(std["text"], sc, 0.8):
            sc.insert(min(i, len(sc)), std)
            content.setdefault("standards_suppressed", []).append(
                {"section": "method_steps", "text": std["text"], "reason": f"restored: rule {rule} removed the deal step in its slot"})


def _traced(content: dict[str, Any], rule: str, fn, *args, **kwargs):
    before = _snap(content)
    lists_before = {k: list(content.get(k) or []) for k in _LIST_SECTION}
    result = fn(*args, **kwargs)
    _slot_back(content, lists_before.get("scope") or [], rule)
    _std_gone(content, f"rule {rule}")
    guarded = _guard_drops(content, rule, before, lists_before) if rule in GUARDED_RULES else []
    kept = {tuple(g["atom_ids"]) for g in guarded}
    content.setdefault("rule_decisions", []).extend(
        [d for d in _decisions(rule, before, _snap(content)) if tuple(d["atom_ids"]) not in kept] + guarded)
    return result


def _route_family(family: str, fs: list[dict[str, Any]]) -> str:
    """The job family: the writer's call, unless it is generic or wrong for
    what the facts plainly are."""
    family = family if family in FAMILIES else "other"
    # The writer's family call wobbles run to run (010082: low_voltage_cabling,
    # then smart_hands_general, losing the cabling standards). A generic call
    # yields to what the facts plainly are.
    # Dismantle-and-pack is for jobs that only take gear out: a job whose
    # scope installs new gear in its place is a replacement (010376 pass U:
    # "Replace a 5K UPS", "install the new UPS and battery pack" routed to
    # decommission_packing, so the UPS standards never fired and its
    # "Hardware installation" exclusion negated the sold install).
    if family == "decommission_packing":
        scope_txt = " ".join(f["text"] for f in fs if f["section"] == "scope_of_work").lower()
        if len(re.findall(r"\b(install(s|ed|ing)?|replace(s|d|ment)?|replacing|upgrade|swap)\b(?![^.]{0,20}\bnot\b)", scope_txt)) >= 2 \
                and not re.search(r"\bno (new )?(install|installation)\b|\bnot (be )?install", scope_txt):
            family = "smart_hands_general"
    if family in ("smart_hands_general", "other"):
        blob = " ".join(f["text"] for f in fs).lower()
        for fam_, pat in (("low_voltage_cabling", r"\bcat ?6a?\b|\bcat ?5e?\b|cable runs?|\bdrops?\b.{0,30}\b(cable|run|data)|terminat"),
                          ("cameras_security", r"\bcameras?\b.{0,40}\b(install|mount)|verkada|axis camera"),
                          ("network_switch_refresh", r"\bswitch(es)?\b.{0,40}\b(replace|refresh|swap)|replace .{0,20}switch"),
                          ("decommission_packing", r"\bdecommission|dismantl|pack(ing)? up|box(ing)? up"),
                          ("pos_install_refresh", r"\bpos\b|point of sale|register lanes?|kitchen printer")):
            if re.search(pat, blob) and len(re.findall(pat, blob)) >= 2:
                family = fam_
                break
        # A phone swap or recurring support call that came back "other" is smart
        # hands, whose standards carry those blocks (010326 pass P: "other", no
        # standards at all).
        # So is technician staffing for an IT transition or onboarding (010419
        # pass V: "one network technician and one help desk technician per
        # community ... before the transition" came back "other", no standards).
        if family == "other" and re.search(r"\b(phones?|handsets?|ata'?s?)\b.{0,60}\b(swap|replace|install|deploy)|weekly (on-?site|visit)|staff aug|monitor arms?"
                                           r"|\b(network|help ?desk|desk-?side|it) (support )?(technicians?|techs?)\b"
                                           r"|\b(it|provider|msp|managed services?|system) transition\b|\bonboarding\b|\bthe transition\b"
                                           r"|\b(it|provider|msp|network|site) (cutover|takeover|take-over)\b|\btak(e|ing) over (the )?(it|support|site|network)\b", blob):
            family = "smart_hands_general"
    return family


def write(envelope: dict[str, Any], pm_handoff: dict[str, Any] | None = None, *,
          model: str | None = None) -> dict[str, Any]:
    """The SOW content for one deal: four focused calls (header, tasks, who
    supplies what, then the assumptions and exclusions against that draft),
    a pass over the facts still unused, and every item checked. One call
    asked for all of it dropped a different rule each run."""
    from concurrent.futures import ThreadPoolExecutor

    model = model or sow_model()
    fs = facts(envelope)
    brief = envelope.get("c3_brief") or {}
    crm = envelope.get("crm") or {}
    crm_rec = {"deal_name": crm.get("deal_name") or (brief.get("deal") or {}).get("deal_name"),
               "account_name": crm.get("account_name") or (brief.get("deal") or {}).get("account_name"),
               # A file's own name can be the only place the end customer is
               # written (000123: "Bevmo locations 1.txt").
               "document_titles": sorted({re.sub(r"^\d{6}-(hs|fireflies)-[\w-]*?-", "", str(d.get("filename") or ""))[:80]
                                          for d in envelope.get("documents") or [] if d.get("filename")})[:25]}
    contacts = [c.get("line") for c in brief.get("contacts") or [] if c.get("line")][:12]
    sites = _sites(envelope, pm_handoff)
    rows = len([x for x in (brief.get("sections") or {}).get("contacts") or [] if x.get("type") == "physical_site"])
    site_info = {"count_in_site_lists": max(rows, len(sites), int((pm_handoff or {}).get("site_count") or 0)),
                 "sites": sites[:15]}
    base = _BASE % (json.dumps(crm_rec), json.dumps(site_info), json.dumps(contacts),
                    "\n".join(_fact_line(f) for f in fs))
    effort = os.environ.get("SOWSMITH_C3_SOW_EFFORT", "medium")
    got: dict[str, Any] = dict(_json_of(_complete(base + _PARTS["header"], model, 3500, effort)) or {})
    family = _route_family(str(got.get("job_family") or "other"), fs)
    got["job_family"] = family
    deal_no = (re.match(r"\s*(\d{6})", str(crm_rec.get("deal_name") or "")) or [None, ""])[1]
    std = standard_items(family, deal_no, [f["text"][:300] for f in fs
                                            if f["section"] in ("scope_of_work", "customer_responsibilities",
                                                                "assumptions", "out_of_scope")][:40])
    fixed = purtera_standards(family)
    # Visible in every SOW: an install without the library wrote with none and
    # nothing failed (the data files were not shipped).
    library_counts = {"library_items": sum(len(v) for v in std.values()), "standards_items": sum(len(v) for v in fixed.values())}
    pick = lambda *secs: json.dumps({**{k: std.get(k, []) for k in secs},
                                     **{f"purtera_standard_{k}": fixed[k] for k in secs if fixed.get(k)}})
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_scope = pool.submit(_complete, base + _PARTS["scope"] % pick("method_steps", "deliverables"), model, 4000,
                              effort)
        f_supply = pool.submit(_complete, base + _PARTS["supply"] % pick("tools", "customer_responsibilities"),
                               model, 3000, effort)
        for fut in (f_scope, f_supply):
            got.update({k: v for k, v in (_json_of(fut.result()) or {}).items() if k != "job_family"})
    draft = {k: [str((x or {}).get("text") if isinstance(x, dict) else x) for x in got.get(k) or []]
             for k in ("overview", "scope", "provider_materials", "customer_materials", "customer_responsibilities")}
    got.update(_json_of(_complete(base + _PARTS["bounds"] % (pick("assumptions", "exclusions"),
                                                              json.dumps(draft)[:10000]), model, 3000, effort)) or {})
    got["job_family"] = family
    content = _normalise(got)
    content["overview"] = content["overview"][:4]
    pool_text = " ".join(f["text"] for f in fs).lower()
    content["scope_lead"] = _lead(str(content.get("scope_lead") or ""), pool_text)
    lines = {f["id"]: (f"[under: {f['under']}] " if f["under"] else "") + f["text"]
             + (f" [supplier head: {_SUPPLIER.get(f['supplier'][0], f['supplier'][0])}]"
                if f.get("supplier") and float(f["supplier"][1] or 0) >= SUPPLIER_BAR
                and f.get("type") in _ITEM_TYPES else "") for f in fs}
    _traced(content, "rejected_source", _rejected_source, content, envelope)
    content["added"] = _cover(content, fs, model)
    content["added"] += _must_include(content, fs, model)
    # The writer's own lists, before any check or rule, so a replay feeds
    # what the live writer said rather than the chain's output (010198 pass T:
    # the replay re-ran the rules on their own output and could not show the
    # live date line).
    content["writer_draft"] = {k: [{kk: x.get(kk) for kk in ("text", "cites", "kind") if kk in x}
                                   for x in content.get(k) or [] if isinstance(x, dict)]
                               for k in LISTS if content.get(k)}
    _model_numbers(content, envelope)  # before the check reads "785 existing" as a count
    content["checks"] = check(content, lines)
    # A schedule or work-window line is a commitment: it needs a line the brief
    # kept, not only one it mentioned in passing (000132's "24/7 on-call",
    # keep 0.21, became "Work Window: 24/7 on-call availability").
    weak = {f["id"] for f in fs if f["section"] in ("also_kept", "unsure", "typed_only", "answered_question")}
    content["schedule"] = [x for x in content.get("schedule") or []
                           if not x.get("cites") or not all(c in weak for c in x["cites"])]
    # The summary commits only what the brief commits: a line the brief merely
    # mentioned (000132's "24/7 on-call", whose caveat sat beside it) is not a
    # promise in the overview.
    promise = re.compile(r"(?i)24\s*/\s*7|on-?call|\bsla\b|guarantee|response time|within \d+ (hours?|minutes?)")
    for k in ("scope", "schedule", "assumptions"):
        content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("cites")
                      and all(c in weak for c in x["cites"]) and promise.search(str(x.get("text"))))]
    ov_kept = [x for x in content.get("overview") or [] if not x.get("cites") or not all(c in weak for c in x["cites"])]
    content["overview"] = ov_kept or content.get("overview") or []
    # A start date before the cutoff is a document's date, not the job's
    # (010120 printed the survey deck's 7/9 as the requested start).
    cut = str((envelope.get("run_scope") or {}).get("as_of") or "")[:10]
    def _before_cut(t: str) -> bool:
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})|(\d{1,2})/(\d{1,2})/(\d{2,4})", t)
        if not (m and cut):
            return False
        y, mo, d = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(6), m.group(4), m.group(5))
        y = ("20" + y) if len(y) == 2 else y
        return f"{y}-{int(mo):02d}-{int(d):02d}" < cut
    content["schedule"] = [x for x in content.get("schedule") or []
                           if not (re.search(r"(?i)\b(start|service|install|scheduled)\b.{0,20}\bdate\b|^\s*requested start", str(x.get("text")))
                                   and _before_cut(str(x.get("text"))))]
    # A resource count needs a line that states staffing ("one tech for two
    # days"), not a named delivery team (010082: "Kelly, Octavian and Chase are
    # our delivery team" became "Number of Resources: three").
    text_of = {f["id"]: f["text"] for f in fs}
    content["schedule"] = [x for x in content.get("schedule") or []
                           if not re.match(r"(?i)\s*number of resources", str(x.get("text")))
                           or any(_STAFF.search(text_of.get(c, "")) for c in x.get("cites") or [])]
    # An empty schedule takes the job's dated window from a milestone fact
    # (010209: "install period 10/13 - 11/19", milestone_phase 0.94).
    if not content.get("schedule"):
        for f in fs:
            if f.get("type") == "milestone_phase" or f["section"] == "timeline":
                m = re.search(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s*(?:-|–|to|through|thru)\s*(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b", f["text"])
                if m:
                    content["schedule"] = [{"text": f"Work Window: {m.group(1)} through {m.group(2)}", "cites": [f["id"]], "kind": "fact"}]
                    break
    # A date from a quoted earlier email is history, not this job's schedule
    # (010043: CDW's May meeting slots became a "work window").
    quoted = {f["id"] for f in fs if f.get("quoted")}
    # Only a dated line is history, and a quoted copy whose original the
    # sources also hold is that original (010209 pass U: the RFQ's "If
    # equipment is received earlier, we can start the project earlier", cited
    # to a forward's quoted copy, never reached the schedule).
    by_text = [f for f in fs if not f.get("quoted")]
    for x in content.get("schedule") or []:
        if x.get("cites") and all(c in quoted for c in x["cites"]):
            src = " ".join(f["text"] for f in fs if f["id"] in x["cites"])
            twin = [f["id"] for f in by_text if _said(src, [f["text"]], 0.8) and _said(f["text"], [src], 0.8)]
            if twin:
                x["cites"] = twin[:2]
    dated = re.compile(r"(?i)\b\d{1,2}/\d{1,2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d|\b\d{4}-\d\d-\d\d\b|"
                       r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b\d{1,2}(:\d\d)?\s*(am|pm)\b")
    content["schedule"] = [x for x in content.get("schedule") or []
                           if not (x.get("cites") and all(c in quoted for c in x["cites"]) and dated.search(str(x.get("text")))
                                   and not re.match(r"(?i)\s*(number of resources|hardware quantity)", str(x.get("text"))))]
    # A timestamp is a record's metadata (a meeting's end, an email's send
    # time), never a schedule value (010264 pass U: "Requested Completion
    # Date: 2026-09-09T15:00:00.001Z", a meeting recap's end).
    content["schedule"] = [x for x in content.get("schedule") or []
                           if not re.search(r"\b\d{4}-\d\d-\d\dT\d\d:\d\d", str(x.get("text") if isinstance(x, dict) else x))]
    sched = content.setdefault("schedule", [])
    if not any("completion date" in str(x.get("text", "")).lower() for x in sched if isinstance(x, dict)):
        for f in fs:
            m = re.search(r"(?i)\b(end|completion|finish)\s*date\s*[:|]\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})", f["text"])
            # The work's own deadline: "... across 1,400 U.S. locations by 1/31/27".
            m = m or (re.search(r"(?i)\b(replace|install|complete|finish|deploy|roll ?out|migrate)\w*\b(?:(?!\.\s)[^?]){0,200}?\b(by)\s+([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})\b",
                                re.sub(r"\b([A-Z])\.(?=[A-Z]?\.?\s)", r"\1", f["text"]))
                      if f["section"] in ("scope_of_work", "timeline", "overview") and not f.get("quoted") else None)
            if m:
                sched.append({"text": f"Requested Completion Date: {m.group(m.lastindex)}", "cites": [f["id"]], "kind": "fact"})
                break
    _reseller_supplies(content, fs)
    # A customer duty cannot rest only on what PurTera itself wrote (010217:
    # Trent's sales intro "deep familiarity with Certinia, Agiloft" became a
    # CDW responsibility).
    ours = {str(a.get("id")) for a in envelope.get("atoms") or []
            if "purtera" in str((a.get("locator") or {}).get("sender") or "").lower()}
    if ours:
        content["customer_responsibilities"] = [x for x in content.get("customer_responsibilities") or []
                                                if not (isinstance(x, dict) and x.get("cites") and all(c in ours for c in x["cites"]))]
    # What the customer says it will provide stays the customer's (010171:
    # "I will provide HDMI and USB cabling" became a Provider task).
    by_id_a = {str(a.get("id")): a for a in envelope.get("atoms") or []}
    def _customer_provides(c: str) -> bool:
        a = by_id_a.get(c) or {}
        sender = str((a.get("locator") or {}).get("sender") or "").lower()
        return bool(re.search(r"(?i)\b(i|we)\s*(will|'ll|am going to|are going to)\s+(be\s+)?provid", str(a.get("text") or ""))) \
            and "purtera" not in sender
    for k in ("scope", "provider_materials"):
        hit = [x for x in content.get(k) or [] if isinstance(x, dict) and x.get("cites") and all(_customer_provides(c) for c in x["cites"])]
        if hit:
            content[k] = [x for x in content.get(k) or [] if x not in hit]
            for x in hit:
                x["text"] = re.sub(r"(?i)^\s*provide( and use)?\s+", "", str(x["text"])).rstrip(".")
                x["text"] = x["text"][:1].upper() + x["text"][1:] + "."
            content.setdefault("customer_materials", []).extend(hit)
    # A "Provider will ..." sentence under Customer Provided Materials is the
    # customer's (reseller's) supply (010083: "Provider will acquire, configure,
    # and ship all required equipment" where CDW said it would).
    for x in content.get("customer_materials") or []:
        if isinstance(x, dict) and re.match(r"(?i)\s*provider will\b", str(x.get("text"))):
            x["text"] = re.sub(r"(?i)^\s*provider will\b", "Customer will", str(x["text"]))
    # An equipment line that is a task belongs in the scope (010198: "Install
    # one (1) Ubiquiti router with cellular backup" sat only under equipment).
    eq = content.get("equipment_in_scope") or []
    tasks = [x for x in eq if isinstance(x, dict) and re.match(r"(?i)\s*(install|mount|replace|configure|connect|deploy|set up)\b", str(x.get("text")))]
    if tasks:
        content["equipment_in_scope"] = [x for x in eq if x not in tasks]
        content.setdefault("scope", []).extend(tasks)
    # Remote tools only when the facts mention remote work (010198, 010082, 000132).
    if not re.search(r"(?i)\bremote(ly)?\b", pool_text):
        content["provider_tools"] = [x for x in content.get("provider_tools") or [] if not re.search(r"(?i)remote|hotspot", str(x.get("text") if isinstance(x, dict) else x))]
    # A start date without a year takes the cutoff's year.
    yr = str((envelope.get("run_scope") or {}).get("as_of") or "")[:4]
    for x in content.get("schedule") or []:
        if isinstance(x, dict) and yr.isdigit() and re.match(r"(?i)\s*requested (start|completion) date: [A-Za-z, ]+\d{1,2}\s*$", str(x.get("text"))):
            x["text"] = str(x["text"]).rstrip() + f", {yr}"
    content["moved_by_supplier_head"] = _follow_supplier(content, fs)
    _ship_list_siblings(content, envelope, fs)
    sup_of = {f["id"]: f.get("supplier") for f in fs}
    cust = lambda c: bool(sup_of.get(c)) and sup_of[c][0] == "customer" and float(sup_of[c][1] or 0) >= SUPPLIER_BAR
    eq = content.get("equipment_in_scope") or []
    to_cust = [x for x in eq if isinstance(x, dict) and x.get("cites") and all(cust(c) for c in x["cites"])]
    # A BOM block the supplier head reads as the customer's is the customer's,
    # row by row (010116: three rows read "customer" at the bar, the rest none).
    bom_row = re.compile(r"^\s*\d{1,4}\s*[:|x]\s*(?:\d{1,4}\s*\|\s*)?\|?\s*[A-Z0-9][A-Z0-9./_-]{3,30}\b")
    bom_facts = [f for f in fs if bom_row.match(f["text"])]
    if bom_facts and any(cust(f["id"]) for f in bom_facts):
        cm = content.setdefault("customer_materials", [])
        for f in bom_facts:
            if not _said(f["text"], cm, 0.6):
                cm.append({"text": re.sub(r"^\s*(\d+)\s*[:|]\s*", r"\1 x ", f["text"]).strip(), "cites": [f["id"]], "kind": "fact"})
    if to_cust:  # 010116: the supplier head read "customer" on every BOM line
        content["equipment_in_scope"] = [x for x in eq if x not in to_cust]
        content.setdefault("customer_materials", []).extend(to_cust)
    # A customer duty every cited line of which the supplier head gives to us
    # is Provider work (010279: proof of delivery, inventory recap); and one
    # atom is never both Provider scope and a customer duty.
    sup = {f["id"]: f.get("supplier") for f in fs}
    ours_cite = lambda c: bool(sup.get(c)) and sup[c][0] == "us" and float(sup[c][1] or 0) >= SUPPLIER_BAR
    scope_cites = {c for x in content.get("scope") or [] if isinstance(x, dict) for c in x.get("cites") or []}
    moved = []
    for x in list(content.get("customer_responsibilities") or []):
        if not isinstance(x, dict) or x.get("standard") or not x.get("cites"):
            continue
        if set(x["cites"]) <= scope_cites:
            content["customer_responsibilities"].remove(x)
        elif all(ours_cite(c) for c in x["cites"]):
            content["customer_responsibilities"].remove(x)
            moved.append(x)
    content.setdefault("scope", []).extend(moved)
    content["unassigned"] = _party_check(content, lines)
    _provider_half(content, fs)
    _dedupe_items(content)
    content["withheld"] = _scrub(content)
    # A number that names a document ("785 POS Termainal Refresh RFP.doc") is a
    # model or project number, never a count: the line is reworded, not
    # dropped (010279 pass T: "replace 785 existing POS terminals at 113
    # sites" went whole, and with it every terminal in the SOW).
    _model_numbers(content, envelope)
    # A heading whose tasks the checks dropped is an empty promise in the SOW
    # (010043: "Camera mounting" with nothing under it).
    sc = content.get("scope") or []
    content["scope"] = [x for i, x in enumerate(sc)
                        if not (isinstance(x, dict) and x.get("kind") == "heading")
                        or (i + 1 < len(sc) and not (isinstance(sc[i + 1], dict) and sc[i + 1].get("kind") == "heading"))]
    # Picture captions never trigger standards (010120: "Two large TVs are
    # mounted on the front wall" fired the display-mounting block).
    writer_snap = _snap(content)
    content["_noise_atoms"], content["_noise_re"] = _noise(envelope, fs)
    # The standards' gates also read the deal lines the brief left out, from
    # the documents the facts come from: not addresses, contacts, chatter or
    # rejected lines (000123: "Gateway Road" is not a gateway).
    rejected = _rejected_atoms(envelope) | set(content.get("_noise_atoms") or [])
    fact_ids = {f["id"] for f in fs}
    fact_docs = {str(a.get("artifact_id")) for a in envelope.get("atoms") or [] if str(a.get("id")) in fact_ids}
    no_gate = ("physical_site", "contact", "person", "signature", "small_talk", "email_header", "deal_metadata",
               "open_question", "action_item", "answered_question")
    gate_atoms = [str(a.get("text") or "") for a in envelope.get("atoms") or []
                  if not (a.get("structured") or {}).get("quoted") and str(a.get("id")) not in rejected
                  and str(a.get("artifact_id")) in fact_docs and "?" not in str(a.get("text") or "")
                  and str(((a.get("structured") or {}).get("c3_label") or {}).get("type") or a.get("atom_type") or "") not in no_gate]
    _traced(content, "standards", _apply_standards, content, family, [f["text"] for f in fs if not f.get("pictured")],
            [str(a.get("text") or "") for a in envelope.get("atoms") or [] if not (a.get("structured") or {}).get("quoted")],
            gate_atoms)
    _traced(content, "zero_rows", _zero_rows, content, envelope)
    _traced(content, "first_person", _first_person_duties, content, envelope)
    _traced(content, "assumption_voice", _assumption_voice, content)
    _traced(content, "moved_gear", _moved_gear, content, family)
    _traced(content, "destination", _destination_owner, content, fs)
    _onsite_designate(content, fs)
    _traced(content, "tidy", _tidy, content, facts_say_permit=bool(re.search(r"(?i)\bpermit", pool_text)))
    _arrival_name_rule(content, fs)
    content["slot_moved"] = _traced(content, "slot_by_head", _slot_by_head, content, envelope)
    # Lines from a floor plan, leasing sheet or brochure are about the building,
    # never the job (010012: Stamford floor-plan captions typed as tasks).
    bro_docs = {str(d.get("artifact_id")) for d in envelope.get("documents") or []
                if re.search(r"(?i)floor ?plan|concept plan|space plan|test[ -]?fit|leasing|brochure|flyer|availability|marketing", str(d.get("filename") or ""))}
    content["_our_people"] = sorted({str(pp.get("name") or "").split()[0] for pp in (envelope.get("deal_roster") or {}).get("people") or []
                                     if pp.get("side") == "ours" and str(pp.get("name") or "").strip()})
    content["_brochure_atoms"] = [str(a.get("id")) for a in envelope.get("atoms") or [] if str(a.get("artifact_id")) in bro_docs]
    # The deal's own hedges on round-the-clock coverage, wherever the brief
    # filed them (000132: "The 24/7 availability can be tough ..." is a
    # commitment atom the facts never carried; only its bare ask was a fact).
    content["_clock_hedges"] = [{"id": str(a.get("id")), "text": str(a.get("text") or "")} for a in envelope.get("atoms") or []
                                if re.search(r"(?i)" + _ROUND_CLOCK + r"|on-?call", str(a.get("text") or ""))
                                and _HEDGE.search(str(a.get("text") or ""))]
    # A line the type head called small talk is never cited (010082: "This
    # work is for a new unit opening" from a signature's menu entry).
    chatter = {str(a.get("id")) for a in envelope.get("atoms") or []
               if str(((a.get("structured") or {}).get("c3_label") or {}).get("type") or a.get("atom_type") or "") in ("small_talk", "signature")}
    if chatter:
        for k in LISTS:
            content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("cites") and all(c in chatter for c in x["cites"]))]
    _traced(content, "grounding", _grounding, content, fs)
    _traced(content, "hours", _hours_basis, content, fs)
    _traced(content, "materials", _materials_hygiene, content, family, fs)
    _traced(content, "one_supplier", _one_supplier, content, fs)
    _traced(content, "third_party", _third_party, content, envelope, fs)
    # An unanswered question is not a fact: a bullet cited only to open
    # questions goes (010085: PurTera's own "no remote access or console work?"
    # became scope), unless it asks the customer to confirm.
    # By the words, not only the label: 010085's "Assume 2 boom lifts are
    # needed?" was typed quantity and became "Two boom lifts are assumed".
    # A question that asks for work is a request, not an open question
    # (010198: "I would also like to add a Ubiquiti router ... - is that
    # something you can help with?" is half the job).
    requests_work = lambda t: re.search(r"(?i)\b(would (also )?like|i'?d like|we need|need (you|us|to)|want to add|add (a|an|the)|can you help|could you|please (install|add|quote))\b", t)
    # Unanswered only when nothing follows the last "?": a Q&A fact is stored as
    # "question? [Betty] answer" and every one of 010083's answers was dropped
    # when any "?" counted (schedule, hours, site mix: 30 items).
    def _unanswered(f: dict[str, Any]) -> bool:
        t = f["text"].strip()
        if f.get("section") == "answered_question" or f.get("type") in ("answered_question", "constraint", "dependency",
                                                                        "milestone_phase", "scope_item"):
            return False
        if "?" in t and len(re.sub(r"\W+", "", t.rsplit("?", 1)[1])) >= 3:
            return False  # an answer follows the question (010083's site-list duty)
        if f.get("type") == "open_question":
            return True
        return "?" in t
    # A working assumption put as a question ("Safe to assume no spotter is
    # required while lifts are in use?") is still the assumption the price
    # rests on (010085 pass T dropped it); a quantity guess is not.
    working = lambda t: re.search(r"(?i)\b(safe to )?assum\w*\b[^?]{0,20}\b(no|not|none|without)\b", t)
    asked = {f["id"] for f in fs if _unanswered(f) and not requests_work(f["text"]) and not working(f["text"])}
    # No "only the listed devices" assumption while a request adds another.
    if any(requests_work(f["text"]) for f in fs):
        content["assumptions"] = [x for x in content.get("assumptions") or [] if not re.match(r"(?i)\s*only the listed", str(x.get("text")))]
    # A dated service day the writer left out ("scheduled for Sunday morning,
    # August 16, 2026") is the requested start (010198 lost its date twice).
    sched = content.setdefault("schedule", [])
    if not any(str(x.get("text")).lower().startswith("requested start") for x in sched if isinstance(x, dict)):
        months = "January|February|March|April|May|June|July|August|September|October|November|December"
        for f in fs:
            if f["section"] in ("timeline", "job_details", "scope_of_work", "schedule"):
                m = re.search(r"(?i)\b(?:on|for)\s+(?:next\s+|this\s+)?((?:mon|tues|wednes|thurs|fri|satur|sun)day[\w\s,]{0,12}?)?(" + months + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?", f["text"])
                if m:
                    day = (m.group(1) or "").strip(" ,")
                    sched.insert(0, {"text": "Requested Start Date: " + ((day.split()[0].capitalize() + ", ") if day else "")
                                     + f"{m.group(2).capitalize()} {m.group(3)}" + (f", {m.group(4)}" if m.group(4) else ""),
                                     "cites": [f["id"]], "kind": "fact"})
                    break
    for k in LISTS:
        content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("cites")
                      and all(c in asked for c in x["cites"]) and not re.match(r"(?i)\s*confirm\b", str(x.get("text"))))]
    _traced(content, "question_not_commitment", _question_not_commitment, content, fs)
    _traced(content, "cell_not_task", _cell_not_task, content, fs)
    _traced(content, "contradiction", _scope_contradictions, content, fs)
    _keep_exceptions(content, fs)
    # A lift in scope: no "lift by change request" assumption, no "specialty
    # lifting" exclusion (010085 rented a 40 ft lift and excluded lifts).
    lift_in = re.search(r"(?i)\b(lift|boom|scissor)\b", " ".join(str(x.get("text")) for k in ("scope", "provider_materials", "provider_tools")
                                                             for x in content.get(k) or [] if isinstance(x, dict)))
    if lift_in:
        content["assumptions"] = [x for x in content.get("assumptions") or [] if not re.search(r"(?i)lift.{0,60}change (request|order)|no (lift|specialized lifting)", str(x.get("text")))]
        content["out_of_scope"] = [x for x in content.get("out_of_scope") or [] if not re.search(r"(?i)lift|rigging", str(x.get("text")))]
    for x in content.get("out_of_scope") or []:
        if isinstance(x, dict):
            x["text"] = re.sub(r"(?i)(?<!wall )\bpatching or painting\b", "wall patching or painting", str(x["text"]))
    svc = content.get("services") or []
    # Recurring onsite support is staff augmentation and support (000132).
    if family in ("smart_hands_general", "other") and re.search(r"(?i)once (a|per) week|weekly (on-?site|visit)|recurring|"
                                                                r"staff aug|dedicated (tech|resource)|on-?site support", pool_text):
        content["services"] = svc = ["staff_augmentation", "support"]
    # A transition or onboarding check is an assessment and implementation
    # visit, not staff augmentation (010419: "IT Transition & Onboarding").
    crm_name = str((envelope.get("crm") or {}).get("deal_name") or "")
    # An IT transition, not a device cutover: 010326's phone "cutover" and
    # 010082's ISP "transition" ticked Assessment on install jobs.
    transition = bool(re.search(r"(?i)\b(it|msp|support|service|services|provider)\s+transition\b|\btransition\s+(and|&)\s+onboarding|\bonboarding\b",
                                pool_text + " " + crm_name.lower())) and not re.search(
            r"(?i)once (a|per) week|weekly (on-?site|visit)|recurring|staff aug|ongoing", pool_text)
    if transition:
        content["services"] = svc = ["assessment", "implementation"]
    if transition:
        _traced(content, "work_no_fact_names", _work_no_fact_names, content, pool_text)
    _traced(content, "duty_owner", _duty_owner, content, fs)
    # A negation its cited source lacks reverses the source (000132: the
    # requested end-user O365 support written as "is excluded").
    n_dec = len(content.get("rule_decisions") or [])
    _traced(content, "source_polarity", _source_polarity, content, fs, envelope)
    for d in (content.get("rule_decisions") or [])[n_dec:]:
        d["why"] = "states a negation its cited source lacks"
    prov = [x for x in content.get("customer_responsibilities") or [] if isinstance(x, dict) and not x.get("standard")
            and re.match(r"(?i)\s*(provide\s+)?(troubleshoot|incident response|end-user support|support for|patch|monitor(ing)? and)", str(x.get("text")))]
    if prov and family in ("smart_hands_general", "other"):
        content["customer_responsibilities"] = [x for x in content["customer_responsibilities"] if x not in prov]
        content.setdefault("scope", []).extend(prov)
    if "implementation" in svc and "custom_work" in svc and family not in ("decommission_packing", "relocation_move"):
        content["services"] = svc = [v for v in svc if v != "custom_work"]  # 010373: haul-away read as packing
    # A replacement job ticks its family's box (010264 pass U: Implementation
    # on a time-clock swap whose family default is Reinstallation).
    if svc == ["implementation"] and _FAMILY_SERVICE.get(family) == "reinstallation" and re.search(r"(?i)\breplac", pool_text):
        content["services"] = svc = ["reinstallation"]
    if "implementation" in svc and "reinstallation" in svc:
        content["services"] = [v for v in svc if v != "implementation"]
    # A library method item that promises a document nobody asked for (as-built
    # floor plans, cable inventory spreadsheets, certification reports) needs a
    # fact behind it (010043: a cabling item pulled into a camera job).
    doc_promise = re.compile(r"(?i)as-built|floor plans?|inventory records?|spreadsheet|certification report")
    for k in ("scope", "deliverables"):
        content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict)
                      and not x.get("standard") and doc_promise.search(str(x.get("text") or x.get("item") or ""))
                      and not doc_promise.search(pool_text))]
        for d in content.get("deliverables") or []:
            if isinstance(d, dict) and not doc_promise.search(pool_text):
                d["description"] = [t for t in d.get("description") or [] if not doc_promise.search(str(t))]
    # A library step about third-party vendors needs a fact that names one
    # (000132: "Support remote direction from ... third party vendors").
    if not re.search(r"(?i)third[- ]party|vendors?\b", pool_text):
        for x in content.get("scope") or []:
            if isinstance(x, dict) and not x.get("standard"):
                x["text"] = re.sub(r"(?i)\s*,?\s*(or|and)\s+(with\s+)?(any\s+)?third[- ]party vendors?", "", str(x.get("text")))
        content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and not x.get("standard")
                            and re.search(r"(?i)third[- ]party vendors?", str(x.get("text"))))]
    content["customer_responsibilities"] = [x for x in content.get("customer_responsibilities") or []
                                            if not (isinstance(x, dict) and not x.get("standard")
                                                    and re.search(r"(?i)construction drawings|cabling paths and idf", str(x.get("text")))
                                                    and not re.search(r"(?i)drawing", pool_text))]
    # A library line ruling out in-wall work the deal asks for (010120: the
    # survey moves the replication cable behind the wall, no visible cables).
    # Read every atom, not only the facts: 010120's "moved behind wall" and "No
    # cables should be visible" never reached the facts, and the line came back.
    all_text = pool_text + " " + " ".join(str(a.get("text") or "") for a in envelope.get("atoms") or []).lower()
    if re.search(r"(?i)behind (the )?walls?|in-?wall|inside (the )?wall|concealed|pass-?through|no cables? (should be |will be )?visible", all_text):
        for k in ("scope", "assumptions", "out_of_scope"):
            content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict)
                          and re.search(r"(?i)without in-?wall|no (in-?wall|concealed)|in-?wall .{0,20}(routing|cabling).{0,20}(excluded|not included)|concealed routing", str(x.get("text"))))]
    # Every stated cable run is a scope line, in code: the writer dropped or
    # mis-cited 010043's back and front door runs two rounds of three.
    scope_text = " ".join(str(x.get("text")) for x in content.get("scope") or [] if isinstance(x, dict))
    for f in fs:
        m = re.match(r"(?i)\s*(.{2,40}?)\s*[:\-–]?\s*(?:approx(?:imately)?\.?\s*)?(\d{2,4})\s*(?:ft|feet|')\b", f["text"])
        if m and re.search(r"(?i)\bto\b|run|drop|idf|mdf", m.group(1)) and not re.search(r"\b" + m.group(2) + r"\b", scope_text) \
                and not re.search(r"(?i)provi|estim|\btech\b|\bneed|quote|price|materials?", m.group(1)):  # 010082 "Tech to provdie estiamted 100ft"
            sc = content.setdefault("scope", [])
            close = next((i for i, x in enumerate(sc) if isinstance(x, dict) and x.get("kind") == "heading"
                          and re.search(r"(?i)clos|clean|wrap|complet", str(x.get("text")))), len(sc))
            sc.insert(close, {"text": f"Run cabling from {m.group(1).strip()}: approximately {m.group(2)} ft.",
                              "cites": [f["id"]], "kind": "fact"})
            scope_text += " " + m.group(2)
    # A ladder and no lift: say so (010043; PurTera writes both lines).
    # Not where the facts talk of a lift the work may need (010284: a 60 ft
    # pole and a boom lift).
    if re.search(r"(?i)\bladders?\b", pool_text) and not re.search(r"(?i)\b(boom|scissor|aerial) lifts?\b|\blift\b.{0,25}\b(need|requir|rent)", pool_text) and not re.search(r"(?i)\b(lift|boom|scissor)\b(?!\s*(is|are)?\s*not)", " ".join(
            str(x.get("text")) for k in ("scope", "provider_tools", "provider_materials") for x in content.get(k) or [] if isinstance(x, dict))):
        asm = content.setdefault("assumptions", [])
        for t in ("No lift is required.", "Standard ladders reach every work location."):
            if not _said(t, asm, 0.6):
                asm.append({"text": t, "cites": [], "kind": "method"})
    # A run-length cap no fact gives is a priced promise nobody made (010120:
    # "All cable runs are under 50 feet" on a job whose runs home-run to the
    # IDF). The cap derived from the stated runs below is the only one.
    content["assumptions"] = [x for x in content.get("assumptions") or [] if not (
        isinstance(x, dict) and not x.get("cites") and (mm := re.search(
            r"(?i)\b(?:cable )?runs?\b.{0,30}?\b(?:under|less than|no more than|not exceed|up to|maximum of)\s*(\d{2,4})\s*(?:feet|ft|')", str(x.get("text"))))
        and not re.search(r"\b%s\s*(?:-?\s*)(?:feet|ft|foot|')" % mm.group(1), pool_text))]
    # "All cable runs are under N feet" when the runs are stated (PurTera writes
    # it from the longest run, rounded up to 50).
    runs = [int(m.group(1)) for f in fs if f["section"] not in ("also_kept", "typed_only")
            for m in [re.search(r"(?i)\b(\d{2,4})\s*(?:ft|feet|')\b", f["text"])] if m]
    # One max-run assumption: a library "under 100 feet" beside the deal's own
    # runs gives way to the derived one (010082 printed 100 and 150 ft).
    if len(runs) >= 2:
        content["assumptions"] = [x for x in content.get("assumptions") or [] if not (isinstance(x, dict) and not x.get("cites")
                                  and re.search(r"(?i)cable runs? (are|is|will be) (under|less than|not exceed)", str(x.get("text"))))]
    # Any line already bounding the runs is the bound (010082: "will be no more
    # than 100 ft" plus a derived "under 150 feet").
    bound = re.compile(r"(?i)(no more than|under|less than|not (to )?exceed|maximum|max\.?|up to)\s*\d{2,4}\s*(ft|feet|')")
    if family in ("low_voltage_cabling", "cameras_security", "av_conference_room", "network_switch_refresh") and len(runs) >= 2 and not any(bound.search(str(x.get("text") if isinstance(x, dict) else x))
                                  for k in ("assumptions", "scope") for x in content.get(k) or []):
        cap = -(-max(runs) // 50) * 50
        text = f"All cable runs are no more than {cap} feet." if cap == max(runs) else f"All cable runs are under {cap} feet."
        content.setdefault("assumptions", []).append({"text": text, "cites": [], "kind": "method"})
    if not content.get("overview"):
        # Never an empty Project Summary: the request's own description of the
        # work (010115 round 17 printed the heading with nothing under it).
        bg = None
        for pat in (r"(?i)project background", r"(?i)description of work|scope of work requested|work requested"):
            bg = bg or next((f for f in fs if re.search(pat, f["text"]) and len(f["text"].split("|")[-1].split()) >= 4
                             and not re.search(r"…|\.\.\.", f["text"])), None)
        bg = bg or next((f for f in fs if f["section"] == "scope_of_work" and not re.search(r"…|\.\.\.", f["text"])
                         and len(f["text"].split()) >= 5), None)
        if bg:
            text = bg["text"].split("|")[-1].strip().rstrip(".") + "."
            content["overview"] = [{"text": text[:1].upper() + text[1:], "cites": [bg["id"]], "kind": "fact"}]
    _fix_dates(content, fs, cut)
    _cited_counts(content, envelope)
    _agreed_arrival(content, envelope)
    _weekend(content, fs)
    _schedule_from_facts(content, fs, envelope)
    _roles_and_hours(content, fs)
    if transition:
        _traced(content, "role_groups", _role_groups, content)
    _traced(content, "polarity", _scope_polarity, content, fs)
    _traced(content, "exclusions_never_tasks", _exclusions_never_tasks, content, fs)
    _traced(content, "service_groups", _service_groups, content)
    _overview_covers_scope(content, fs)
    _boundary_verbs(content)
    _traced(content, "partner_phase", _partner_phase, content, fs)
    _contact_roles(content, fs)
    _traced(content, "form_chrome", _form_chrome, content, fs)
    _planning_basis(content, fs)
    _room_counts(content)
    # "the above" points into the source note's list, not the SOW (000123
    # pass T: "Use the cords for the above.").
    for k in LISTS:
        for x in content.get(k) or []:
            if isinstance(x, dict) and isinstance(x.get("text"), str):
                x["text"] = re.sub(r"(?i)\b(for|with|to|of|on) the above(?:[- ](?:gear|items?|equipment|devices?|list|hardware))?\b",
                                   r"\1 the installed equipment", x["text"])
    _traced(content, "exclusion_not_task", _no_exclusion_negates_task, content)
    _traced(content, "own_documents", _own_documents, content)
    _traced(content, "our_word_on_disposal", _our_word_on_disposal, content, envelope)
    _traced(content, "preconfigured", _preconfigured, content)
    _traced(content, "not_job_terms", _not_job_terms, content)
    _traced(content, "summary_consistent", _summary_consistent, content)
    _traced(content, "facts_to_action_heading", _facts_to_action_heading, content)
    _traced(content, "room_from_source", _room_from_source, content, fs, envelope)
    _traced(content, "room_from_section", _room_from_section, content, envelope)
    _traced(content, "explanation_not_exclusion", _explanation_not_exclusion, content, fs)
    _traced(content, "summary_order", _summary_order, content)
    _deliverables_hygiene(content)
    _reconcile_dropped(content)
    # A one-room job says "the room" (010120: "Verify the equipment for each
    # room" on a single conference room).
    if family == "av_conference_room" and not re.search(r"(?i)\brooms\b", pool_text):
        def _one_room(t: str) -> str:
            t = re.sub(r"(?i)\beach room('s)?\b", lambda m_: "the room" + (m_.group(1) or ""), t)
            return re.sub(r"\bthe rooms\b", "the room", t)
        for k in LISTS:
            for x in content.get(k) or []:
                if isinstance(x, dict) and isinstance(x.get("text"), str):
                    x["text"] = _one_room(x["text"])
        for d in content.get("deliverables") or []:
            if isinstance(d, dict):
                d["description"] = [_one_room(str(t)) for t in d.get("description") or []]
    # Assessment is ticked for assessment work the task list does: a survey
    # the summary mentions, or one offered as optional, is not it (010326's
    # "Twilio survey integration" under a separate SOW, 010082's optional survey).
    if not transition and "assessment" in (content.get("services") or []) and len(content["services"]) > 1 and not any(
            isinstance(x, dict) and x.get("kind") != "heading" and re.search(r"(?i)\b(survey|assess|audit|walk-?through|inventory)", str(x.get("text")))
            and not re.search(r"(?i)\boptional|if (requested|needed|required)|may be|separate", str(x.get("text")))
            for x in content.get("scope") or []):
        content["services"] = [v for v in content["services"] if v != "assessment"]
    content.update(library_counts)
    content["facts_given"] = len(fs)
    # Everything the rules changed after the writer, rule by rule where traced,
    # the rest as "writer_rules": the queue SowSmith shows as "Teach the heads".
    _std_gone(content, "writer rules (untraced)")
    content.pop("_noise_atoms", None)
    content.pop("_slots", None)
    content.pop("_noise_re", None)
    content["teachable"] = {k: v[0] for k, v in TEACHABLE.items()}
    traced = {(tuple(d["atom_ids"]), d["head"]) for d in content.get("rule_decisions") or []}
    content["rule_decisions"] = (content.get("rule_decisions") or []) + [
        d for d in _decisions("writer_rules", writer_snap, _snap(content)) if (tuple(d["atom_ids"]), d["head"]) not in traced]
    return content


#: A cleanup sentence: it hauls or clears trash, debris or packaging (not
#: "clean up the existing cabling", not disposing of replaced gear).
_NOMINAL_VERB = {"installation": "instal", "install": "instal", "installing": "instal", "mounting": "mount", "removal": "remov",
                 "replacement": "replac", "configuration": "configur", "cabling": "cabl", "testing": "test",
                 "relocation": "relocat", "decommissioning": "decommission", "packing": "pack", "labeling": "label"}
_GENERIC_OBJ = frozenset("hardware equipment device devices system systems new the any all of".split())


def _no_exclusion_negates_task(content: dict[str, Any]) -> None:
    """An exclusion never takes back a task the SOW sells: the item of an
    out-of-scope list that names a cited task's work on generic gear goes
    from the list ("Hardware installation, configuration, ..." beside
    "Install one (1) APC Smart-UPS", 010376 pass U); an item naming other
    gear stays ("installation of new in-wall cabling")."""
    tasks = [str(x.get("text")) for x in content.get("scope") or [] if isinstance(x, dict) and x.get("kind") != "heading"
             and x.get("cites") and not re.search(r"(?i)\b(not|no|never|except)\b", str(x.get("text")))]
    verb_of = lambda t: (lambda v: v[:-1] if v.endswith("e") else v)(_stem((re.findall(r"[a-z]+", t.lower()) or [""])[0]))
    by_verb: dict[str, set[str]] = {}
    for t in tasks:
        by_verb.setdefault(verb_of(t), set()).update(_object_toks(t))
    if not by_verb:
        return
    out = []
    for x in content.get("out_of_scope") or []:
        t = str(x.get("text") if isinstance(x, dict) else x)
        items = [p_ for p_ in re.split(r",\s*(?:or\s+|and\s+)?|\s+or\s+", t.rstrip(".")) if p_.strip()]
        # A list's trailing "of the existing displays" is every item's object.
        tail = re.search(r"(?i)\b(?:of|for)\s+(.+)$", t.rstrip("."))
        shared = {w for w in re.findall(r"[a-z]+", tail.group(1).lower())} - _GENERIC_OBJ - set(_NOMINAL_VERB) if tail else set()
        keep = []
        for it in items:
            if re.search(r"(?i)\b(outside|beyond|other than|unless|except)\b", it):
                keep.append(it.strip())
                continue
            words = re.findall(r"[a-z]+", it.lower())
            # "packing materials" is a supply, not packing work.
            verbs = [_NOMINAL_VERB[w] for i, w in enumerate(words) if w in _NOMINAL_VERB
                     and not (i + 1 < len(words) and words[i + 1] in ("materials", "material", "supplies", "tape", "boxes"))]
            own = {w for w in words if w not in _NOMINAL_VERB and w not in _GENERIC_OBJ}
            objs = own or shared
            hit = verbs and verbs[0] in by_verb and (not objs or {_stem(w) for w in objs} & by_verb[verbs[0]])
            if hit and len(items) > 1 and not (shared and not own and "existing" in shared):
                continue
            keep.append(it.strip())
        if len(keep) < len(items) and isinstance(x, dict):
            body = _join(keep)
            if re.search(r"\bor\b", t) and not re.search(r"\band\b", t):
                body = re.sub(r",? and (?!.*\band\b)", ", or " if len(keep) > 2 else " or ", body)
            x = {**x, "text": body[:1].upper() + body[1:] + "."}
        out.append(x)
    content["out_of_scope"] = out


#: A source that explains data (why rows of a sheet have no address).
_EXPLAINS_DATA = re.compile(r"(?i)\b(sheet|spreadsheet|rows?|entries|tab|columns?|listed|unaddressed|the list)\b")


def _explanation_not_exclusion(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A source that explains data is never an exclusion (010326 pass U: "The
    other unaddressed phones within the sheet are ... virtual extensions ...,
    fax lines, or phones connected via ATA or RJ11" became "exclude the
    virtual, fax, call-routing, ATA, and RJ11 entries", against the 18 ATAs
    the job installs). The exclusion clause goes from a task; an exclusion
    resting only on such sources goes."""
    by = {f["id"]: f for f in fs}
    excl = re.compile(r"(?i)\b(exclud\w*|except\w*|not (be )?(included|installed|in scope))\b")
    def explains(c: str) -> bool:
        t = str((by.get(c) or {}).get("text") or "")
        return bool(_EXPLAINS_DATA.search(t)) and not re.search(r"(?i)\b(not (in|part of) (the )?scope|do not install|don'?t install|skip|out of scope)\b", t)
    for k in ("scope", "assumptions", "out_of_scope"):
        keep = []
        for x in content.get(k) or []:
            if not (isinstance(x, dict) and not x.get("standard") and x.get("cites") and excl.search(str(x.get("text")))):
                keep.append(x)
                continue
            exp = [c for c in x["cites"] if explains(c)]
            # Only when no other cite of the line states an exclusion.
            if not exp or any(c not in exp and excl.search(str((by.get(c) or {}).get("text") or "")) for c in x["cites"]):
                keep.append(x)
                continue
            if k == "scope":
                t = _drop_clause(str(x["text"]), excl)
                if t:
                    keep.append({**x, "text": t})
                continue
            if all(c in exp for c in x["cites"]):
                continue
            keep.append(x)
        content[k] = keep


def _not_job_terms(content: dict[str, Any]) -> None:
    """An order's billing block is not the summary, a contracting remark is not
    the schedule, and a travel remark is an assumption, not a date (010326
    pass U: "Project billing/shipping details are ... Accts Payable" in the
    summary; "Quick signature of the main SOW is needed ..." and "The project
    sites are within a 3-5 mile radius ..." under Scheduling)."""
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    content["overview"] = [x for x in content.get("overview") or [] if not re.search(
        r"(?i)\b(billing|bill[- ]to|ship[- ]to|shipping details|accounts? payable|accts\.? payable|remit to)\b", tx(x))] or content.get("overview")
    labeled = re.compile(r"^[\w /#-]{2,40}:\s")
    keep = []
    for x in content.get("schedule") or []:
        t = tx(x)
        if not labeled.match(t) and re.search(r"(?i)\bsignature\b|\bsign(ed|ing)? (the |off on the )?(main |master )?(sow|contract|agreement)\b", t):
            continue
        if not labeled.match(t) and re.search(r"(?i)\b(mile radius|miles? (apart|away|of each other)|travel time)\b", t):
            content.setdefault("assumptions", []).append(x)
            continue
        keep.append(x)
    content["schedule"] = keep


#: Kinds of work a library or writer line can carry onto a job that has none.
_WORK_KINDS = ((r"\bcabling\b|\blow[- ]voltage\b|\bcable (pathways?|organization|management|testers?)\b", r"\bcabl|\bwir(e|ing)\b|\bpatch|\bdrops?\b|low[- ]voltage|\bcat ?[56]"),
               (r"\bmigrat\w*\b|\bequipment moves?\b|\bmoves?\b", r"\bmigrat|\bmov(e|es|ing)\b|\brelocat"))


def _work_no_fact_names(content: dict[str, Any], pool: str) -> None:
    """On an IT transition, a line about cabling, moves or migrations on a job
    whose facts name none of them goes (010419 pass U: "Assist with
    workstation cabling and cable organization", the low-voltage pathway
    assumptions, and tools "to perform workstation migrations" and "for
    equipment moves")."""
    for line_pat, fact_pat in _WORK_KINDS:
        if re.search(fact_pat, pool, re.I):
            continue
        for k in ("scope", "assumptions", "provider_tools", "provider_materials"):
            keep = []
            for x in content.get(k) or []:
                t = str(x.get("text") if isinstance(x, dict) else x)
                if isinstance(x, dict) and not x.get("standard") and x.get("kind") != "heading" \
                        and re.search(line_pat, t, re.I):  # no fact names it, so no cite states it
                    # A tool line keeps its other purpose ("... for equipment
                    # moves and cable management" goes whole; "... to perform
                    # workstation migrations and troubleshooting" too).
                    continue
                keep.append(x)
            content[k] = keep


def _duty_owner(content: dict[str, Any], fs: list[dict[str, Any]] | None = None) -> None:
    """A Provider task never claims the work a customer duty gives the
    customer: with "Provide all device configurations and troubleshooting
    support" or a remote engineer who owns configuration, "hands on
    troubleshooting" leaves the task list (010419 pass U: "Perform basic issue
    triage and hands on troubleshooting")."""
    duty = " ".join(str(x.get("text")) for x in content.get("customer_responsibilities") or [] if isinstance(x, dict)).lower()
    owned = [w for w, pat in (("troubleshoot", r"troubleshoot"), ("configur", r"configur")) if re.search(pat, duty)]
    if not owned:
        return
    pat = re.compile(r"(?i)\b(?:basic |hands[- ]on |technical |advanced )*(%s)\w*" % "|".join(owned))
    by = {f["id"]: f["text"] for f in fs or []}
    keep = []
    for x in content.get("scope") or []:
        t = str(x.get("text") if isinstance(x, dict) else x)
        cited = " ".join(by.get(c, "") for c in (x.get("cites") or []) if isinstance(x, dict))
        if not (isinstance(x, dict) and not x.get("standard") and not pat.search(cited) and x.get("kind") != "heading" and pat.search(t)) \
                or re.search(r"(?i)\b(escalat|coordinat|under (the )?direction|as directed|support (the )?customer)", t):
            keep.append(x)
            continue
        m = re.match(r"(?i)^(\s*\w+\s+)(.+?)\.?$", t)
        items = [i for i in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(2)) if i.strip()] if m else []
        left = [i for i in items if not pat.search(i)]
        if left and len(left) < len(items):
            keep.append({**x, "text": m.group(1) + _join(left) + "."})
    content["scope"] = keep


_ROLE_AREAS = (("Network", r"(?i)\bnetwork|switch|firewall|wireless|wi-?fi|access points?|circuit|\bisp\b|router|\brack|closet|\bidf\b|\bmdf\b|patch|uplink"),
               ("End User Computing", r"(?i)end[- ]user|workstation|desktop|laptop|printer|\busers?\b|help ?desk|peripheral|o365|m365|e-?mail|desk-?side"))


def _role_groups(content: dict[str, Any]) -> None:
    """A transition staffed by role groups its tasks by role, not under
    "Installation" or a site name (010419 pass U: one network technician and
    one help desk technician; the PM wrote Network / EUC / Shared)."""
    if not re.search(r"(?i)\btechnicians?\b.*\band\b.*\btechnicians?\b", str(content.get("scope_lead") or "")):
        return
    sc = content.get("scope") or []
    closing = re.compile(r"(?i)clos|clean|wrap|complet")
    groups = {name: [] for name, _ in _ROLE_AREAS}
    shared, close, in_close = [], [], False
    for x in sc:
        if isinstance(x, dict) and x.get("kind") == "heading":
            in_close = bool(closing.search(str(x.get("text"))))
            continue
        if in_close:
            close.append(x)
            continue
        t = str(x.get("text") if isinstance(x, dict) else x)
        hits = [name for name, pat in _ROLE_AREAS if re.search(pat, t)]
        (groups[hits[0]] if len(hits) == 1 else shared).append(x)
    head = lambda n: {"text": n, "cites": [], "kind": "heading"}
    out = []
    for name, _ in _ROLE_AREAS:
        if groups[name]:
            out += [head(name)] + groups[name]
    if shared:
        # Nothing role-specific: one onsite group, not "Installation" and a
        # site name.
        out += [head("Shared" if out else "Onsite Support")] + shared
    content["scope"] = out + ([head("Closeout")] + close if close else [])


def _summary_consistent(content: dict[str, Any]) -> None:
    """A summary sentence never retains what the task list removes (010120
    pass U: "... and twelve (12) speakers while retaining the existing room
    displays, TV mounts, and ceiling speaker system")."""
    removed = set()
    for x in content.get("scope") or []:
        t = str(x.get("text") if isinstance(x, dict) else x)
        if isinstance(x, dict) and x.get("kind") != "heading" and re.match(r"(?i)\s*(also\s+)?(remove|decommission|dismantle|take down)\b", t):
            removed |= {w.rstrip("s") for w in re.findall(r"[a-z]+", t.lower())} & {
                "speaker", "display", "tv", "camera", "soundbar", "mount", "projector", "screen", "phone", "switch", "rack", "ups"}
    if not removed:
        return
    for x in content.get("overview") or []:
        if not isinstance(x, dict):
            continue
        t = str(x.get("text"))
        m = re.search(r"(?i)(,?\s*(?:while|and)\s+(?:retaining|keeping|leaving)\s+)(.+?)([.;]|$)", t)
        if not m:
            continue
        items = [i for i in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(2)) if i.strip()]
        keep = [i for i in items if not ({w.rstrip("s") for w in re.findall(r"[a-z]+", i.lower())} & removed
                                         and not re.search(r"(?i)\bmounts?\b", i))]
        if len(keep) == len(items):
            continue
        x["text"] = (t[:m.start()] + (m.group(1) + _join([k.strip() for k in keep]) if keep else "") + m.group(3) + t[m.end():]).strip()


def _facts_to_action_heading(content: dict[str, Any]) -> None:
    """A cited deal step sits under the heading whose work it does, not under
    a testing or closing group (010120 pass U: speaker removal and the wall
    mount under "Installation and Testing", the cable relocation and "Also
    remove the circular ... speakers" under Closeout)."""
    sc = content.get("scope") or []
    heads = [i for i, x in enumerate(sc) if isinstance(x, dict) and x.get("kind") == "heading"]
    if len(heads) < 3:
        return
    generic = re.compile(r"(?i)test|clos|clean|wrap|complet|all work areas")
    verb = lambda t: _stem((re.findall(r"[a-z]+", re.sub(r"(?i)^\s*also\s+", "", t).lower()) or [""])[0])
    groups: list[list[Any]] = []
    for x in sc:
        if isinstance(x, dict) and x.get("kind") == "heading":
            groups.append([x])
        elif groups:
            groups[-1].append(x)
        else:
            groups.append([None, x])
    for g in groups:
        if not (g[0] and generic.search(str(g[0].get("text")))):
            continue
        for x in list(g[1:]):
            if not (isinstance(x, dict) and x.get("cites") and not x.get("standard")) \
                    or re.search(r"(?i)debris|clean|trash|photo|confirm|notify|document", str(x.get("text"))):
                continue
            t = str(x.get("text"))
            v = verb(t)
            obj = _object_toks(t)
            home = next((h for h in groups if h is not g and h[0] and not generic.search(str(h[0].get("text")))
                         and (obj & _object_toks(str(h[0].get("text")))
                              or any(isinstance(y, dict) and y.get("cites") and verb(str(y.get("text"))) == v
                                     and obj & _object_toks(str(y.get("text"))) for y in h[1:]))), None)
            if home is None and re.match(r"(?i)\s*(also\s+)?(remove|decommission|disconnect)\b", t):
                home = next((h for h in groups if h[0] and re.search(r"(?i)decommission|removal|remove", str(h[0].get("text")))), None)
            if home is not None:
                g.remove(x)
                home.append(x)
    for g in groups:
        for x in g[1:]:
            if isinstance(x, dict) and isinstance(x.get("text"), str) and re.match(r"(?i)\s*also\s+\w", x["text"]):
                body = re.sub(r"(?i)^\s*also\s+", "", x["text"])
                x["text"] = body[:1].upper() + body[1:]
    content["scope"] = [x for g in groups for x in g if x is not None]


#: A room a line names ("the small conference room", "executive office").
_ROOM = re.compile(r"(?i)\b(?:(?:main|small|large|big|second|first|executive|training|board|huddle|break|front|back)\s+)?"
                   r"(?:conference|meeting|training|board|executive|huddle|break)\s+(?:room|office)s?\b|\bexecutive (?:room|office)s?\b|\bboardroom\b")


def _room_from_source(content: dict[str, Any], fs: list[dict[str, Any]], envelope: dict[str, Any]) -> None:
    """A room is never invented: a task naming a room its cited lines do not
    name takes the room its source points at ("this specific room" is the one
    the sentence before it names) and moves under that room's heading (010171 pass U: "Install one (1) Logitech Sight in the main
    conference room" where the source's room is the small conference room;
    "main" was the customer's existing, out-of-scope room)."""
    atoms = [a for a in envelope.get("atoms") or [] if isinstance(a, dict)]
    by_id = {str(a.get("id")): a for a in atoms}
    key = lambda a: ((a.get("locator") or {}).get("message_index") or 0, (a.get("locator") or {}).get("line_start") or 0,
                     (a.get("locator") or {}).get("sentence_index") or 0)
    def antecedent(c: str) -> str | None:
        a = by_id.get(c)
        if not a or not re.search(r"(?i)\b(this|that|the same|this specific|that specific)\s+(room|office|space|area)\b", str(a.get("text"))):
            return None
        same = sorted((b for b in atoms if b.get("artifact_id") == a.get("artifact_id")), key=key)
        i = next((j for j, b in enumerate(same) if b is a), 0)
        for b in reversed(same[max(0, i - 4):i]):
            mm = _ROOM.search(str(b.get("text")))
            if mm:
                return mm.group(0).lower()
        return None
    sc = content.get("scope") or []
    for x in list(sc):
        if not (isinstance(x, dict) and x.get("cites") and x.get("kind") != "heading"):
            continue
        t = str(x.get("text"))
        mm = _ROOM.search(t)
        if not mm:
            continue
        cited = " ".join(str((by_id.get(c) or {}).get("text") or "") for c in x["cites"]).lower()
        if mm.group(0).lower().rstrip("s") in cited:
            continue
        room = next((r for r in (antecedent(c) for c in x["cites"]) if r), None)
        if room:
            x["text"] = t[:mm.start()] + room + t[mm.end():]
            words = {w for w in re.findall(r"[a-z]+", room) if w not in ("room", "office", "the")}
            home = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                         and words & set(re.findall(r"[a-z]+", str(y.get("text")).lower()))), None)
            if home is not None:
                sc.remove(x)
                nxt = next((i for i, y in enumerate(sc) if i > home and isinstance(y, dict) and y.get("kind") == "heading"), len(sc))
                sc.insert(nxt, x)

    content["scope"] = sc


def _room_from_section(content: dict[str, Any], envelope: dict[str, Any]) -> None:
    """A row's room is its sheet section's: a task citing rows whose
    section_path ends in another area ("Common Area Displays") names that
    area and sits under its heading, not under a heading the writer coined
    (010116 pass U: two QM65C displays "in the Boardroom" under "Boardroom
    Displays")."""
    by_id = {str(a.get("id")): a for a in envelope.get("atoms") or [] if isinstance(a, dict)}
    def area(c: str) -> str:
        sp = (by_id.get(c, {}).get("locator") or {}).get("section_path") or []
        last = str(sp[-1]).strip() if sp else ""
        return "" if not last or re.fullmatch(r"(?i)sheet\s*\d*|table\s*\d*|summary|totals?|bom|quote", last) else last
    sc = content.get("scope") or []
    heads = {str(x.get("text")).strip().lower(): x for x in sc if isinstance(x, dict) and x.get("kind") == "heading"}
    for x in list(sc):
        if not (isinstance(x, dict) and x.get("cites") and x.get("kind") != "heading"):
            continue
        areas = {area(c) for c in x["cites"]} - {""}
        if len(areas) != 1:
            continue
        a = areas.pop()
        t = str(x.get("text"))
        mm = re.search(r"\s+in the ([A-Z][\w-]*(?: [A-Z][\w-]*){0,3})(?=[.,;]|$)", t)
        if not mm or mm.group(1).lower() == a.lower() or a.lower() in t.lower():
            continue
        place = re.sub(r"(?i)\s+(displays?|devices|equipment|items)$", "", a)
        x["text"] = t[:mm.start()] + f" in the {place}" + t[mm.end():]
        sc.remove(x)
        h = heads.get(a.lower())
        if h is None:
            h = {"text": a, "cites": [], "kind": "heading"}
            at = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                       and re.search(r"(?i)all work areas|test|clos", str(y.get("text")))), len(sc))
            sc.insert(at, h)
            heads[a.lower()] = h
        j = next(i for i, y in enumerate(sc) if y is h)
        nxt = next((i for i, y in enumerate(sc) if i > j and isinstance(y, dict) and y.get("kind") == "heading"), len(sc))
        sc.insert(nxt, x)
    # A heading left empty goes.
    content["scope"] = [y for i, y in enumerate(sc) if not (isinstance(y, dict) and y.get("kind") == "heading"
                        and (i + 1 >= len(sc) or (isinstance(sc[i + 1], dict) and sc[i + 1].get("kind") == "heading"))
                        and not re.search(r"(?i)clos", str(y.get("text"))))]


def _our_word_on_disposal(content: dict[str, Any], envelope: dict[str, Any]) -> None:
    """PurTera's own word on disposal outranks a customer's agenda bullet: when
    PurTera wrote that it is not approved for ITAD or asked to leave removed
    devices onsite, the SOW never promises disposal; removed gear stays onsite
    (010264 pass U: "Retire and appropriately dispose of replaced InTouch 9000
    clocks" against "we are not currently approved to perform ITAD services")."""
    ours = " ".join(str(a.get("text") or "") for a in envelope.get("atoms") or []
                    if "purtera" in str((a.get("locator") or {}).get("sender") or "").lower())
    if not re.search(r"(?i)\bnot\b[^.]{0,30}\b(approved|certified|licensed)\b[^.]{0,40}\b(itad|dispos\w*|recycl\w*)|"
                     r"\b(left|leave)\s+(the\s+)?(removed\s+)?(devices|equipment|clocks|units|hardware)?\s*onsite\b|"
                     r"removed devices left onsite", ours):
        return
    disp = re.compile(r"(?i)\b(retire\w* and (appropriately )?dispos\w*|dispos\w*|itad|recycl\w*|e-?waste|data destruction)\b")
    keep_line = "Leave the removed equipment onsite at the Customer-designated location."
    for k in ("overview", "scope", "deliverables_text"):
        out = []
        for x in content.get(k) or []:
            t = str(x.get("text") if isinstance(x, dict) else x)
            if not (isinstance(x, dict) and disp.search(t)) or re.search(r"(?i)\b(not|no|excluded|outside)\b", t):
                out.append(x)
                continue
            t2 = _drop_clause(t, disp)
            if t2 and len(t2.split()) >= 5 and not disp.search(t2):
                out.append({**x, "text": t2})
            elif k == "scope" and not _said(keep_line, out, 0.8):
                out.append({**x, "text": keep_line})
        content[k] = out
    for d in content.get("deliverables") or []:
        if isinstance(d, dict):
            d["description"] = [t for t in d.get("description") or [] if not disp.search(str(t))]
    oos = content.setdefault("out_of_scope", [])
    ex = "Disposal, recycling or ITAD of removed equipment."
    if not any(disp.search(str(x.get("text") if isinstance(x, dict) else x)) for x in oos):
        oos.append({"text": ex, "cites": [], "kind": "method"})


def _preconfigured(content: dict[str, Any]) -> None:
    """Devices configured and provisioned before they ship leave configuration,
    provisioning and Wi-Fi setup with the customer: no Provider task or
    summary clause claims them (010264 pass U: "Provider will standardize
    configuration and provisioning ..." and "Configure Wi-Fi or network
    settings after installation" beside "Replacement clocks ... are
    configured and provisioned before shipment")."""
    terms = " ".join(str(x.get("text") if isinstance(x, dict) else x) for k in ("assumptions", "customer_responsibilities")
                     for x in content.get(k) or [])
    if not re.search(r"(?i)(pre-?configur\w*|configured|provisioned)[^.]{0,40}\bbefore (shipment|shipping|ship|arrival|delivery|deployment)|"
                     r"arrive[^.]{0,30}pre-?configured", terms):
        return
    pat = re.compile(r"(?i)\b(configur\w*|provision\w*|wi-?fi|network settings|clock setup|device setup)\b")
    content["scope"] = [x for x in content.get("scope") or [] if not (
        isinstance(x, dict) and not x.get("standard") and x.get("kind") != "heading"
        and re.match(r"(?i)\s*(configure|provision|set up wi-?fi|program)\b", str(x.get("text"))))]
    for x in content.get("overview") or []:
        if not isinstance(x, dict):
            continue
        t = str(x.get("text"))
        mm = re.match(r"(?i)^(\s*provider will\s+)(.+?)\.?$", t)
        if not (mm and pat.search(t)):
            continue
        parts = [p_ for p_ in re.split(r",\s*(?:and\s+)?|\s+and\s+", mm.group(2)) if p_.strip()]
        left = [p_ for p_ in parts if not pat.search(p_)]
        # The part that carried the verb went: the rest is no sentence.
        verb_ok = bool(left) and (left[0] is parts[0] or re.match(
            r"(?i)(install|replace|remove|validate|verify|mount|connect|coordinate|perform|provide|support|test|deploy)\b", left[0].strip()))
        x["text"] = (mm.group(1) + _join(left) + ".") if verb_ok and len(" ".join(left).split()) >= 3 else ""
    content["overview"] = [x for x in content.get("overview") or [] if not (isinstance(x, dict) and not str(x.get("text")).strip())]


def _own_documents(content: dict[str, Any]) -> None:
    """Provider supplies its own certificate of insurance; the customer gives
    the site's requirements (010376 pass U: "Provide PurTera's certificate of
    insurance (COI) before site work begins" as a customer duty)."""
    own = re.compile(r"(?i)\b(purtera|provider)'?s\s+(certificate of insurance|coi|insurance certificate|w-?9)\b")
    for x in list(content.get("customer_responsibilities") or []):
        if not (isinstance(x, dict) and own.search(str(x.get("text")))):
            continue
        doc = "certificate of insurance (COI)" if re.search(r"(?i)coi|insurance", str(x["text"])) else "W-9"
        when = " before site work begins" if re.search(r"(?i)before (site )?work|before arrival|prior to", str(x["text"])) else ""
        x["text"] = f"Provide the site's {doc} requirements{when}."
        asm = content.setdefault("assumptions", [])
        line = f"Provider will provide its {doc} to the site{when}, as the site requires."
        if not _said(line, asm, 0.6):
            asm.append({"text": line, "cites": list(x.get("cites") or []), "kind": "fact"})


_CLEANUP = re.compile(r"(?i)^(?!.*\b(install|mount|replace|configure|connect)(s|ed|ing)?\b).*\b(trash|debris|packing materials?|packaging)\b")


def _summary_order(content: dict[str, Any]) -> None:
    """The summary leads with the work: a cleanup or haul-away sentence comes
    after the install sentences, as its step closes the task list (010373
    pass U: "Provider will haul away installation trash" opened the summary)."""
    ov = [x for x in content.get("overview") or [] if isinstance(x, dict)]
    # One sentence per claim: a sentence whose counts another one with the
    # same verb gives in full goes, the fuller one taking its place (010209
    # pass U: "replace store switches ... thirty (30) sites ... two (2) to five
    # (5) switches" and "replace two (2) to five (5) switches per store across
    # approximately thirty (30) sites, with the project estimate at ...").
    nums = lambda x: set(re.findall(r"\d+", _digits(str(x.get("text")))))
    verb = lambda x: _action(re.sub(r"(?i)^\s*provider will\s+", "", str(x.get("text"))))[0]
    for i, x in enumerate(list(ov)):
        if len(nums(x)) < 2 or not any(x is y for y in ov):
            continue
        fuller = next((y for y in ov if y is not x and verb(y) == verb(x) and nums(x) < nums(y)), None)
        if fuller is not None:
            j = next(k for k, y in enumerate(ov) if y is x)
            ov = [y for y in ov if y is not fuller]
            ov[j:j + 1] = [fuller]
    content["overview"] = ov
    work = [x for x in ov if not _CLEANUP.search(str(x.get("text")))]
    if work and len(work) < len(ov):
        content["overview"] = work + [x for x in ov if all(x is not w for w in work)]


_ROLE_WORDS = re.compile(r"(?i)^(customer|client|provider|we|they|it|seller|vendor|subcontractor|contractors?|service provider|"
                         r"partner|technician|tech|team|purtera|puretera|the [a-z]+)$")
_TP_SAYS = re.compile(r"\b([A-Z][A-Za-z0-9&.-]{1,}(?: [A-Z][A-Za-z0-9&.-]+)?) (?:will|shall|can|would) (?:also )?"
                      r"(?:provide|supply|ship|furnish|deliver|bring|recycle|collect|pick up|transport|issue)\b")
_QUOTE_DOC = re.compile(r"(?i)quote|proposal|invoice|estimate|\bpo\b|purchase order")
_PRICE_TERM = re.compile(r"(?i)\btax(es)?\b|invoice|freight|\$\s?\d|\bprice|pricing|payment|quote (is )?valid|per (pound|lb)|/\s?(lb|pound)\b|"
                         r"purchase order|\bpo\b|commodity value")


def _third_parties(envelope: dict[str, Any], content: dict[str, Any]) -> tuple[dict[str, str], set[str]]:
    """Named other vendors on the deal and the atoms they wrote: a company
    that says "<Name> will provide ..." and writes from its own mail domain
    (010376: ThinkTLS, the recycler, kgrove@thinktls.com, and its pickup
    proposal). Not PurTera, the account, the end customer or a person."""
    atoms = envelope.get("atoms") or []
    crm = envelope.get("crm") or {}
    own = " ".join(str(v or "") for v in (crm.get("account_name"), content.get("end_customer"), crm.get("deal_name"))).lower()
    own_key = re.sub(r"[^a-z0-9]", "", own)
    domains = {str(m_.group(1)).lower() for a in atoms for m_ in [re.search(r"@([\w.-]+)", str((a.get("locator") or {}).get("sender") or ""))] if m_}
    names: dict[str, str] = {}
    for a in atoms:
        for mm in _TP_SAYS.finditer(str(a.get("text") or "")):
            n = mm.group(1).strip()
            key = re.sub(r"[^a-z0-9]", "", n.lower())
            if len(key) < 3 or _ROLE_WORDS.match(n) or key in own_key or "purtera" in key:
                continue
            if any(key in re.sub(r"[^a-z0-9]", "", d.split(".")[0]) for d in domains if "purtera" not in d):
                names[key] = n
    if not names:
        return {}, set()
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for a in atoms:
        by_doc.setdefault(str(a.get("artifact_id")), []).append(a)
    ids: set[str] = set()
    for a in atoms:
        sender = str((a.get("locator") or {}).get("sender") or "").lower()
        dom = (re.search(r"@([\w.-]+)", sender) or [None, ""])[1]
        if dom and any(k in re.sub(r"[^a-z0-9]", "", dom.split(".")[0]) for k in names):
            ids.add(str(a.get("id")))
    for doc, group in by_doc.items():
        # A document (not a mail thread) in which the vendor states its supply
        # is the vendor's document, row by row ("Box Truck w/ Liftgate").
        if any(str((a.get("locator") or {}).get("sender") or "") for a in group):
            continue
        if any(re.sub(r"[^a-z0-9]", "", mm.group(1).lower()) in names for a in group for mm in _TP_SAYS.finditer(str(a.get("text") or ""))):
            ids |= {str(a.get("id")) for a in group}
    return names, ids


def _third_party(content: dict[str, Any], envelope: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """Another party's supply, terms and work are never Provider's (010376:
    the recycler's box truck, vehicles and packing supplies in Technician
    Provided Tools and Materials; its commodity $/lb and LTL notes and the
    hardware quote's tax line as PurTera assumptions; "Provider will ...
    eWaste the removed UPS" while the recycler recycles). A Provider
    materials or tools line no supplier answer stands behind goes too."""
    sup = {f["id"]: f.get("supplier") for f in fs}
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    names, tp = _third_parties(envelope, content)
    name = next(iter(names.values()), "")
    asm = content.setdefault("assumptions", [])
    for k in ("provider_materials", "provider_tools"):
        kept = []
        for x in content.get(k) or []:
            cites = x.get("cites") or [] if isinstance(x, dict) else []
            if not cites or x.get("standard"):
                kept.append(x)
                continue
            if tp and all(c in tp for c in cites):
                t = tx(x).strip().rstrip(".")
                t = re.sub(r"(?i)^\s*(provide|supply|furnish)\s+", "", t)
                line = {**x, "text": f"{name} provides {t[:1].lower() + t[1:]}."}
                if not _said(line["text"], asm, 0.6):
                    asm.append(line)
                continue
            if all(not sup.get(c) for c in cites):
                continue  # no supplier answer at all: not ours to supply
            kept.append(x)
        content[k] = kept
    docs = {str(d.get("artifact_id")): str(d.get("filename") or "") for d in envelope.get("documents") or []}
    art = {str(a.get("id")): str(a.get("artifact_id")) for a in envelope.get("atoms") or []}
    quote_term = lambda c: bool(_QUOTE_DOC.search(docs.get(art.get(c, ""), ""))) and "purtera" not in docs.get(art.get(c, ""), "").lower()
    content["assumptions"] = [x for x in asm if not (
        isinstance(x, dict) and x.get("cites") and not x.get("standard") and not tx(x).startswith(f"{name} provides " if name else "\0")
        and (all(c in tp for c in x["cites"]) or (all(quote_term(c) for c in x["cites"]) and _PRICE_TERM.search(tx(x)))))]
    if not tp:
        return
    said = " ".join(str(a.get("text") or "") for a in envelope.get("atoms") or [] if str(a.get("id")) in tp).lower()
    recycles = bool(re.search(r"recycl|e-?waste|dispos|settlement|certificate of (recycling|destruction)|\besg\b", said))
    hauls = bool(re.search(r"\btruck|transport|pick ?up|collect|vehicle", said))
    if not (recycles or hauls):
        return
    work = r"e-?waste|recycl\w*|dispos\w*( of)?" if recycles else r"(?!x)x"
    # A summary sentence that is the other vendor's work, or rests only on
    # its lines, is that vendor's sentence (010376 pass U: "Provider will use
    # a box truck with liftgate ..." from "ThinkTLS will provide dedicated
    # vehicles"; "Provider will eWaste the removed UPS").
    theirs = r"(?:" + work + r"|use (a |the )?(box )?truck|transport\w*|haul\w*|pick ?up)" if hauls else r"(?:" + work + r")"
    for x in content.get("overview") or []:
        if isinstance(x, dict) and name and re.match(r"(?i)\s*provider will\b", tx(x)) and (
                (x.get("cites") and all(c in tp for c in x["cites"])) or re.match(r"(?i)\s*provider will\s+" + theirs + r"\b", tx(x))):
            x["text"] = re.sub(r"(?i)^\s*provider will\b", f"{name} will", tx(x))
    for x in content.get("overview") or []:
        if isinstance(x, dict) and name and tx(x).startswith(f"{name} will"):
            continue
        if isinstance(x, dict):
            t = re.sub(r"(?i)\s*(,\s*|\band\s+)+(?:then\s+)?(?:" + work + r")\b[^,.;]*", "", tx(x))
            x["text"] = t if t.endswith(".") else t.rstrip(" ,;") + "."
    for x in content.get("scope") or []:
        if isinstance(x, dict) and x.get("kind") != "heading":
            x["text"] = re.sub(r"(?i)\s+(?:for|to)\s+(?:" + work + r")(?=[.,;]|$)", "", tx(x))
    what = " and ".join(w for w, on in (("offsite transport", hauls), ("recycling", recycles)) if on)
    line = f"{name} performs the {what} of the removed equipment."
    if not _said(line, asm, 0.6):
        content["assumptions"].append({"text": line, "cites": sorted(tp & {f["id"] for f in fs})[:3], "kind": "fact"})
    oos = content.setdefault("out_of_scope", [])
    ex = "Offsite transport, recycling, disposal and recycling certificates for the removed equipment." if recycles \
        else "Offsite transport of the removed equipment."
    if not _said(ex, oos, 0.5):
        oos.append({"text": ex, "cites": [], "kind": "method"})


_LINE_COUNT = re.compile(r"(?:\b([a-z]+(?:-[a-z]+)?) \((\d{1,4})\)|(?<![\w.$/-])(\d{1,4})(?![\w.%/:-]))\s+(?:approximately\s+)?([a-z][a-z-]+)", re.I)
_ATOM_COUNT = re.compile(r"(?<![\w.$/-])(\d{1,4})(?![\w.%/:-])\s+([a-z][a-z-]+)", re.I)


def _cited_counts(content: dict[str, Any], envelope: dict[str, Any]) -> None:
    """A count a line writes for a thing is the count its cited lines give
    that thing (010083: "About ninety-two (92) sites will be single-switch
    locations" cited "Apx 83 sites will be singles"; 92 was the site total).
    Only when the cited lines give that thing exactly one count and never
    the written number."""
    text_of = {str(a.get("id")): str(a.get("text") or "") for a in envelope.get("atoms") or []}
    for k in LISTS:
        for x in content.get(k) or []:
            if not (isinstance(x, dict) and x.get("cites") and x.get("kind") != "heading"):
                continue
            pool = " ".join(text_of.get(c, "") for c in x["cites"])
            if not pool:
                continue
            nums = {int(n) for n in re.findall(r"(?<![\w.])(\d{1,4})(?![\w.])", pool)}
            by_noun: dict[str, dict[int, str]] = {}
            for sent in re.split(r"(?<=[.!?])\s+|\n", pool):
                for mm in _ATOM_COUNT.finditer(sent):
                    by_noun.setdefault(_stem(mm.group(2).lower()), {})[int(mm.group(1))] = sent
            t = str(x.get("text") or "")
            for mm in reversed(list(_LINE_COUNT.finditer(t))):
                n = int(mm.group(2) or mm.group(3))
                said = by_noun.get(_stem(mm.group(4).lower())) or {}
                if n in nums or len(said) != 1:
                    continue
                if mm.group(2) and _words_value(mm.group(1)) not in (None, n):
                    continue  # the words and the digits already disagree: leave it
                right, sent = next(iter(said.items()))
                # The same statement: the cited sentence's words are the line's
                # (a total is not corrected to a sub-count stated beside it).
                clause = re.split(r"[,;.]", t[mm.start():])[0]
                if not _said(sent, [clause], 0.5):
                    continue
                new = f"{_words(right)} ({right})" if mm.group(2) else str(right)
                t = t[:mm.start()] + new + t[(mm.end(2) + 1) if mm.group(2) else mm.end(3):]
            if t != x.get("text"):
                x["text"] = t[:1].upper() + t[1:] if t[:1].islower() and str(x.get("text"))[:1].isupper() else t


_ZERO_ROW = re.compile(r"[:|]\s*0\s*$")


def _zero_rows(content: dict[str, Any], envelope: dict[str, Any]) -> None:
    """A form row answered 0 is nothing to do and nothing to exclude (010147:
    "6U-10U Server Quantity: 0" became "Provide 6U-10U servers are not
    included" in the task list, and four zero cabinet rows a "Leave ...
    out of scope" task). A line most of whose cites are such rows goes."""
    zero = {str(a.get("id")) for a in envelope.get("atoms") or [] if _ZERO_ROW.search(str(a.get("text") or ""))}
    if not zero:
        return
    for k in LISTS:
        content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("cites") and not x.get("standard")
                      and sum(c in zero for c in x["cites"]) > 0.5 * len(x["cites"]))]


_WE_WILL = re.compile(r"(?i)\b(?:we|i)\s*(?:will|'ll|are going to|am going to)\s+(?:also\s+)?(?:be\s+)?([a-z]+)\b([^.;]*)")


def _first_person_duties(content: dict[str, Any], envelope: dict[str, Any]) -> None:
    """What the customer says it will do itself is its duty, never a Provider
    task (010147: "We will also unrack and ready any cables and equipment",
    the customer's questionnaire answer, became the Provider task "Unrack and
    ready any cables and equipment"). Only a line no PurTera person wrote."""
    by = {str(a.get("id")): a for a in envelope.get("atoms") or []}
    # A reseller's "we" may be the delivery team it sells (010171: SHI's "We
    # can mount the NUC"): only the end customer's own words count.
    acct = re.sub(r"[^a-z0-9]", "", str((envelope.get("crm") or {}).get("account_name") or "").lower())
    cr = content.setdefault("customer_responsibilities", [])
    keep = []
    for x in content.get("scope") or []:
        if not (isinstance(x, dict) and x.get("cites") and x.get("kind") != "heading" and not x.get("standard")):
            keep.append(x)
            continue
        t = str(x.get("text") or "")
        verb = _stem((re.findall(r"[a-z]+", t.lower()) or [""])[0])
        theirs = False
        for c in x["cites"]:
            a = by.get(c) or {}
            sender = str((a.get("locator") or {}).get("sender") or "").lower()
            if "purtera" in sender or (acct and acct in re.sub(r"[^a-z0-9@]", "", sender).split("@")[-1]):
                continue
            for mm in _WE_WILL.finditer(str(a.get("text") or "")):
                if _stem(mm.group(1).lower()) == verb and _said(mm.group(1) + mm.group(2), [t], 0.6):
                    theirs = True
        if not theirs:
            keep.append(x)
            continue
        line = {**x, "text": re.sub(r"(?i)\s+as part of [^.]*", "", t).rstrip(" .") + "."}
        if not _said(line["text"], cr, 0.6):
            cr.append(line)
    content["scope"] = keep


_ORDER_ASM = re.compile(r"(?i)^\s*(direct|maintain|provide|ensure|confirm|keep|coordinate|designate|identify|supply|grant|give|arrange|notify|schedule)\b")
_NO_NEED_ASM = re.compile(r"(?i)^\s*(?:avoid|require no|need no|use no)\s+(?:any\s+)?(.+?)(?:\s+for (?:this|the) (?:job|work|project))?\.?$")


def _assumption_voice(content: dict[str, Any]) -> None:
    """An assumption is a condition, never an order with no party (010147:
    "Direct what is packed ..." and "Maintain clear access ..." are the
    customer's duties; "Avoid any lift rental or specialty rigging for this
    job" is "No lift rental or specialty rigging is required", and goes when
    an exclusion already says it; 010376: "Require no lift rental ...")."""
    cr = content.setdefault("customer_responsibilities", [])
    oos = content.get("out_of_scope") or []
    keep = []
    for x in content.get("assumptions") or []:
        if not isinstance(x, dict) or x.get("standard"):
            keep.append(x)
            continue
        t = str(x.get("text") or "").strip()
        nm = _NO_NEED_ASM.match(t)
        if nm:
            t2 = "No " + nm.group(1).rstrip(" .") + " is required."
            if _said(t2, oos, 0.6) or _said(t2, keep, 0.6):
                continue
            keep.append({**x, "text": t2})
        elif _ORDER_ASM.match(t):
            if not _said(t, cr, 0.6):
                cr.append(x)
        else:
            keep.append(x)
    content["assumptions"] = keep


_PACKING_SUPPLY = re.compile(r"(?i)packing|\btape\b|bubble|pallets?\b|shrink|gaylord|foam|crates?|empty boxes|moving boxes|cardboard|labels?\b|carts?\b|dolly|dollies")


def _moved_gear(content: dict[str, Any], family: str) -> None:
    """On a move or pack-up the customer's gear is what is moved, not
    material the customer supplies for the work (010147: the 25-30 curved
    monitors and three document boxes under Customer Provided Materials)."""
    if family not in ("relocation_move", "decommission_packing"):
        return
    # Counted gear only: "The networking rack" for the new site (010038) is
    # material the customer supplies; "Twenty-five (25) ... monitors" is moved.
    gear = [x for x in content.get("customer_materials") or [] if isinstance(x, dict) and x.get("cites")
            and not _PACKING_SUPPLY.search(str(x.get("text"))) and re.search(r"\d", str(x.get("text")))]
    if not gear:
        return
    content["customer_materials"] = [x for x in content.get("customer_materials") or [] if not any(x is g for g in gear)]
    eq = content.setdefault("equipment_in_scope", [])
    for g in gear:
        if not _said(str(g.get("text")), eq, 0.6):
            eq.append(g)


def _destination_owner(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """When the customer manages the destination, Provider does not unload,
    stage or set up there (010147: "We will manage things at the
    destination" beside "Unload and stage the equipment at the destination")."""
    if not any(re.search(r"(?i)\b(we|customer|they|our team)\b.{0,20}\b(manage|handle|take care of|receive)\b.{0,30}\bdestination", f["text"]) for f in fs):
        return
    content["scope"] = [x for x in content.get("scope") or [] if not (
        isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")
        and re.search(r"(?i)\bdestination|delivery (site|location)", str(x.get("text")))
        and re.match(r"(?i)\s*(unload|stage|unpack|unbox|install|set up|place|position|rack|reinstall|connect)\b", str(x.get("text"))))]


def _onsite_designate(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A contact whose onsite designate is still to be named is not the person
    the technician meets (010147: "Bill Tripp - Onsite designate to be named."
    printed "Check in with Bill Tripp"; the office has no employees left)."""
    for f in fs:
        mm = re.search(r"(?i)\b([A-Z][a-z]+(?: [A-Z][a-z]+)?)\s*[-–:,]\s*(?:an?\s+)?onsite designate (?:to be named|tbd|to be determined)", f["text"])
        if not mm:
            continue
        who = mm.group(1)
        for k in ("scope", "arrival_protocol"):
            for x in content.get(k) or []:
                if isinstance(x, dict) and re.search(r"(?i)\b(check in|meet)\w*\s+with\s+" + re.escape(who), str(x.get("text"))):
                    x["text"] = re.sub(r"(?i)\b(check in|meet)(\w*)\s+with\s+" + re.escape(who),
                                       lambda m_: f"{m_.group(1)}{m_.group(2)} with the onsite designate named by {who}", str(x["text"]))
        cr = content.setdefault("customer_responsibilities", [])
        line = "Provide an onsite designate at the origin location and name them before the visit."
        if not any(re.search(r"(?i)onsite designate", str(c.get("text") if isinstance(c, dict) else c)) for c in cr):
            cr.append({"text": line, "cites": [f["id"]], "kind": "fact"})
        break


def _rejected_source(content: dict[str, Any], envelope: dict[str, Any]) -> None:
    """Lines the writer built on what is not a fact of the job. A line citing
    a rejected atom (see _rejected_atoms) whose words come from that atom
    goes (010082: "This work is for a new unit opening" cited the signature's
    "New Unit Opening" beside a real site line; "Include network consultation
    ... ISP upgrade support" cited the menu). An absence note goes ("No
    testing or certification tools were specified" beside the testing tools;
    "The paying contact is unknown"). A summary or assumption naming two or
    more of the deal's people is staffing colour, not the job (010082: "led by
    AJ Deuel, with Chase, Kelly, and Octavian"). A verb glued to a negation
    ("Provide do not install a patch cable") keeps the negation."""
    rejected = _rejected_atoms(envelope)
    text_of = {str(a.get("id")): str(a.get("text") or "") for a in envelope.get("atoms") or []}
    people = [{w for w in str(pp.get("name") or "").split() if len(w) >= 3 and w[0].isupper()}
              for pp in (envelope.get("deal_roster") or {}).get("people") or []]
    def built_on_rejected(x: dict[str, Any]) -> bool:
        bad = [c for c in x.get("cites") or [] if c in rejected]
        if not bad:
            return False
        good = " ".join(text_of.get(c, "") for c in x.get("cites") or [] if c not in rejected).lower()
        words = {w for c in bad for w in re.findall(r"[a-z]{4,}", text_of.get(c, "").lower())}
        return len(bad) == len(x.get("cites") or []) or any(w in str(x.get("text")).lower() and w not in good for w in words)
    for k in LISTS:
        keep = []
        for x in content.get(k) or []:
            if not isinstance(x, dict) or x.get("kind") == "heading":
                keep.append(x)
                continue
            t = str(x.get("text") or "")
            if built_on_rejected(x) or (k != "schedule" and _ABSENCE_LINE.match(t)):
                continue
            if k in ("overview", "assumptions") and sum(1 for ws in people if ws and any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in ws)) >= 2:
                continue
            x["text"] = re.sub(r"^(?:Provide|Perform|Ensure|Complete|Handle)\s+(do not|don't|not)\s+(\w)",
                               lambda m_: "Do not " + m_.group(2), t)
            keep.append(x)
        content[k] = keep


_PROVIDER_SUPPLIES = re.compile(r"(?i)\b(?:provider|technicians?|techs?|technican|purtera)\s+will\s+(?:supply|provide|provdie|bring|furnish)\s+([^;.]+)")
_TOOL_WORDS = re.compile(r"(?i)\b(finder|drill|driver|ladder|tester|toner|tool|tools|level|crimper|punch)\b")


def _provider_half(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """The Provider half of a line that splits who supplies what. 010082's
    "Provider will supply backing board; AJ/CDW will provide/order the 9U
    rack" gave the customer its rack and the backing board went nowhere;
    "Provider will bring toggle bolts and stud finder" left only "anchors"."""
    have = list(content.get("provider_materials") or []) + list(content.get("provider_tools") or [])
    for f in fs:
        if f.get("pictured") or f["section"] in ("figure", "unsure"):
            continue
        for m_ in _PROVIDER_SUPPLIES.finditer(f["text"]):
            for part in re.split(r"\s*(?:,|\band\b)\s*", m_.group(1)):
                part = re.sub(r"(?i)^(the|a|an|all|any)\s+", "", part.strip()).strip()
                if not part or "/" in part or not 1 <= len(part.split()) <= 5 or not re.search(r"[a-z]{3}", part) \
                        or re.search(r"(?i)\b(provider|customer|vendor|contractor|team|support|services?|labor|access|help|"
                                     r"assistance|update|information|details|quote|pricing)\b", part) \
                        or _said(part, have, 0.6):
                    continue
                item = {"text": part[:1].upper() + part[1:] + ".", "cites": [f["id"]], "kind": "fact"}
                content.setdefault("provider_tools" if _TOOL_WORDS.search(part) else "provider_materials", []).append(item)
                have.append(item)


_EXCEPTION = re.compile(r"(?i)\b(?:the )?exceptions? (?:being|is|are|would be)\s+(?:the\s+)?([^.;]{3,90})|\bexcept(?: for)?\s+(?:the\s+)?([^.;]{3,90})|"
                        r"\bother than\s+(?:the\s+)?([^.;]{3,90})|\bbut not\s+(?:the\s+)?([^.;]{3,90})")


def _keep_exceptions(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A removal line keeps the exception its cited line states. 010120 cited
    "Almost all of this will be removed ... The exception being the HDMI over
    ethernet adapter and HDMI Replicator" for "Remove the existing front-facing
    cameras, mounted soundbar, and remaining Yealink components", which told
    the technician to strip the gear driving the displays the customer keeps."""
    by = {f["id"]: f for f in fs}
    for k in ("scope", "overview"):
        for x in content.get(k) or []:
            if not isinstance(x, dict) or x.get("kind") == "heading" or not x.get("cites"):
                continue
            t = str(x.get("text") or "")
            if not re.search(r"(?i)\b(remove|removal|decommission|disconnect|dispose|haul|demo(lish)?|uninstall|take down|strip)", t):
                continue
            for c in x["cites"]:
                for m_ in _EXCEPTION.finditer(by.get(c, {}).get("text", "")):
                    obj = next(g for g in m_.groups() if g).strip(" ,")
                    want = {w for w in _toks(obj) if len(w) > 2}
                    if want and len(want & _toks(t)) < 0.5 * len(want):
                        t = t.rstrip(" .") + f", except the {obj}."
            x["text"] = t


_REQUESTS_WORK = re.compile(r"(?i)\b(would (also )?like|i'?d like|we need|need (you|us|to)|want to add|add (a|an|the)|can you help|"
                            r"could you|please (install|add|quote))\b")


def _only_asks(text: str) -> bool:
    """A line that asks and is not answered on the same line: nothing of
    substance follows its last question mark."""
    t = str(text).strip()
    return "?" in t and len(re.sub(r"\W+", "", t.rsplit("?", 1)[1])) < 3


def _question_not_commitment(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A Provider task or schedule line resting only on questions is not a
    commitment. 010326's "is there any type of registering, any type of E911
    ...?" and "Are you talking about the E911 stuff?" became "Provider will
    complete ... E911 verification"; the answer was "we're not doing any of
    that". The type head's answered_question label marks the question as
    answered, so the general open-question filter lets it through; the answer
    is the fact, and the question that asks for work stays (010198's router)."""
    by = {f["id"]: f for f in fs}
    # An open checklist item ("Any testing, labeling, certification, or
    # as-built documentation requirements", 010082) asks too.
    checklist = re.compile(r"(?i)^\s*(any|is there|are there|what|which|does|do|will)\b[^.?!]*\b(requirements?|needs?|constraints?|"
                           r"restrictions?|preferences?|expectations?)\s*\??\s*$")
    def asks(c: str) -> bool:
        f = by.get(c)
        return bool(f) and ((bool(f.get("asked")) and _only_asks(f["text"]) and not _REQUESTS_WORK.search(f["text"]))
                            or bool(checklist.match(f["text"])))
    # Nor is it equipment or a summary claim (010326 pass V: "Are all devices
    # from the BOM actually in scope for onsite installation?", answered no,
    # became "All devices on the BOM are in scope for onsite installation.").
    for k in ("scope", "schedule", "provider_materials", "equipment_in_scope", "customer_materials", "overview"):
        content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("kind") != "heading"
                      and not x.get("standard") and x.get("cites") and all(asks(c) for c in x["cites"]))]


def _cell_not_task(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """An inventory row's description cell says what a line is for, never
    work (010326 pass V: the analog-port sheet's "Description: Dial up
    connection for Meter", "Lift pump dialers for Sewer System" and "Public
    accessable Emergency POTS phone" became "Provider will handle ..."). A
    task or summary line every cite of which is such a cell goes; the
    same family as explanation_not_exclusion."""
    by = {f["id"]: f for f in fs}
    # One labeled cell with no count (010279's "Description | Part # |
    # Quantity Needed | ..." is the equipment table itself).
    cell = re.compile(r"(?i)^\s*(description|desc|purpose|usage|function)\s*:\s*[^|\d]+$")
    is_cell = lambda c: c in by and bool(cell.match(by[c]["text"]))
    for k in ("scope", "overview"):
        content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("kind") != "heading"
                      and not x.get("standard") and x.get("cites") and all(is_cell(c) for c in x["cites"]))]


#: Words that say how or that work is done, not what it acts on: two lines
#: sharing only these do not name the same thing.
_WORK_WORDS = frozenset("""provider customer will also help handle handling perform provide install installed
installation installing equipment hardware gear each all any where required need needed work onsite site location locations
existing new including include includes such other them their phone phones device devices""".split())
_WORK_STEMS: set[str] = set()


def _object_toks(t: str) -> set[str]:
    if not _WORK_STEMS:
        _WORK_STEMS.update(_stem(w) for w in _WORK_WORDS)
    return {w for w in _toks(t) if w not in _WORK_STEMS and not w.isdigit() and len(w) > 2
            and not re.fullmatch(r"(?:%s)" % _NUM_WORDS, w)}


def _collides(a: str, b: str) -> bool:
    """Two lines name the same work: at least two object words in common,
    covering half of the first line's ("help with racking, stacking" and
    "rack & stack gear, including the gateways")."""
    # A helping verb names no work of its own (010326 pass V: "assist with
    # racking, stacking, and wall mounting" counted "assist" as a fifth
    # object, so rack and stack were under half and the line survived).
    helpers = {"assist", "help", "handle", "support", "provide", "perform", "manage"}
    ta, tb = _object_toks(a) - helpers, _object_toks(b) - helpers
    both = ta & tb
    return len(both) >= 2 and len(both) >= 0.5 * len(ta)


def _scope_contradictions(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A SOW never says both sides do a thing. A Provider task that names the
    same work as a customer duty or an exclusion goes when the other side is
    backed by a cited fact: a cited customer duty, or an exclusion restating
    one. 010326 printed "Provider will also help with racking, stacking" and
    "Handle rack ears and brackets" beside the customer's "rack & stack gear,
    including the gateways" and the exclusion "Rack hardware, rack ears, and
    gateway installation". A Provider line whose cited facts are all dated
    after the other side's keeps its place instead (a later fact replaces an
    earlier one), and an exclusion nothing backs gives way to the task: the
    draft never excludes what it includes. Standard steps are the template's
    and are not judged here."""
    by = {f["id"]: f for f in fs}
    # A hedged offer ("on-site staff may help mount the back of the clock if
    # needed", 010264) is not a role split.
    duties = [x for x in content.get("customer_responsibilities") or [] if isinstance(x, dict) and x.get("cites")
              and not re.search(r"(?i)\b(may|might|could|can)\b|\bif needed\b|\bassist|\bhelp\b", str(x.get("text")))]
    def latest(cites: list[str]) -> str:
        return max((by[c]["when"] for c in cites if c in by and by[c].get("when") and not by[c].get("quoted")), default="")
    def later(x: dict[str, Any], other: list[str]) -> bool:
        mine = [c for c in x.get("cites") or [] if c in by]
        dated = [by[c]["when"] for c in mine if by[c].get("when") and not by[c].get("quoted")]
        theirs = latest(other)
        return bool(mine) and len(dated) == len(mine) and bool(theirs) and min(dated) > theirs
    weak_oos: list[dict[str, Any]] = []
    keep = []
    for x in content.get("scope") or []:
        if not isinstance(x, dict) or x.get("kind") == "heading" or x.get("standard"):
            keep.append(x)
            continue
        t = str(x.get("text") or "")
        hit = None
        for d in duties:
            dt = str(d.get("text"))
            # A task the customer is to decide on is not yet Provider work
            # (010082: "Perform a project survey" beside "Decide whether to
            # proceed with the optional site survey").
            optional = re.search(r"(?i)\b(optional|decide whether|whether to (proceed|add|do))\b", dt) \
                and len(_object_toks(t) & _object_toks(dt)) >= max(1, 0.5 * len(_object_toks(t)))
            if _collides(t, dt) or optional:
                hit = list(d.get("cites") or [])
                break
        if hit is None:
            for o in content.get("out_of_scope") or []:
                if not isinstance(o, dict) or not _collides(t, str(o.get("text"))):
                    continue
                backing = list(o.get("cites") or []) or [c for d in duties if _collides(str(o.get("text")), str(d.get("text")))
                                                         for c in d.get("cites") or []]
                if backing:
                    hit = backing
                    break
                if x.get("cites"):
                    weak_oos.append(o)
        if hit is not None and not later(x, hit):
            continue
        keep.append(x)
    content["scope"] = keep
    # An assumption that the customer did a task before arrival while the task
    # list has Provider doing it (010120: "Removed Yealink equipment is powered
    # down, disconnected, and released ... before Provider arrives" beside
    # "Disconnect and remove the existing Yealink collaboration solution").
    leads = {m_.group(1).lower()[:6] for x in keep if isinstance(x, dict) and x.get("kind") != "heading"
             for m_ in [re.match(r"(?i)\s*(disconnect|remove|unrack|uninstall|pack|power down|decommission|dismantle)", str(x.get("text")))] if m_}
    if leads:
        content["assumptions"] = [x for x in content.get("assumptions") or [] if not (
            isinstance(x, dict) and re.search(r"(?i)before (the )?(provider|technicians?|crew|installers?)\b.{0,20}\barriv", str(x.get("text")))
            and any(w[:6] in leads for w in re.findall(r"(?i)\b(disconnect|remov|unrack|uninstall|pack|power(?:ed)? down|decommission|dismantl)", str(x.get("text")))
                    for w in [w.lower().replace("powered", "power")]))]
    if weak_oos:
        content["out_of_scope"] = [o for o in content.get("out_of_scope") or [] if not any(o is w for w in weak_oos)]


def _ship_list_siblings(content: dict[str, Any], envelope: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """Every member of the customer's ship-out list is customer-provided
    material (000123: "we'd ship out: ... Printer" and the printer vanished).
    Runs after the supplier head has filed its items, so the list's anchors
    are there (000123's tablet and hub reached the list only then, and the
    printer, card reader and cords beside them never followed)."""
    cm = content.setdefault("customer_materials", [])
    # Short item lines right beside a cited customer-provided item, in the
    # same document, are its list siblings.
    atoms = envelope.get("atoms") or []
    line_of = lambda a: (a.get("locator") or {}).get("line_start") if isinstance((a.get("locator") or {}).get("line_start"), int) else a.get("order")
    pos = {str(a.get("id")): (str(a.get("artifact_id") or ""), line_of(a)) for a in atoms}
    anchors = [pos.get(c) for x in cm if isinstance(x, dict) for c in x.get("cites") or [] if pos.get(c)]
    near = lambda art, ln: any(art == aa and (ln is None or oo is None or abs(int(ln) - int(oo)) <= 6) for aa, oo in anchors)
    for a in atoms:
        t = str(a.get("text") or "").strip()
        art, ln = str(a.get("artifact_id") or ""), line_of(a)
        if not t or len(t) > 60 or re.search(r"[.!?]\s|\b(will|we|you|please)\b", t, re.I) \
                or re.match(r"(?i)(reuse|re-use|install|use|unpack|notify|power|request|mount|remove|connect|test|confirm|provide)\b", t):
            continue  # an item, never a task line beside the list
        if near(art, ln) and not _said(t, cm, 0.6) \
                and re.fullmatch(r"(?i)[\w\s()/+&.-]{3,60}", t) \
                and re.search(r"(?i)\b(printer|reader|pin ?pad|scanner|cash drawer|tablet|terminal|stand|hub|cords? for|cables?)\b", t):
            cm.append({"text": t, "cites": [str(a.get("id"))], "kind": "fact"})
    for f in fs:
        if re.search(r"(?i)we'?d ship( out)?|we (will|'ll) (ship|provide)|customer (will )?provide|provided by (the )?customer|ship(ping)? out", f.get("under") or "") \
                and _ITEM_LINE.match(f["text"].strip()) and not _said(f["text"], cm, 0.6):
            cm.append({"text": f["text"].strip().rstrip("."), "cites": [f["id"]], "kind": "fact"})


def _noise(envelope: dict[str, Any], fs: list[dict[str, Any]]) -> tuple[list[str], str]:
    """Atoms (and a line pattern) whose drop by a rule is noise, never
    guarded: pictures, small talk and signatures, brochure and floor-plan
    documents, open questions, quoted history, existing gear, optional
    surveys, and on a labor-only bid the RFP's logistics duties PurTera
    declined (010279: accept delivery, store, tag, transport)."""
    bro_ids = {str(d.get("artifact_id")) for d in envelope.get("documents") or []
               if re.search(r"(?i)floor ?plan|concept plan|space plan|test[ -]?fit|leasing|brochure|flyer|availability|marketing", str(d.get("filename") or ""))}
    labor_only = any(re.search(r"(?i)\blabor[- ]only\b|installation labor only|onsite labor only|field labor only", f["text"]) for f in fs)
    logi = (r"accept,? (and )?(verify )?(the )?deliver|receiv(e|ing) (the )?(equipment|hardware|shipments?)|\bwarehous|"
            r"\bstor(e|ed|ing|age)\b|\btransport|\bkitting\b|proof of delivery|inventory recap|asset[- ]tag") if labor_only else ""
    noise_txt = re.compile(r"(?i)\b(we|they) (already |currently )?have\b|\balready (have|installed|in place)\b|"
                           r"survey.{0,60}\b(optional|contingent|if needed|probably not|not needed)\b" + ("|" + logi if logi else ""))
    ids = {f["id"] for f in fs if f.get("pictured") or f.get("quoted") or noise_txt.search(f["text"])
           or (f.get("asked") and f.get("section") != "answered_question")
           or re.search(r"(?i)open questions?|questions? for (the )?customer|to confirm|tbd items", str(f.get("under") or ""))}
    ids |= {str(a.get("id")) for a in envelope.get("atoms") or []
            if str(a.get("artifact_id")) in bro_ids
            or str(((a.get("structured") or {}).get("c3_label") or {}).get("type") or a.get("atom_type") or "") in ("small_talk", "signature")}
    return sorted(ids), logi


COVER_FACTS = int(os.environ.get("SOWSMITH_C3_SOW_COVER_FACTS", "300"))
COVER_BATCH = int(os.environ.get("SOWSMITH_C3_SOW_COVER_BATCH", "50"))


_SALIENT = re.compile(r"\b(\d+(?:\.\d+)?\s*(?:ft|feet|foot|in|inch|\"|u|mm|lbs?)?|[A-Za-z]+\d+[A-Za-z0-9-]*|"
                      r"\d+[A-Za-z]+[A-Za-z0-9-]*)\b")
_PRODUCT = re.compile(r"\b([A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})?)\b")
MUST_SECTIONS = ("scope_of_work", "customer_furnished", "answered_question", "customer_responsibilities",
                 "assumptions", "out_of_scope")


def _must_include(content: dict[str, Any], fs: list[dict[str, Any]], model: str | None) -> int:
    """A job line carrying a count, size, model or product name that no SOW
    line mentions is handed to one more pass that must place it: the writer
    left out 010171's "2 x 85\" and 1 x 75\"" TVs and the Rally Bar ask, and
    010082's "Tech to provide 100ft black CAT6", all of them given as facts."""
    text = " ".join(x["text"] for k in LISTS for x in content.get(k) or []).lower()
    used = {c for k in LISTS for x in content.get(k) or [] for c in x["cites"]}
    missing = []
    for f in fs:
        if f.get("pictured"):
            continue
        tm = _TERM.search(f["text"]) if f["section"] in MUST_SECTIONS or f.get("type") in ("scope_item", "constraint", "task") else None
        if tm:  # a quoted copy too: 010209's Q&A answers live only in forwards
            key = next((k for k in _TERM_WORDS if k in tm.group(0).lower()), "")
            if key and not re.search(_TERM_WORDS[key], " ".join(x["text"] for k2 in LISTS for x in content.get(k2) or []
                                                              if f["id"] in (x.get("cites") or [])), re.I):
                missing.append(f)
                continue
        if f["id"] in used:
            continue
        # A written instruction about the work, or what stays in place, is
        # placed whatever its section: 010120's survey deck lines ("should be
        # moved behind wall", "No cables should be visible after mount", "These
        # 4 TV's to Stay in Place") were handed over and left out.
        if _DIRECTIVE.search(f["text"]) and not f.get("quoted"):
            missing.append(f)
            continue
        if f["section"] not in MUST_SECTIONS:
            continue
        toks = {m.group(1).lower().strip() for m in _SALIENT.finditer(f["text"])}
        toks |= {m.group(1).lower() for m in _PRODUCT.finditer(f["text"]) if m.start() > 0}
        # An acronym names a thing too (010376: "PurTera's COI (certificate of
        # insurance)" was kept by the brief and silently left out).
        toks |= {m.group(1).lower() for m in re.finditer(r"\b([A-Z]{3,5})s?\b", f["text"]) if m.group(1) not in ("ASAP", "FYI", "TBD", "PST", "EST", "CST")}
        toks = {t for t in toks if len(t) >= 2 and t not in ("the", "we", "you", "and")}
        # A number counts as written only as a number of its own: "20" cables
        # is not in "2026-08-25" (010147 lost its 20 cables that way).
        if toks and not any((re.search(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", text) if re.fullmatch(r"\d+", t) else t in text) for t in toks):
            missing.append(f)
    if not missing:
        return 0
    draft = json.dumps({k: [x["text"] for x in content.get(k) or []] for k in LISTS})[:12000]
    raw = _complete(_MUST_PROMPT % (json.dumps(list(LISTS)), draft, "\n".join(_fact_line(f) for f in missing[:60])),
                    model, 3000, os.environ.get("SOWSMITH_C3_SOW_EFFORT", "medium"))
    n = 0
    for a in (_json_of(raw) or {}).get("add") or []:
        k = str((a or {}).get("list") or "")
        item = _item(a)
        if k in LISTS and item and item["cites"]:
            item["kind"] = "fact"
            content[k].append(item)
            n += 1
    return n


def _cover(content: dict[str, Any], fs: list[dict[str, Any]], model: str | None) -> int:
    """Offer the writer every job fact it left uncited, in focused batches run
    in parallel, and add the ones a SOW needs. One call over 120 facts left
    010082's materials list and 010171's Logitech device lines out."""
    from concurrent.futures import ThreadPoolExecutor

    used = {c for k in LISTS for x in content.get(k) or [] for c in x["cites"]}
    unused = [f for f in fs if f["id"] not in used and f["section"] not in ("acceptance", "terms", "change_control")
              and not f.get("pictured")]
    # A PurTera person's own "assume ..." goes first: 010115's "assume they will
    # have all the packing materials onsite" was left out run after run.
    unused.sort(key=lambda f: 0 if re.search(r"(?i)\bassum", f["text"]) else 1)
    unused = unused[:COVER_FACTS]
    if not unused:
        return 0
    draft = json.dumps({k: [x["text"] for x in content.get(k) or []] for k in LISTS})[:12000]
    batches = [unused[i:i + COVER_BATCH] for i in range(0, len(unused), COVER_BATCH)]

    def one(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raw = _complete(_COVER_PROMPT % (json.dumps(list(LISTS)), draft,
                                         "\n".join(_fact_line(f) for f in batch)), model, 3000)
        return list((_json_of(raw) or {}).get("add") or [])

    with ThreadPoolExecutor(max_workers=max(1, len(batches))) as pool:
        adds = [a for got in pool.map(one, batches) for a in got]
    n = 0
    for a in adds:
        k = str((a or {}).get("list") or "")
        item = _item(a)
        if k in LISTS and item and item["cites"]:
            item["kind"] = "fact"
            content[k].append(item)
            n += 1
    return n


#: A line that names an item rather than says something: "Samsung Tablet +
#: Stand", "Cords for the above", never "Delivery team will bring...".
_ITEM_LINE = re.compile(r"^(?!(?:we|they|he|she|i|you|the team|delivery|provider|customer)\b)[^.!?]{2,90}$", re.I)
_ITEM_TYPES = frozenset({"bom_line"})
_SUPPLY_VERB = re.compile(r"^(?:provide|supply|bring|furnish|deliver|use)s?\s+", re.I)


_LEAD = re.compile(r"^Provider will (?:dispatch|provide) (?:(%s) \((\d+)\) )?(?:[a-z]+ ){0,2}(?:field )?"
                   r"(?:technicians?|technician|resources?|engineers?) to perform the following:?$" % _NUM_WORDS, re.I)


def _lead(lead: str, pool: str) -> str:
    """PurTera's one task-list lead, or the plain one: a crew count only when a
    fact states it (010082's lead named the meeting's attendees as the crew)."""
    lead = lead.strip()
    m = _LEAD.match(lead)
    if not m:
        return "Provider will perform the following:"
    if m.group(2) and not re.search(r"\b(%s|%s)\s+(?:\(\d+\)\s+)?(?:\w+\s+)?(?:techs?|technicians?|resources?)"
                                    % (m.group(1), m.group(2)), pool, re.I):
        return "Provider will perform the following:"
    return lead if lead.endswith(":") else lead + ":"


_STOPWORDS = frozenset("""and the for with from per all any each other standard required needed install installation
materials material equipment items item supplies supply provided provide customer provider onsite site""".split())
_NOT_SUPPLY = re.compile(r"(?i)^(timeline|schedule|date)\b|\b\d+\s*(days?|weeks?)\b|\binstallation of\b|\bpictures?\b|"
                         r"\bphotos?\b|\breuse\b|\bsurvey\b")


def _grounded(x: dict[str, Any], lines: dict[str, str]) -> bool:
    """A supply item names things its cited lines name: at least half its
    words are in them (010043's writer listed a "backer board" off a line about
    running the cabling)."""
    # A count is its value however it is spelled (010373 pass U: "Supply
    # sixteen (16) monitor arms and forty-eight (48) monitors" failed on
    # "sixteen", "forty" and "eight" against "16 ... arms", "48 monitors").
    words = [w for w in re.findall(r"[a-z0-9]{2,}", _digits(x["text"]).lower())
             if w not in _STOPWORDS and (len(w) >= 3 or w.isdigit())]
    if not words:
        return True
    cited = _digits(" ".join(lines.get(c, "") for c in x["cites"])).lower()
    return sum(1 for w in words if w in cited or w.rstrip("s") in cited) >= max(1, (len(words) + 1) // 2)


def _party_check(content: dict[str, Any], lines: dict[str, str]) -> int:
    """A supply item stays with a party only when a cited line says that party
    supplies it; the rest go to the neutral Equipment in Scope list, as
    000123's SOW lists hardware "provided by SHI, the customer, or another
    party" (the writer gave Provider its printers on a bare "Printer")."""
    for k in ("provider_materials", "customer_materials", "equipment_in_scope"):
        content[k] = [x for x in content.get(k) or []
                      if not _NOT_SUPPLY.search(x["text"]) and (not x["cites"] or _grounded(x, lines))]
    items = [(k, x) for k in ("provider_materials", "customer_materials") for x in content.get(k) or []
             if x.get("by") != "col:supplier"]
    if not items:
        return 0
    body = "\n".join(f"{i}. [{('Provider' if k == 'provider_materials' else 'Customer')}] {x['text']}\n"
                     + "\n".join(f"   [{c}] {lines.get(c, '')[:300]}" for c in x["cites"])
                     for i, (k, x) in enumerate(items))
    got = _json_of(_complete(_PARTY_PROMPT % body, sow_model(), 800)) or {}
    ok = {int(i) for i in got.get("stated") or [] if str(i).isdigit()}
    moved = 0
    for i, (k, x) in enumerate(items):
        if i in ok:
            continue
        content[k] = [y for y in content[k] if y is not x]
        content.setdefault("equipment_in_scope", []).append(x)
        moved += 1
    return moved


def _model_numbers(content: dict[str, Any], envelope: dict[str, Any]) -> None:
    """A number a document's file name carries ("785 POS Termainal Refresh
    RFP.doc") is a model or project number: where a line writes it as a
    count it is reworded as the model ("seven hundred eighty-five (785) POS
    terminals" -> "785 POS terminals", "replace 785 existing POS terminals"
    -> "replace existing 785 POS terminals"); the line stays (010279)."""
    titled = {n for d in envelope.get("documents") or [] for n in re.findall(r"\b(\d{3,4})\b", str(d.get("filename") or ""))
              if not re.fullmatch(r"(19|20)\d\d", n)}
    if not titled:
        return
    alt = "|".join(sorted(titled))
    def fix(t: str) -> str:
        t = re.sub(r"(?i)\b" + _NUMW + r"(?:[\s-]+" + _NUMW + r")*\s*\((%s)\)" % alt, r"\1", t)
        t = re.sub(r"(?i)\(?\b(%s)\)?\s+(existing|old|current|legacy|replaced|removed)\s+" % alt, r"\2 \1 ", t)
        t = re.sub(r"(?i)\b(replace|replacing|swap)\s+(all\s+)?(the\s+)?(%s)\b(?=[^.;]{0,60}\bwith\b)" % alt,
                   lambda m: f"{m.group(1)} {m.group(2) or ''}{m.group(3) or ''}existing {m.group(4)}".replace("  ", " "), t)
        return t
    for k in LISTS:
        for x in content.get(k) or []:
            if isinstance(x, dict) and isinstance(x.get("text"), str):
                x["text"] = fix(x["text"])
    for d in content.get("deliverables") or []:
        if isinstance(d, dict):
            d["description"] = [fix(str(t)) for t in d.get("description") or []]


def _dedupe_items(content: dict[str, Any]) -> None:
    """One line per thing in each list: the writer and the coverage pass both
    wrote 010082's drop-off arrangement."""
    # Lines that name different rooms or sites are two lines, however alike
    # their first words (010012 pass U: "Install one (1) Poly X72 bar ... in
    # Conference Room 2102" and "... in Shared Office 2105" became one).
    names = lambda t: frozenset(re.findall(r"\b(?:[A-Z][A-Za-z]+|\d+)\b", t)[1:])
    for k in LISTS:
        seen, shorts, keep = set(), {}, []
        for x in content.get(k) or []:
            key = re.sub(r"\W+", " ", x["text"].lower()).strip()
            short = " ".join(key.split()[:8])
            if key in seen or names(x["text"]) in shorts.get(short, ()):
                continue
            seen.add(key)
            shorts.setdefault(short, set()).add(names(x["text"]))
            keep.append(x)
        content[k] = keep


def _follow_supplier(content: dict[str, Any], fs: list[dict[str, Any]]) -> int:
    """The supplier head decides which party's list a supply item sits in: an
    item whose cited lines the head reads, sure, as the other party's moves
    there (000123's mounting hardware, customer 0.80, written as Provider's).
    Items read as plain nouns, as PurTera lists them."""
    # The supplier head is a table-column head (col:supplier): on held-out
    # deals it equals a constant guess overall (0.659), so it decides only on
    # BOM rows, the lines it was trained on; on 010082's sentences it read the
    # backboard Provider brings as the customer's.
    sup = {f["id"]: f["supplier"][0] for f in fs
           if f.get("supplier") and float(f["supplier"][1] or 0) >= SUPPLIER_BAR and f.get("type") in _ITEM_TYPES}
    side = {"us": "provider_materials", "partner": "provider_materials", "customer": "customer_materials"}
    moved = 0
    # An item line the head assigns is listed whatever the writer did: the
    # writer dropped 000123's tablet, hub and cords on one run of two.
    cited = {c for k in ("provider_materials", "customer_materials") for x in content.get(k) or [] for c in x["cites"]}
    for f in fs:
        if f["id"] in cited or sup.get(f["id"]) not in side or not _ITEM_LINE.match(f["text"].strip()):
            continue
        if f.get("type") not in _ITEM_TYPES:
            continue
        content.setdefault(side[sup[f["id"]]], []).append(
            {"text": f["text"].strip().rstrip("."), "cites": [f["id"]], "kind": "fact", "by": "col:supplier"})
    for k in ("provider_materials", "customer_materials"):
        keep = []
        for x in content.get(k) or []:
            x["text"] = _SUPPLY_VERB.sub("", x["text"]).strip()
            x["text"] = x["text"][:1].upper() + x["text"][1:]
            votes = [side[sup[c]] for c in x["cites"] if sup.get(c) in side]
            target = max(set(votes), key=votes.count) if votes else k
            if target != k:
                content.setdefault(target, []).append(x)
                moved += 1
            else:
                keep.append(x)
        content[k] = keep
    for x in content.get("provider_tools") or []:
        x["text"] = _SUPPLY_VERB.sub("", x["text"]).strip()
        x["text"] = x["text"][:1].upper() + x["text"][1:]
    return moved


def _item(x: Any) -> dict[str, Any] | None:
    if isinstance(x, str):
        x = {"text": x, "cites": [], "kind": "method"}
    if not isinstance(x, dict) or not str(x.get("text") or "").strip():
        return None
    cites = [str(c) for c in x.get("cites") or [] if str(c).strip()]
    text = strip_cites(str(x["text"]))
    # Ids the writer put in the text itself: "(atm_6c5ab..., atm_8cde...)".
    text = re.sub(r"\s*[\(\[]?\b(?:atm|pmq|ent|q|declared_scope)_[0-9A-Za-z_]+(?:\s*,\s*(?:atm|pmq|ent|q|declared_scope)_[0-9A-Za-z_]+)*[\)\]]?", "", text)
    # "(Provider responsibility)" tags: the list says whose it is.
    text = re.sub(r"\s*\((?:provider|customer|end customer)(?: duty| responsibility)?[^)]*\)", "", text, flags=re.I)
    kind = str(x.get("kind"))
    return {"text": re.sub(r"\s+", " ", text).strip(), "cites": cites,
            "kind": kind if kind in ("fact", "heading") else "method"}


SERVICES = {"assessment", "configuration", "design", "implementation", "knowledge_transfer", "project_management",
            "reconfiguration", "reinstallation", "staff_augmentation", "support", "training", "custom_work"}
_FAMILY_SERVICE = {"network_switch_refresh": "reinstallation", "pos_install_refresh": "reinstallation",
                   "time_clock_kiosk": "reinstallation", "decommission_packing": "custom_work",
                   "relocation_move": "custom_work", "smart_hands_general": "custom_work", "site_survey": "assessment"}


def _normalise(got: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: got.get(k) or "" for k in ("project_name", "end_customer", "scope_lead",
                                                          "pricing_model", "job_family")}
    # The writer sometimes returns services as items ({"text": "implementation", ...}); the
    # checkbox row matches on the bare name, so every box printed empty.
    svc = got.get("services") or []
    svc = [svc] if isinstance(svc, (str, dict)) else svc
    out["services"] = [t for t in (str(s.get("text") if isinstance(s, dict) else s).strip().lower().replace(" ", "_")
                                   for s in svc) if t in SERVICES]
    # The writer can answer in prose ("install_three_(3)_verkada_cameras"), which
    # ticks nothing; the job family gives the box.
    if not out["services"]:
        out["services"] = [_FAMILY_SERVICE.get(str(got.get("job_family") or ""), "implementation")]
    # A decommission or packing job is Custom Work only, as PurTera ticks it
    # (010115 v1); the writer also ticked Staff Augmentation for "one tech, two days".
    if str(got.get("job_family") or "") == "decommission_packing":
        out["services"] = ["custom_work"]
    elif str(got.get("job_family") or "") == "network_switch_refresh":
        out["services"] = ["reinstallation"]  # as PurTera ticks a switch swap (010083, 010085 v1s)
    for k in LISTS:
        v = got.get(k) or []
        # A list returned as one string is one item: 000123's overview came back
        # as a string and was read letter by letter ("I n s t").
        if isinstance(v, (str, dict)):
            v = [v]
        out[k] = [i for i in (_item(x) for x in v) if i]
    dels = []
    for d in got.get("deliverables") or []:
        if not isinstance(d, dict):
            continue
        desc = d.get("description")
        desc = [strip_cites(str(t.get("text") if isinstance(t, dict) else t)) for t in
                (desc if isinstance(desc, list) else [desc])] if desc else []
        dels.append({"item": str(d.get("item") or "Work Completion"), "description": [t for t in desc if t],
                     "format": str(d.get("format") or "PDF")})
    out["deliverables"] = dels
    return out


_STD_LIST = {"method_steps": "scope", "customer_responsibilities": "customer_responsibilities",
             "assumptions": "assumptions", "exclusions": "out_of_scope", "tools": "provider_tools",
             "materials": "provider_materials"}
_TOK = re.compile(r"[a-z0-9]+")
_FILLER = frozenset("the a an and or of to for in on at by with is are be any all each its their".split())


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 4 and w[-1] == w[-2]:
        w = w[:-1]
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


def _toks(t: str) -> set[str]:
    return {_stem(w) for w in _TOK.findall(str(t).lower()) if w not in _FILLER}


_OBJ_SKIP = frozenset("each every all the a an of new both its their".split())
_OBJ_STOP = frozenset("on onto in into at to for from with along per under above behind across then and using "
                      "before after once so that as".split())


def _action(t: str) -> tuple[str, list[str]]:
    """A step's verb and the object it acts on: "Install each of the sixteen
    (16) monitor arms on its desk" -> ("instal", ["monitor", "arm"])."""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*|[,.;:()]", _digits(str(t)).lower())
    if not words or not re.match(r"[a-z]", words[0]):
        return "", []
    obj: list[str] = []
    for w in words[1:]:
        if w in "()":
            continue
        if not obj and (w in _OBJ_SKIP or w.isdigit()):
            continue
        if w in _OBJ_STOP or w in ",.;:":
            break
        obj.append(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return _stem(words[0]), obj


def _same_action(std: str, deal: str) -> bool:
    """The deal step does the standard step's work on the same thing: same
    verb, same object (its head noun and every word the standard names)."""
    v1, o1 = _action(std)
    v2, o2 = _action(deal)
    return bool(v1 and o1 and o2) and v1 == v2 and o1[-1] == o2[-1] and set(o1) <= set(o2)


def _said(text: str, items: list[Any], share: float = 0.5) -> bool:
    """Does some item already say ``text`` (most of its words)?"""
    want = _toks(text)
    return bool(want) and any(len(want & _toks(x.get("text") if isinstance(x, dict) else x)) >= share * len(want)
                              for x in items)


def _apply_standards(content: dict[str, Any], family: str, facts_text: list[str] | None = None,
                     context_text: list[str] | None = None, gate_text: list[str] | None = None) -> int:
    """Write PurTera's fixed handling of the job type (purtera_standards.json)
    into the SOW. Offered to the writer as hints they came back missing,
    paraphrased or out of order (010115 rounds 12-14), so the standards are the
    list: every item goes in, in the file's order, and a writer item saying the
    same thing goes (a scope step keeps its place only when it carries a count,
    date or name of its own: the deal's facts stay)."""
    fixed = purtera_standards(family)
    n = 0
    # A standard item a deal fact contradicts is not written (the file's own
    # rule): 010120's survey says the TVs stay in place, and the AV standards
    # wrote "mount each display" and plywood backing.
    # Only the words beside the keep word: what stays is what it is said of.
    # Read over the whole line, 010326's "Existing wall mounts will be reused
    # by swapping old mounts for new ones with new phones" made every phone
    # standard (deployment list, staging area for removed phones) contradicted.
    keep_word = re.compile(r"(?i)\b(stay|stays|remain|remains|no change|left in place|already (mounted|installed|on the wall|in place)|"
                           r"keep (the )?existing|not (be )?(moved|replaced|removed)|reuse|re-use|reusing|reused)\b")
    reuse_word = re.compile(r"(?i)^(reuse|re-use|reusing|reused)$")
    # A reused thing still carries the new work: reusing the mounts rules out
    # removing them, not mounting the new displays on them (010217: "Mount
    # each display at the approved location" and "Adjust the mounts" were
    # never written because the existing mounts are reused).
    stays_all = " ".join(str(f)[max(0, k.start() - 45):k.end() + 30] for f in facts_text or [] for k in keep_word.finditer(str(f))).lower()
    stays_fixed = " ".join(str(f)[max(0, k.start() - 45):k.end() + 30] for f in facts_text or [] for k in keep_word.finditer(str(f))
                           if not reuse_word.match(k.group(0))).lower()
    def contradicted(t: str) -> bool:
        stays = stays_all if re.match(r"(?i)\s*(remove|replace|relocate|dismantle|disconnect)\b", t) else stays_fixed
        if not stays:
            return False
        # Only a line that does the work: a check, a review or a supply duty
        # names the displays without moving them (010120: "Confirm display
        # connection, camera, microphone, speaker ..." and "Provide all
        # displays, hardware, cables and mounts" went because the TVs stay).
        if not re.match(r"(?i)\s*(mount|install|remove|replace|relocate|dismantle|dress|connect|power|adjust|align|inspect|disconnect)\b", t) \
                and not re.search(r"(?i)backing", t):
            return False
        # The object of the work is what the verb acts on, not every noun the
        # line names (010120: "Mount each video bar ... connect it to its
        # display" was suppressed by "the TVs stay").
        vm = re.search(r"(?i)\b(mount|install|remove|replace|relocate|dismantle|dress|connect|power|adjust|align|inspect)\w*\b", t)
        if vm and re.search(r"(?i)^\W*(each |the |all )?(of the )?(\w+ )?(video bars?|table controllers?|controllers?|neat|bar|soundbar|network switch|switch)\b", t[vm.end():vm.end() + 30]):
            return False
        t = t[max(0, vm.start() - 25):vm.end() + 60] if vm and not re.search(r"(?i)backing", t) else t
        nouns = {w for w in re.findall(r"[a-z]+", t.lower()) if w in ("display", "displays", "tv", "tvs", "mount", "mounts", "backing",
                                                                       "monitor", "monitors", "speaker", "speakers", "camera",
                                                                       "cameras", "rack", "switch", "switches", "phone", "phones")}
        verbs = re.search(r"(?i)\b(mount|install|remove|replace|relocate|backing|dismantle|dress|connect|power|adjust|align)", t)
        groups = [r"display|tv|television|monitor|screen", r"mount|bracket", r"speaker", r"camera", r"rack",
                  r"switch", r"phone|handset"]
        pats = [g for g in groups if any(re.fullmatch(r"(?:" + g + r")s?", n) for n in nouns)]
        if re.search(r"(?i)backing|wall mount|\bmounts?\b", t):
            pats.append(groups[0])  # backing and wall mounts exist for the displays
        return bool(nouns and verbs and any(re.search(r"\b(?:" + g + r")(?:'?s)?\b", stays) for g in pats))
    # The customer doing its own testing or configuration rules out the test
    # steps (010171: "I will do the device configuration and testing").
    self_test = any(re.search(r"(?i)\b(i|we|customer|our (team|it))\b.{0,25}\b(will|'ll|would)\b.{0,15}\b(do|handle|perform|take care of)\b.{0,40}\btest", str(f))
                    for f in (facts_text or []) + (context_text or []))
    # Every standard item that applies and is not written says why
    # (graders found applicable items nowhere in the SOW, not even dropped).
    why = content.setdefault("standards_suppressed", [])
    def suppressed(sec: str, t: str, reason: str) -> None:
        why.append({"section": sec, "text": t, "reason": reason})
    # A gate reads the deal's own lines, not only the brief's facts (000123:
    # "we'd ship out:" was a deal_state line the brief left out, so "No
    # materials are required to be provided by Provider" never applied).
    gate = list(facts_text or []) + list(gate_text or [])
    for sec, lst in _STD_LIST.items():
        std = []
        for t0 in fixed.get(sec) or []:
            ok, t = _standard_applies(t0, gate, facts_text or [])
            about_reuse = bool(re.match(r"^\{when:[^}]*(reuse|existing)", t0))
            if ok and self_test and re.search(r"(?i)\btest|confirm .{0,90}\bfunction|verify (basic )?operation|verify\b.{0,40}\b(signal|video|source)|"
                                              r"remote (engineering|av|network) support", t):
                suppressed(sec, t, "the customer does its own testing")
                continue
            if ok and (about_reuse or not contradicted(t)):
                std.append({"text": t, "cites": [], "kind": "method", "standard": True})
            elif ok:
                suppressed(sec, t, "a fact says what it acts on stays in place")
        if not std:
            continue
        items = content.setdefault(lst, [])
        share = 0.4 if lst == "scope" else 0.5
        def covered(x: Any) -> bool:
            if not isinstance(x, dict) or x.get("kind") == "heading" or x.get("standard"):
                return False
            t = str(x.get("text"))
            if x.get("cites") and re.search(r"(?i)\b(stay|remain|retain|except|unless|existing|reuse|no|not)\b", t):
                return False  # a deal line carrying an exception is never a duplicate of a standard
            # A deal line naming something no standard line names (Windows,
            # Linux, O365) is its own fact (000132's OS patching line).
            names = {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z0-9/+-]{1,}\b", t)[1:]}
            std_words = " ".join(str(it["text"]) for it in std).lower()
            if names and any(n not in std_words for n in names):
                return False
            if lst == "scope":
                # The same step, verb and object: "Label all cables before
                # removing the old switches" is not the rack-space check that
                # also mentions cables and removal (010209).
                verb = lambda u: _stem((re.findall(r"[a-z]+", u.lower()) or [""])[0])
                return not re.search(r"\d", t) and any(verb(str(it["text"])) == verb(t) and _said(t, [it], max(share, 0.55)) for it in std)
            return _said(t, std, max(share, 0.55))
        items[:] = [x for x in items if not covered(x) and not (isinstance(x, dict) and x.get("standard"))]
        if lst == "scope":
            # A counted deal step is the standard step with the deal's numbers:
            # keep it, not both (010373: install/mount/cable/trash written twice).
            counted = [x for x in items if isinstance(x, dict) and x.get("kind") != "heading" and re.search(r"\d", str(x.get("text")))]
            # So is a cited deal step kept for its exception word ("existing",
            # "reuse"): 010326 printed "Remove each existing phone and install
            # its replacement using the existing cabling" and the standard's
            # "Disconnect and remove each existing phone ... reusing existing cabling".
            counted += [x for x in items if isinstance(x, dict) and x.get("kind") != "heading" and x.get("cites")
                        and re.search(r"(?i)\b(existing|reuse|reusing|reused)\b", str(x.get("text")))]
            # A cited counted step doing the standard step's work on the same
            # thing takes the step's slot, so the list keeps the standard order
            # (010373 pass U: "Install sixteen (16) Humanscale monitor arms"
            # sat after cable management while "Install each monitor arm"
            # stayed in its slot).
            slotted = []
            for j, it in enumerate(std):
                twin = next((x for x in counted if x.get("cites") and re.search(r"\d", _digits(str(x.get("text"))))
                             and all(x is not y for y in slotted) and _same_action(it["text"], str(x.get("text")))), None)
                if twin is None:
                    continue
                # The standard's own method word stays: "align and secure them"
                # replaced by "... and secure them" keeps "align".
                co = re.search(r"(?i)\b(\w+) and (\w+) (them|it)\b", it["text"])
                tt = str(twin.get("text"))
                if co and re.search(r"(?i)\b%s %s\b" % (co.group(2), co.group(3)), tt) and not re.search(r"(?i)\b%s\b" % co.group(1), tt):
                    tt = re.sub(r"(?i)(,?\s*and\s+)?\b%s %s\b" % (co.group(2), co.group(3)),
                                lambda m: (", then " if m.group(1) else "") + f"{co.group(1).lower()} and {co.group(2)} {co.group(3)}", tt, count=1)
                    twin["text"] = tt
                suppressed(sec, it["text"], "a cited deal step with its own count does the same work and takes its place")
                items[:] = [x for x in items if x is not twin]
                std[j] = twin
                slotted.append(twin)
                content.setdefault("_slots", {})[id(twin)] = it
            for it in std:
                if any(it is y for y in slotted):
                    continue
                if _said(it["text"], counted, 0.6):
                    suppressed(sec, it["text"], "a cited deal step with its own count or exception says it")
            std = [it for it in std if any(it is y for y in slotted) or not _said(it["text"], counted, 0.6)]
        if lst == "customer_responsibilities":
            # A counted, cited customer-supply line is the standard supply duty
            # with the deal's numbers: keep it, not the uncounted standard
            # (010373 pass U: "Provide all monitor arms, monitors, mounting
            # hardware and cables" beside "Sixteen (16) monitor arms and
            # forty-eight (48) monitors ...").
            cm = [x for x in content.get("customer_materials") or [] if isinstance(x, dict) and x.get("cites")
                  and re.search(r"\d", _digits(str(x.get("text"))))]
            supply = lambda t: re.sub(r"(?i)^\s*(provide|supply|furnish)\s+(all\s+)?(the\s+)?", "", t)
            gone = [it for it in std if re.match(r"(?i)\s*(provide|supply|furnish)\b", it["text"]) and _said(supply(it["text"]), cm, 0.7)]
            for it in gone:
                suppressed(sec, it["text"], "a cited customer-supply line with the deal's counts says it")
            std = [it for it in std if it not in gone]
            if not std:
                continue
        n += len(std)
        if lst == "provider_tools":
            # The family's tool list is the list: the writer's own tool lines were
            # wrong-job ("equipment moves", 010373) or someone else's ("Delivery
            # team will bring its own tools", 010082).
            items[:] = []
        if lst != "scope":
            items.extend(std)
            continue
        # Scope: the standard steps lead the work group; the closing steps go
        # under the closing heading.
        tail = [it for it in std if it["text"].lower().startswith(("notify", "confirm completion", "clean", "inform", "remove debris",
                                                                     "remove all debris", "dispose", "haul"))
                or re.match(r"(?i)remove\b.{0,30}\bdebris", it["text"])]
        # Test and validation steps follow the work they test (000123: tests
        # and the test print ahead of the device installs).
        checks = [it for it in std if it not in tail and re.match(
            r"(?i)(perform (functional |basic )?test|test |verify|validate|confirm (the installed|display|each)|power on each)", it["text"])
            and not re.search(r"(?i)\b(required|present|onsite|on site|available|undamaged|before installation)\b", it["text"])]
        # Testing comes before the fixes it finds (010012: "Correct any
        # workmanship deficiencies identified during testing" ahead of the
        # power-on and function checks).
        fixes = [it for it in std if it not in tail and it not in checks and re.match(r"(?i)correct\b.{0,40}\bdeficienc", it["text"])]
        checks = checks + fixes
        lead = [it for it in std if it not in tail and it not in checks]
        heads = [i for i, x in enumerate(items) if isinstance(x, dict) and x.get("kind") == "heading"]
        work = [i for i in heads if not re.search(r"(?i)arriv|access|check|start|prep|clos|clean|wrap|complet", str(items[i].get("text")))]
        arrival_end = next((i for i in heads if i > 0), len(items)) if heads and heads[0] == 0 else 0
        # Their own group ahead of the deal's room or area groups, never inside
        # the first one (010012: seven generic steps under "Shared Office").
        generic = work and re.search(r"(?i)install|work|mount|cabl|scope|procedure|task|decommission|pack|replace|refresh|deploy|run",
                                     str(items[work[0]].get("text")))
        # When the deal has its own work groups, preparation leads and the
        # standard follow-on steps (router configuration, failover) come after
        # the deal's installs (010198: router config ahead of the register).
        pre = [it for it in lead if re.match(r"(?i)(arrive|review|confirm (the |each )?(install|mount|location)|verify|unbox|unpack|inspect|check in|meet)", it["text"])]
        post = [it for it in lead if it not in pre]
        deal_work = work and sum(1 for x in items if isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")
                                    and re.match(r"(?i)\s*(install|mount|replace|connect|set up|deploy|place)\b", str(x.get("text")))) >= 1
        if deal_work and post and family in ("pos_install_refresh", "av_conference_room", "network_switch_refresh",
                                             "cameras_security", "low_voltage_cabling"):
            items[work[0] + 1:work[0] + 1] = pre
            close0 = next((i for i, x in enumerate(items) if isinstance(x, dict) and x.get("kind") == "heading"
                           and re.search(r"(?i)clos|clean|wrap|complet", str(x.get("text")))), len(items))
            # With two or more rooms or areas, the steps for every one of them
            # get their own heading, not the last room's (010217: dress, power
            # and verify sat under "Room 247").
            many = len(work) >= 2 and not (close0 > 0 and isinstance(items[close0 - 1], dict) and items[close0 - 1].get("kind") == "heading")
            areas = any(re.search(r"(?i)\b(room|area|floor|office|suite|noc|closet|idf|mdf|lobby|wing|building|site)s?\b|\b\d{2,4}\b",
                                  str(items[i].get("text"))) for i in work)
            items[close0:close0] = ([{"text": "All Work Areas" if areas else "Installation and Testing", "cites": [], "kind": "heading"}]
                                    if many else []) + post
        elif generic:
            items[work[0] + 1:work[0] + 1] = lead
        elif work:
            name = {"decommission_packing": "Dismantle and Pack", "relocation_move": "Pack and Move",
                    "low_voltage_cabling": "Cabling", "network_switch_refresh": "Switch Replacement"}.get(family, "Installation")
            if any(re.search(r"(?i)once (a|per) week|weekly|recurring|staff aug|on-?site (it )?support", t) for t in facts_text or []) \
                    and family == "smart_hands_general":
                name = "Onsite Support"
            items[work[0]:work[0]] = [{"text": name, "cites": [], "kind": "heading"}] + lead
        else:
            items[arrival_end:arrival_end] = lead
        close = next((i for i, x in enumerate(items) if isinstance(x, dict) and x.get("kind") == "heading"
                      and re.search(r"(?i)clos|clean|wrap|complet", str(x.get("text")))), None)
        if close is None:
            items.append({"text": "Closeout", "cites": [], "kind": "heading"})
            close = len(items) - 1
        for it in checks:
            if _said(it["text"], items, 0.6):
                suppressed(sec, it["text"], "a scope line already says it")
        items[close:close] = [it for it in checks if not _said(it["text"], items, 0.6)]
        close += len([it for it in checks if any(y is it for y in items)])
        for it in tail:
            if _said(it["text"], items[close + 1:], 0.5):
                suppressed(sec, it["text"], "a closing line already says it")
        items[close + 1:close + 1] = [it for it in tail if not _said(it["text"], items[close + 1:], 0.5)]
        # One closing group: "Cleanup" and "Closeout" both held closing steps.
        items[close]["text"] = "Closeout"
        items[:] = [x for i, x in enumerate(items) if not (i > close and isinstance(x, dict) and x.get("kind") == "heading"
                    and re.search(r"(?i)clos|clean|wrap|complet", str(x.get("text"))))]
    outcomes = []
    for ok, t in (_standard_applies(o, gate, facts_text or []) for o in fixed.get("deliverable_outcomes") or []):
        if not ok:
            continue
        if contradicted(t) or (self_test and re.search(r"(?i)\btest|verification|verif(y|ied)", t)):
            suppressed("deliverable_outcomes", t, "contradicted by a fact or the customer's own testing")
            continue
        outcomes.append(t)
    if outcomes:
        dl = content.setdefault("deliverables", [])
        if not dl:
            dl.append({"item": "Work Completion Report", "description": [], "format": "PDF"})
        d0 = dl[0] if isinstance(dl[0], dict) else {"item": "Work Completion Report", "description": [], "format": "PDF"}
        own = [str(t) for t in d0.get("description") or [] if str(t).strip()]
        # The writer's echo of a standard outcome the facts gate off goes
        # (010083: "Serial numbers ... if requested" beside a customer who
        # requires serials).
        off = [t for o in fixed.get("deliverable_outcomes") or [] if o.startswith("{unless:")
               for ok, t in [_standard_applies(o, gate, facts_text or [])] if not ok]
        own = [t for t in own if not any(_said(t, [o], 0.8) and _said(o, [t], 0.8) for o in off)]
        # The customer doing its own testing: no test or verification outcome
        # either (010171 pass U: "Power-on and video-source verification for
        # each display").
        if self_test:
            own = [t for t in own if not re.search(r"(?i)\btest|verification|verif(y|ied)", t)]
        # Said by a line about the same thing: "Serial numbers of each new
        # switch" is not "New switches installed and cabled".
        head = lambda u: _stem((re.findall(r"[a-z]+", str(u).lower()) or [""])[0])
        # Said only by a line that carries most of it: "Hardware dismantled as
        # directed by the onsite contact" is not "Document completion
        # confirmation from the onsite contact" (010115 pass U).
        said_o = lambda o: _said(o, own, 0.7) and (head(o) in {head(t) for t in own} or not o.lower().startswith(("serial", "posted", "photo")))
        for o in outcomes:
            if said_o(o):
                suppressed("deliverable_outcomes", o, "the report already says it")
        d0["description"] = own + [o for o in outcomes if not said_o(o)]
        # Photographs ship as images beside the report (010116, 010209: "PDF, Jpeg").
        if any(re.search(r"(?i)photo", str(t)) for t in d0["description"]) and "JPEG" not in str(d0.get("format") or "").upper():
            d0["format"] = (str(d0.get("format") or "PDF").strip() or "PDF") + ", JPEG"
        dl[0] = d0
    content["standards_added"] = n
    content["_std_objs"] = [[id(x), str(x.get("text"))] for k in _STD_LIST.values() for x in content.get(k) or []
                            if isinstance(x, dict) and x.get("standard")]
    return n


_ACCESS = re.compile(r"(?i)\b(badges?|parking|pass(es)?|escort|keys?|access cards?)\b")
_LEDGER = re.compile(r"(?i)tax[- ]exempt|tax status|\bexempt\b|cost cent(er|re)|\bcorp(orate)? service\b|\bsegment\b|\bgl code\b|"
                     r"request date|\bSR ?#|unified sr|\bproject name\s*:|\bticket\s*#")
_FAMILY_TOOLS_NOT = {"decommission_packing": re.compile(r"(?i)\brack\b|\bcable\b|crimp|punch|tester|toner|termination"),
                     "pos_install_refresh": re.compile(r"(?i)network testing|validation tools|certifier|toner"),
                     "av_conference_room": re.compile(r"(?i)termination tools?|cable testers?|certifier|toner|punch")}


_CONSUMABLE = re.compile(r"(?i)\b(cat ?5e?|cat ?6a?|cable|wire|anchors?|screws?|nuts?|bolts?|fasteners?|j-?hooks?|velcro|zip ties?|"
                         r"rj-?45|connectors?|keystones?|jacks?|faceplates?|biscuit|backing board|plywood|supports?|labels?)\b")


def _materials_hygiene(content: dict[str, Any], family: str, fs: list[dict[str, Any]] | None = None) -> None:
    """Equipment in Scope is the gear worked on; on a cabling job the cable and
    fasteners are Provider's materials, and one line per thing (010082: Cat6
    listed twice, anchors as equipment). Who carries the materials to site is
    PurTera logistics, never a material line ('Chase Smith will handle small
    material runs')."""
    for k in ("provider_materials", "customer_materials", "equipment_in_scope"):
        content[k] = [x for x in content.get(k) or []
                      if not re.search(r"\b(?!Provider\b|Customer\b|The\b)[A-Z][a-z]+ [A-Z][a-z]+ will (?i:handle|bring|pick up|deliver|run)\b|(?i:\bmaterial runs?\b)",
                                       str(x.get("text") if isinstance(x, dict) else x))]
    # The customer ships the POS gear: one Customer Provided list, not a split
    # with an ownerless Equipment in Scope (000123: printer and pin pad apart
    # from the tablet and hub, all shipped by the customer).
    if family == "pos_install_refresh" and content.get("customer_materials") and content.get("equipment_in_scope"):
        cm = content["customer_materials"]
        for x in content["equipment_in_scope"]:
            if not _said(str(x.get("text") if isinstance(x, dict) else x), cm, 0.7):
                cm.append(x)
        content["equipment_in_scope"] = []
    for k in ("customer_materials", "provider_materials", "equipment_in_scope"):
        for x in content.get(k) or []:
            if isinstance(x, dict):
                x["text"] = re.sub(r"(?i)\bthe above( gear| equipment| items)?\b", "the listed equipment", str(x.get("text")))
    # A material the cited line says the tech provides is Provider's (010082:
    # "Tech to provdie estiamted 100ft ... CAT6" under Customer Provided).
    by = {f["id"]: f["text"] for f in fs or []}
    ours = re.compile(r"(?i)^\s*(tech|technician|provider|purtera|we|our tech)\w*\s+(to|will|shall|would)?\s*(provi|provdi|supply|bring|furnish)")
    theirs = re.compile(r"(?i)customer (will )?(provide|supply|furnish)|supplied by (the )?customer|client (will )?provide")
    pm = content.setdefault("provider_materials", [])
    keep_cm = []
    for x in content.get("customer_materials") or []:
        cited = [by.get(c, "") for c in (x.get("cites") or [])] if isinstance(x, dict) else []
        t = str(x.get("text") if isinstance(x, dict) else x)
        to_ours = any(ours.search(c) for c in cited) or (family == "low_voltage_cabling" and _CONSUMABLE.search(t)
                                                         and not any(theirs.search(c) for c in cited))
        if to_ours:
            if not _said(t, pm, 0.6):
                pm.append(x)
        else:
            keep_cm.append(x)
    content["customer_materials"] = [x for x in keep_cm if not _said(str(x.get("text") if isinstance(x, dict) else x), pm, 0.7)]
    # And back: a Provider material whose cited line has the customer or the
    # reseller providing or ordering it (010082: "AJ/CDW will provide/order the
    # 9U rack"), or that a customer duty says to have onsite.
    cust_side = re.compile(r"(?i)\b(cdw|shi|customer|client|they|aj)\b[^.]{0,25}\b(will |to )?(provide|order|supply|furnish)|customer[- ]provided|provided by (the )?(customer|cdw)")
    duties = " ".join(str(y.get("text") if isinstance(y, dict) else y) for y in content.get("customer_responsibilities") or []).lower()
    back = []
    for x in content.get("provider_materials") or []:
        if not isinstance(x, dict) or x.get("standard"):
            continue
        cited = [by.get(c, "") for c in x.get("cites") or []]
        key = [w for w in re.findall(r"[a-z0-9]{3,}", str(x.get("text")).lower()) if w not in ("the", "and", "for", "with")][:3]
        strict = family == "low_voltage_cabling" and _CONSUMABLE.search(str(x.get("text")))
        if (not strict and (any(cust_side.search(c) and not ours.search(c) for c in cited) or (key and all(w in duties for w in key)))) \
                or (strict and any(theirs.search(c) for c in cited)):
            back.append(x)
    if back:
        content["provider_materials"] = [x for x in content["provider_materials"] if not any(x is y for y in back)]
        content.setdefault("customer_materials", []).extend(back)
    if family == "low_voltage_cabling":
        pm = content.setdefault("provider_materials", [])
        keep = []
        for x in content.get("equipment_in_scope") or []:
            t = str(x.get("text") if isinstance(x, dict) else x)
            if _CONSUMABLE.search(t):
                if not _said(t, pm, 0.6):
                    pm.append(x)
            else:
                keep.append(x)
        content["equipment_in_scope"] = keep


SLOT_BY_HEAD = os.environ.get("SOWSMITH_C3_SOW_SLOT", "off").strip().lower() in ("1", "true", "yes", "on")
_SLOT_LIST = {"scope_of_work": "scope", "customer_responsibilities": "customer_responsibilities",
              "assumptions": "assumptions", "out_of_scope": "out_of_scope"}


def _slot_by_head(content: dict[str, Any], envelope: dict[str, Any]) -> int:
    """The section head decides the slot; the writer phrases (owner rule:
    heads decide, the parser extracts). A bullet every cited line of which the
    head placed in one SOW section moves to that section when the writer put
    it elsewhere. Refit 2026-10-18 on 345 real-SOW alignments: held-out
    agreement with PurTera's own SOWs 0.36 -> 0.78 (vs the writer's ~0.5)."""
    if not SLOT_BY_HEAD:
        return 0
    head_sec = {x["id"]: k for k, v in ((envelope.get("c3_brief") or {}).get("sections") or {}).items()
                for x in v if isinstance(x, dict) and str(x.get("section_by") or "") == "base:read:sow_section"}
    moved = 0
    for src in ("scope", "customer_responsibilities", "assumptions", "out_of_scope"):
        keep = []
        for x in content.get(src) or []:
            if not isinstance(x, dict) or x.get("standard") or x.get("kind") == "heading" or not x.get("cites"):
                keep.append(x)
                continue
            secs = {head_sec.get(c) for c in x["cites"]}
            dst = _SLOT_LIST.get(next(iter(secs))) if len(secs) == 1 and None not in secs else None
            if dst and dst != src:
                if dst == "scope":
                    sc = content.setdefault("scope", [])
                    close = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                                  and re.search(r"(?i)clos|clean|wrap|complet", str(y.get("text")))), len(sc))
                    sc.insert(close, x)
                else:
                    content.setdefault(dst, []).append(x)
                x["slot_by"] = "base:read:sow_section"
                moved += 1
            else:
                keep.append(x)
        content[src] = keep
    return moved


def _grounding(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """Lines whose only support is not the job (010012 round K: the summary
    built from a leasing render's captions; 010082: a request-form 'Project
    survey' row written as a firm task the notes call optional)."""
    by = {f["id"]: f for f in fs}
    pictured = lambda x: bool(x.get("cites")) and all(by.get(c, {}).get("pictured") for c in x.get("cites") or [])
    ov = [x for x in content.get("overview") or [] if isinstance(x, dict)]
    room_desc = lambda x: re.match(r"(?i)\s*(the )?[\w ()#-]{0,40}\b(contains|features|is equipped with|is furnished with)\b", str(x.get("text")))
    keep = [x for x in ov if not pictured(x) and not room_desc(x)]
    if len(keep) < len(ov):
        if not keep:
            # Only captions: the writer's own summary drafts the check set aside
            # (010012: "Conference Room 2102 contains a large rectangular table").
            keep = [{k: v for k, v in x.items() if k not in ("list", "reason")} for x in content.get("dropped") or []
                    if isinstance(x, dict) and x.get("list") == "overview" and x.get("text") and not pictured(x) and not room_desc(x)][:2]
        if keep:
            content["overview"] = keep
    # A picture caption is no duty, assumption or material (010012: "Building
    # access is via the central core ..." from a floor-plan caption).
    for k in ("customer_responsibilities", "assumptions", "customer_materials", "provider_materials", "equipment_in_scope", "arrival_protocol"):
        content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and pictured(x))]
    # "The attached floor plan": nothing is attached to a SOW.
    for k in LISTS:
        content[k] = [x for x in content.get(k) or []
                      if not re.search(r"(?i)\battached (floor ?plan|drawing|diagram|layout|map)", str(x.get("text") if isinstance(x, dict) else x))]
    # An exclusion needs a source that excludes: 000132's requested end-user
    # O365 support (a service in both notes) came out as Out of Scope.
    # A zero-quantity row excludes too (010147: "6U-10U servers x 0").
    neg = re.compile(r"(?i)\b(not|no|exclud\w*|except|outside|without|won'?t|n't|never|customer'?s? (team|staff|it)|by (the )?customer|handled by|responsib)|"
                     r"(\bx|\bqty:?|\bquantity:?|[:|])\s*0\s*$|^\s*0\s*(x|\|)")
    unsourced = [x for x in content.get("out_of_scope") or []
                 if isinstance(x, dict) and x.get("cites") and not x.get("standard") and all(c in by for c in x["cites"])
                 and not any(neg.search(by[c]["text"]) for c in x["cites"])]
    content["out_of_scope"] = [x for x in content.get("out_of_scope") or [] if not any(x is y for y in unsourced)]
    # A requested service the writer flipped into an exclusion is scope.
    # Its wording turns with it: 000132's "End-user support ... is excluded
    # from the service scope." printed as "Provide end-user support ... is
    # excluded" in the task list. Only when a cited line names that service:
    # 010284's lift, core-drilling and disposal exclusions cited call chatter
    # about poles and penetrations, and became "Provide lift rental" tasks.
    # Otherwise the exclusion stays, as the writer's own uncited line.
    # Grounding never turns an exclusion into a commitment (010082 pass U:
    # "Testing, certification, labeling, or as-built documentation", from an
    # open-question checklist line, became a Provider task). A requested
    # service the writer flipped into an exclusion is turned back by
    # _source_polarity, which reads the source's polarity; here the line
    # only loses the cite that does not exclude it.
    for x in unsourced:
        if _LINE_NEG.search(str(x.get("text"))):
            content["out_of_scope"].append(x)  # _source_polarity reads its source
        else:
            content["out_of_scope"].append({**x, "cites": [], "kind": "method"})
    # An assumption that hands a requested service to the customer (000132
    # pass O: "Customer will provide end-user support ... outside of
    # Provider's onsite support scope" from a bare service bullet) or excludes
    # it is reversed by _source_polarity, after the drop guard: dropped here,
    # the guard put it back as the only line citing the service.
    # Recurring onsite support has no install site conditions to price.
    facts_all = " ".join(f["text"] for f in fs)
    if re.search(r"(?i)\b(weekly|monthly|recurring|break[/ -]?fix|staff aug|ongoing (it )?support|onsite (it )?support)\b", facts_all) \
            and not re.search(r"(?i)\b(install|mount|cabling|run cable|decommission)\w*", facts_all):
        content["assumptions"] = [x for x in content.get("assumptions") or []
                                  if not re.search(r"(?i)hidden site conditions|undocumented cabling", str(x.get("text") if isinstance(x, dict) else x))]
    # The overview is PurTera's summary, never a pasted email (010198: "Hope
    # you're both doing well ... this Sunday morning, 8/2").
    greet = re.compile(r"(?i)^\s*(hi|hello|hey|good (morning|afternoon)|hope (you|all|everyone)|thanks|thank you|happy)\b|hope you'?re|\bboth doing well\b")
    ov = content.get("overview") or []
    keep = [x for x in ov if not greet.search(str(x.get("text") if isinstance(x, dict) else x))]
    if not keep and ov:
        # Every line was a pasted email: the writer's own summary drafts the
        # fact check set aside are the better overview.
        keep = [{k: v for k, v in x.items() if k not in ("list", "reason")} for x in content.get("dropped") or []
                if isinstance(x, dict) and x.get("list") == "overview" and x.get("text") and not greet.search(str(x["text"]))][:2]
    if keep:
        content["overview"] = keep
    _speaker_in_overview(content, by)
    # The reseller's warehouse work in the summary (010209: "Provider will
    # manage hardware receipt, benching, configuration, shipping ...") against
    # the SOW's own onsite-only lines.
    wh = r"(?i)\b(receipt|receiving|bench\w*|staging warehouse|configuration|shipping|disposal|serial numbers?)\b"
    content["overview"] = [x for x in content.get("overview") or []
                           if len(set(m.lower() for m in re.findall(wh, str(x.get("text") if isinstance(x, dict) else x)))) < 3] or content.get("overview")
    for k in ("scope", "deliverables", "customer_responsibilities"):
        content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and re.search(r"(?i)\bbenched\b|\bbenching\b", str(x.get("text"))))]
    # Gear the job does not touch is not Equipment in Scope (010171: the main
    # room's existing Rally Plus kit).
    content["equipment_in_scope"] = [x for x in content.get("equipment_in_scope") or []
                                     if not re.search(r"(?i)\bexisting\b", str(x.get("text") if isinstance(x, dict) else x))]
    # "Provider will supply no external microphones" is an exclusion, not a duty.
    for k in ("provider_materials", "scope", "assumptions"):
        for x in list(content.get(k) or []):
            t = str(x.get("text") if isinstance(x, dict) else "")
            m = re.match(r"(?i)\s*provider will (?:supply|provide|install|include) no ([^.]+)\.?$", t)
            if m:
                content[k].remove(x)
                x["text"] = m.group(1)[0].upper() + m.group(1)[1:] + " are not included."
                content.setdefault("out_of_scope", []).append(x)
    # PurTera's sales blurb is not this job's overview (010120 "3,500 vetted
    # engineers across 80+ countries").
    content["overview"] = [x for x in content.get("overview") or []
                           if not re.search(r"(?i)\bvetted\b|\d+\+? countries|engineers across|our capabilities|global (network|footprint)",
                                            str(x.get("text") if isinstance(x, dict) else x))] or content.get("overview")
    # A description of the room is not a task (010120: "Two large TVs are
    # mounted on the front wall" under Closeout): it is an assumption.
    desc = [x for x in content.get("scope") or [] if isinstance(x, dict) and x.get("kind") != "heading"
            and not x.get("standard") and re.match(r"(?i)\s*(the |two |three |one |an? |\d+ |existing )?[\w ()-]{0,40}\b(is|are) "
                                                 r"(currently |already )?(mounted|installed|located|present|in place)\b", str(x.get("text")))]
    # And a statement about the sites (000123: "Most stores have four register
    # lanes" as a Closeout task).
    desc += [x for x in content.get("scope") or [] if isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")
             and x not in desc and re.match(r"(?i)\s*(most|all|each|every|some|the)?\s*(stores?|sites?|locations?|rooms?|offices?|buildings?)\s+(have|has|are|is|use|contain)\b", str(x.get("text")))]
    desc += [x for x in content.get("scope") or [] if isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")
             and x not in desc and re.search(r"(?i)\b(will )?(stay|stays|remain|remains)\b.{0,20}\b(in place|mounted|in (their |its )?current|as is|on the)", str(x.get("text")))
             and not re.match(r"(?i)\s*(install|mount|remove|replace|leave|keep|retain|protect)\b", str(x.get("text")))]
    # Presales paperwork is not an onsite step (010120 "documentation
    # requirements needed to complete the SOW").
    content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and re.search(
        r"(?i)complete the sow|for the sow|to quote|for (the )?quot(e|ing)|pricing purposes", str(x.get("text"))))]
    # A verb-less description of a room or its furniture is a caption, never a
    # task (010012: "Conference Room 2102 contains a large rectangular table
    # with ten chairs", a leasing floor-plan caption, in the task list).
    noise_ids = set(content.get("_noise_atoms") or [])
    capt = [x for x in content.get("scope") or [] if isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")
            and x not in desc and (room_desc(x) or re.match(r"(?i)\s*(the )?[\w ()#-]{0,40}\b(is shown with|has (a|an|two|three|\d+))\b", str(x.get("text"))))
            and not re.match(r"(?i)\s*(install|mount|remove|replace|connect|run|place|provide|confirm|verify)\b", str(x.get("text")))]
    for x in capt:
        content["scope"].remove(x)
        if not (pictured(x) or (x.get("cites") and all(c in noise_ids for c in x["cites"]))):
            desc.append(x)  # a site condition the job states: an assumption
    for x in desc:
        if x in content["scope"]:
            content["scope"].remove(x)
        content.setdefault("assumptions", []).append(x)
    # Provider supplies materials: no blanket procurement exclusion (010120's
    # floor raceway and cable).
    if content.get("provider_materials"):
        content["out_of_scope"] = [x for x in content.get("out_of_scope") or []
                                   if not re.match(r"(?i)\s*procurement of (hardware|equipment)( not provided by the customer)?\.?$", str(x.get("text") if isinstance(x, dict) else x))]
    # Every line starts with a capital ("wall patching or painting.", 010373).
    for k in LISTS:
        for x in content.get(k) or []:
            if isinstance(x, dict) and isinstance(x.get("text"), str) and x["text"][:1].islower():
                x["text"] = x["text"][0].upper() + x["text"][1:]
    # A tools line in the task list is a tool (010115: "common tools" alone
    # under a "POS Hardware" heading).
    tl = [x for x in content.get("scope") or [] if isinstance(x, dict) and x.get("kind") != "heading"
          and re.match(r"(?i)\s*(use |bring |carry )?(common|standard|basic|general|hand)\b.{0,40}\btools?\b", str(x.get("text")))]
    for x in tl:
        content["scope"].remove(x)
        if not _said(str(x["text"]), content.setdefault("provider_tools", []), 0.5):
            content["provider_tools"].append(x)
    # A license is not an install material (010043: Verkada license increases
    # under Customer Provided Materials beside the licensing exclusion).
    for k in ("customer_materials", "provider_materials", "equipment_in_scope"):
        content[k] = [x for x in content.get(k) or [] if not re.search(r"(?i)\blicen[cs]e", str(x.get("text") if isinstance(x, dict) else x))]
    # "Provide and install" claims Provider supplies it: only with a Provider
    # materials line saying so (010043's cable runs).
    if not content.get("provider_materials"):
        for k in ("scope", "overview", "deliverables_text", "assumptions"):
            for x in content.get(k) or []:
                if isinstance(x, dict):
                    x["text"] = re.sub(r"(?i)\b(provide|furnish|supply)(,)? and install\b", lambda m: "install" if m.group(0)[0].islower() else "Install", str(x.get("text")))
    # Customer duties are the customer's: a Provider action moves to scope, an
    # open question or the quote's approval route goes (010082: 21 duties with
    # 5 provider tasks, 2 questions and the sign-off route).
    keep_cr = []
    for x in content.get("customer_responsibilities") or []:
        t = str(x.get("text") if isinstance(x, dict) else x)
        if t.rstrip().endswith("?") or re.search(r"(?i)\b(quote|pricing|proposal|purchase order|po)\b.{0,40}\b(approv|sign|review)|\b(approv|sign)\w*\b.{0,30}\b(quote|pricing|proposal)\b", t):
            continue
        if isinstance(x, dict) and re.match(r"(?i)\s*(provider|technician|tech|purtera|our tech)s? (will|to|shall|must)\b", t):
            x["text"] = re.sub(r"(?i)^\s*(provider|technician|tech|purtera|our tech)s? (will|to|shall|must)\s+", "", t)
            x["text"] = x["text"][:1].upper() + x["text"][1:]
            sc = content.setdefault("scope", [])
            close = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                          and re.search(r"(?i)clos|clean|wrap|complet", str(y.get("text")))), len(sc))
            if not _said(x["text"], sc, 0.6):
                sc.insert(close, x)
            continue
        std_x = isinstance(x, dict) and x.get("standard")
        # Two standard duties are two duties ("an active primary internet
        # connection" beside "an active cellular service ... for the backup
        # internet connection", 010198): only a writer line repeats one.
        # A standard duty goes only for a writer line that says it back
        # (010279 pass U: "Provide all hardware, cables, adapters, accessories,
        # labels and asset tags" went for a hardware line without the labels).
        if any(_said(t, [y], 0.7) and (not std_x or _said(str(y.get("text") if isinstance(y, dict) else y), [t], 0.7))
               for y in keep_cr if not (std_x and isinstance(y, dict) and y.get("standard"))):
            continue
        keep_cr.append(x)
    content["customer_responsibilities"] = keep_cr
    # A duty that hands something TO the customer is not the customer's own
    # (010279: the RFP's "Supply all terminal MAC addresses to Harris Teeter"
    # and "report any issues ... to Harris Teeter Systems Support Desk" became
    # customer responsibilities). It is the installer's task.
    names = [n for n in {str(content.get("end_customer") or "").strip(), "the customer", "customer"} if len(n) > 2]
    to_cust = re.compile(r"(?i)\b(provide|supply|submit|send|report|deliver|communicate|return|hand over)\w*\b[^.]{0,120}?\bto (" +
                         "|".join(re.escape(n) for n in names) + r")\b(?!'s|’s)")
    handed = [x for x in content.get("customer_responsibilities") or [] if isinstance(x, dict) and not x.get("standard")
              and x.get("cites") and to_cust.search(str(x.get("text")))]
    for x in handed:
        content["customer_responsibilities"].remove(x)
        sc = content.setdefault("scope", [])
        close = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                      and re.search(r"(?i)clos|clean|wrap|complet", str(y.get("text")))), len(sc))
        if not _said(str(x["text"]), sc, 0.6):
            sc.insert(close + 1 if close < len(sc) else close, x)
    # A statement of fact filed as a duty is an assumption (010171: "Customer
    # states there is no existing equipment in place").
    facts_as_duty = [x for x in content.get("customer_responsibilities") or [] if isinstance(x, dict)
                     and re.match(r"(?i)\s*(customer (states|confirms|says|notes)|there (is|are)|no existing)\b", str(x.get("text")))]
    for x in facts_as_duty:
        content["customer_responsibilities"].remove(x)
        x["text"] = re.sub(r"(?i)^\s*customer (states|confirms|says|notes)( that)?\s*", "", str(x["text"]))
        x["text"] = x["text"][:1].upper() + x["text"][1:]
        content.setdefault("assumptions", []).append(x)
    # A priced maybe is not a task (010171: "Zoom support available ...
    # depending on cost").
    content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and x.get("kind") != "heading"
                        and re.search(r"(?i)depending on (cost|price|budget)|if (the )?budget|for an additional (cost|fee)", str(x.get("text"))))]
    # Site hours are the template's business-hours line, not a customer duty.
    content["customer_responsibilities"] = [x for x in content.get("customer_responsibilities") or []
                                            if not re.search(r"(?i)\b(operates|open|hours)\b.{0,30}monday.{0,15}friday", str(x.get("text") if isinstance(x, dict) else x))]
    # A deliverable the exclusions rule out (010115: "Chain of custody for boxed
    # equipment transported from site" beside "transport ... excluded").
    oos = " ".join(str(x.get("text") if isinstance(x, dict) else x) for x in content.get("out_of_scope") or []).lower()
    for d in content.get("deliverables") or []:
        if isinstance(d, dict):
            desc, seen_d = [], []
            for t in d.get("description") or []:
                if re.search(r"(?i)chain of custody|transport|shipp?(ed|ing)|freight", str(t)) and re.search(r"transport|ship|freight|logistic", oos):
                    continue
                if _said(str(t), seen_d, 0.6):
                    continue
                desc.append(t); seen_d.append(str(t))
            d["description"] = desc
    # A heading left with nothing under it goes.
    sc = content.get("scope") or []
    content["scope"] = [x for i, x in enumerate(sc)
                        if not (isinstance(x, dict) and x.get("kind") == "heading")
                        or (i + 1 < len(sc) and not (isinstance(sc[i + 1], dict) and sc[i + 1].get("kind") == "heading"))]
    # A standard step says "each <thing>"; the deal says how many: write the
    # count in (010373: "Install each monitor arm" where the facts say 16
    # Humanscale arms, 48 monitors).
    facts_txt = " ".join(f["text"] for f in fs)
    for x in content.get("scope") or []:
        if not (isinstance(x, dict) and x.get("standard")):
            continue
        m = re.search(r"(?i)\beach (monitor arm|monitor|display|tv|camera|video bar|table controller|switch|register|printer|access point|speaker)\b", x["text"])
        if not m:
            continue
        noun = m.group(1).lower()
        counts = [int(n) for n, b in re.findall(r"(?i)\b(\d{1,4})\s+(?:new\s+|existing\s+)?(?:[A-Z][\w-]+\s+)?(?:\d+\"?\s*)?" + re.escape(noun) + r"(s)?\b", facts_txt) if 1 < int(n) < 1000]
        if counts:
            n = max(counts, key=counts.count)
            x["text"] = x["text"][:m.start()] + f"each of the {_words(n)} ({n}) {noun}s" + x["text"][m.end():]
    # A ladder or lift line on a job with no height in the sources (010373 is
    # desk-height work).
    if not re.search(r"(?i)\b(ladder|lift|ceiling|height|high|ft|feet|foot|elevated|scissor|boom)\b", facts_txt):
        content["assumptions"] = [x for x in content.get("assumptions") or []
                                  if not re.search(r"(?i)\bladders?\b|\bno lift\b|\blifts?\b|\blift (rental|operation)", str(x.get("text") if isinstance(x, dict) else x))]
    # A hedged ask is not a promise (000132: "24/7 on-call can be tough ...
    # depending on volume" became "24/7 on-call availability" in the summary,
    # beside its own no-SLO and business-hours assumptions).
    # The round-the-clock term however it is written: the writer spells
    # counts, and "24/7" came back "twenty-four (24) by seven (7)", which the
    # strip below left as the whole line (000132 pass V).
    clock = _ROUND_CLOCK
    promise = re.compile(r"(?i)" + clock + r"|on-?call|guaranteed response|response time|\bsla\b")
    hedge_facts = [f for f in fs if promise.search(f["text"]) and _HEDGE.search(f["text"])] + list(content.pop("_clock_hedges", []) or [])
    hedged = bool(hedge_facts)
    withheld = False
    no_slo = any(re.search(r"(?i)no (service[- ]level|slo|sla)|business hours", str(x.get("text") if isinstance(x, dict) else x))
                 for x in content.get("assumptions") or [])
    if hedged or no_slo:
        for k in ("overview", "scope", "schedule"):
            kept = []
            for x in content.get(k) or []:
                t = str(x.get("text") if isinstance(x, dict) else x)
                if isinstance(x, dict) and promise.search(t):
                    # The whole clause goes, not its object (000132 pass U:
                    # "... and remain available for.").
                    t1 = _drop_clause(t, promise)
                    if t1 and len(t1) >= 25 and not promise.search(t1):
                        x["text"] = t1
                        kept.append(x)
                        continue
                    t2 = re.sub(r"(?i),?\s*(including |and |with |provide |offer |providing )*(" + clock + r")?\s*on-?call( availability| support| coverage)?"
                                r"|,?\s*(with |and |provide |providing )*(" + clock + r")( availability| support| coverage)?", "", t).strip()
                    t2 = re.sub(r"\s+,", ",", re.sub(r",\s*,", ",", t2))
                    withheld = True
                    # What is left must still say some work: "Provider will
                    # provide." is the promise's verb with nothing after it.
                    if len(t2) < 25 or promise.search(t2) or re.match(r"(?i)\s*(provider will )?(provide|offer|maintain|be)\s*[.]?\s*$", t2):
                        continue
                    x["text"] = t2
                kept.append(x)
            if kept or not content.get(k):
                content[k] = kept
        # A hedged round-the-clock ask the SOW does not promise is said as
        # what it is: not included unless agreed (000132: "The 24/7
        # availability can be tough because you're paying for someone to be
        # on call"). Only when a line was withheld and nothing says it yet.
        if hedged and withheld and not any(promise.search(str(x.get("text") if isinstance(x, dict) else x))
                                           for k in ("assumptions", "out_of_scope") for x in content.get(k) or []):
            content.setdefault("out_of_scope", []).append(
                {"text": "24/7 on-call availability or guaranteed response times, unless separately agreed in writing.",
                 "cites": [hedge_facts[0]["id"]], "kind": "fact"})
    # A service the writer filed as an assumption is a task (000132: "End-user
    # support includes O365/M365 clients").
    svc = [x for x in content.get("assumptions") or [] if isinstance(x, dict)
           and re.match(r"(?i)\s*[\w /-]{0,30}\bsupport (includes|covers|will include)\b", str(x.get("text")))]
    for x in svc:
        content["assumptions"].remove(x)
        x["text"] = "Provide " + re.sub(r"(?i)\bsupport (includes|covers|will include)\b", "support, including", str(x["text"]))[0].lower() + \
            re.sub(r"(?i)\bsupport (includes|covers|will include)\b", "support, including", str(x["text"]))[1:]
        sc = content.setdefault("scope", [])
        close = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                      and re.search(r"(?i)clos|clean|wrap|complet", str(y.get("text")))), len(sc))
        sc.insert(close, x)
    # PurTera's sales pitch is never a term of the job (010217: "Customer has
    # deep familiarity with Certinia ... so no handholding is required", from
    # our own capability email).
    pitch = re.compile(r"(?i)handholding|hand-holding|deep familiarity|engineers? (onsite|on site) within|do not charge for (pm|project man)|"
                       r"\bvetted\b|\d+\+? countries|nationwide coverage|we pride|our (capabilities|team has)|white[- ]glove")
    for k in LISTS:
        content[k] = [x for x in content.get(k) or [] if not pitch.search(str(x.get("text") if isinstance(x, dict) else x))] \
            if k != "overview" else ([x for x in content.get(k) or [] if not pitch.search(str(x.get("text") if isinstance(x, dict) else x))] or content.get(k))
    # Closing steps written inside a room group go under Closeout (010217:
    # "verify displays", "remove debris" under Room 247).
    sc = content.get("scope") or []
    ci = next((i for i, x in enumerate(sc) if isinstance(x, dict) and x.get("kind") == "heading"
               and re.search(r"(?i)clos|clean|wrap|complet", str(x.get("text")))), None)
    if ci is not None:
        early = [x for x in sc[:ci] if isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")
                 and re.match(r"(?i)\s*(clean|remove (all )?(debris|packaging|trash)|dispose|haul away|perform a (final )?walk|walk ?through|"
                              r"confirm completion|obtain sign)", str(x.get("text")))]
        if early:
            rest = [x for x in sc if not any(x is y for y in early)]
            ci2 = next(i for i, x in enumerate(rest) if x is sc[ci])
            content["scope"] = rest[:ci2 + 1] + early + rest[ci2 + 1:]
    # A "no new cabling" term beside the job's own cable runs (010120: "Route
    # cable drops from the ceiling" and "No new structured network cabling").
    scope_txt = " ".join(str(x.get("text")) for x in content.get("scope") or [] if isinstance(x, dict))
    if re.search(r"(?i)cable drops?|run (new )?cable|pull (new )?cable|new cat ?6|cable run", scope_txt):
        for k in ("assumptions", "out_of_scope"):
            content[k] = [x for x in content.get(k) or [] if not re.search(
                r"(?i)no new (structured )?(network )?cabling|new (structured )?cabling (or|and) (network )?drops|new network drops|new structured cabling",
                str(x.get("text") if isinstance(x, dict) else x))]
    # "Visible cable routing" against a no-visible-cables instruction.
    if any(re.search(r"(?i)no cables? (should|will) be visible|not (be )?visible", f["text"]) for f in fs):
        for d in content.get("deliverables") or []:
            if isinstance(d, dict):
                d["description"] = [re.sub(r"(?i)\bvisible cable routing\b", "concealed cable routing", str(t)) for t in d.get("description") or []]
        for k in ("scope", "assumptions"):
            for x in content.get(k) or []:
                if isinstance(x, dict):
                    x["text"] = re.sub(r"(?i)\broute and dress visible cabling\b", "Route and conceal cabling", str(x.get("text")))
    # Removed gear the SOW leaves onsite is not shipped (010083: "stage them
    # for shipment ... for disposal" beside "does not ship" and "left onsite").
    terms = " ".join(str(x.get("text") if isinstance(x, dict) else x) for k in ("assumptions", "out_of_scope") for x in content.get(k) or [])
    if re.search(r"(?i)does not (package|ship)|not ship|left onsite|remain onsite|placed in the customer-designated onsite|return logistics", terms):
        for k in ("scope", "customer_responsibilities"):
            content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")
                          and re.search(r"(?i)for shipment|ship (them |it )?to|shipping to|for disposal|ship it to a", str(x.get("text"))))]
    # After-hours work is not business hours (010083).
    if any(re.search(r"(?i)after[- ]hours|after (store |branch )?clos|overnight|before (store |branch )?open|minutes before (store |branch )?clos", f["text"]) for f in fs):
        content["assumptions"] = [x for x in content.get("assumptions") or []
                                  if not re.search(r"(?i)(normal|regular|standard) business hours", str(x.get("text") if isinstance(x, dict) else x))]
    # Before-work photos and checks lead the work (010083: prework photos last).
    sc = content.get("scope") or []
    pre_ph = [x for x in sc if isinstance(x, dict) and x.get("kind") != "heading" and re.search(r"(?i)\b(pre-?work|before (starting|work|removal)|existing condition|before)\b.{0,40}photo|photo.{0,40}\b(before (starting|work|removal|any)|pre-?work|existing condition)", str(x.get("text")))
              and not re.search(r"(?i)after|completed|completion", str(x.get("text")))]
    # Labeling the gear before it ships is the customer's staging (010209: the
    # customer "will supply, configure, label and deliver"; Provider labeled to
    # the naming convention too).
    if re.search(r"(?i)label.{0,30}deliver|label.{0,30}prior to (shipment|deployment)", " ".join(str(y.get("text")) for y in content.get("customer_responsibilities") or [] if isinstance(y, dict))):
        content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and re.search(r"(?i)label (all )?(the )?(new )?equipment with|naming convention", str(x.get("text"))))]
        sc = content.get("scope") or []
    if pre_ph:
        rest = [x for x in sc if not any(x is y for y in pre_ph)]
        at = 1 if rest and isinstance(rest[0], dict) and rest[0].get("kind") == "heading" else 0
        # Under the first work heading: past a leading prep group, never into
        # a closing one (010209 pass V: with "Switch replacement" then
        # "Closeout", the prework photos went under Closeout).
        if at and re.search(r"(?i)arriv|prep|getting started|common|general|site access|before", str(rest[0].get("text"))):
            nxt = next((i for i, x in enumerate(rest) if i > 0 and isinstance(x, dict) and x.get("kind") == "heading"), None)
            if nxt is not None and not _CLOSING_HEAD.search(str(rest[nxt].get("text"))):
                at = nxt + 1
        # After the arrival and review steps that open the group.
        while at < len(rest) and isinstance(rest[at], dict) and rest[at].get("kind") != "heading" \
                and re.match(r"(?i)\s*(arrive|check in|review|verify available|validate receipt|meet)\b", str(rest[at].get("text"))):
            at += 1
        content["scope"] = rest[:at] + pre_ph + rest[at:]
    # A task the SOW also excludes goes (010326: "Provide help with racking/
    # stacking/wall mounting" beside rack-and-stack in Out of Scope).
    # The deal's own exclusions: a generic standard exclusion ("additional
    # work from missing equipment") does not rule out the task of reporting it.
    oos_items = [x for x in content.get("out_of_scope") or [] if isinstance(x, dict) and not x.get("standard")]
    content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and x.get("kind") != "heading"
                        and not x.get("standard") and oos_items and _said(re.sub(r"(?i)^\s*(provide help with|assist with|help with)\s*", "", str(x.get("text"))), oos_items, 0.6))]
    # Vendor lease or financing terms are not SOW terms (010326 CDW lease text).
    for k in LISTS:
        content[k] = [x for x in content.get(k) or [] if not re.search(r"(?i)residual value|lease term|end of the lease|financing (terms|agreement)",
                                                                         str(x.get("text") if isinstance(x, dict) else x))]
    # "Labor only" overrides an RFP's vendor logistics (010279: "We are pricing
    # physical installation labor only"; the SOW still accepted delivery,
    # stored, tagged and transported gear to 113 stores).
    if any(re.search(r"(?i)\blabor[- ]only\b|\blabor only\b|installation labor only|onsite labor only|field labor only", f["text"]) for f in fs):
        logi = re.compile(r"(?i)accept (the )?deliver|receiv(e|ing) (the )?(equipment|hardware|shipments?)|\bwarehous|\bstor(e|ing|age)\b[^.]{0,25}\b(equipment|hardware)|accept,? (and )?verify deliver|"
                          r"asset[- ]tag|transport|\bkitting\b|\bstaging (area|facility|center|warehouse)|proof of delivery|inventory recap")
        # A step the source files under its storage or staging section is that
        # logistics too (010279 pass U: "Record all MAC addresses for terminals
        # by store" and "Provide all terminal MAC addresses ... the prior week",
        # RFP 2.1 "Storage / Staging of Equipment"). The customer's own
        # standard duties (labels and asset tags, retired-gear logistics) are
        # the customer's and stay.
        staged = {f["id"] for f in fs if re.search(r"(?i)\b(storage|staging|warehous\w*|receiving|kitting)\b", str(f.get("under") or ""))}
        content["_noise_atoms"] = list(set(content.get("_noise_atoms") or []) | staged)  # the guard leaves them out
        for k in ("scope", "customer_responsibilities", "deliverables_text"):
            content[k] = [x for x in content.get(k) or [] if not (
                isinstance(x, dict) and x.get("kind") != "heading" and not (k == "customer_responsibilities" and x.get("standard"))
                and (logi.search(str(x.get("text"))) or (k == "scope" and not x.get("standard") and x.get("cites")
                                                        and all(c in staged for c in x["cites"]))))]
        # A logistics task filed as a customer material is the material it
        # handles ("Apply Asset tag for each POS terminal" -> "Asset tags for
        # each POS terminal.").
        for x in content.get("customer_materials") or []:
            if isinstance(x, dict) and re.match(r"(?i)\s*(apply|affix|attach|place|put)\s+", str(x.get("text"))):
                t = re.sub(r"(?i)^\s*(apply|affix|attach|place|put)\s+(an?\s+|the\s+)?", "", str(x["text"])).strip().rstrip(".")
                t = re.sub(r"(?i)^(\w+ \w*?)(tag|label|sticker)\b(?!s)", r"\1\2s", t)
                x["text"] = t[:1].upper() + t[1:] + "."
        # A summary listing the RFP's verbs keeps the labor ones (010279:
        # "stage, tag, transport, de-install, install, label, and test the
        # equipment using the required checklists" went whole, and with it the
        # only de-install and checklist line in the SOW).
        lverb = re.compile(r"(?i)^(stage|tag|asset[- ]tag|transport|receive|accept( delivery)?|store|warehouse|kit|ship|deliver|inventory)$")
        for x in content.get("overview") or []:
            if isinstance(x, dict) and logi.search(str(x.get("text"))):
                x["text"] = _strip_verbs(str(x["text"]), lverb)
        ov = [x for x in content.get("overview") or [] if not (isinstance(x, dict) and logi.search(str(x.get("text"))))]
        # Gear shipped to Provider is warehousing a labor-only bid declines
        # (010279: "... will directly ship the 71 self-checkout terminals to
        # Provider"): the clause goes, the rest of the supply line stays.
        for x in list(content.get("customer_materials") or []):
            if isinstance(x, dict) and re.search(r"(?i)\bship\w*\b[^.;]{0,80}\bto (the )?(provider|vendor)\b", str(x.get("text"))):
                parts = [p_ for p_ in re.split(r";\s*", str(x["text"])) if not re.search(r"(?i)\bship\w*\b[^.;]{0,80}\bto (the )?(provider|vendor)\b", p_)]
                if parts:
                    x["text"] = "; ".join(parts).rstrip(" .;") + "."
                else:
                    content["customer_materials"].remove(x)
        content["overview"] = ov or [{"text": "Provider will provide onsite installation labor at each location.", "cites": [], "kind": "method"}]
        content["overview"] = [x for x in content["overview"] if not re.search(r"(?i)appears to be a basic|this looks like|seems like a", str(x.get("text") if isinstance(x, dict) else x))] or content["overview"]
        asm = content.setdefault("assumptions", [])
        if not any(re.search(r"(?i)labor only|onsite (field )?labor", str(x.get("text") if isinstance(x, dict) else x)) for x in asm):
            asm.insert(0, {"text": "Pricing covers onsite installation labor only; hardware is delivered to each location before technicians arrive.", "cites": [], "kind": "method"})
    # The customer is not given the configuration Provider's own scope does.
    if re.search(r"(?i)\bconfigur", " ".join(str(x.get("text")) for x in content.get("scope") or [] if isinstance(x, dict) and x.get("kind") != "heading")):
        content["customer_responsibilities"] = [x for x in content.get("customer_responsibilities") or []
                                                if not re.search(r"(?i)perform all device configurations", str(x.get("text") if isinstance(x, dict) else x))]
    # A device the customer asked to add gets an install line (010198: the
    # router's setup steps but no "Install the Ubiquiti router").
    sc_txt = " ".join(str(x.get("text")) for x in content.get("scope") or [] if isinstance(x, dict))
    for f in fs:
        m = re.search(r"(?i)\b(?:add|install|put in)\s+(?:an?|one|the)?\s*((?:[A-Z][\w-]*\s+){0,2}(router|gateway|firewall|access point|switch)(?:\s+with cellular backup)?)", f["text"])
        if m and not re.search(r"(?i)\binstall\b[^.]{0,60}\b" + m.group(2), sc_txt) and not f.get("asked", False) is None:
            sc = content.setdefault("scope", [])
            close = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                          and re.search(r"(?i)clos|clean|wrap|complet", str(y.get("text")))), len(sc))
            first_std = next((i for i, y in enumerate(sc[:close]) if isinstance(y, dict) and y.get("standard")
                              and re.search(r"(?i)" + m.group(2), str(y.get("text")))), close)
            sc.insert(first_std, {"text": f"Install the {m.group(1).strip()}.", "cites": [f["id"]], "kind": "fact"})
            sc_txt += " install " + m.group(2)
    # "We have X" is the customer's existing gear, never an install (010171:
    # "in our main conference room we have a Logitech Rally Plus").
    by_t = {f["id"]: f["text"] for f in fs}
    content["scope"] = [x for x in content.get("scope") or [] if not (
        isinstance(x, dict) and x.get("cites") and re.match(r"(?i)\s*install\b", str(x.get("text")))
        and all(re.search(r"(?i)\b(we|they) (already |currently )?have\b|\balready (have|installed|in place)\b|\bcurrently (have|has|installed)\b", by_t.get(c, ""))
                for c in x["cites"]))]
    # An uncited "limited to basic ..." cap is library text that narrows the
    # contract below the deal's own scope (000132: vSphere, NAS and patching
    # capped as "basic on-site support, triage, equipment moves").
    content["assumptions"] = [x for x in content.get("assumptions") or [] if not (
        isinstance(x, dict) and not x.get("cites") and not x.get("standard")
        and re.match(r"(?i)\s*(services|work|provider'?s? (work|scope|services|support)|provider support|support) (are|is) limited to (basic|general|light)\b", str(x.get("text"))))]
    # "Only pack and deliver; we manage the destination": no destination work
    # (010147: a Destination Office section of unpack, place, reconnect).
    if any(re.search(r"(?i)only (be )?(a )?pack(ing)? and deliver|we will (manage|handle|take care of) (things |everything )?at the destination|"
                     r"(customer|we) (will )?(unpack|set up) at (the )?destination", f["text"]) for f in fs):
        sc = content.get("scope") or []
        out, skip = [], False
        for x in sc:
            if isinstance(x, dict) and x.get("kind") == "heading":
                skip = bool(re.search(r"(?i)destination|unpack|setup at|reinstall", str(x.get("text"))))
                if skip:
                    continue
            # A standard step stays: the job's own standards already say what
            # the move does at the dock (010147 pass U: the tag-verification
            # step went with a writer "Destination" heading).
            if (skip and not (isinstance(x, dict) and x.get("standard"))) or (
                    isinstance(x, dict) and not x.get("standard")
                    and re.match(r"(?i)\s*(unpack|reconnect|re-?install|place (the )?equipment at|set up)\b", str(x.get("text")))):
                continue
            out.append(x)
        content["scope"] = out
        content["assumptions"] = [x for x in content.get("assumptions") or [] if not re.search(r"(?i)\b(unpack|reconnect)", str(x.get("text") if isinstance(x, dict) else x))]
    # "No rack modification" against sites without a dedicated rack (010083).
    if any(re.search(r"(?i)(no|not have a|without a) (dedicated )?rack|may not have (a )?rack|consolidat", f["text"]) for f in fs):
        content["assumptions"] = [x for x in content.get("assumptions") or [] if not re.search(r"(?i)no rack modification|rack (modification|fabrication) is (not )?required",
                                                                                                 str(x.get("text") if isinstance(x, dict) else x))]
    # A leasing brochure is not the job (010012: the landlord's leasing team
    # named "the site contacts"; the summary built from its floor-plan text).
    brochure = re.compile(r"(?i)leasing (team|agent|office)|contact our leasing|vr view|virtual tour|for lease|rsf\b|rentable|"
                          r"waiting area|pantry|reception area|available space")
    bro = {f["id"] for f in fs if brochure.search(f["text"])}
    # Lines from a plan or brochure document: never a duty, assumption, summary
    # or arrival step (they describe the building), but they may still name
    # the room a scope task happens in.
    bro_doc = set(content.pop("_brochure_atoms", []) or [])
    for k in ("customer_responsibilities", "assumptions", "arrival_protocol", "overview"):
        kept = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("cites") and all(c in bro_doc for c in x["cites"]))]
        if kept or k != "overview":
            content[k] = kept
    for k in LISTS:
        kept = [x for x in content.get(k) or [] if not (isinstance(x, dict) and ((x.get("cites") and all(c in bro for c in x["cites"]))
                                                                                 or brochure.search(str(x.get("text")))))]
        if kept or k != "overview":
            content[k] = kept
    if not content.get("overview") or all(isinstance(x, dict) and ((x.get("cites") and all(c in bro for c in x["cites"])) or brochure.search(str(x.get("text"))))
                                          for x in content.get("overview") or []):
        alt = [{k2: v for k2, v in x.items() if k2 not in ("list", "reason")} for x in content.get("dropped") or []
               if isinstance(x, dict) and x.get("list") == "overview" and not all(c in bro for c in x.get("cites") or ["-"])
               and not brochure.search(str(x.get("text")))][:2]
        if not alt:
            # No clean draft: the summary is the job's own install lines.
            inst = [x for x in content.get("scope") or [] if isinstance(x, dict) and x.get("cites")
                    and re.match(r"(?i)\s*(install|mount|replace|remove|run|deploy|pack|swap)\b", str(x.get("text")))][:2]
            alt = [{"text": "Provider will " + str(x["text"])[0].lower() + str(x["text"])[1:], "cites": x.get("cites") or [], "kind": "fact"} for x in inst]
        if alt:
            content["overview"] = alt
    # Gear the SOW excludes is not in its equipment lists (010171: the stale
    # bundle "+ 2 Mic Pod" beside "external microphones are not included").
    oos_txt = " ".join(str(x.get("text") if isinstance(x, dict) else x) for x in content.get("out_of_scope") or []).lower()
    banned = [w for w in ("mic pod", "microphone", "sight", "nuc", "license", "lift") if w in oos_txt]
    if banned:
        for k in ("equipment_in_scope", "customer_materials", "provider_materials", "scope", "deliverables_text"):
            content[k] = [x for x in content.get(k) or [] if not (any(b in str(x.get("text") if isinstance(x, dict) else x).lower() for b in banned)
                                                                  and not (isinstance(x, dict) and x.get("standard")))]
    # A relative date copied from an email means nothing in a SOW (010171:
    # "Have the network drops installed next week").
    for k in LISTS:
        for x in content.get(k) or []:
            if isinstance(x, dict) and isinstance(x.get("text"), str):
                x["text"] = re.sub(r"(?i)\s+(next week|this week|tomorrow|later this week|early next week|by end of (the )?week)\b", "", x["text"])
    # PurTera's own "Safe to assume X?" is the assumption PurTera prices on
    # (010085: "Safe to assume no spotter is required while lifts are in use?").
    asm = content.setdefault("assumptions", [])
    for f in fs:
        m = re.match(r"(?i)\s*(?:is it |it'?s )?safe to assume (?:that )?(.{8,160}?)\s*\?\s*$", f["text"])
        if m:
            t = m.group(1).strip().rstrip(".")
            t = t[0].upper() + t[1:] + "."
            if not _said(t, asm, 0.6):
                asm.append({"text": t, "cites": [f["id"]], "kind": "fact"})
    # A dropped scope line every number and most words of which the sources
    # state comes back (010085: "Install approximately twenty (20) network
    # switches across the one (1) MDF and thirteen (13) raised IDFs").
    all_facts = " ".join(f["text"] for f in fs)
    have_nums = set(re.findall(r"\d+", all_facts))
    sc = content.setdefault("scope", [])
    for x in content.get("dropped") or []:
        if not (isinstance(x, dict) and x.get("list") == "scope" and re.match(r"(?i)\s*(install|replace|remove|mount|run)\b", str(x.get("text")))):
            continue
        nums = set(re.findall(r"\d+", str(x["text"])))
        if nums and nums <= have_nums and _stated_by(re.sub(r"\d+", "", str(x["text"])), all_facts) and not _said(str(x["text"]), sc, 0.7):
            close = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                          and re.search(r"(?i)clos|clean|wrap|complet", str(y.get("text")))), len(sc))
            sc.insert(close, {k: v for k, v in x.items() if k not in ("list", "reason")})
    # A line from an "Open Questions" list is a question, never a duty (010082:
    # the parking/loading-dock question became a customer responsibility).
    oq = {f["id"] for f in fs if re.search(r"(?i)open questions?|questions? for (the )?customer|to confirm|tbd items", str(f.get("under") or ""))}
    if oq:
        for k in ("customer_responsibilities", "assumptions", "scope"):
            content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("cites") and all(c in oq for c in x["cites"]))]
    # PurTera's own people are never named in the customer's summary (010082:
    # "Chase, Kelly and Octavian" in the overview).
    ours = [n for n in content.pop("_our_people", []) or [] if len(n) > 2]
    if ours:
        pat = re.compile(r"(?i)\b(" + "|".join(re.escape(n) for n in ours) + r")\b")
        content["overview"] = [x for x in content.get("overview") or [] if not (isinstance(x, dict) and pat.search(str(x.get("text"))))] or content.get("overview")
    # Per-lane installs when the sites have several register lanes (000123:
    # "one (1) printer at each Customer location" with four lanes a store).
    if any(re.search(r"(?i)\b(two|three|four|five|six|\d)\s+(register |pos )?lanes?\b", f["text"]) for f in fs):
        for x in content.get("scope") or []:
            if isinstance(x, dict):
                x["text"] = re.sub(r"(?i)\bone \(1\) ([\w +/()-]{2,60}?) at each (customer )?(location|store|site)\b",
                                   r"\1 at each register lane", str(x.get("text")))
    # A site condition is an assumption, not a closing step (010082: the 12 ft
    # ceiling and the safe under Closeout).
    cond = [x for x in content.get("scope") or [] if isinstance(x, dict) and re.match(r"(?i)\s*site condition:\s*", str(x.get("text")))]
    for x in cond:
        content["scope"].remove(x)
        x["text"] = re.sub(r"(?i)^\s*site condition:\s*", "", str(x["text"]))
        content.setdefault("assumptions", []).append(x)
    # "Only one (1) register and one (1) printer are included" contracts out
    # the rest of the job (010198's router).
    content["assumptions"] = [x for x in content.get("assumptions") or []
                              if not re.match(r"(?i)\s*only\b.{0,120}\b(is|are) included\b", str(x.get("text") if isinstance(x, dict) else x))]
    # "Provider-supplied" hardware when Provider supplies nothing (000123).
    if not content.get("provider_materials"):
        content["assumptions"] = [x for x in content.get("assumptions") or []
                                  if not re.search(r"(?i)provider[- ]supplied", str(x.get("text") if isinstance(x, dict) else x))]
    _weekend(content, fs)
    # Work steps written after the closing heading go before it (010198: the
    # router install after "Confirm completion before leaving site").
    sc = content.get("scope") or []
    close = next((i for i, x in enumerate(sc) if isinstance(x, dict) and x.get("kind") == "heading"
                  and re.search(r"(?i)clos|clean|wrap|complet", str(x.get("text")))), None)
    if close is not None:
        late = [x for x in sc[close + 1:] if isinstance(x, dict) and x.get("kind") != "heading"
                and re.match(r"(?i)\s*(install|mount|run|replace|configure|connect|deploy|set up|terminate|remove|pack|unbox)\b", str(x.get("text")))]
        if late:
            rest = [x for x in sc if not any(x is y for y in late)]
            ci = next(i for i, x in enumerate(rest) if x is sc[close])
            content["scope"] = rest[:ci] + late + rest[ci:]
    # A survey the facts call optional or conditional is not a task.
    if any(re.search(r"(?i)survey.{0,60}\b(optional|contingent|if needed|as needed|if required|probably not|not needed|may not)\b|"
                     r"\b(optional|contingent)\b.{0,40}survey", f["text"]) for f in fs):
        content["scope"] = [x for x in content.get("scope") or []
                            if not (isinstance(x, dict) and x.get("kind") != "heading" and re.search(r"(?i)\bsurvey\b", str(x.get("text"))))]
    # One line per exclusion, and arrival lines that repeat a customer duty go
    # (010115: transport/packing/disposal two or three times each).
    oos, kept_o = content.get("out_of_scope") or [], []
    for x in oos:
        t = str(x.get("text") if isinstance(x, dict) else x)
        if not _said(t, kept_o, 0.6):
            kept_o.append(x)
    content["out_of_scope"] = kept_o
    cr_items = content.get("customer_responsibilities") or []
    content["arrival_protocol"] = [x for x in content.get("arrival_protocol") or []
                                   if not (isinstance(x, dict) and re.match(r"(?i)\s*customer\b", str(x.get("text"))) and _said(str(x.get("text")), cr_items, 0.5))]
    for x in content.get("arrival_protocol") or []:
        if isinstance(x, dict) and re.match(r"(?i)\s*require an onsite manager", str(x.get("text"))):
            x["text"] = re.sub(r"(?i)^\s*require an onsite manager", "Customer provides an onsite manager", str(x["text"]))
    # The customer's own staff commitment is the customer's (010147: "Provide a
    # member of the Lafayette team to meet the carrier" as a Provider step).
    mv = [x for x in content.get("scope") or [] if isinstance(x, dict) and re.match(r"(?i)\s*(provide|have) (a|one) (member|representative|person) (of|from) (the|their|our) ", str(x.get("text")))]
    for x in mv:
        content["scope"].remove(x)
        if not _said(str(x["text"]), content.setdefault("customer_responsibilities", []), 0.5):
            content["customer_responsibilities"].append(x)
    # Customer duties written into the technician's arrival protocol are the
    # customer's (010082: inform Henry of the schedule, provide credentials).
    arr_keep = []
    for x in content.get("arrival_protocol") or []:
        t = str(x.get("text") if isinstance(x, dict) else x)
        if isinstance(x, dict) and re.match(r"(?i)\s*(customer|the customer|site staff|onsite staff|have the onsite|provide (any |the )?(security|credentials|access|badges?|escorts?))\b", t):
            if not _said(t, content.setdefault("customer_responsibilities", []), 0.5):
                content["customer_responsibilities"].append(x)
            continue
        arr_keep.append(x)
    content["arrival_protocol"] = arr_keep
    # Assessment is ticked only when an assessment or survey survives in scope.
    if "assessment" in (content.get("services") or []) and not re.search(
            r"(?i)\bsurvey|assess|walk-?through|evaluat", " ".join(str(x.get("text")) for x in content.get("scope") or [] if isinstance(x, dict))):
        content["services"] = [v for v in content["services"] if v != "assessment"] or ["implementation"]
    # A possible follow-up visit is never a planned one (010419: "some
    # follow-up needed the next day" became a scheduled return on a one-day fee).
    fu = _FOLLOW_UP
    had = False
    for k in ("overview", "schedule", "scope", "assumptions"):
        kept = [x for x in content.get(k) or [] if not (isinstance(x, dict) and not x.get("standard") and fu.search(str(x.get("text"))))]
        had = had or len(kept) < len(content.get(k) or [])
        if kept or k != "overview":
            content[k] = kept
    if had:
        oos = content.setdefault("out_of_scope", [])
        t = "Follow-up or return visits, unless authorized in writing or by Change Order."
        if not _said(t, oos, 0.5):
            oos.append({"text": t, "cites": [], "kind": "method"})
    # An overview left opening on "also" lost its lead line: put it back
    # (010116: "Provider will also install ..." alone).
    ov = content.get("overview") or []
    if ov and isinstance(ov[0], dict) and re.match(r"(?i)\s*(provider will )?also\b", str(ov[0].get("text"))):
        lead = [{k: v for k, v in x.items() if k not in ("list", "reason")} for x in content.get("dropped") or []
                if isinstance(x, dict) and x.get("list") == "overview" and not re.match(r"(?i)\s*(provider will )?also\b", str(x.get("text")))][:1]
        if lead:
            content["overview"] = lead + ov
    # A stated deadline the schedule lacks goes in (010115: "dismantled and
    # boxed up by end of Business 8/7/26" never reached the schedule).
    sched_txt = " ".join(str(x.get("text") if isinstance(x, dict) else x) for x in content.get("schedule") or [])
    if not re.search(r"(?i)complet|deadline|by end of|no later than", sched_txt):
        for f in fs:
            m = re.search(r"(?i)\b(by (the )?end of (business|day)|completed? by|no later than)\b[^.]{0,30}?(\d{1,2}/\d{1,2}(/\d{2,4})?|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2})", f["text"])
            # A deadline for receiving, invoicing or shipping hardware is the
            # supply chain's, not the job's completion (010209 pass T: "All
            # hardware must be received into COI no later than 9/25").
            if m and re.search(r"(?i)\b(receiv\w*|invoic\w*|ship\w*|deliver\w*|order\w*|procur\w*|arriv\w*)\b", f["text"][max(0, m.start() - 60):m.end()]):
                continue
            if m and f.get("type") in ("deadline", "milestone_phase", "scope_item", "constraint", "", None):
                content.setdefault("schedule", []).append({"text": f"Requested Completion: {m.group(0).strip()}", "cites": [f["id"]], "kind": "fact"})
                break
    # An empty schedule takes back the cited schedule line the check set aside
    # (010043: "Week of July 6, 2026" dropped with no reason).
    if not content.get("schedule"):
        # Never a line the check dropped for a date its cites do not state
        # (010198: the writer's "2026-08-09" for "next Sunday, August 16th").
        back = [{k: v for k, v in x.items() if k not in ("list", "reason")} for x in content.get("dropped") or []
                if isinstance(x, dict) and x.get("list") == "schedule" and x.get("cites")
                and _dates_ok(str(x.get("text")), " ".join(by.get(c, {}).get("text", "") for c in x["cites"]))][:2]
        if back:
            content["schedule"] = back
    # Form labels are not prose (010115: "... Specific Location(s): Sodexo
    # Dining CAFE." in the summary).
    for x in content.get("overview") or []:
        if isinstance(x, dict):
            x["text"] = re.sub(r"(?i)\s*specific location\(?s?\)?\s*:\s*[^.]*\.?", "", str(x.get("text"))).strip()
    content["overview"] = [x for x in content.get("overview") or [] if not (isinstance(x, dict) and len(str(x.get("text"))) < 12)] or content.get("overview")
    # A scope line that only repeats the summary goes (010115 "POS Hardware"
    # heading over "Assist Bernie Donnelly ...").
    # Only a line that restates a whole summary sentence: a task the summary
    # merely mentions among others is the contract's own line and stays
    # (010198: "Install one (1) Square register" went because the summary
    # says "install one register and one printer"; 000123 lost the scanner
    # reuse and mounting-hardware tasks the same way).
    ov_items = [x for x in content.get("overview") or [] if isinstance(x, dict)]
    def _restates(x: dict[str, Any]) -> bool:
        t = str(x.get("text"))
        # A counted task is the contract's own line, whatever the summary says
        # (000132 pass U: "Perform weekly on-site visits lasting approximately
        # four (4) to eight (8) hours" went once the summary sentence read clean).
        # A site count is the summary's; a count of the work itself is not,
        # unless the line also spreads it across the sites and a summary
        # sentence gives every number it does (010209 pass U: "Replace two (2)
        # to five (5) switches per store across approximately thirty (30)
        # sites, with the project estimate at ..." as a task).
        site_n = r"(?i)\b\d+\)?\s+(?:\w+\s+)?(locations?|sites?|stores?|branches|restaurants?)\b"
        if re.search(r"\d", re.sub(site_n, "", _digits(t))):
            nums = set(re.findall(r"\d+", _digits(t)))
            spread = r"(?i)\b(per|across|at each|for each)\s+(\w+\s+)?(\d+\)?\s+)?(\w+\s+)?(locations?|sites?|stores?|branches|restaurants?)\b"
            return bool(re.search(spread, _digits(t)) and re.search(site_n, _digits(t))) and any(
                nums <= set(re.findall(r"\d+", _digits(str(o.get("text"))))) for o in ov_items)
        return any(_said(t, [o], 0.75) and _said(re.sub(r"(?i)^\s*provider will\s+", "", str(o.get("text"))), [t], 0.9)
                   for o in ov_items)
    content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")
                        and ov_items and _restates(x))]
    # A generic "install cable runs" line beside the measured runs repeats them
    # (010043: "Install CAT6 cable runs to the camera locations").
    sc_all = content.get("scope") or []
    if any(isinstance(x, dict) and re.search(r"(?i)\b\d{2,4}\s*(ft|feet)\b", str(x.get("text"))) and re.search(r"(?i)\b(run|cable)", str(x.get("text"))) for x in sc_all):
        content["scope"] = [x for x in sc_all if not (isinstance(x, dict) and not x.get("standard")
                            and re.match(r"(?i)\s*(install|run|pull)\s+(the\s+)?(cat ?6a?|cat ?5e?|low[- ]voltage)?\s*cable runs?\b", str(x.get("text")))
                            and not re.search(r"\d", str(x.get("text"))))]
    # Uncited padding about hidden repairs (010115).
    for k in ("assumptions", "out_of_scope"):
        content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and not x.get("cites") and not x.get("standard")
                      and re.search(r"(?i)hidden (site )?(repairs?|infrastructure|conditions)", str(x.get("text"))))]
    # The form's own tools line, when the tools list came back empty (010115:
    # "Use Of Common Tools (Screwdriver, Wrenches)").
    if not content.get("provider_tools"):
        for f in fs:
            m = re.search(r"(?i)use of common tools\s*\(([^)]+)\)", f["text"])
            if m:
                content["provider_tools"] = [{"text": f"Common hand tools, including {m.group(1).strip().lower()}.", "cites": [f["id"]], "kind": "fact"}]
                break
    # Last: a heading left with nothing under it goes (010209 "Switch replacement").
    sc = content.get("scope") or []
    content["scope"] = [x for i, x in enumerate(sc)
                        if not (isinstance(x, dict) and x.get("kind") == "heading")
                        or (i + 1 < len(sc) and not (isinstance(sc[i + 1], dict) and sc[i + 1].get("kind") == "heading"))]


def _ruled_out_verbs(content: dict[str, Any]) -> set[str]:
    """The stems of the work the SOW's own limits rule out ("Provider does not
    procure, receive, stage, configure, package or ship")."""
    verbs: set[str] = set()
    for k in ("assumptions", "out_of_scope"):
        for x in content.get(k) or []:
            t = str(x.get("text") if isinstance(x, dict) else x)
            for mm in re.finditer(r"(?i)\b(?:does not|do not|will not|won'?t|is not responsible for|not)\s+((?:[a-z]+(?:,\s*|\s+or\s+|\s+and\s+|,\s*or\s+|,\s*and\s+)){1,8}[a-z]+)", t):
                for v in re.split(r",\s*(?:or\s+|and\s+)?|\s+or\s+|\s+and\s+", mm.group(1)):
                    v = v.strip().lower()
                    if len(v) >= 4 and v not in ("include", "included", "perform", "provide", "responsible", "the", "any", "work"):
                        verbs.add(re.sub(r"(ing|es|e|s)$", "", v)[:6])
    if verbs & {"receiv", "stag", "procur"}:
        verbs |= {"bench", "kit"}  # warehouse work goes with receiving and staging
    return verbs


#: The noun and verb forms of warehouse work, by the stem a limit line gives.
_PHASE_WORDS = {"receiv": r"receiv\w*|receipt", "ship": r"ship(s|ped|ping|ment|ments)?\b|track shipment", "bench": r"bench\w*",
                "procur": r"procur\w*", "configur": r"configuration (center|partner)|pre-?configur\w*", "packag": r"packag\w*|pack\b",
                "stag": r"staging (center|warehouse|facility)", "kit": r"\bkitting\b"}


def _partner_phase(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """Another party's phase of the project is not this SOW's work: under a
    limit that rules out receiving, benching, configuring or shipping, a line
    whose every cited source is a step of that phase (it names the work, or
    sits in the same source list as one that does) is neither a Provider task
    nor a customer duty, and a standard step whose only trigger is such a step
    goes (010209 pass U: the RFQ's "Validate receipt with store manager" as a
    Provider site-start step, the bench list's "Label all equipment with RL
    standard naming convention" as a customer duty, its "serial numbers of
    benched equipment" firing the serial-capture standard). A Provider task
    that does the ruled-out work itself (pack removed switches for shipment)
    goes too; staging removed gear onsite stays."""
    verbs = _ruled_out_verbs(content)
    pats = [p for v, p in _PHASE_WORDS.items() if v in verbs]
    if not pats:
        return
    phase = re.compile(r"(?i)\b(" + "|".join(pats) + r")")
    onsite = re.compile(r"(?i)\b(onsite|on site|on-site|customer[- ]designated|in place|existing|removed|old|legacy|decommission\w*)\b")
    hit = {f["id"] for f in fs if phase.search(f["text"]) and not onsite.search(f["text"])}
    groups = {f["under"] for f in fs if f["id"] in hit and f.get("under")}
    hit |= {f["id"] for f in fs if f.get("under") and f["under"] in groups}
    for k in ("scope", "customer_responsibilities", "arrival_protocol", "schedule", "customer_materials"):
        content[k] = [x for x in content.get(k) or [] if not (
            isinstance(x, dict) and not x.get("standard") and x.get("kind") != "heading" and x.get("cites")
            and all(c in hit for c in x["cites"])
            # When this job starts is this job's (the RFQ's "If equipment is
            # received earlier, we can start the project earlier").
            and not (k == "schedule" and re.search(r"(?i)\b(start|begin|commence)\w*\b", str(x.get("text")))))]
    # A task doing the ruled-out work itself.
    lead = lambda t: re.sub(r"(ing|es|e|s)$", "", (re.findall(r"[a-z]+", t.lower()) or [""])[0])[:6]
    alias = {"pack": "packag"}
    content["scope"] = [x for x in content.get("scope") or [] if not (
        isinstance(x, dict) and not x.get("standard") and x.get("kind") != "heading"
        and alias.get(lead(str(x.get("text"))), lead(str(x.get("text")))) in verbs - {"stag"}
        and re.search(r"(?i)\bship|freight|return|dispos", str(x.get("text"))))]
    # A standard serial step whose every serial fact is that phase's.
    serial = [f for f in fs if re.search(r"(?i)\bserial", f["text"])]
    if serial and all(f["id"] in hit for f in serial):
        content["scope"] = [x for x in content.get("scope") or [] if not (
            isinstance(x, dict) and x.get("standard") and re.match(r"(?i)\s*(capture|record|provide)\b[^.]{0,30}\bserial", str(x.get("text"))))]


#: A form row: "Label | value" (the value may be empty).
_FORM_ROW = re.compile(r"^\s*([^|\n]{2,140}?)\s*\|\s*(.*)$", re.S)


def _form_chrome(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A request form's empty or N/A field, and the footnote of a field left
    blank, are not terms of this job (010115 pass U: "Hardware to be
    Installed (By Category) | N/A" became "No hardware is to be installed
    ...; the category is N/A", and the blank "Rush?*" field's footnote "*A
    rush fee is added ..." a 20% rush-fee assumption). A line resting only
    on such rows goes from the assumptions, duties and exclusions."""
    by = {f["id"]: f for f in fs}
    def chrome(c: str) -> bool:
        f = by.get(c)
        if not f:
            return False
        mm = _FORM_ROW.match(str(f["text"]))
        if not mm:
            return False
        label, value = mm.group(1).strip(), mm.group(2).strip()
        if re.fullmatch(r"(?i)(n/?a|none|tbd|-+|—)?\.?", value) or re.search(r"(?i)[:|]\s*(n/?a|tbd)\s*\.?$", value):
            return True  # "1875 Lawrence Street, Suite 750: N/A" (010147: N/A read as "No")
        # The field is blank and the cell holds only its own footnote.
        return label.endswith("*") and value.startswith("*")
    for k in ("assumptions", "customer_responsibilities", "out_of_scope"):
        content[k] = [x for x in content.get(k) or [] if not (
            isinstance(x, dict) and not x.get("standard") and x.get("cites") and all(chrome(c) for c in x["cites"]))]


def _contact_roles(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A person is the onsite contact only when a fact names them so: with
    another person named as the on-site contact, a line calling someone else
    the "onsite" contact loses the word (010082 pass U: "Provide AJ Deuel as
    the main onsite technical contact", AJ being the form's remote TECH
    contact and Henry Hankins its On Site Contact Name)."""
    onsite = set()
    for f in fs:
        for mm in re.finditer(r"(?i)on[- ]?site contact(?: name)?\s*[:|]\s*([A-Z][a-z]+(?: [A-Z][a-z'-]+)+)", f["text"]):
            onsite.add(mm.group(1).split()[0].lower())
    if not onsite:
        return
    for k in ("customer_responsibilities", "arrival_protocol", "assumptions", "scope"):
        for x in content.get(k) or []:
            if not isinstance(x, dict) or x.get("standard"):
                continue
            t = str(x.get("text"))
            names = {w.lower() for w in re.findall(r"\b([A-Z][a-z]+|[A-Z]{2})\b(?= [A-Z][a-z]+)", t)}
            if names and not names & onsite and re.search(r"(?i)\bon[- ]?site\s+(\w+\s+)?contact", t):
                x["text"] = re.sub(r"(?i)\bon[- ]?site\s+(?=(\w+\s+)?contact)", "", t)


def _boundary_verbs(content: dict[str, Any]) -> None:
    """The summary and the task list never claim work the SOW's own limits
    rule out: the verbs of an assumption or exclusion that says Provider
    "does not procure, receive, stage, configure, package or ship" leave a
    "Provider will receive, bench, configure, ship, install ..." line (010209
    pass T: the overview contradicted the physical-only boundary)."""
    verbs = _ruled_out_verbs(content)
    if not verbs:
        return
    bad = re.compile(r"(?i)^(%s)" % "|".join(sorted(re.escape(v) for v in verbs)))
    for k in ("overview", "scope"):
        for x in content.get(k) or []:
            if isinstance(x, dict) and not x.get("standard") and x.get("kind") != "heading":
                x["text"] = _strip_verbs(str(x.get("text")), bad)


def _strip_verbs(text: str, bad: re.Pattern) -> str:
    """"Provider will stage, tag, install and test the equipment" without the
    verbs ``bad`` matches; the text unchanged when it is not such a list."""
    m = re.match(r"(?i)^(\s*provider will\s+)(.+?)(\s+(?:the|all|each|every|new|existing)\b.*)$", text)
    if not m:
        return text
    verbs = [v.strip() for v in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(2)) if v.strip()]
    if len(verbs) < 2 or any(len(v.split()) > 2 for v in verbs):
        return text
    keep = [v for v in verbs if not bad.match(v)]
    if not keep or len(keep) == len(verbs):
        return text
    return m.group(1) + _join(keep) + m.group(3)


def _join(items: list[str]) -> str:
    """"a", "a and b", "a, b, and c"."""
    return items[0] if len(items) == 1 else (" and ".join(items) if len(items) == 2 else ", ".join(items[:-1]) + ", and " + items[-1])


def _speaker_in_overview(content: dict[str, Any], by: dict[str, dict[str, Any]]) -> None:
    """A summary clause "Provider will <X>" whose cited line is the customer
    saying "we will <X>" is the customer's, when the line sits under a customer
    heading or the SOW's own exclusions rule <X> out (000123: "We will
    preconfigure all items before shipping them", under "Customer notes:",
    became "Provider will preconfigure the equipment", beside the exclusion
    "Preconfiguration, imaging, staging ..."). The clause moves to the
    Customer; the rest stays Provider's."""
    oos_stems = {w[:8] for x in content.get("out_of_scope") or []
                 for w in re.findall(r"[a-z]{8,}", str(x.get("text") if isinstance(x, dict) else x).lower())}
    for x in content.get("overview") or []:
        if not isinstance(x, dict):
            continue
        m = re.match(r"(?i)^(\s*provider will\s+)(.+)$", str(x.get("text")))
        if not m:
            continue
        cited = [by[c] for c in x.get("cites") or [] if c in by]
        said = []  # stems of what a customer "we will" line commits the customer to
        for f in cited:
            for cl in re.findall(r"(?i)\bwe\s*(?:will|'ll|are going to|plan to)\s+([^.;]{3,120})", f["text"]):
                said += [(w[:8], f) for w in re.findall(r"[a-z]{6,}", cl.lower())]
        if not said:
            continue
        phrases = [p_ for p_ in re.split(r",\s*(?:then\s+|and\s+)*|\s+and then\s+", m.group(2)) if p_.strip()]
        moved, kept = [], []
        for p_ in phrases:
            first = (re.findall(r"[a-z]+", p_.lower()) or [""])[0]
            hit = len(first) >= 6 and next((f for st, f in said if st == first[:8]), None)
            if hit and (re.search(r"(?i)customer", str(hit.get("under") or "")) or first[:8] in oos_stems):
                moved.append(p_.strip().rstrip("."))
            else:
                kept.append(p_.strip())
        if not moved or not kept:
            continue
        x["text"] = "Customer will " + _join(moved) + "; Provider will " + _join([k.rstrip(".") for k in kept]) + "."


_MIXED_HOURS = re.compile(r"(?i)\b(mix|combination|blend) of (normal |regular )?business hours and after[- ]?hours|"
                          r"both (during )?business hours and after[- ]?hours|business hours (and|with) some after[- ]?hours")
_CLOCK = r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?"


def _hours_basis(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """The deal's own hours replace the template's business-hours-only basis.
    "A mix of business hours and after hours" becomes the work window and the
    assumption (010085 kept "All Services ... during normal Business Hours"),
    and a stated daytime window other than 8-5 redefines Business Hours
    (010279: arrival 6 AM, departure 5 PM, so every visit's first two hours
    would bill as after-hours)."""
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    sched = content.setdefault("schedule", [])
    mix = next((f for f in fs if _MIXED_HOURS.search(f["text"]) and not f["text"].rstrip().endswith("?")), None)
    if mix:
        if not any(tx(x).lower().startswith("work window") for x in sched):
            sched.append({"text": "Work Window: A mix of Business Hours and Customer-approved after-hours windows",
                          "cites": [mix["id"]], "kind": "fact"})
        content["assumptions"] = [x for x in content.get("assumptions") or []
                                  if not re.search(r"(?i)(normal|regular|standard) business hours|during business hours", tx(x))]
        # First, where the template's business-hours basis stands (010085: the
        # corrective line came last, after the line it corrects).
        content["assumptions"].insert(0, {"text": "Work is performed during a mix of Business Hours and Customer-approved after-hours "
                                                  "windows, not during Business Hours only.", "cites": [mix["id"]], "kind": "fact"})
        content["hours_basis"] = "mixed"
        cr = content.setdefault("customer_responsibilities", [])
        line = "Confirm which areas may be worked during Business Hours and which require after-hours windows."
        if not _said(line, cr, 0.6):
            cr.append({"text": line, "cites": [mix["id"]], "kind": "fact"})
    win = next((x for x in sched if re.match(r"(?i)\s*work window", tx(x))), None)
    # An after-hours-only window replaces the business-hours basis (010083:
    # "After hours, no weekends, no overnights" under the template's "All
    # Services will be performed during normal Business Hours").
    if not mix and win and re.search(r"(?i)after[- ]?hours", tx(win)) and not re.search(r"(?i)business hours|mix|or after|daytime|weekday days", tx(win)):
        content["assumptions"] = [x for x in content.get("assumptions") or []
                                  if not re.search(r"(?i)(normal|regular|standard) business hours|during business hours", tx(x))]
        line = "All Services will be performed After Hours" + (" during the week" if re.search(r"(?i)no weekends?", tx(win)) else "") + "."
        if not any(re.search(r"(?i)performed after hours", tx(x)) for x in content["assumptions"]):
            content["assumptions"].insert(0, {"text": line, "cites": list(win.get("cites") or []), "kind": "fact"})
        content["hours_basis"] = "after_hours"
        content["customer_responsibilities"] = [x for x in content.get("customer_responsibilities") or []
                                                if not re.search(r"(?i)during business hours and which require after", tx(x))]
    m = win and re.search(_CLOCK + r"\s*(?:-|–|to|until)\s*" + _CLOCK, tx(win), re.I)
    if m:
        h1 = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        h2 = int(m.group(4)) % 12 + (12 if m.group(6).lower() == "p" else 0)
        if 5 <= h1 <= 10 and 15 <= h2 <= 20 and (h1, h2) != (8, 17):
            fmt = lambda h, mi: f"{(h - 1) % 12 + 1}:{mi or '00'} {'AM' if h < 12 else 'PM'}"
            a, b = fmt(h1, m.group(2)), fmt(h2, m.group(5))
            days = re.search(r"(?i)\b(monday|mon)\s*(?:-|–|through|to)\s*(thursday|thu|friday|fri|saturday|sat)\b", tx(win))
            dtxt = f", {days.group(0)}" if days else ""
            line = f"Business Hours for this SOW are {a} to {b} local time{dtxt}; After Hours are {b} to {a}."
            asm = content.setdefault("assumptions", [])
            if not any(re.search(r"(?i)business hours for this sow", tx(x)) for x in asm):
                asm.insert(0, {"text": line, "cites": list(win.get("cites") or []) if isinstance(win, dict) else [], "kind": "fact"})


_NEGATED = re.compile(r"(?i)\b(is|are|will be|remains?)\s+(excluded|not included|out of scope|outside (of )?(the )?(service |work )?scope|not part of)\b|"
                      r"\bexcluded from\b|\bnot in scope\b|\bout of scope\b")


def _as_scope(text: str) -> str:
    """An exclusion's wording turned into a task ("End-user support is
    excluded from the service scope." -> "Provide end-user support."), or ""
    when it cannot be turned cleanly. A scope bullet never says excluded."""
    t = re.sub(r"(?i)^\s*(no |excludes? |exclusion of |excluded:\s*)", "", text).strip()
    t = re.sub(r"(?i),?\s*(?:is|are|will be|remains?)\s+(?:excluded|not included|out of scope|outside (?:of )?(?:the )?(?:service |work )?scope|not part of)\b[^.;]*", "", t)
    t = t.strip(" ,.;")
    if not t or _NEGATED.search(t):
        return ""
    t = ("Provide " + t[0].lower() + t[1:]) if not re.match(r"(?i)\s*(provide|support|perform|assist|install|mount|run|replace|remove|configure)\b", t) else t[0].upper() + t[1:]
    return t + "."


def _scope_polarity(content: dict[str, Any], fs: list[dict[str, Any]] | None = None) -> None:
    """A task line never says the work is excluded: whatever rule moved it,
    the wording turns with the section, or the line goes (000132). Only a
    line a cited source names as work turns into a task; an uncited
    exclusion, or one citing lines that do not name the work, stays an
    exclusion (010419 pass T: library exclusions citing an email subject
    printed "Provide managed IT services", "Provide lift rental" under
    Closeout)."""
    text_of = {f["id"]: f["text"] for f in fs or []}
    kept = []
    for x in content.get("scope") or []:
        if isinstance(x, dict) and x.get("kind") != "heading" and _NEGATED.search(str(x.get("text"))):
            t = _as_scope(str(x.get("text")))
            cited = [text_of[c] for c in x.get("cites") or [] if c in text_of]
            named = bool(t) and bool(x.get("cites")) and (fs is None or _said(re.sub(r"(?i)^provide\s+", "", t), cited, 0.6))
            if not named:
                ex = re.sub(r"(?i),?\s*(?:is|are|will be|remains?)\s+(?:excluded|not included|out of scope|outside (?:of )?(?:the )?(?:service |work )?scope|not part of)\b[^.;]*", "",
                            str(x.get("text"))).strip(" ,.;")
                ex = re.sub(r"(?i)^\s*(provide|perform)\s+", "", ex)
                oos = content.setdefault("out_of_scope", [])
                if ex and not _NEGATED.search(ex) and not _said(ex, oos, 0.6):
                    oos.append({"text": ex[:1].upper() + ex[1:] + ".", "cites": [], "kind": "method"})
                continue
            x["text"] = t
        kept.append(x)
    content["scope"] = kept


def _known_exclusions() -> list[str]:
    """Every exclusion PurTera writes: the method library's and the
    standards', all families, their gates stripped."""
    out = [str(it.get("text") or "") for fam in _library().values() if isinstance(fam, dict)
           for it in fam.get("exclusions") or [] if isinstance(it, dict)]
    try:
        data = json.loads((Path(__file__).parent / "purtera_standards.json").read_text())
    except Exception:
        data = {}
    for fam in data.values():
        if isinstance(fam, dict):
            out += [re.sub(r"^(\{(when|unless):.*?\}\s+)+(?=[A-Z])", "", str(t)) for t in fam.get("exclusions") or []]
    return [t for t in out if t]


def _exclusions_never_tasks(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A Provider task that is one of PurTera's exclusions word for word
    ("Provide managed IT services or ongoing help desk coverage.") is that
    exclusion, unless a line it cites names the work: it goes back to Out
    of Scope (010419 pass T: five exclusions became Closeout tasks)."""
    text_of = {f["id"]: f["text"] for f in fs}
    known = _known_exclusions()
    oos = content.setdefault("out_of_scope", [])
    # What the writer itself wrote as an exclusion or a negated assumption,
    # by its cites (c3_sow.writer_draft): it never comes back as a task
    # unless a line it cites names the work (010085 pass T: "No rack
    # modification" became "Provide rack modification or cabinet
    # modification." in the task list).
    draft = content.get("writer_draft") if isinstance(content.get("writer_draft"), dict) else {}
    was_excl = {tuple(x.get("cites") or []): str(x.get("text") or "") for k in ("out_of_scope", "assumptions", "customer_responsibilities")
                for x in draft.get(k) or [] if isinstance(x, dict) and x.get("cites")
                and (k == "out_of_scope" or _LINE_NEG.search(str(x.get("text") or "")) or _NEGATED.search(str(x.get("text") or ""))
                     or re.match(r"(?i)\s*(no|not|never|without)\b", str(x.get("text") or "")))}
    kept = []
    for x in content.get("scope") or []:
        t = str(x.get("text") if isinstance(x, dict) else x)
        if not isinstance(x, dict) or x.get("standard") or x.get("kind") == "heading":
            kept.append(x)
            continue
        mm = re.match(r"(?i)\s*(provide|perform|handle|supply)\s+(.{6,})$", t)
        rest = (mm.group(2) if mm else t).strip().rstrip(".")
        cited = [text_of[c] for c in x.get("cites") or [] if c in text_of]
        orig = was_excl.get(tuple(x.get("cites") or []))
        library = bool(mm) and any(_said(rest, [e], 0.8) and _said(e, [rest], 0.8) for e in known)
        if (library or orig is not None) and not _said(rest, cited, 0.6):
            line = orig if orig is not None and not library else rest[:1].upper() + rest[1:] + "."
            if not _said(line, oos, 0.6):
                oos.append({**x, "text": line, "kind": "method"})
            continue
        kept.append(x)
    content["scope"] = kept


#: A line that excludes work: "excluded", "not included", "will not
#: perform", "no <service>". A bound or a site condition ("does not exceed
#: 200 feet", "No concealed utilities are present") is not one.
_LINE_NEG = re.compile(r"(?i)\b(excluded|exclusion|not (be )?(included|provided|performed|supported|covered|part of|in scope)|"
                       r"out of scope|outside (of )?(the |provider'?s? )?(\w+ ){0,2}scope|"
                       r"(will not|won'?t|does not|do not|is not to)( be)? (provide|perform|support|include|cover|install|handle|do)\w*)\b|"
                       r"^\s*no\b[^.;]{0,40}\b(support|services?|maintenance|troubleshoot\w*|patch\w*|install\w*|administration|work)\b"
                       r"(?![^.;]{0,30}\b(is|are)\s+(present|expected|anticipated|required|needed)\b)")
#: A source line that carries a negation, a limit, or hands the work to
#: another party: an exclusion or a "Customer will" line may lean on it.
_SOURCE_NEG = re.compile(r"(?i)\b(not|no|none|never|neither|nor|without|exclud\w*|except\w*|outside|only|limited|solely|"
                         r"won'?t|dont|wont|cant|cannot|doesnt|isnt|arent|n/a|optional|separate(ly)?|instead)\b|n't\b|"
                         r"\b(customer|client|they|their|we|i|in-?house|internal(ly)?|reseller|vendor|others?)\b.{0,40}\b(will|to|handles?|provides?|does|owns?|manages?)\b|"
                         r"\b(by|from) (the )?(customer|client|others|their|reseller|vendor)|handled by|responsib")
#: A source that names work: a service, a task.
_SERVICE_WORDS = re.compile(r"(?i)\b(support|maintenance|maintain|troubleshoot\w*|install\w*|patch\w*|upgrades?|repair\w*|"
                            r"configur\w*|monitor\w*|management|coordinat\w*|response|services?|deploy\w*|replace\w*|"
                            r"migrat\w*|administration|assistance|help ?desk|desk-?side)\b")
_SERVICE_TYPES = frozenset({"scope_item", "task", "deliverable"})


def _service_heading(text: str) -> str:
    """A heading for a service line: its object up to the first comma,
    "including" or "for" ("Provide end-user support, including O365" ->
    "End User Support")."""
    t = re.sub(r"(?i)^\s*(provide|perform|support for|support)\s+", "", text).strip()
    t = re.split(r"(?i),|\bincluding\b|\bfor\b|\bat\b|\bduring\b|\(", t)[0].strip(" .;")
    words = [w for w in re.split(r"[\s-]+", t) if w][:4]
    return " ".join(w[:1].upper() + w[1:] for w in words) if words else ""


def _source_polarity(content: dict[str, Any], fs: list[dict[str, Any]], envelope: dict[str, Any]) -> None:
    """A line that states an exclusion or a negation ("excluded", "not
    included", "no X", "will not") when none of its cited lines carries a
    negation, a limit or another party's duty says the opposite of its
    source (000132: the requested "End-user support including O365/M365
    clients" written as "End-user support ... is excluded." under
    assumptions). A source that names work (the parser or a head read it as
    a scope item, or the heads put it in the scope) turns into a task in the
    source's polarity, under its own service heading; any other source makes
    the line go. Standard and uncited lines are the template's."""
    by = {f["id"]: f for f in fs}
    atoms = {str(a.get("id")): a for a in envelope.get("atoms") or []}
    def src(c: str) -> str:
        return str((by.get(c) or {}).get("text") or (atoms.get(c) or {}).get("text") or "")
    def reads_as_work(c: str) -> bool:
        a = atoms.get(c) or {}
        st = a.get("structured") or {}
        types = {str(a.get("atom_type") or ""), str((by.get(c) or {}).get("type") or ""),
                 str((st.get("c3_label") or {}).get("was") or "")} | {str(t) for t in st.get("alt_atom_types") or []}
        return bool(types & _SERVICE_TYPES or (by.get(c) or {}).get("section") == "scope_of_work") and bool(_SERVICE_WORDS.search(src(c)))
    sc = content.setdefault("scope", [])
    for k in ("assumptions", "customer_responsibilities", "out_of_scope", "overview"):
        for x in list(content.get(k) or []):
            if not isinstance(x, dict) or x.get("standard") or not x.get("cites") or x.get("kind") == "heading":
                continue
            t = str(x.get("text") or "")
            cites = [str(c) for c in x["cites"]]
            if not _LINE_NEG.search(t) or not all(src(c) for c in cites) or any(_SOURCE_NEG.search(src(c)) for c in cites):
                continue
            content[k].remove(x)
            work = [c for c in cites if reads_as_work(c)]
            if not work or k == "overview":
                continue  # a summary sentence goes; the task list says the work
            line = _as_scope(t)
            if not line or not _said(re.sub(r"(?i)^provide\s+", "", line), [src(c) for c in work], 0.6):
                body = src(work[0]).strip().rstrip(".")
                line = body[:1].upper() + body[1:] + "." if re.match(r"(?i)(provide|perform|support|assist|install|coordinate)\b", body) \
                    else "Provide " + body[:1].lower() + body[1:] + "."
            if _said(line, [y for y in sc if isinstance(y, dict) and y.get("kind") != "heading"], 0.8):
                continue
            new = {"text": line, "cites": cites, "kind": "fact"}
            # Under the heading that names its service, else its own heading
            # ahead of the closing group.
            mine = _toks(_service_heading(line)) - {"support", "provid"}
            at = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                       and mine and mine <= _toks(y.get("text"))), None)
            if at is not None:
                end = next((j for j in range(at + 1, len(sc)) if isinstance(sc[j], dict) and sc[j].get("kind") == "heading"), len(sc))
                sc.insert(end, new)
                continue
            close = next((i for i, y in enumerate(sc) if isinstance(y, dict) and y.get("kind") == "heading"
                          and re.search(r"(?i)clos|clean|wrap|complet", str(y.get("text")))), len(sc))
            head = _service_heading(line)
            group = ([{"text": head, "cites": [], "kind": "heading"}] if head and any(
                isinstance(y, dict) and y.get("kind") == "heading" for y in sc) else []) + [new]
            sc[close:close] = group


#: IT support service areas and the words that name each, so a standard
#: step finds the heading of its service when the two word it differently
#: ("virtual servers, host visibility" under "Virtualization Support",
#: "switches, wireless access points" under "Network Support").
_SERVICE_AREAS = {
    "network": "network switch wireless lan wlan access uplink port cabling cable wap",
    "server": "server nas rack",
    "virtualization": "virtualization virtual vmware vsphere hypervisor host vm",
    "patching": "patching patch-management upgrade update",
    "end user": "end-user end user workstation peripheral printer desk-side deskside desktop laptop o365 m365",
}


def _area_hits(text: str, area: str) -> int:
    low = text.lower()
    words = _SERVICE_AREAS[area].split()
    if area == "end user":
        return sum(1 for w in words if w not in ("end", "user") and re.search(r"\b" + re.escape(w) + r"s?\b", low)) \
            + (1 if re.search(r"\bend[- ]users?\b", low) else 0)
    return sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"(s|es|ing)?\b", low))


def _heading_area(text: str) -> str | None:
    hits = {a: _area_hits(text, a) for a in _SERVICE_AREAS}
    best = max(hits, key=hits.get)
    return best if hits[best] else None


def _service_groups(content: dict[str, Any]) -> None:
    """Recurring support (staff augmentation): each standard step sits under
    the service heading it supports, the rest under the general visit
    heading; a writer line a standard step in its group already says goes;
    "remote direction" from the customer's IT team is coordination with
    its designated IT contact (000132: fifteen standard steps under "Onsite
    Support", one under "Network Support", the visit-validation step under
    "Windows and Linux Patching", patch validation written twice)."""
    if not ({"support", "staff_augmentation"} & set(content.get("services") or [])):
        return
    sc = content.get("scope") or []
    heads = [i for i, x in enumerate(sc) if isinstance(x, dict) and x.get("kind") == "heading"]
    if len(heads) < 3:
        return
    closing = re.compile(r"(?i)clos|clean|wrap|complet")
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    lead: list[dict[str, Any]] = []
    for x in sc:
        if isinstance(x, dict) and x.get("kind") == "heading":
            groups.append((x, []))
        elif groups:
            groups[-1][1].append(x)
        else:
            lead.append(x)
    work = [g for g in groups if not closing.search(str(g[0].get("text")))]
    if len(work) < 2:
        return
    # The general group: the one the standard steps were written into.
    general = max(work, key=lambda g: sum(1 for y in g[1] if isinstance(y, dict) and y.get("standard")))
    if not any(isinstance(y, dict) and y.get("standard") for y in general[1]):
        return
    remote = re.compile(r"(?i)^\s*(support|take|follow|accept|work under|receive)\s+(remote\s+)?(direction|guidance|instructions?)\s+from\b[^.]*\bremote\b|"
                        r"^\s*(support|take|follow|accept|work under|receive)\s+remote\s+(direction|guidance|instructions?)\b")
    reworded: list[dict[str, Any]] = []
    for g in work:
        for y in list(g[1]):
            if isinstance(y, dict) and not y.get("standard") and remote.search(str(y.get("text"))):
                y["text"] = "Coordinate with Customer's designated remote IT contact during each visit."
                reworded.append(y)
                if g is not general:
                    g[1].remove(y)
                    general[1].append(y)
    # Each service heading's area; a standard step goes to the one area its
    # words name most, else to the general group.
    services = [(g, _heading_area(str(g[0].get("text")))) for g in work if g is not general]
    services = [(g, a) for g, a in services if a]
    for g in work:
        for y in list(g[1]):
            if not (isinstance(y, dict) and y.get("standard")):
                continue
            score = sorted(((_area_hits(str(y.get("text")), a), n) for n, (_, a) in enumerate(services)), reverse=True)
            best = services[score[0][1]][0] if score and score[0][0] >= 1 and (len(score) == 1 or score[0][0] > score[1][0]) else general
            if best is not g:
                g[1].remove(y)
                best[1].append(y)
    # Said once per group: a writer line a standard step in its group says.
    for g in work:
        std = [y for y in g[1] if isinstance(y, dict) and y.get("standard")]
        g[1][:] = [y for y in g[1] if not (isinstance(y, dict) and not y.get("standard") and not y.get("cites")
                                           and not any(y is r for r in reworded) and any(_said(str(y.get("text")), [s], 0.75) for s in std))]
    out = list(lead)
    for h, items in groups:
        if items or closing.search(str(h.get("text"))):
            out += [h] + items
    content["scope"] = out


def _schedule_voice(c: dict[str, Any], as_of: str, theirs: set[str] | frozenset[str] = frozenset()) -> None:
    """Schedule lines in SOW voice, not the customer's: "we"/"our" is the
    Customer, a date carries its month name and year (the next one after the
    cutoff when the month has passed), and a clock range reads "7:00 AM to
    10:00 AM local site time" (010209 pass U: "All work 7-10am Monday,
    Tuesday, Thursday" and "scheduled once we are out of holiday blackout
    starting mid-January", on a rollout across time zones)."""
    # Work hours are the site's hours, never a call or meeting slot (010082
    # pass U: "Work Hours: tomorrow (Wednesday 7/8) 2-3pm EST"; 010264: a
    # meeting invite's "When:" line).
    slot = re.compile(r"(?i)\b(tomorrow|today|tonight|yesterday|subject:|when:|meeting|call|intro|session|webinar|demo)\b|"
                      r"\?|#\s*of\b|\b(will|is|are|does|do|can)\s+(work|it|the work)\s+be\b")  # a questionnaire's question (010147)
    c["schedule"] = [x for x in c.get("schedule") or [] if not (isinstance(x, dict)
                     and re.match(r"(?i)\s*work (hours|window)\s*:", str(x.get("text"))) and slot.search(str(x.get("text"))))]
    try:
        y0, m0 = int(as_of[:4]), int(as_of[5:7])
    except ValueError:
        return
    year = lambda mo: y0 + 1 if mo < m0 - 2 else y0

    def md(m: re.Match) -> str:
        mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        yy = (2000 + int(y) if len(y) == 2 else int(y)) if y else year(mo)
        return f"{_MONTHS[mo - 1].capitalize()} {d}, {yy}"

    def clock(m: re.Match) -> str:
        h1, m1, a1, h2, m2, a2 = m.groups()
        a2 = a2.upper().replace(".", "")
        a1 = (a1 or ("AM" if a2 == "PM" and int(h1) > int(h2) else a2)).upper().replace(".", "")
        return f"{int(h1)}:{m1 or '00'} {a1} to {int(h2)}:{m2 or '00'} {a2}"

    for x in c.get("schedule") or []:
        if not isinstance(x, dict) or not isinstance(x.get("text"), str):
            continue
        t = x["text"]
        if x.get("cites") and all(cc in theirs for cc in x["cites"]):
            # "we" in a customer's or reseller's line is the Customer; in
            # PurTera's own it stays for the PM to read.
            t = re.sub(r"\b[Ww]e are\b|\b[Ww]e're\b", "the Customer is", t)
            t = re.sub(r"\b[Ww]e\b", "the Customer", t)
            t = re.sub(r"\b[Oo]ur\b", "the Customer's", t)
            t = re.sub(r"^([\w /-]{2,30}:\s+)the Customer", r"\1The Customer", t)
        t = re.sub(r"(?i)^(work (hours|window)\s*:\s*)all work\s+", r"\1", t)
        t = re.sub(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\d/])", md, t)
        months = "|".join(mo.capitalize() for mo in _MONTHS if mo != "may")  # "May" also opens a sentence
        t = re.sub(r"\b((?:[Mm]id-|[Ee]arly |[Ll]ate )?)(%s)\b(?!\s+\d)(?!,?\s+(19|20)\d\d)" % months,
                   lambda m: f"{m.group(1)}{m.group(2)} {year(_MONTHS.index(m.group(2).lower()) + 1)}", t)
        mon = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
        full = lambda a: next(mo.capitalize() for mo in _MONTHS if mo.startswith(a.lower()[:3]))
        t = re.sub(r"\b%s\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s*((?:19|20)\d\d)\b)?(?!:)" % mon,
                   lambda m: f"{full(m.group(1))} {m.group(2)}, {m.group(3) or year(_MONTHS.index(full(m.group(1)).lower()) + 1)}", t)
        t2 = re.sub(r"(?i)\b(\d{1,2})(?::(\d\d))?\s*(am|pm|a\.m\.|p\.m\.)?\s*(?:-|–|to)\s*(\d{1,2})(?::(\d\d))?\s*(am|pm|a\.m\.|p\.m\.)", clock, t)
        if t2 != t and not re.search(r"(?i)\blocal\b|\b(e|c|m|p)[sd]?t\b|eastern|central|mountain|pacific", t2):
            t2 = re.sub(r"(\d{1,2}:\d\d [AP]M to \d{1,2}:\d\d [AP]M)(\s+(?=[A-Za-z]))?",
                        lambda m: m.group(1) + " local site time" + (", " if m.group(2) else ""), t2, count=1)
        t = t2
        # A customer's sentence is not a slot value: "Work Hours: This can be
        # during office hours Monday and Friday (8am-5PM) or outside of
        # business hours" reads "Work Hours: Monday through Friday, 8:00 AM to
        # 5:00 PM local site time" (010171 pass U); an after-hours option is a
        # change order under the template's overtime line.
        hm = re.match(r"(?i)\s*work hours\s*:\s*(.*)$", t)
        if hm and re.search(r"(?i)\b(this|we|it|can|could|either|or outside|would|should)\b", hm.group(1)):
            clk = re.search(r"\d{1,2}:\d\d [AP]M to \d{1,2}:\d\d [AP]M", hm.group(1))
            body = hm.group(1).lower()
            if clk:
                days = "Monday through Friday" if (re.search(r"\bmon", body) and re.search(r"\bfri", body)) or \
                    re.search(r"office hours|business hours", body) else ""
                t = "Work Hours: " + (days + ", " if days else "") + clk.group(0) + " local site time"
        x["text"] = t[:1].upper() + t[1:]


def _decisions_match_output(c: dict[str, Any]) -> None:
    """The decision logs say what the SOW shows (010217 pass U: a guard
    "keep" whose line was absent, an 86-inch line logged dropped but present,
    dropped_unsupported lines that appear). The last sow_keep decision on each
    fact is checked against the final lists: a keep whose fact no line cites
    and whose text is nowhere becomes a drop, a drop whose exact text is shown
    becomes a keep; both say so (superseded). A dropped_unsupported line the
    SOW shows is not dropped."""
    lines = [x for k in LISTS for x in c.get(k) or [] if isinstance(x, dict) and x.get("kind") != "heading"]
    shown = {str(x.get("text")).strip() for x in lines}
    cited = {a for x in lines for a in x.get("cites") or []}
    last: dict[tuple, dict[str, Any]] = {}
    for d in c.get("rule_decisions") or []:
        if d.get("head") == "sow_keep":
            last[tuple(d.get("atom_ids") or [])] = d
    for ids, d in last.items():
        text = str(d.get("text") or "").strip()
        if d.get("value") == "keep" and text not in shown and not (set(ids) & cited):
            d.update(value="drop", superseded="keep: absent from the final SOW")
        elif d.get("value") == "drop" and text and text in shown:
            d.update(value="keep", superseded="drop: the same line is in the final SOW")
    c["dropped_unsupported"] = [x for x in c.get("dropped_unsupported") or []
                                if not (isinstance(x, dict) and str(x.get("text") or "").strip() in shown)]


def _materials_counted(c: dict[str, Any]) -> None:
    """One line per item, with its count: an uncounted line ("The 32-65 inch
    articulating TV mount.") goes beside the counted one ("Three (3) new 32-65
    inch articulating TV mounts."), and a bare "(1) new ..." row reads "One (1)
    new ..." (010217 pass U: a singular articulating mount and the 55-100 inch
    mount listed twice)."""
    count = re.compile(r"(?i)^\s*(?:\(\d+\)|\d+\b|%s(?:[\s-]+%s)*\s*\(\d+\))" % (_NUMW, _NUMW))
    toks = lambda t: {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", str(t).lower())} - {"the", "new", "a", "an", "provide", "and"} - {
        w for w in _ONES + _TENS[2:] + ["hundred"]}
    for k in ("customer_materials", "provider_materials", "equipment_in_scope"):
        items = [x for x in c.get(k) or [] if isinstance(x, dict)]
        for x in items:
            mm = re.match(r"^\s*\((\d+)\)\s+", str(x.get("text")))
            if mm:
                n = int(mm.group(1))
                body = str(x["text"])[mm.end():].strip().rstrip(".")
                x["text"] = f"{_words(n).capitalize()} ({n}) {body}."
        # A merged spreadsheet row is its rows ("2 x 2 | QM65C: TS525TU |
        # Samsung 65IN ...: Chief Thinkstall ..." is two lines), and a
        # subscription term on a SKU ("w/Navigator -36mo") is not the item
        # (010116 pass U).
        for x in c.get("scope") or []:
            if isinstance(x, dict) and isinstance(x.get("text"), str):
                x["text"] = re.sub(r"\s+-\d{1,3}\s*mo\b", "", x["text"])
        parts: dict[int, list[dict[str, Any]]] = {}
        for x in items:
            t = re.sub(r"\s+-\d{1,3}\s*mo\b", "", str(x.get("text")))
            x["text"] = t
            mm = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*\|\s*([^:|]+):\s*([^|]+?)\s*\|\s*([^:|]+):\s*(.+?)\s*$", t)
            if mm:
                parts[id(x)] = [{**x, "text": f"{mm.group(1)} x {mm.group(3).strip()} {mm.group(5).strip()}"},
                                {**x, "text": f"{mm.group(2)} x {mm.group(4).strip()} {mm.group(6).strip()}"}]
        if parts:
            c[k] = [y for x in c.get(k) or [] for y in parts.get(id(x), [x])]
            items = [x for x in c[k] if isinstance(x, dict)]
        counted = [x for x in items if count.match(str(x.get("text")))]
        keep = []
        for x in c.get(k) or []:
            if isinstance(x, dict) and x in items and not count.match(str(x.get("text"))):
                tx_ = toks(x.get("text")) - {str(d) for d in range(10)}
                if tx_ and any(tx_ <= toks(y.get("text")) for y in counted):
                    continue
            if isinstance(x, dict) and any(y is not x and str(y.get("text")).lower() == str(x.get("text")).lower() for y in keep):
                continue
            keep.append(x)
        c[k] = keep


def _locations_cover_sites(c: dict[str, Any]) -> None:
    """The locations list names every site the site table holds when both are
    "City, ST" lists (000132 pass U: the writer listed five cities, the sites
    six; Delphos, OH was only in the table)."""
    locs = [x for x in c.get("locations") or [] if isinstance(x, dict)]
    city_st = re.compile(r"^\s*[A-Za-z][A-Za-z .'-]{1,30},\s*[A-Z]{2}\.?\s*$")
    if not locs or not all(city_st.match(str(x.get("text"))) for x in locs):
        return
    have = " ".join(str(x.get("text")) for x in locs).lower()
    for s_ in c.get("sites") or []:
        ci, st = str(s_.get("city") or "").strip(), str(s_.get("state") or "").strip()
        if ci and re.fullmatch(r"[A-Z]{2}", st) and f"{ci}, {st}".lower() not in have:
            locs.append({"text": f"{ci}, {st}.", "cites": [], "kind": "method", "from": "sites"})
    c["locations"] = locs


#: Round-the-clock coverage however the writer puts it ("24/7", "24x7",
#: "twenty-four (24) by seven (7)", "24 hours a day").
_ROUND_CLOCK = (r"24\s*/\s*7|24\s*x\s*7|24\s+by\s+7|twenty-four\s*(?:\(24\)\s*)?(?:by|x|/)\s*seven(?:\s*\(7\))?"
                r"|24 hours a day|around the clock|round-the-clock")


#: Words that make an ask a hedge, not a commitment ("can be tough", "depending on volume").
_HEDGE = re.compile(r"(?i)tough|depend|might|may |not sure|tbd|hard to|difficult|\?")


#: A sentence that ends on a word that needs an object ("remain available for.").
_DANGLING = re.compile(r"(?i)(?<![-/\w])(for|to|with|of|by|at|on|in|from|into|onto|via|than|including|and|or|the|a|an|such as)\s*[.:;]?\s*$")


def _clip_dangling(t: str) -> str | None:
    """A line that lost its object ends on a preposition: its last clause goes
    whole ("... as needed and remain available for." -> "... as needed.");
    with no clause left, the line goes (None)."""
    t = str(t).strip()
    if not _DANGLING.search(t) or re.search(r"(?i)\b(check|log|sign|plug|power|turn|switch|hand|fill|lock|clock|tie|dial|opt|carry|"
                                            r"follow|come|move|drop|pick|swap|back|set|hook|roll|bring|wire|stay|stays|kept|left)"
                                            r"(s|ed|ing|ged)?\s+(in|on|at|by|to)\s*[.:;]?\s*$", t):
        return t
    body = t.rstrip(" .:;")
    cut = max((m.start() for m in re.finditer(r",\s+(?:and\s+|or\s+)?|\s+(?:and|or|then)\s+|;\s*", body)), default=-1)
    if cut <= 0:
        return None
    head = body[:cut].rstrip(" ,;")
    if len(head.split()) < 4 or _DANGLING.search(head):
        return None
    return head + "."


def _drop_clause(t: str, rx: re.Pattern[str]) -> str:
    """Remove the whole clause a match sits in, never just the matched words
    (000132 pass U: stripping "24/7 on-call support" left "and remain
    available for."). The sentence's first clause going takes the line."""
    m = rx.search(t)
    if not m:
        return t
    seps = list(re.finditer(r",\s+(?:and\s+|or\s+)?|\s+(?:and|or|then)\s+(?=[a-z])|;\s*", t))
    left = max((x.start() for x in seps if x.end() <= m.start()), default=None)
    right = min((x.start() for x in seps if x.start() >= m.end()), default=None)
    if left is None:
        return ""
    # A clause the match leads ("and exclude the virtual, fax, ATA ...
    # entries") runs to the sentence's end, list commas and all (010326).
    lead_end = max(x.end() for x in seps if x.start() == left)
    if len(t[lead_end:m.start()].split()) <= 2:
        semi = t.find(";", m.end())
        right = semi if semi >= 0 else None
    end = right if right is not None else len(t.rstrip(" ."))
    out = (t[:left] + t[end:]).strip()
    return out if out.endswith((".", "?", "!", ":")) else out.rstrip(" ,;") + "."


#: Work that puts gear in place.
_INSTALL_WORK = re.compile(r"(?i)\b(install\w*|mount\w*|hang\w*|rack\w*|cabl\w*|replac\w*|swap\w*|deploy\w*|relocat\w*|moves?|moving|"
                           r"setup|set up|refresh\w*|wir(e|es|ing)|drill\w*)\b")
#: Install-job wording a non-install SOW reads in its own words.
_INSTALL_REWORD = ((r"(?i)\b(device )?installation activities\b", "onsite support activities"),
                   (r"(?i)\binstallation (location|area|site)s?\b", r"work \1"),
                   (r"(?i)\binstallation exceptions\b", "exceptions"),
                   (r"(?i),?\s*\b(mounting (materials|components|hardware)|damaged fittings)\b(?=,|\s+(and|or)\b)", ""),
                   (r"(?i)\bdevice type, quantity, and device-specific information\b", "site, system and access information"))


def _no_install_wording(c: dict[str, Any], atoms: list[dict[str, Any]]) -> None:
    """A job that installs nothing reads in its own words (010419 pass V: an
    IT transition's SOW asked for "mounting materials", "the installation
    location" and "device installation activities"). Only when no deal line
    and no deal task names install work; a writer line is reworded where its
    install word has a plain equivalent, and an uncited one still built on
    install work goes."""
    if any(_INSTALL_WORK.search(str(a.get("text") or "")) for a in atoms
           if str(a.get("atom_type") or "") not in ("small_talk", "signature", "deal_metadata", "stakeholder")):
        return
    # A task that does install work leads with it ("Install ...", "Provider
    # will replace ..."); a report line naming "installation exceptions" is
    # the wording this rule fixes, not install work.
    lead = re.compile(r"(?i)^\s*(provider will (also )?)?(install|mount|hang|rack|cable|replace|swap|deploy|relocate|move|set up|"
                      r"refresh|wire|drill|remove and replace)\b")
    if any(lead.search(str(x.get("text"))) for k in ("overview", "scope") for x in c.get(k) or []
           if isinstance(x, dict) and not x.get("standard") and x.get("kind") != "heading"):
        return
    still = re.compile(r"(?i)\b(install\w*|mount\w*|fittings?)\b")
    for k in LISTS:
        keep = []
        for x in c.get(k) or []:
            if not isinstance(x, dict) or x.get("standard") or not still.search(str(x.get("text"))):
                keep.append(x)
                continue
            t = str(x["text"])
            for pat, rep in _INSTALL_REWORD:
                t = re.sub(pat, rep, t)
            t = re.sub(r"\s+,", ",", re.sub(r",\s*,", ",", t))
            if still.search(t) and not x.get("cites"):
                continue
            x["text"] = t
            keep.append(x)
        c[k] = keep


#: A follow-up, return or next-day visit.
_FOLLOWUP = re.compile(r"(?i)\b(next|following|second|additional) (business )?day\b|\bfollow-?ups?\b|\breturn (visit|trip)s?\b|"
                       r"\bremaining items\b|\bcome back\b|\brevisits?\b")
#: Words that make a follow-up a habit or a maybe, not a term.
_HABIT = re.compile(r"(?i)\b(usually|typically|normally|often|sometimes|some|may|might|if needed|as needed|possibly|depend\w*|can be)\b")


def _followup_not_promised(c: dict[str, Any], atoms: list[dict[str, Any]]) -> None:
    """A follow-up the deal hedges is never a commitment of any kind, not a
    task, a schedule term or an assumption (010419 pass V: "usually a full
    day of work, with some follow-up needed the next day" became the
    assumption "... with any remaining items handled the next day if
    needed"). The follow-up clause goes whole; a line that is nothing but
    the follow-up goes; and the SOW says follow-up visits need a change
    order, cited to the deal's hedged line."""
    hedges = [a for a in atoms if _FOLLOWUP.search(str(a.get("text") or "")) and _HABIT.search(str(a.get("text") or ""))
              and str(a.get("atom_type") or "") not in ("small_talk", "signature")]
    if not hedges:
        return
    firm = re.compile(r"(?i)\b(unless|change order|not included|excluded|authori[sz]ed in writing)\b")
    withheld = False
    for k in ("overview", "scope", "schedule", "assumptions", "customer_responsibilities", "arrival_protocol"):
        keep = []
        for x in c.get(k) or []:
            t = str(x.get("text") if isinstance(x, dict) else x)
            if not isinstance(x, dict) or x.get("kind") == "heading" or x.get("standard") or not _FOLLOWUP.search(t) or firm.search(t):
                keep.append(x)
                continue
            withheld = True
            t1 = _drop_clause(t, _FOLLOWUP)
            if t1 and len(t1.split()) >= 4 and not _FOLLOWUP.search(t1):
                x["text"] = t1
                keep.append(x)
        c[k] = keep
    oos = c.setdefault("out_of_scope", [])
    if withheld and not any(_FOLLOWUP.search(str(x.get("text") if isinstance(x, dict) else x)) for x in oos):
        oos.append({"text": "Follow-up visits unless authorized in writing or by Change Order.",
                    "cites": [str(hedges[0].get("id"))], "kind": "fact"})


#: A closing group's heading.
_CLOSING_HEAD = re.compile(r"(?i)clos|clean|wrap|complet")


def _replacement_order(body: list[Any]) -> list[Any]:
    """Nothing new is installed, dressed or validated before the old gear
    comes out (010209 pass V: rack hardware, power cords, Velcro and remote
    validation were listed before "Remove the existing switches"). In a
    group with a step that removes gear, an install step listed before it
    moves to just after it, and a dressing or validation step to after the
    group's last install step; every other line keeps its place. A line
    that leads with removal and also installs ("Remove ... and install
    ...") is the removal. Only the existing gear's removal counts (not a
    cable access plug), and never one the deal puts "after installation"
    (010319's old infrastructure comes out last)."""
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    lines = [isinstance(x, dict) and x.get("kind") != "heading" for x in body]
    removal = re.compile(r"(?i)^\s*(also\s+)?(remove|disconnect|unrack|uninstall|decommission|dismantle)\b.{0,30}\b(existing|old|legacy|current)\b"
                         r"(?!.*\bafter (the )?(install|installation|cutover|migration)\b)")
    install = re.compile(r"(?i)^\s*(also\s+)?(install|mount|rack|migrate|connect|power|reinstall|seat)\b")
    finish = re.compile(r"(?i)^\s*(also\s+)?(manage|dress|bundle|tidy|support (the )?(available )?[\w ]{0,20}remote|validate|test)\b")
    loose = re.compile(r"(?i)^\s*(also\s+)?(disconnect and remove|remove|disconnect|unrack|uninstall|decommission|dismantle)\b"
                       r"(?!.{0,40}\b(debris|packaging|trash|waste|cardboard|boxes|plug|cover|cap|film|tabs?)\b)")
    # The group's first removal must be the existing gear's (010387: "remove
    # the two bad TVs" already leads; a later standard removal is no anchor).
    r = next((i for i, x in enumerate(body) if lines[i] and loose.search(tx(x))), None)
    if r is None or not removal.search(tx(body[r])):
        return body
    early_i = [x for i, x in enumerate(body[:r]) if lines[i] and install.search(tx(x))]
    early_f = [x for i, x in enumerate(body[:r]) if lines[i] and finish.search(tx(x))]
    if not early_i and not early_f:
        return body
    rest = [x for x in body if not any(x is y for y in early_i + early_f)]
    r = next(i for i, x in enumerate(rest) if x is body[r])
    rest[r + 1:r + 1] = early_i
    last = max((i for i, x in enumerate(rest) if i > r and isinstance(x, dict) and x.get("kind") != "heading" and install.search(tx(x))), default=r)
    rest[last + 1:last + 1] = early_f
    return rest


def _standard_exclusion_survives(c: dict[str, Any]) -> None:
    """An exclusion that is one of PurTera's standard exclusions needs no
    source to state it, so the fact check never takes it (010326 pass V:
    "Disposal, shipping, data destruction, or asset disposition of removed
    phones." dropped as "the cited lines do not state it"; the standards
    say it of removed equipment). It comes back unless a deal task does the
    work it excludes."""
    try:
        data = json.loads((Path(__file__).parent / "purtera_standards.json").read_text())
    except Exception:
        return
    std = [re.sub(r"^(\{(when|unless):.*?\}\s+)+", "", str(t)) for fam, v in data.items() if not fam.startswith("_") and isinstance(v, dict)
           for t in v.get("exclusions") or []]
    tasks = [str(x.get("text")) for x in c.get("scope") or [] if isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")]
    oos = c.setdefault("out_of_scope", [])
    back = []
    for d in c.get("dropped") or []:
        if not (isinstance(d, dict) and d.get("list") == "out_of_scope" and d.get("text")):
            continue
        t = str(d["text"])
        if _said(t, std, 0.6) and not _said(t, oos, 0.6) and not any(_collides(t, k) for k in tasks):
            oos.append({k: v for k, v in d.items() if k not in ("list", "reason")})
            back.append(d)
    if back:
        c["dropped"] = [d for d in c.get("dropped") or [] if not any(d is b for b in back)]


def _materials_onsite_only(c: dict[str, Any]) -> None:
    """Materials are what is on site for the work (010326 pass V: Customer
    Provided Materials listed "Cisco Unified Communications Security
    licenses", "Smart Net Total Care service" and the router and VG400
    gateways the customer racks itself). A license, subscription or service
    contract is never a material; gear a cited customer duty has the
    customer install or rack itself is the customer's own work, not
    material for Provider's."""
    contract = re.compile(r"(?i)\blicen[cs]es?\b|\bsubscriptions?\b|smart ?net|\bservice (agreement|contract)\b|\bsupport contract\b|"
                          r"\bwarrant(y|ies)\b|\bsaas\b")
    gear = re.compile(r"(?i)\b(gateways?|routers?|firewalls?|servers?|switch(es)?|racks?)\b")
    # The duty must have the customer doing the work on that gear, not
    # supplying it (010082: "Provide the 9U rack" leaves the rack material).
    own = [m_.group(0) for d in c.get("customer_responsibilities") or [] if isinstance(d, dict) and d.get("cites")
           for m_ in [re.search(r"(?i)\b(handle|install|rack(ing)? ?(and|&|/) ?stack|mount|manage)\w*\b.{0,60}", str(d.get("text")))] if m_]
    own_gear = {g.group(1).lower().rstrip("s").rstrip("e") if g.group(1).lower().endswith("es") else g.group(1).lower().rstrip("s")
                for t in own for g in gear.finditer(t)}
    for k in ("customer_materials", "provider_materials", "equipment_in_scope"):
        keep = []
        for x in c.get(k) or []:
            t = str(x.get("text") if isinstance(x, dict) else x)
            if isinstance(x, dict) and not x.get("standard") and contract.search(t):
                continue
            if k == "customer_materials" and isinstance(x, dict) and not x.get("standard"):
                mine = {g.group(1).lower().rstrip("s").rstrip("e") if g.group(1).lower().endswith("es") else g.group(1).lower().rstrip("s")
                        for g in gear.finditer(t)}
                if mine and mine <= own_gear:
                    continue
            keep.append(x)
        c[k] = keep


def _schedule_not_estimates(c: dict[str, Any]) -> None:
    """A customer's per-unit hour estimate is a pricing input, never a
    schedule term (010209 pass V: "2 switches are estimated at 2.0 hours
    per store" and three more, from the RFQ's "My estimated time", read as
    duration commitments). An unlabeled schedule line giving estimated hours
    per store, site or unit goes; a labeled line PurTera writes
    ("Estimated Duration: ...") stays."""
    est = re.compile(r"(?i)\bestimat\w*\b.{0,40}\bhours?\b.{0,20}\bper\b|\bhours?\b.{0,10}\bper (store|site|location|unit|device|switch|room)\b.{0,40}\bestimat")
    c["schedule"] = [x for x in c.get("schedule") or []
                     if not (isinstance(x, dict) and est.search(str(x.get("text")))
                             and not re.match(r"^\s*[A-Z][\w /-]{2,30}:", str(x.get("text"))))]


def _window_rollover(c: dict[str, Any]) -> None:
    """A work window the deal itself extends says so (010209 pass V: "Work
    Window: October 13, 2026 through November 19, 2026" beside "Any site not
    complete by November 19, 2026 will be scheduled ... starting mid-January
    2027"; the template's "All Services will be performed during the
    Customer's work window" then ruled the rollover out)."""
    sch = [x for x in c.get("schedule") or [] if isinstance(x, dict)]
    win = next((x for x in sch if re.match(r"(?i)\s*work window:.*\bthrough\b", str(x.get("text")))), None)
    if win is None:
        return
    end = re.search(r"(?i)\bthrough\s+(.+?)\s*$", str(win["text"])).group(1).rstrip(". ")
    for x in sch:
        t = str(x.get("text"))
        m = re.search(r"(?i)not (be )?complete[d]? by\s+" + re.escape(end) + r".*?\b(?:starting|from|beginning|resuming|after)\s+"
                      r"((?:early|mid|late)-?\s*[A-Z][a-z]+(?:\s+\d{4})?|[A-Z][a-z]+ \d{1,2},? \d{4}|[A-Z][a-z]+ \d{4})", t)
        if x is not win and m and "not complete by then" not in str(win["text"]):
            win["text"] = f"{str(win['text']).rstrip('. ')}; any site not complete by then is scheduled from {m.group(2)}"
            win["cites"] = list(dict.fromkeys(list(win.get("cites") or []) + list(x.get("cites") or [])))
            return


def _qualified_over_universal(c: dict[str, Any]) -> None:
    """One step said twice, once "where required" and once "at every store",
    is the qualified step (010209 pass V: "Install rack mount hardware at
    every store" beside "... where required", from the RFQ's "Depending on
    switch, rack mount hardware must be installed"). The universal copy
    goes."""
    qual = re.compile(r"(?i)\b(where|as|if|when) (required|needed|applicable|necessary)\b")
    univ = re.compile(r"(?i)\b(at|in|for|to) (every|each|all) (stores?|sites?|locations?|rooms?)\b")
    for k in ("scope",):
        lst = c.get(k) or []
        qs = [x for x in lst if isinstance(x, dict) and x.get("kind") != "heading" and qual.search(str(x.get("text")))]
        strip = lambda t: _toks(univ.sub(" ", qual.sub(" ", t)))
        gone = []
        for x in lst:
            t = str(x.get("text") if isinstance(x, dict) else x)
            if not isinstance(x, dict) or x.get("kind") == "heading" or not univ.search(t) or qual.search(t):
                continue
            a = strip(t)
            if any(a and len(a & strip(str(q.get("text")))) >= 0.8 * max(len(a), len(strip(str(q.get("text"))))) for q in qs):
                gone.append(x)
        c[k] = [x for x in lst if not any(x is g for g in gone)]


def _field_sequence(c: dict[str, Any]) -> None:
    """Steps in the order a technician does them, within each work group
    (010209 pass V: "Take after photos of the completed switch replacement"
    before the switches were removed, and "Confirm the switches ... are
    onsite before starting" after the labeling and tracing it gates). A
    photo of finished work goes to the end of its group; a check made
    "before starting" goes ahead of the group's first hands-on step."""
    sc = c.get("scope") or []
    groups: list[list[Any]] = [[]]
    for x in sc:
        if isinstance(x, dict) and x.get("kind") == "heading":
            groups.append([x])
        else:
            groups[-1].append(x)
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    after_ph = re.compile(r"(?i)^\s*(take|capture|provide)\b.{0,20}\b(after|completion|completed|post-?work|final)\b.{0,20}photo"
                          r"|^\s*(take|capture)\b.{0,10}photos? of the (completed|finished|installed)")
    gate = re.compile(r"(?i)^\s*(confirm|verify|check)\b.{0,120}\b(onsite|on site|available|delivered)\b.{0,30}\bbefore (starting|beginning|work)")
    hands = re.compile(r"(?i)^\s*(label|trace|remove|disconnect|install|mount|use the|capture|migrate|unrack|dismantle|replace|take (pre-?work|before))\b")
    out: list[Any] = []
    for g in groups:
        head = g[:1] if g and isinstance(g[0], dict) and g[0].get("kind") == "heading" else []
        body = g[len(head):]
        if head and _CLOSING_HEAD.search(tx(head[0])):
            out += g
            continue
        late = [x for x in body if isinstance(x, dict) and after_ph.search(tx(x))]
        body = [x for x in body if not any(x is y for y in late)]
        body = _replacement_order(body)
        first = next((i for i, x in enumerate(body) if hands.search(tx(x))), None)
        if first is not None:
            moved = [x for i, x in enumerate(body) if i > first and isinstance(x, dict) and gate.search(tx(x))]
            if moved:
                body = [x for x in body if not any(x is y for y in moved)]
                body[first:first] = moved
        out += head + body + late
    c["scope"] = out


def _sentence_integrity(c: dict[str, Any]) -> None:
    """No line ends on a preposition or a conjunction: a clause that lost its
    object is clipped, a line with nothing left goes."""
    for k in LISTS:
        keep = []
        for x in c.get(k) or []:
            if isinstance(x, dict) and x.get("kind") != "heading" and isinstance(x.get("text"), str):
                t = _clip_dangling(x["text"])
                if t is None:
                    continue
                x["text"] = t
            keep.append(x)
        c[k] = keep


#: Visit logistics, not a service the summary names.
_LOGISTICS = re.compile(r"(?i)\b(arriv\w*|site access|check[- ]?in|parking|badg\w*|escort\w*|onsite contact|on-site contact|sign[- ]?in)\b")

_NOMINAL = ((r"provide\s+", ""), (r"perform\s+", ""), (r"support\s+", "support for "), (r"coordinate\s+", "coordination "),
            (r"document\s+", "documentation of "), (r"troubleshoot\s+", "troubleshooting of "), (r"respond to\s+", "response to "),
            (r"maintain\s+", "maintenance of "), (r"monitor\s+", "monitoring of "))


def _per_site_lines(c: dict[str, Any], atoms: list[dict[str, Any]]) -> None:
    """Per-site lines are one per resolved site (000132 pass V: "Visit X once
    per week" for five of the six sites; Delphos, OH, in the roster, the
    locations and the summary, had no visit line). Lines that say the same
    thing but for the one site each names are a per-site family; when the
    family covers at least half the sites, each site it leaves out gets the
    same line, cited to what the family shares plus the site's own line.
    A family must end up with exactly one line per site, else nothing is
    added."""
    sites = [x for x in c.get("sites") or [] if isinstance(x, dict)]
    if not 2 <= len(sites) <= 40:
        return
    def pats(st: dict[str, Any]) -> list[str]:
        out = []
        ci, sta = str(st.get("city") or "").strip(), str(st.get("state") or "").strip()
        nm = str(st.get("name") or "").strip()
        if ci and re.fullmatch(r"[A-Z]{2}", sta):
            out.append(re.escape(ci) + r",\s*" + re.escape(sta) + r"\b")
        if nm and len(nm) >= 4:
            out.append(re.escape(nm))
        if ci and len(ci) >= 4:
            out.append(r"\b" + re.escape(ci) + r"\b")
        return out
    site_pats = [pats(st) for st in sites]
    if any(not p for p in site_pats):
        return
    for k in ("scope", "schedule"):
        lst = c.get(k) or []
        fams: dict[str, list[tuple[int, int, str]]] = {}
        for i, x in enumerate(lst):
            if not isinstance(x, dict) or x.get("kind") == "heading" or x.get("standard"):
                continue
            t = str(x.get("text") or "")
            hits = []
            for j, ps in enumerate(site_pats):
                m = next((mm for p in ps for mm in [re.search(p, t)] if mm), None)
                if m:
                    hits.append((j, m))
            if len(hits) != 1:
                continue
            j, m = hits[0]
            fams.setdefault(t[:m.start()] + "\x00" + t[m.end():], []).append((i, j, m.group(0)))
        adds: list[tuple[int, dict[str, Any]]] = []
        for tmpl, rows in fams.items():
            have = {j for _, j, _ in rows}
            if len(rows) < 2 or len(have) != len(rows) or len(rows) * 2 < len(sites) or len(have) == len(sites):
                continue
            with_state = all("," in g for _, _, g in rows)
            group = [lst[i] for i, _, _ in rows]
            shared = [cc for cc in group[0].get("cites") or [] if all(cc in (g.get("cites") or []) for g in group)]
            last = max(i for i, _, _ in rows)
            for j, st in enumerate(sites):
                if j in have:
                    continue
                ci, sta = str(st.get("city") or "").strip(), str(st.get("state") or "").strip()
                label = f"{ci}, {sta}" if with_state and ci and sta else (ci or str(st.get("name") or ""))
                own = [str(a.get("id")) for a in atoms if str(a.get("atom_type") or "") == "physical_site"
                       and any(re.search(p, str(a.get("text") or "")) for p in site_pats[j][:1])][:1]
                adds.append((last, {"text": tmpl.replace("\x00", label), "cites": shared + own,
                                    "kind": group[0].get("kind") or "fact", "from": "sites"}))
        # After the family's last line, in roster order (a later anchor
        # first, so an earlier anchor's index still holds).
        for at in sorted({a for a, _ in adds}, reverse=True):
            lst[at + 1:at + 1] = [line for a, line in adds if a == at]
        c[k] = lst


def _summary_services_tasked(c: dict[str, Any], atoms: list[dict[str, Any]]) -> None:
    """On recurring support, a service the summary promises is also a task
    (000132 pass V: "end-user support including O365/M365 clients" was in the
    summary only; the deal's line saying it was typed deal_metadata, so no
    fact carried it to the task list). The service is the deal's own short
    line, every word of which the summary says, that no task says yet; it is
    written as a task cited to that line. Never a hedged or round-the-clock
    ask, a question, or a line the SOW excludes."""
    if not {"support", "staff_augmentation"} & set(c.get("services") or []):
        return
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    ov_toks = _toks(" ".join(tx(x) for x in c.get("overview") or []))
    sc = c.setdefault("scope", [])
    oos = [x for x in (c.get("out_of_scope") or []) + (c.get("assumptions") or [])]
    clock = re.compile(r"(?i)" + _ROUND_CLOCK + r"|on-?call")
    seen: list[str] = []
    adds = []
    for a in atoms:
        t = re.sub(r"\s+", " ", str(a.get("text") or "")).strip().rstrip(".")
        if not (2 <= len(t.split()) <= 10) or "?" in t or clock.search(t) or _HEDGE.search(t):
            continue
        if not re.search(r"(?i)\bsupport\b", t) or str(a.get("atom_type") or "") in ("physical_site", "question", "small_talk", "signature", "contact"):
            continue
        want = _toks(t)
        if len(want) < 2 or not want <= ov_toks or _said(t, sc, 0.6) or _said(t, oos, 0.6) or _said(t, seen, 0.8):
            continue
        seen.append(t)
        t = re.sub(r"(?i)\s+including\b", ", including", t)
        if re.match(r"(?i)(support|perform|provide|troubleshoot|maintain|coordinate)\b", t):
            line = t[:1].upper() + t[1:] + "."
        else:
            line = "Provide " + (t[:1].lower() + t[1:] if t[1:2].islower() else t) + "."
        adds.append({"text": line, "cites": [str(a.get("id"))], "kind": "fact", "from": "overview"})
    if not adds:
        return
    # Beside the deal's other cited service tasks.
    at = max((i for i, x in enumerate(sc) if isinstance(x, dict) and x.get("kind") != "heading" and x.get("cites")
              and not x.get("standard") and re.match(r"(?i)\s*(support|perform|provide|troubleshoot|maintain)\b", tx(x))), default=len(sc) - 1)
    sc[at + 1:at + 1] = adds[:3]


def _summary_names_job(c: dict[str, Any]) -> None:
    """A few-site job's summary names its sites, and a recurring support
    job's summary names every service its cited task lines carry (000132
    pass T: the summary shrank to four technical areas, without the six
    cities, the end-user support or the stakeholder coordination). The
    task line that already names every site is the site sentence; else
    the cities are listed. At most two sentences are added."""
    ov = c.setdefault("overview", [])
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    cities = [(str(x.get("city") or "").strip(), str(x.get("state") or "").strip()) for x in c.get("sites") or []]
    # Only a clean city list names the sites (010376: a street and a suite
    # number parsed as cities).
    clean = all(re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,30}", ci) and re.fullmatch(r"[A-Z]{2}", st) for ci, st in cities)
    cities = [(ci, st) for ci, st in cities if ci] if clean else []
    support = bool({"support", "staff_augmentation"} & set(c.get("services") or []))
    tasks = [x for x in c.get("scope") or [] if isinstance(x, dict) and x.get("kind") != "heading"
             and not x.get("standard") and x.get("cites")]
    added = 0
    ov_txt = " ".join(tx(o) for o in ov)
    if 2 <= len(cities) <= 8 and len({ci for ci, _ in cities}) == len(cities) \
            and not any(re.search(r"\b" + re.escape(ci) + r"\b", ov_txt) for ci, _ in cities):
        named = next((x for x in tasks if all(re.search(r"\b" + re.escape(ci) + r"\b", tx(x)) for ci, _ in cities)), None)
        if named is not None and re.match(r"(?i)(perform|provide|support|install|deploy|replace|visit)\b", tx(named)):
            t = tx(named).strip().rstrip(".")
            line = {"text": "Provider will " + t[:1].lower() + t[1:] + ".", "cites": list(named["cites"]), "kind": "fact"}
        else:
            places = [f"{ci}, {st}" if st else ci for ci, st in cities]
            what = "provide recurring on-site IT support" if support else "perform the work"
            line = {"text": f"Provider will {what} at {_words(len(places))} ({len(places)}) Customer locations: "
                            + ("; ".join(places[:-1]) + "; and " if len(places) > 2 else places[0] + " and ") + places[-1] + ".", "cites": [], "kind": "method"}
        ov.insert(min(1, len(ov)), line)
        added += 1
    if support:
        said_by = [o for o in ov]
        names = []
        for x in tasks:
            t = tx(x).strip().rstrip(".")
            if cities and sum(1 for ci, _ in cities if re.search(r"\b" + re.escape(ci) + r"\b", t)) >= 2:
                continue  # the site line
            if _said(t, said_by, 0.5) or len(t.split()) > 10 or not _SERVICE_WORDS.search(t) or _LOGISTICS.search(t):
                continue  # a service is named in a few words; a long line is a task, not a service
            for pat, rep in _NOMINAL:
                mm = re.match(r"(?i)\s*" + pat, t)
                if mm:
                    names.append(rep + t[mm.end():])
                    break
        if names:
            body = names[0] if len(names) == 1 else "; ".join(names[:-1]) + ("; and " if len(names) > 2 else " and ") + names[-1]
            if len(names) == 2:
                body = body.replace("; and ", " and ")
            ov.append({"text": "Provider will also provide " + body + ".",
                       "cites": [cc for x in tasks if any(_said(n, [x], 0.6) for n in names) for cc in x.get("cites") or []],
                       "kind": "fact"})
            added += 1


_PLAN_BASIS = re.compile(r"(?i)\b(most|each|every|all|typical(?:ly)?|the average)\b[^.;?]{0,30}?\b(stores?|sites?|locations?|restaurants?|branches)\b[^.;?]{0,15}?"
                         r"\b(have|has|take|takes|would take|will take|need|needs|require|requires|run|runs)\b[^.;?]{3,60}")


def _planning_basis(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A per-site planning fact the brief kept and no line cites ("Most
    stores have four register lanes", "each store would take 2 hours") is
    a pricing assumption (000123 pass T: the lane count was never written).
    At most two."""
    said_by: dict[str, str] = {}
    for k in LISTS:
        for x in content.get(k) or []:
            if isinstance(x, dict):
                for c in x.get("cites") or []:
                    said_by[c] = said_by.get(c, "") + " " + str(x.get("text")).lower()
    asm = content.setdefault("assumptions", [])
    added = 0
    for f in fs:
        if added >= 2 or f["section"] in ("also_kept", "unsure", "typed_only", "answered_question") or f.get("quoted"):
            continue
        mm = _PLAN_BASIS.search(f["text"])
        if not mm or not re.search(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|dozen)\b", mm.group(0), re.I):
            continue
        clause = re.split(r"(?i)\s+and\s+there\b|,\s+(?:and|but|so)\b|\s+but\b", mm.group(0))[0].strip(" ,")
        num = re.search(r"\d+|\b(one|two|three|four|five|six|seven|eight|nine|ten|dozen)\b", clause, re.I)
        if not num or num.group(0).lower() in said_by.get(f["id"], ""):
            continue  # a line citing it already states the number
        line = "The Customer has indicated that " + clause[:1].lower() + clause[1:] + "."
        if _said(line, asm, 0.6):
            continue
        asm.append({"text": line, "cites": [f["id"]], "kind": "fact"})
        added += 1


def _one_supplier(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """One item is never both Provider material and customer-provided. When
    a Provider material line names an item that a customer material, a
    customer duty ("Ensure the 9U rack is on site") or an assumption
    ("The Customer-provided 9U rack ...") also gives the customer, the side
    whose cited line is dated later wins; undated, the side more lines give
    it to (010082 pass T: the 9U rack the customer orders sat under Provider
    Materials beside two lines saying the customer provides it)."""
    when = {f["id"]: str(f.get("when") or "") for f in fs}
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    latest = lambda xs: max((when.get(c, "") for x in xs if isinstance(x, dict) for c in x.get("cites") or []), default="")
    cust_say = re.compile(r"(?i)customer[- ](provided|supplied|furnished)|customer (will )?(provides?|supplies|orders?)|\bensure\b.{0,40}\b(on ?site|delivered|available)")
    pm = content.get("provider_materials") or []
    for x in list(pm):
        if not isinstance(x, dict) or x.get("standard"):
            continue
        key = {w for w in _toks(tx(x)) if not re.fullmatch(r"one|two|three|four|five|\d+|provid|suppl|all|required|need", w)}
        if not key or len(key) > 4:
            continue
        hits = lambda k, need_say: [y for y in content.get(k) or [] if isinstance(y, dict) and y is not x
                                    and key <= _toks(tx(y)) and (not need_say or cust_say.search(tx(y)))]
        theirs = hits("customer_materials", False) + hits("customer_responsibilities", False) + hits("assumptions", True)
        if not theirs:
            continue
        ours = [x] + [y for y in content.get("scope") or [] if isinstance(y, dict) and key <= _toks(tx(y))
                      and re.search(r"(?i)\b(provide|supply|furnish|bring)\b", tx(y))]
        a, b = latest(ours), latest(theirs)
        customer_wins = (b > a) if (a or b) and a != b else len(theirs) >= len(ours)
        if customer_wins:
            pm.remove(x)
            cm = content.setdefault("customer_materials", [])
            if not _said(tx(x), cm, 0.7):
                cm.append(x)
        else:
            for k in ("customer_materials",):
                content[k] = [y for y in content.get(k) or [] if not any(y is t for t in theirs)]


def _room_counts(content: dict[str, Any]) -> None:
    """Per-room install counts add up to the equipment count. When the task
    lines naming a model install fewer than Equipment in Scope lists, the
    lines that name it (summary, tasks, report) carry the full count across
    the rooms instead of one room's (010012 pass T: "one (1) Poly X72 bar ...
    in the Conference Room" beside "Two (2) Poly X72 bars" and a second room)."""
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    num = r"(?:(%s)\s*\((\d{1,3})\)|(\d{1,3}))" % _NUMW
    for eq in content.get("equipment_in_scope") or []:
        mm = re.match(r"(?i)\s*" + num + r"\s+(.+)$", tx(eq))
        if not mm:
            continue
        n_eq = int(mm.group(2) or mm.group(3) or 0)
        models = [w for w in re.findall(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9-]{2,}\b", mm.group(4))]
        if n_eq < 2 or not models:
            continue
        def count_in(t: str) -> int | None:
            k = re.search(r"(?i)" + num + r"\s+(?:[\w-]+\s+){0,2}" + re.escape(models[0]) + r"\b", t)
            return int(k.group(2) or k.group(3)) if k else None
        tasks = [x for x in content.get("scope") or [] if isinstance(x, dict) and not x.get("standard") and x.get("kind") != "heading"
                 and count_in(tx(x)) is not None]
        total = sum(count_in(tx(x)) or 0 for x in tasks)
        if not tasks or total >= n_eq:
            continue
        def fix(t: str) -> str:
            if count_in(t) is None:
                return t
            t = re.sub(r"(?i)\b(one|a|an)\s*\(1\)\s+", f"{_words(n_eq)} ({n_eq}) ", t)
            t = re.sub(r"(?i)\b(bar|controller|display|camera|unit|device|kit|mount|speaker|panel)\b(?!s)", r"\1s", t)
            t = re.sub(r"\s+(?:in|at) the [A-Z][\w-]*(?: [A-Z][\w-]*){0,3}(?=[.,;]|$)", " across the rooms in scope", t)
            return t[:1].upper() + t[1:]
        for k in ("overview", "scope"):
            for x in content.get(k) or []:
                if isinstance(x, dict) and not x.get("standard") and x.get("kind") != "heading":
                    x["text"] = fix(tx(x))
        for d in content.get("deliverables") or []:
            if isinstance(d, dict):
                d["description"] = [fix(str(t)) for t in d.get("description") or []]


def _overview_covers_scope(content: dict[str, Any], fs: list[dict[str, Any]] | None = None) -> None:
    """The summary names what the cited task list counts: a cited work line
    whose numbers the summary lacks, and a cited haul-away, are added as
    summary sentences (010373: the 48 monitors and the trash haul-away were
    in the tasks, the summary named only the 16 arms). At most four lines."""
    ov = content.setdefault("overview", [])
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    room = min(4, len(ov) + 2)
    # One of several per-room or per-site lines is not the job's count: the
    # summary would name one room's share (010012 pass U: "Provider will
    # install one (1) Poly X72 bar ... in Conference Room 2102" beside the
    # Shared Office's own line).
    first8 = lambda y: " ".join(re.sub(r"\W+", " ", tx(y).lower()).split()[:8])
    tally: dict[str, int] = {}
    for y in content.get("scope") or []:
        if isinstance(y, dict) and y.get("kind") != "heading":
            tally[first8(y)] = tally.get(first8(y), 0) + 1
    for x in content.get("scope") or []:
        if isinstance(x, dict) and tally.get(first8(x), 0) > 1:
            continue
        if len(ov) >= room:
            break
        hauls = any(re.search(r"(?i)\bhaul|trash|debris removal|dispos", f["text"]) for f in fs or [])
        if not (isinstance(x, dict) and x.get("kind") != "heading"
                and (x.get("cites") or (hauls and re.search(r"(?i)\bhaul|trash", tx(x))))):
            continue
        t = tx(x).strip()
        if not re.match(r"(?i)(install|mount|replace|remove|deploy|relocate|haul|dispose)\b", t):
            continue
        if re.search(r"°|%|humidity|above (the )?floor|\b(?=[\w-]*\d)(?=[\w-]*[A-Z])[A-Z0-9][A-Z0-9-]{4,}\b|\b\d{6,}\b", t):
            continue  # specs, heights and part numbers are not summary material
        ov_txt = " ".join(tx(o) for o in ov).lower()
        counts = {a or b for a, b in re.findall(r"\b[a-z-]+ \((\d{1,4})\)|\b(\d{1,4}) (?:new )?(?:[A-Z]\w*|[a-z]+s)\b", t)}
        missing_count = bool(counts) and not counts <= set(re.findall(r"\d+", ov_txt)) and not _said(t, ov, 0.5)
        haul = bool(re.search(r"(?i)\bhaul|dispos|trash", t)) and not re.search(r"(?i)\bhaul|dispos|trash|debris", ov_txt)
        if (missing_count or haul) and not _said(t, ov, 0.7):
            ov.append({"text": "Provider will " + t[0].lower() + t[1:].rstrip(".") + ".", "cites": list(x["cites"]), "kind": "fact"})


def _weekend(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A weekend date is weekend work, not "regular business hours"."""
    # A weekend date is weekend work, not "regular business hours" (010198 Sunday).
    sched = " ".join(str(x.get("text") if isinstance(x, dict) else x) for x in content.get("schedule") or [])
    # The calendar date says the weekday when the line does not (010198:
    # "August 16, 2026" is a Sunday).
    from datetime import date as _date
    for mo, dy, yr_ in re.findall(r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(20\d\d)", sched):
        try:
            wd = _date(int(yr_), ["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                  "october", "november", "december"].index(mo.lower()) + 1, int(dy)).weekday()
        except ValueError:
            continue
        if wd >= 5:
            sched += " " + ("saturday" if wd == 5 else "sunday")
    if re.search(r"(?i)\b(saturday|sunday)\b", sched) and not re.search(
            r"(?i)\b(no|not|never|excluding|except|without)\b[^.]{0,40}\b(saturday|sunday|weekend)", sched + " " + " ".join(f["text"] for f in fs)):
        for k in ("schedule", "assumptions"):
            content[k] = [x for x in content.get(k) or [] if not re.search(r"(?i)regular business hours|monday.{0,12}friday", str(x.get("text") if isinstance(x, dict) else x))]
        asm = content.setdefault("assumptions", [])
        if not any(re.search(r"(?i)weekend", str(x.get("text") if isinstance(x, dict) else x)) for x in asm):
            day = re.search(r"(?i)\b(saturday|sunday)\b", sched).group(1).capitalize()
            asm.append({"text": f"All Services will be performed during the weekend ({day}, as scheduled); site access is available that day.", "cites": [], "kind": "method"})
        # The window says it too, so no template "regular business hours"
        # line stands unanswered beside a Sunday date (010198 W1).
        day = re.search(r"(?i)\b(saturday|sunday)\b", sched).group(1).capitalize()
        sch = content.setdefault("schedule", [])
        if not any(re.match(r"(?i)\s*work window", str(x.get("text") if isinstance(x, dict) else x)) for x in sch):
            sch.append({"text": f"Work Window: Weekend ({day})", "cites": [c for x in sch if isinstance(x, dict) for c in x.get("cites") or []][:2],
                        "kind": "fact"})


_MONTH_RE = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?"
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _dates_in(text: str) -> list[tuple[re.Match, int, int, int | None]]:
    """(match, month, day, year or None) for each written date."""
    out = []
    for m in re.finditer(r"\b(20\d\d)-(\d{2})-(\d{2})\b", text):
        out.append((m, int(m.group(2)), int(m.group(3)), int(m.group(1))))
    for m in re.finditer(r"(?i)\b(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day,?\s+)?" + _MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b", text):
        mo = next(i for i, n in enumerate(_MONTHS) if n.startswith(m.group(1).lower()[:3])) + 1
        out.append((m, mo, int(m.group(2)), int(m.group(3)) if m.group(3) else None))
    for m in re.finditer(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\d/])", text):
        y = m.group(3)
        out.append((m, int(m.group(1)), int(m.group(2)), (int(y) + (2000 if len(y) == 2 else 0)) if y else None))
    return [d for d in out if 1 <= d[1] <= 12 and 1 <= d[2] <= 31]


def _dates_ok(text: str, cited: str) -> bool:
    """Every date the line writes is a date its cited lines state."""
    have = {(mo, d) for _m, mo, d, _y in _dates_in(cited)}
    return all((mo, d) in have for _m, mo, d, _y in _dates_in(text)) if have or _dates_in(text) else True


def _fix_dates(content: dict[str, Any], fs: list[dict[str, Any]], as_of: str) -> None:
    """The writer never re-dates a dated fact: a schedule or summary date its
    cited lines do not state takes the month and day they do state (010198:
    "next Sunday, August 16th" printed as 2026-08-09, the as-of Sunday), with
    the weekday the calendar gives."""
    from datetime import date as _date
    text_of = {f["id"]: f["text"] for f in fs}
    yr0 = int(as_of[:4]) if as_of[:4].isdigit() else None
    for k in ("schedule", "overview"):
        for x in content.get(k) or []:
            if not (isinstance(x, dict) and x.get("cites")):
                continue
            cited = " ".join(text_of.get(c, "") for c in x["cites"])
            stated = _dates_in(cited)
            if not stated:
                continue
            have = {(mo, d) for _m, mo, d, _y in stated}
            t = str(x.get("text"))
            for m, mo, d, y in sorted(_dates_in(t), key=lambda z: -z[0].start()):
                if (mo, d) in have:
                    continue
                _m2, mo2, d2, y2 = stated[-1]
                y2 = y2 or y or yr0
                if not y2:
                    continue
                try:
                    dd = _date(y2, mo2, d2)
                except ValueError:
                    continue
                t = t[:m.start()] + f"{_WEEKDAYS[dd.weekday()]}, {_MONTHS[mo2 - 1].capitalize()} {d2}, {y2}" + t[m.end():]
            x["text"] = re.sub(r"\b(Sunday|Saturday|Monday|Tuesday|Wednesday|Thursday|Friday),\s+(?=(Sunday|Saturday|Monday|Tuesday|Wednesday|Thursday|Friday),)", "", t)
    # A schedule date no fact states, on a line whose cites state no date
    # (uncited, or citing an undated line), never renders when the deal
    # states its own date: it takes the date of the deal's latest dated
    # timeline or milestone line, or goes (010198 live, pass T: an uncited
    # "Requested Start Date: 2026-08-09", the as-of Sunday, beside "next
    # Sunday, August 16th").
    pool = {(mo, d) for f in fs for _m, mo, d, _y in _dates_in(f["text"])}
    if not pool:
        return
    dated = [f for f in fs if (f.get("type") in ("milestone_phase", "deadline") or f.get("section") in ("timeline", "schedule"))
             and _dates_in(f["text"])]
    kept = []
    for x in content.get("schedule") or []:
        if not isinstance(x, dict):
            kept.append(x)
            continue
        t = str(x.get("text"))
        cited_dates = [z for c in x.get("cites") or [] for z in _dates_in(text_of.get(c, ""))]
        wrong = [z for z in _dates_in(t) if (z[1], z[2]) not in pool]
        if cited_dates or not wrong:
            kept.append(x)
            continue
        if any(re.match(r"(?i)\s*requested (start|service)", str(y.get("text"))) for y in kept if isinstance(y, dict)) \
                and re.match(r"(?i)\s*requested (start|service)", t):
            continue  # the deal's own start line is already there
        if dated and re.match(r"(?i)\s*requested (start|service)", t):
            f = dated[-1]
            _m2, mo2, d2, y2 = _dates_in(f["text"])[-1]
            y2 = y2 or wrong[0][3] or yr0
            try:
                dd = _date(int(y2), mo2, d2)
            except (TypeError, ValueError):
                continue
            for m, *_rest in sorted(wrong, key=lambda z: -z[0].start()):
                t = t[:m.start()] + f"{_WEEKDAYS[dd.weekday()]}, {_MONTHS[mo2 - 1].capitalize()} {d2}, {dd.year}" + t[m.end():]
            if not any(str(y.get("text")) == t for y in kept if isinstance(y, dict)):
                kept.append({**x, "text": t, "cites": [f["id"]]})
            continue
        # no dated line to take it from: the date goes with its line
    content["schedule"] = kept


def _agreed_arrival(content: dict[str, Any], envelope: dict[str, Any]) -> None:
    """A start time one side proposes as a question ("How would 8am Sunday
    work for you?") and the other side accepts in a later message ("Yes, we
    can hit that date!") is the arrival time (010198)."""
    sch = content.setdefault("schedule", [])
    if any(re.match(r"(?i)\s*arrival time", str(x.get("text") if isinstance(x, dict) else x)) for x in sch):
        return
    atoms = envelope.get("atoms") or []
    sent = lambda a: _when_sent(str((a.get("locator") or {}).get("sent_at") or ""))
    who = lambda a: str((a.get("locator") or {}).get("sender") or "").lower()
    for a in atoms:
        t = str(a.get("text") or "")
        m = re.search(r"(?i)\b(how (would|does|about)|would|does|can we do|is)\b[^?]{0,20}?\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\b[^?]{0,40}\?", t)
        if not m or (a.get("structured") or {}).get("quoted") or sent(a) is None:
            continue
        ok = next((b for b in atoms if b is not a and sent(b) and 0 < sent(b) - sent(a) < 4 * 86400 and who(b) and who(b) != who(a)
                   and re.match(r"(?i)\s*(yes|yep|yeah|sounds good|that works|works (great|for)|we can (hit|do|make)|confirmed|perfect)\b", str(b.get("text") or ""))), None)
        if ok:
            h = int(m.group(3))
            sch.append({"text": f"Arrival Time: {h}:{m.group(4) or '00'} {m.group(5).upper()}M", "cites": [str(a.get("id")), str(ok.get("id"))], "kind": "fact"})
            return


#: Schedule terms PurTera writes from the facts: a dated window, a blackout,
#: allowed days and hours, a pace limit (010209: "installs will start 10/13
#: and continue through 11/19", "holiday blackout", "All work 7-10am Monday,
#: Tuesday, Thursday", "Only 1 store per time zone").
_SCHED_TERM = re.compile(r"(?i)\bblackout\b|\bper time zone\b|\b(start|begin)\w*\b.{0,40}\b(through|until|thru|continue)\b|"
                         r"\b\d{1,2}\s*(:\d{2})?\s*(am|pm)?\s*(-|–|to)\s*\d{1,2}\s*(:\d{2})?\s*(am|pm)\b.{0,40}\b(mon|tue|wed|thu|fri|sat|sun)|"
                         r"\b(mon|tue|wed|thu|fri|sat|sun)\w*\b.{0,40}\b\d{1,2}\s*(:\d{2})?\s*(am|pm)?\s*(-|–|to)\s*\d{1,2}\s*(:\d{2})?\s*(am|pm)\b|"
                         r"\b(one|1) (store|site|location) (per|a|each) (day|time zone)\b|\b(stores?|sites?) per (day|week)\b")


def _schedule_from_facts(content: dict[str, Any], fs: list[dict[str, Any]], envelope: dict[str, Any]) -> None:
    """Schedule terms a fact states reach the schedule even when the writer
    left them out, from the facts and from atoms the type head dated
    (milestone_phase, deadline) that no section kept. A term in a quoted
    reply is still the customer's term (010209's answers were all quoted in
    the next email); only meeting and call slots stay out."""
    sch = content.setdefault("schedule", [])
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    # A pace and an arrival lead time are schedule terms too (010083: "8
    # scheduled a week" and "arrive no less than 30 minutes before closing
    # time" were cited and never written).
    for f in fs:
        t = f["text"]
        pm = re.search(r"(?i)\b(\d{1,3})\s+(?:(sites?|stores?|locations?|branches|installs?)\s+)?(?:scheduled\s+|completed\s+|installed\s+|done\s+)?(?:a|per|each)\s+week\b", t)
        if pm and not any(tx(x).lower().startswith("pace") for x in sch) and not re.search(r"(?i)\b(hours?|days?|calls?|meetings?)\s+(a|per|each)\s+week", pm.group(0)):
            n, noun = int(pm.group(1)), (pm.group(2) or "sites").lower()
            noun = noun if noun.endswith("s") else noun + "s"
            sch.append({"text": f"Pace: {_words(n)} ({n}) {noun} per week", "cites": [f["id"]], "kind": "fact"})
        am = re.search(r"(?i)\barriv\w*\s+((?:no less than|at least|about|approximately)\s+)?(\d{1,3})\s*(minutes?|mins?|hours?|hrs?)\s+(before|prior to|ahead of)\s+([a-z ]{3,30}?)(?=[.,;)]|$)", t)
        if am and not any(tx(x).lower().startswith("arrival time") for x in sch):
            unit = "minutes" if am.group(3).lower().startswith("m") else ("hour" if am.group(2) == "1" else "hours")
            sch.append({"text": f"Arrival Time: {(am.group(1) or '').strip() + ' ' if am.group(1) else ''}{am.group(2)} {unit} {am.group(4).lower()} {am.group(5).strip()}".replace("  ", " "),
                        "cites": [f["id"]], "kind": "fact"})
    own = {}
    for a in envelope.get("atoms") or []:
        if not (a.get("structured") or {}).get("quoted"):
            own.setdefault(re.sub(r"\W+", " ", str(a.get("text") or "").lower()).strip(), str(a.get("id")))
    cands = [(f["id"], f["text"]) for f in fs if not str(f["text"]).rstrip().endswith("?")]
    seen = {i for i, _ in cands}
    for a in envelope.get("atoms") or []:
        lab = (a.get("structured") or {}).get("c3_label") or {}
        if a.get("id") not in seen and lab.get("type") in ("milestone_phase", "deadline") and float(lab.get("p") or 0) >= 0.5 \
                and 10 < len(str(a.get("text") or "")) < 300:
            cands.append((str(a["id"]), str(a["text"])))
    added = 0
    for i, t0 in cands:
        if added >= 4 or re.search(r"(?i)\b(call|meeting|meet|webex|zoom|teams|demo)\b|hours of operation|days of operation|store hours|\boperates\b", t0) \
                or "|" in t0:
            continue  # meeting slots, the site's own opening hours, form rows
        sent = next((x for x in re.split(r"(?<=[.!?])\s+", t0) if _SCHED_TERM.search(x)), None)
        if not sent or len(sent) > 220:
            continue
        t = sent.strip().rstrip(".")
        have = " ".join(tx(x) for x in sch)
        if _said(t, sch, 0.6):
            continue
        cite = own.get(re.sub(r"\W+", " ", t0.lower()).strip(), i)
        win = re.search(r"\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s*(?:-|–|to|through|thru)\s*(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b", t) \
            or re.search(r"(?i)\bstart\w*\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b.{0,30}\b(?:through|until|thru)\s+(\d{1,2}/\d{1,2}(?:/\d{2,4})?)", t)
        times = re.search(r"(?i)\d\s*(:\d{2})?\s*(am|pm)\b", t)
        if win:
            if re.search(r"(?i)work window: (\d|from|beginning|week)", have):
                continue
            line = f"Work Window: {win.group(1)} through {win.group(2)}"
        elif times and re.search(r"(?i)\b(mon|tue|wed|thu|fri|sat|sun)", t):
            if re.search(r"(?i)work window:.*\d\s*(:\d{2})?\s*(am|pm)", have):
                continue
            line = "Work Hours: " + t
        elif re.search(r"(?i)\b(start|begin)\w*\b.{0,40}\b(through|until|thru|continue)\b", t) and re.search(r"(?i)work window", have):
            continue
        else:
            line = "Scheduling Constraint: " + t
        sch.append({"text": line, "cites": [cite], "kind": "fact"})
        added += 1


def _roles_and_hours(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """Staffing by role and the visit length the facts give (010419: "one
    network technician and one help desk technician", "usually a full day of
    work"): the task-list lead names the roles; a full day is one 8-hour
    workday per technician."""
    roles = None
    for f in fs:
        m = re.search(r"(?i)\b(one|two|three|\d)\s+((?:[\w-]+\s+){0,2}?)(technician|tech|engineer)s?\s+and\s+(one|two|three|\d)\s+((?:[\w-]+\s+){0,2}?)(technician|tech|engineer)s?\b", f["text"])
        if m:
            roles = (m, f["id"])
            break
    num = lambda w: int(w) if w.isdigit() else ("one two three".split().index(w.lower()) + 1)
    if roles:
        m = roles[0]
        n1, n2 = num(m.group(1)), num(m.group(4))
        r1 = (m.group(2).strip() + " technician").strip() + ("s" if n1 > 1 else "")
        r2 = (m.group(5).strip() + " technician").strip() + ("s" if n2 > 1 else "")
        content["scope_lead"] = f"Provider will dispatch {_words(n1)} ({n1}) {r1} and {_words(n2)} ({n2}) {r2} to perform the following:"
        techs = n1 + n2
    else:
        nr = next((re.search(r"\((\d+)\)|\b(\d+)\b", str(x.get("text"))) for x in content.get("schedule") or []
                   if isinstance(x, dict) and re.match(r"(?i)\s*number of resources", str(x.get("text")))), None)
        techs = int(nr.group(1) or nr.group(2)) if nr else 0
    day = next((f for f in fs if re.search(r"(?i)\b(a |one )?full (work ?)?day\b|\bone[- ]day (visit|job|engagement)\b", f["text"])
                and not f["text"].rstrip().endswith("?")), None)
    sch = content.setdefault("schedule", [])
    # "two technicians for one business day" (010376 pass U: the staffing and
    # the duration sat only inside an assumption; Duration read TBD).
    if not roles:
        for f in fs:
            m2 = re.search(r"(?i)\b(one|two|three|four|\d)\s+(?:[\w-]+\s+)?(?:technicians?|techs?)\s+for\s+(?:one|a|1)\s+(?:full\s+)?"
                           r"(?:business |work |onsite )?day\b", f["text"])
            if m2 and not f["text"].rstrip().endswith("?"):
                w = m2.group(1).lower()
                techs = techs or (int(w) if w.isdigit() else "one two three four".split().index(w) + 1)
                day = day or f
                if not any(re.match(r"(?i)\s*number of resources", str(x.get("text"))) for x in sch if isinstance(x, dict)):
                    sch.append({"text": f"Number of Resources: {_words(techs)} ({techs}) technician{'s' if techs > 1 else ''}",
                                "cites": [f["id"]], "kind": "fact"})
                break
    if day and techs and not any(re.match(r"(?i)\s*(estimated duration|duration)", str(x.get("text"))) for x in sch if isinstance(x, dict)):
        sch.append({"text": f"Estimated Duration: one (1) onsite workday of eight (8) hours per technician; {_words(techs)} ({techs}) "
                            f"technicians, {_words(8 * techs)} ({8 * techs}) combined hours",
                    "cites": [day["id"]] + ([roles[1]] if roles else []), "kind": "fact"})


def _deliverables_hygiene(content: dict[str, Any]) -> None:
    """The report promises only work the SOW keeps: no outcome restating a
    line a rule or check dropped, and none the SOW's own terms exclude
    (010209: "removed legacy switch shipment for wipe and disposal" and
    "serial numbers for the benched equipment", both dropped from scope)."""
    # dropped_unsupported too: a sentence the check dropped from the scope is
    # not promised in the report (010147 pass U: "five network switches, two
    # Cisco 1U switches" counted the same switches twice).
    gone = [str(d.get("text")) for d in content.get("rule_decisions") or [] if d.get("value") == "drop"] + \
           [str(x.get("text")) for x in content.get("dropped") or [] if isinstance(x, dict)]
    unsupported = [str(x.get("text")) for x in content.get("dropped_unsupported") or [] if isinstance(x, dict)]
    # A line moved into the report is not gone (010083: tidy moved "Provide
    # before/after pictures and serial numbers" from the customer duties to
    # the report, and that "drop" then struck the serial-number outcome).
    report = [str(t) for d in content.get("deliverables") or [] if isinstance(d, dict) for t in d.get("description") or []]
    gone = [g for g in gone if not any(_said(g, [t], 0.8) for t in report)]
    terms = " ".join(str(x.get("text") if isinstance(x, dict) else x) for k in ("assumptions", "out_of_scope")
                     for x in content.get(k) or []).lower()
    # A place named for shipping ("the designated shipment area") is where
    # packed gear waits onsite, not shipping work (010115 pass U lost
    # "Packed equipment staged in the designated shipment area").
    excl = [(r"(?i)\bship(ment|ped|ping)?\b(?!\s+(area|zone|location|cage|room|dock))|\bfreight\b|\btransport", r"ship|freight|transport|logistic"),
            (r"(?i)\bdispos|\bwipe\b|data destruction|recycl", r"dispos|recycl|wipe|destruction"),
            (r"(?i)\bbench(ed|ing)?\b", r"bench|stag")]
    # A standard outcome this removes says so (standards_suppressed).
    try:
        std_out = {_standard_applies(o, [], [])[1] for o in purtera_standards(str(content.get("job_family") or "")).get("deliverable_outcomes") or []}
    except Exception:  # noqa: BLE001 - the record is a courtesy, never a failure
        std_out = set()
    why = content.setdefault("standards_suppressed", [])
    for d in content.get("deliverables") or []:
        if not isinstance(d, dict):
            continue
        keep = []
        for t in d.get("description") or []:
            t = str(t)
            if any(_said(t, [g], 0.6) for g in gone) or any(_said(t, [g], 0.8) and _said(g, [t], 0.8) for g in unsupported):
                if t in std_out:
                    why.append({"section": "deliverable_outcomes", "text": t, "reason": "deliverables: restates a dropped line"})
                continue
            hit = [pat for pat, ex in excl if re.search(pat, t) and re.search(ex, terms)]
            if hit:
                # Strip the excluded clause from a list sentence, or drop it.
                parts = [p_ for p_ in re.split(r",\s*(?:and\s+)?", t) if not any(re.search(h, p_) for h in hit)]
                t2 = ", ".join(p_.strip() for p_ in parts).strip(" ,.")
                if len(parts) >= 2:
                    t2 = ", ".join(p_.strip() for p_ in parts[:-1]) + ", and " + parts[-1].strip().rstrip(".")
                if len(t2.split()) < 3:
                    if t in std_out:
                        why.append({"section": "deliverable_outcomes", "text": t, "reason": "deliverables: the SOW's own terms exclude it"})
                    continue
                t = t2 + "."
            keep.append(t[:1].upper() + t[1:])
        d["description"] = keep


def _reconcile_dropped(content: dict[str, Any]) -> None:
    """content["dropped"] lists only what the SOW leaves out: a line a rule
    took back (a schedule line the check set aside) is no longer dropped."""
    shown = {(k, str(x.get("text"))) for k in LISTS for x in content.get(k) or [] if isinstance(x, dict)}
    content["dropped"] = [x for x in content.get("dropped") or [] if not (isinstance(x, dict) and (x.get("list"), str(x.get("text"))) in shown)]


def _strip_people(name: str, people: list[str]) -> str:
    """The project name without a person's name: the full name, or any first
    name before the person's surname, and a trailing "w/ <Name>"."""
    pn = re.sub(r"(?i)\s+(w/|with)\s+[A-Z][a-z]+\b.*$", "", name)
    for nm in people:
        parts = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", nm) if len(w) >= 3]
        if len(parts) >= 2:
            pn = re.sub(r"\s*[-–]?\s*\b(?:[A-Z][a-z]+\s+)?" + re.escape(parts[-1]) + r"\b", "", pn)
    pn = re.sub(r"\s{2,}", " ", pn).strip(" -–,")
    return pn if len(pn.split()) >= 2 else name


def _arrival_name_rule(content: dict[str, Any], fs: list[dict[str, Any]]) -> None:
    """A facility rule that the technician's name be sent ahead is written once,
    as Provider's duty, with its notice period (010115: 'Need Name provided 48
    Hours Prior'; the writer dropped it, then inverted it, then dropped it)."""
    arr = content.setdefault("arrival_protocol", [])
    if any(re.search(r"(?i)\bname\b", str(x.get("text") if isinstance(x, dict) else x)) for x in arr):
        return
    cands = [f for f in fs if re.search(r"(?i)\bname\b", f["text"]) and re.search(r"(?i)prior|in advance|ahead|before", f["text"])]
    cands.sort(key=lambda f: 0 if re.search(r"(?i)\d+\s*(hours?|hrs?|days?)", f["text"]) else 1)
    for f in cands:
        t = f["text"]
        if True:
            m = re.search(r"(?i)(\d+)\s*(hours?|hrs?|days?|business days?)", t)
            when = f"{m.group(1)} {m.group(2).lower().replace('hrs', 'hours').replace('hr', 'hour')}" if m else "in advance"
            arr.insert(0, {"text": f"Provider submits the technician's name {when} before arrival.", "cites": [f["id"]], "kind": "fact"})
            return


def _tidy(content: dict[str, Any], facts_say_permit: bool = False) -> None:
    """Deterministic placement the writer kept getting wrong (010115 round 12)."""
    # A visitor badge or parking pass is something the site hands over, not
    # equipment worked on.
    eq = content.get("equipment_in_scope") or []
    access = [x for x in eq if _ACCESS.search(str(x.get("text") if isinstance(x, dict) else x))]
    if access:
        content["equipment_in_scope"] = [x for x in eq if x not in access]
        names = " and ".join(str(x.get("text") if isinstance(x, dict) else x).strip(" .").lower() for x in access)
        cr = content.setdefault("customer_responsibilities", [])
        line = f"Provide the {names} at check-in."
        if not _said(line, cr):
            cr.append({"text": line, "cites": sum((x.get("cites") or [] for x in access if isinstance(x, dict)), []),
                       "kind": "fact"})
    # Internal accounting fields (cost center, segment, tax status) are not a
    # description of the work.
    content["overview"] = [x for x in content.get("overview") or []
                           if not _LEDGER.search(str(x.get("text") if isinstance(x, dict) else x))]
    # Exclusions read as sentences: a bare word another exclusion already
    # covers goes; the rest start with a capital and end with a period.
    oos = content.get("out_of_scope") or []
    kept = []
    for x in oos:
        t = str(x.get("text") if isinstance(x, dict) else x).strip()
        others = [y for y in oos if y is not x and len(_toks(y.get("text") if isinstance(y, dict) else y)) > 2]
        if len(t.split()) <= 2 and _said(t, others, 1.0):
            continue
        t = re.sub(r"(?i)^exclud(e|es|ed|ing)\s*:?\s+", "", t)
        t = t[:1].upper() + t[1:]
        if not t.endswith((".", "!", "?")):
            t += "."
        if isinstance(x, dict):
            x["text"] = t
        kept.append(x if isinstance(x, dict) else {"text": t, "cites": [], "kind": "method"})
    content["out_of_scope"] = kept
    tx = lambda x: str(x.get("text") if isinstance(x, dict) else x)
    # Said once: a scope step the arrival protocol states, an assumption a
    # customer responsibility states (010115 said "packing materials" four times).
    arr = [re.sub(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", "", tx(a), flags=re.I) for a in content.get("arrival_protocol") or []]
    content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and x.get("kind") != "heading"
                        and not x.get("standard")
                        and _said(re.sub(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", "", tx(x), flags=re.I), arr, 0.5))]
    cr = content.setdefault("customer_responsibilities", [])
    asm = []
    for x in content.get("assumptions") or []:
        t = re.sub(r"(?i)^assume(s|d)?\s+(that\s+)?", "", tx(x)).strip()
        if not (isinstance(x, dict) and x.get("standard")) and _ACCESS.search(t) \
                and re.search(r"(?i)\b(provid|issue|furnish|receive)", t):
            # The site handing over a badge or pass is the customer's duty.
            line = "Provide the visitor badge and parking pass at check-in."
            if not _said(line, cr, 0.5):
                cr.append({"text": line, "cites": x.get("cites") or [] if isinstance(x, dict) else [], "kind": "fact"})
            continue
        std = isinstance(x, dict) and x.get("standard")
        # A standard goes only for a line that says it back: a standard item
        # nothing else carries stays (010115 pass U: "Onsite parking is
        # available; any parking fee is reimbursable" went as a repeat).
        twin = [y for y in asm if _said(t, [y], 0.6) and (not std or _said(tx(y), [t], 0.6))]
        if not t or _LEDGER.search(t) or (not std and _said(t, cr, 0.5)) or twin:
            continue
        t = t[:1].upper() + t[1:]
        if isinstance(x, dict):
            x["text"] = t
        asm.append(x if isinstance(x, dict) else {"text": t, "cites": [], "kind": "method"})
    arr_items = content.get("arrival_protocol") or []
    window = any(tx(x).lower().startswith("work window") for x in content.get("schedule") or [])
    content["assumptions"] = [a for a in asm if (isinstance(a, dict) and a.get("standard"))
                              or not (_said(tx(a), arr_items, 0.5)
                                      or (window and re.search(r"(?i)monday|business hours|\d+:\d{2}\s*(am|pm)", tx(a))))]
    if any(re.search(r"(?i)powered down", tx(a)) for a in content.get("assumptions") or []):
        content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and not x.get("standard")
                            and re.search(r"(?i)\bpower(s|ing)? down\b", tx(x)))]
    # A standard, convention, matrix or document is information the customer
    # provides, not equipment (010209: "The RL standard naming convention for
    # equipment labels" under Equipment in Scope).
    info = [x for x in content.get("equipment_in_scope") or [] if isinstance(x, dict) and re.search(
        r"(?i)naming convention|\bstandards?\b(?! (power|cords?|cables?|mounts?))|\bmatrix\b|documentation|diagram|\binstructions\b|\bschedule\b|\bconvention\b", tx(x))]
    for x in info:
        content["equipment_in_scope"].remove(x)
        line = {**x, "text": "Provide " + re.sub(r"(?i)^\s*the\s+", "the ", tx(x)).rstrip(".") + "."}
        if not _said(tx(line), cr, 0.6):
            cr.append(line)
    form_label = re.compile(r"(?i)\bN/A\b|to be installed,? by category|no hardware is to be installed")
    # A described picture is not material (010012: "Abstract metal sculpture with
    # spiral and spherical elements", a leasing brochure photo, as customer material).
    pictured = re.compile(r"(?i)\b(sculpture|artwork|abstract|photograph|image of|picture of|logo|illustration|rendering)\b")
    for k in ("customer_materials", "provider_materials", "equipment_in_scope"):
        content[k] = [x for x in content.get(k) or [] if not pictured.search(tx(x))]
    for k in ("equipment_in_scope", "out_of_scope"):
        content[k] = [x for x in content.get(k) or [] if not form_label.search(tx(x))]
    # Contacts belong in the contacts block, never the overview (a phone in the
    # summary was read off a form label with no value, 010115 round 15).
    ov = content.get("overview") or []
    kept_ov = [x for x in ov if not re.search(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}|escalation|backup contact|"
                                               r"\b(main|primary|tech) contact\b", tx(x), re.I)]
    content["overview"] = kept_ov or ov  # an empty Project Summary is worse than a contact in it
    voiced = []
    for x in content["overview"]:
        t = tx(x).strip()
        if re.match(r"^[A-Z][\w /&-]{1,30}:\s", t) and len(content["overview"]) > 1:
            continue  # a "Label: value" line is form residue, not a summary sentence
        if re.match(r"(?i)(dismantle|provide|install|assist|box|pack|send|remove|mount|run|perform|replace|deploy|survey)\b", t):
            t = "Provider will " + t[:1].lower() + t[1:]
        t = re.sub(r"\b(\d+\)?|one|two|three)\s+techs?\b", lambda m: m.group(1) + " technician" + ("s" if m.group(0).endswith("s") else ""), t)
        if isinstance(x, dict):
            x["text"] = t
        voiced.append(x if isinstance(x, dict) else {"text": t, "cites": [], "kind": "fact"})
    fam_words = str(content.get("job_family") or "").replace("_", " ").strip()
    if fam_words:
        for x in voiced:
            if isinstance(x, dict):
                x["text"] = re.sub(r"(?i)\bthe " + re.escape(fam_words) + r"\b", "the work", x["text"])
                x["text"] = re.sub(r"(?i)\b" + re.escape(fam_words) + r"\b", "the work", x["text"])
    content["overview"] = voiced or content["overview"]
    # A permit assumption needs a fact behind it.
    content["assumptions"] = [x for x in content.get("assumptions") or []
                              if not (re.search(r"(?i)\bpermit", tx(x)) and not facts_say_permit
                                      and not (isinstance(x, dict) and x.get("standard")))
                              and not (not (isinstance(x, dict) and x.get("standard"))
                                       and re.match(r"(?i)\s*(limit|support|have|provide|ensure|use|send|keep|make)\b", tx(x)))]
    # A scope step that meets a named person repeats the standard meet step.
    if any(isinstance(x, dict) and x.get("standard") and tx(x).lower().startswith("meet the onsite contact")
           for x in content.get("scope") or []):
        content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and not x.get("standard")
                            and x.get("kind") != "heading" and re.match(r"(?i)\s*(arrive\b.*\band\s+)?meet\b", tx(x)))]
    # Schedule: only its labeled lines once start and end dates are there (the
    # form's raw "Timeline/Schedule: 8/8 and 8/7" contradicted them, 010115).
    labels = ("requested start date", "requested completion date", "arrival time", "work window",
              "number of resources", "hardware quantity", "pace", "work hours", "scheduling constraint")
    sched = content.get("schedule") or []
    # The site's hours are a labeled line, never dropped as prose (010082 pass
    # U: "Site hours were stated as Monday through Friday, 8:00am to 3:00pm"
    # went, and a call slot filled Work Hours).
    hours = re.compile(r"(?i)\b(mon(day)?|tue(sday)?|wed(nesday)?|thu(rsday)?|fri(day)?|sat(urday)?|sun(day)?)\b.{0,40}?"
                       r"\d{1,2}(:\d\d)?\s*(am|pm|a\.m\.|p\.m\.)?\s*(?:-|–|to)\s*\d{1,2}(:\d\d)?\s*(am|pm|a\.m\.|p\.m\.)")
    for x in sched:
        if isinstance(x, dict) and not tx(x).lower().startswith(labels) and not re.match(r"^[\w /-]{2,30}:", tx(x)):
            hm = hours.search(tx(x))
            if hm:
                x["text"] = "Work Hours: " + hm.group(0)
    if sum(1 for x in sched if tx(x).lower().startswith(labels[:2])) == 2:
        content["schedule"] = [x for x in sched if tx(x).lower().startswith(labels)]
    sched_dates = set(re.findall(r"\d{1,2}/\d{1,2}", " ".join(tx(x) for x in content.get("schedule") or [])))
    std_scope = [x for x in content.get("scope") or [] if isinstance(x, dict) and x.get("standard")]
    std_words = set().union(*[_toks(tx(x)) for x in std_scope]) if std_scope else set()
    std_verbs = {_stem(re.findall(r"[a-z]+", tx(x).lower())[0]) for x in std_scope if re.findall(r"[a-z]+", tx(x).lower())}
    def repeats(x: Any) -> bool:
        if not isinstance(x, dict) or x.get("standard") or x.get("kind") == "heading":
            return False
        t = tx(x)
        if set(re.findall(r"\d{1,2}/\d{1,2}", t)) & sched_dates and not re.search(r"(?i)\b\d+\s*(ft|feet|x|units?|pcs)\b", t):
            return True  # restates the schedule
        # A deal line with a count, size or model is the deal's own fact, never a
        # repeat (010217: "Install two (2) new 86 inch Philips displays in
        # training room 250 ..." dropped as the standard install step).
        if re.search(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten)\b|\b[A-Z][A-Za-z]+\b(?<!^)", t[1:]):
            return False
        first = next(iter(re.findall(r"[a-z]+", t.lower())), "")
        if x.get("cites"):
            # A cited deal line repeats only a standard step that says it on
            # its own, verb and object (000123: "Install the printer at each
            # location" went because the standard steps' words, pooled, held
            # "install", "printer" and "location" in three different lines).
            return bool(first) and any(re.findall(r"[a-z]+", tx(s).lower())[:1] == [first]
                                       and len(_toks(t) & _toks(tx(s))) >= 0.75 * max(1, len(_toks(t))) for s in std_scope) \
                or bool(content.get("provider_tools") and re.search(r"(?i)\btools?\b|screwdriver|wrench", t))
        if first and _stem(first) in std_verbs and len(_toks(t) & std_words) >= 0.6 * max(1, len(_toks(t))):
            return True  # the standard step with that verb says it
        if content.get("provider_tools") and re.search(r"(?i)\btools?\b|screwdriver|wrench", t):
            return True  # the tools list says it
        w = _toks(t)
        return bool(std_words) and not re.search(r"\d", t) and len(w) >= 3 and len(w & std_words) >= 0.55 * len(w)
    content["scope"] = [x for x in content.get("scope") or [] if not repeats(x)]
    # The site's own access duties, from the arrival protocol (the fact check
    # dropped the writer's badge line, 010115 round 16).
    arr_text = " ".join(tx(a) for a in content.get("arrival_protocol") or []).lower()
    cr = content.setdefault("customer_responsibilities", [])
    cited_arr = " ".join(tx(a) for a in content.get("arrival_protocol") or [] if isinstance(a, dict) and a.get("cites")).lower()
    if re.search(r"badge|parking pass", cited_arr) and not any(re.search(r"(?i)badge|parking pass", tx(c)) for c in cr):
        cr.append({"text": "Provide visitor badges and parking passes upon technician arrival.", "cites": [], "kind": "method"})
    if "security" in arr_text and not any(re.search(r"(?i)security", tx(c)) and re.search(r"(?i)access|coordinat", tx(c)) for c in cr):
        cr.append({"text": "Coordinate site access with security.", "cites": [], "kind": "method"})
    # A rule the arrival protocol gives Provider is not also a customer duty
    # (010115: "Provide name 48 hours prior" in both, from the same cites).
    arr_cites = {c for a in content.get("arrival_protocol") or [] if isinstance(a, dict) for c in a.get("cites") or []}
    cr[:] = [c for c in cr if not (isinstance(c, dict) and c.get("cites") and set(c["cites"]) <= arr_cites
                                   and re.search(r"(?i)\bname\b", tx(c)))]
    # Photos and serial numbers are the technician's deliverables, not a
    # customer duty (010083); and a duty that contradicts a standard
    # stage-onsite assumption goes ("identify a single site for disposal").
    # Only a line that is about the photos or serials (010319: "Provide required
    # documentation ... cable diagrams, patch schedules, and serial numbers" is
    # the customer's documentation, not the technician's report).
    photo = [c for c in cr if re.match(r"(?i)\s*((provide|capture|take|send|submit|post)\s+)?(the\s+|all\s+)?((before|after|completion)[/ \w-]{0,14}\s+)?(photo|picture|serial)", tx(c))
             and not (isinstance(c, dict) and c.get("standard"))]
    if photo:
        cr[:] = [c for c in cr if c not in photo]
        dl = content.setdefault("deliverables", [])
        if not dl:
            dl.append({"item": "Work Completion Report", "description": [], "format": "PDF"})
        if isinstance(dl[0], dict):
            dl[0].setdefault("description", []).extend(re.sub(r"(?i)^provide\s+", "", tx(c)).rstrip(".") + "." for c in photo)
    # What the technician does is not the customer's duty (010083: "Customer
    # remote resource will provide verbal validation at completion, and the
    # technician may leave once the posted completion time is reached").
    for c in cr:
        if isinstance(c, dict) and not c.get("standard"):
            t = re.sub(r"(?i)\s*(,\s*and|;|, then)\s+(the\s+)?(technicians?|techs?|provider)\b\s+(may|will|can|is|are|must|should)\b.*$", "", tx(c))
            if t != tx(c) and len(t.split()) >= 4:
                c["text"] = t.rstrip(" ,;.") + "."
    if any(re.search(r"(?i)left onsite|staged? onsite|stage the removed", tx(a)) for a in (content.get("assumptions") or []) + (content.get("scope") or [])):
        cr[:] = [c for c in cr if not re.search(r"(?i)site for disposal|ship (the )?removed|return (the )?removed", tx(c))]
    # Hardware Quantity belongs in the Schedule only where PurTera puts it
    # there (a decommission job); elsewhere the count lives in the scope
    # (010043, 010083, 010085: "Schedule: Hardware Quantity" flagged misplaced).
    if str(content.get("job_family") or "") != "decommission_packing":
        content["schedule"] = [x for x in content.get("schedule") or [] if not tx(x).lower().startswith("hardware quantity")]
    # The template already has "Ensure all customer-provided equipment is on site".
    cr[:] = [c for c in cr if (isinstance(c, dict) and c.get("standard"))
             or not re.search(r"(?i)customer[- ]provided equipment.{0,40}on ?site|equipment is on ?site and available", tx(c))]
    # The site, not Provider, hands the technician credentials, badges or passes
    # (010115: a passive "must be provided with credentials" read as Provider's).
    for k in ("arrival_protocol", "scope"):
        moved = [x for x in content.get(k) or [] if isinstance(x, dict) and re.match(
            r"(?i)\s*provider (provides|issues|will provide|supplies)\b.{0,40}\b(credentials?|badges?|passes?|access cards?)", tx(x))]
        for x in moved:
            content[k].remove(x)
            x["text"] = re.sub(r"(?i)^\s*provider (provides|issues|will provide|supplies)", "Provide", tx(x))
            if not _said(tx(x), cr, 0.5):
                cr.append(x)
    # Granting access is the site's; the work the access is for stays
    # Provider's (010279: "Provide POS access for terminal identification and
    # configuration tasks" as a Provider task, from the RFP's "This will
    # include POS access to complete specific tasks to IDing terminals").
    for x in [y for y in content.get("scope") or [] if isinstance(y, dict) and y.get("kind") != "heading" and not y.get("standard")]:
        mm = re.match(r"(?i)\s*provide\s+((?:[\w-]+\s+){0,3}(?:access|credentials|logins?))\s+(?:for|to)\s+(?:complete\s+|perform\s+)?(.{8,})$", tx(x))
        if not mm:
            continue
        i = content["scope"].index(x)
        content["scope"][i] = {**x, "text": "Perform " + mm.group(2).rstrip(".") + "."}
        line = {**x, "text": "Provide " + mm.group(1) + " for " + mm.group(2).rstrip(".") + "."}
        if not _said(tx(line), cr, 0.6):
            cr.append(line)
    # Standard 8-5 site hours repeat the template's Business Hours line.
    content["assumptions"] = [a for a in content.get("assumptions") or [] if (isinstance(a, dict) and a.get("standard"))
                              or not re.search(r"(?i)monday.{0,15}friday.{0,30}8:?0?0?\s*(am|a\.m\.).{0,10}5:?0?0?\s*(pm|p\.m\.)", tx(a))]
    # The template already asks for a point of contact; a standard that asks
    # for more (by phone during the visits, authority to approve a shutdown)
    # is not that line (000132: "Provide a Customer point of contact
    # available by phone during scheduled visits" never rendered).
    poc = "Customer to provide a point of contact for the project and onsite visit."
    content["customer_responsibilities"] = [c for c in cr if not (re.search(r"(?i)point of contact|onsite contact for", tx(c))
                                                                  and not (isinstance(c, dict) and c.get("standard") and not _said(tx(c), [poc], 0.75)))
                                            and not re.search(r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}", tx(c))]
    # The standards win a conflict: removed gear left onsite and return
    # logistics excluded, so no writer line packs and ships it (010209).
    std_all = " ".join(tx(x) for k in ("assumptions", "out_of_scope") for x in content.get(k) or []
                       if isinstance(x, dict) and x.get("standard")).lower()
    if re.search(r"left onsite|return logistics", std_all):
        ship = re.compile(r"(?i)return ship|ship(ping|ped)? (it |them |back|for return|to )|pack .{0,40}ship|shipping documentation|"
                          r"coordinate .{0,40}ship")
        for k in ("scope", "overview", "deliverables", "assumptions"):
            content[k] = [x for x in content.get(k) or [] if (isinstance(x, dict) and x.get("standard"))
                          or not ship.search(str(x.get("text") or x.get("item") or "") if isinstance(x, dict) else str(x))]
    # One "limited to" assumption: the standard one stands.
    asm = content.get("assumptions") or []
    if any(isinstance(a, dict) and a.get("standard") and "limited to" in tx(a).lower() for a in asm):
        content["assumptions"] = [a for a in asm if (isinstance(a, dict) and a.get("standard")) or "limited to" not in tx(a).lower()]
    scope_text = " ".join(tx(x) for k in ("scope", "equipment_in_scope") for x in content.get(k) or []).lower()
    excludes_scope = []
    for dev, words in (("router", r"switch|firewall|network device|router"), ("firewall", r"firewall"),
                       ("switch", r"\bswitch"), ("access point", r"access point|wireless"), ("camera", r"camera"),
                       ("cabl", r"cabl"), ("printer", r"printer"), ("tv|display|monitor", r"display|monitor|\btv\b")):
        if re.search(dev, scope_text):
            excludes_scope.append(words)
    content["out_of_scope"] = [x for x in content.get("out_of_scope") or []
                               if (isinstance(x, dict) and x.get("standard"))
                               or not (re.search(r"(?i)additional devices not listed", tx(x))
                                       or any(re.search(w, tx(x), re.I) and not re.search(r"(?i)\b(existing|other|new)\b.*\b(configur|programm)", tx(x))
                                              for w in excludes_scope))]
    std_oos = [x for x in content.get("out_of_scope") or [] if isinstance(x, dict) and x.get("standard")]
    if std_oos:
        oos_words = set().union(*[_toks(tx(x)) for x in std_oos])
        content["out_of_scope"] = [x for x in content.get("out_of_scope") or [] if (isinstance(x, dict) and x.get("standard"))
                                   or len(_toks(tx(x)) & oos_words) < 0.6 * max(1, len(_toks(tx(x))))]
    # A heading whose tasks all went (repeats of the arrival protocol) is empty.
    sc = content.get("scope") or []
    content["scope"] = [x for i, x in enumerate(sc)
                        if not (isinstance(x, dict) and x.get("kind") == "heading")
                        or (i + 1 < len(sc) and not (isinstance(sc[i + 1], dict) and sc[i + 1].get("kind") == "heading"))]
    fam = str(content.get("job_family") or "")
    oos_not = {"av_conference_room": r"(?i)network|switch|firewall|licens|portal|cloud",
               "decommission_packing": r"(?i)licens|portal|cloud|network redesign",
               "pos_install_refresh": r"(?i)network redesign|firewall|licens"}
    # A desk/monitor mounting job: no network tools, no lift/network/cloud exclusions (010373).
    if re.search(r"(?i)monitor arm|monitor mount|desk mount", " ".join(tx(x) for x in content.get("scope") or [])):
        content["provider_tools"] = [x for x in content.get("provider_tools") or []
                                     if not re.search(r"(?i)console|tester|troubleshoot|network|crimp|punch|toner", tx(x))]
        content["out_of_scope"] = [x for x in content.get("out_of_scope") or [] if (isinstance(x, dict) and x.get("standard"))
                                   or not re.search(r"(?i)lift|network|firewall|switch|cloud|licens|core drill|portal", tx(x))]
    if fam == "pos_install_refresh":  # 000123: uncited "basic troubleshooting" filler
        content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and not x.get("cites")
                            and not x.get("standard") and re.search(r"(?i)troubleshoot", tx(x)))]
    if fam in oos_not:
        content["out_of_scope"] = [x for x in content.get("out_of_scope") or [] if (isinstance(x, dict) and (x.get("standard") or x.get("cites")))
                                   or not re.search(oos_not[fam], tx(x))]
    if fam in _FAMILY_TOOLS_NOT:
        content["provider_tools"] = [x for x in content.get("provider_tools") or [] if not _FAMILY_TOOLS_NOT[fam].search(tx(x))]


def check(content: dict[str, Any], lines: dict[str, str]) -> dict[str, int]:
    """Drop what the facts do not support: a fact bullet whose cited lines do
    not state it, and a method bullet that carries a number or a name no fact
    gives. Returns how many of each were dropped."""
    pool = " ".join(lines.values()).lower()
    dropped = {"fact": 0, "method": 0, "uncited_fact": 0}
    to_verify: list[tuple[str, int, dict[str, Any]]] = []
    # Every drop says why (010279 pass T: the terminal overview line sat in
    # "dropped" with no reason anywhere).
    unsupported = content.setdefault("dropped_unsupported", [])
    for k in LISTS:
        keep = []
        for i, x in enumerate(content.get(k) or []):
            if x["kind"] == "heading":
                if k == "scope" and len(x["text"].split()) <= 7:
                    keep.append(x)
                continue
            x["cites"] = [c for c in x["cites"] if c in lines]
            if not x["cites"] and k not in ("overview",):
                # An uncited line a given fact states is that fact's line:
                # cite it, never drop it (010326 pass U: "Provide two hundred
                # fifty-nine (259) Cisco Desk Phone 9851 units." beside the
                # quote line "Cisco Desk Phone 9851 ... | QTY: 259").
                dt = _digits(str(x["text"]))
                if re.search(r"(?<![\w-])\d+(?![\w-])", dt):  # a count, not a model number
                    src = next((i for i, ln in lines.items() if _stated_by(dt, _digits(ln))), None)
                    if src:
                        x["cites"], x["kind"] = [src], "fact"
            if x["kind"] == "fact" and not x["cites"]:
                # A deal claim with nothing behind it reads as method only if
                # it carries nothing deal-specific.
                if _unsupported_specifics(x["text"], pool):
                    dropped["uncited_fact"] += 1
                    unsupported.append({"list": k, **x, "reason": "uncited fact with a number or name no fact gives"})
                    continue
                x["kind"] = "method"
            if x["kind"] == "method" and (k in ("customer_materials", "provider_materials", "locations")
                                          or (k == "provider_tools" and _unsupported_specifics(x["text"], pool))):
                # Who supplies what is a deal fact, never a default.
                dropped["method"] += 1
                unsupported.append({"list": k, **x, "reason": "uncited supply, location or tool line"})
                continue
            if x["kind"] == "method" and _unsupported_specifics(x["text"], pool):
                dropped["method"] += 1
                unsupported.append({"list": k, **x, "reason": "method line with a number or name no fact gives"})
                continue
            keep.append(x)
        content[k] = keep
        # Supply items are checked by _party_check (who) and _grounded (what):
        # judged here too, a bare order line "does not say" the customer
        # supplies it and 010043's cameras were dropped outright.
        if k not in ("provider_materials", "customer_materials", "equipment_in_scope"):
            to_verify += [(k, i, x) for i, x in enumerate(keep) if x["kind"] == "fact"]
    ok = _verified([x for _, _, x in to_verify], lines)
    bad = {id(x) for _, _, x in to_verify} - {id(x) for x in ok}
    # A scope line rejected for one trailing clause keeps its main claim
    # (010217: Training Room 250's whole install line went over "after
    # confirming fit").
    for k, _i, x in to_verify:
        if id(x) not in bad or k not in ("scope", "assumptions"):
            continue
        head = re.split(r",?\s+(?:after|once|provided|if|when|so that|pending|subject to)\b|;\s*", str(x["text"]), maxsplit=1)[0].strip()
        cited = " ".join(lines.get(c, "") for c in x["cites"])
        if head != str(x["text"]).strip() and len(head.split()) >= 4 and _stated_by(head, cited):
            x["text"] = head.rstrip(".,;") + "."
            bad.discard(id(x))
            continue
        # Rejected only for a count its lines do not state: keep the step, drop the
        # count (010120: "Install two (2) suspended Neat camera devices" went whole).
        bare = re.sub(r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s*(\(\d+\)\s*)?(?=[a-z])", "", str(x["text"])).strip()
        bare = re.sub(r"\s{2,}", " ", bare)
        if k == "scope" and bare != str(x["text"]).strip() and len(bare.split()) >= 3 and _stated_by(bare, cited):
            x["text"] = bare[:1].upper() + bare[1:]
            bad.discard(id(x))
            continue
        # Cited to the wrong copy: another given line states it (010082 pass U:
        # the metal-studs assumption cited the call's small talk "It may have
        # metal studs." while the packet says "Walls may have metal studs").
        for cand in (str(x["text"]), head):
            if len(cand.split()) < 4:
                continue
            other = next((i for i, ln in lines.items() if i not in x["cites"] and _stated_by(cand, ln)), None)
            if other:
                x["cites"] = [other]
                if cand is head and head != str(x["text"]).strip():
                    x["text"] = head.rstrip(".,;") + "."
                bad.discard(id(x))
                break
    gone = content.setdefault("dropped", [])
    for k in LISTS:
        before = len(content[k])
        gone += [{"list": k, **x, "reason": "check: the cited lines do not state it"} for x in content[k] if id(x) in bad]
        content[k] = [x for x in content[k] if id(x) not in bad]
        dropped["fact"] += before - len(content[k])
    return dropped


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december")


def _date_stated(y: str, mo: str, d: str, pool: str) -> bool:
    """The pool states this date in some form: "August 16", "Aug 16", "8/16"."""
    m, dd = int(mo), int(d)
    if not (1 <= m <= 12):
        return False
    name = _MONTHS[m - 1]
    return bool(re.search(rf"\b({name}|{name[:3]})\.?\s+{dd}(st|nd|rd|th)?\b", pool)
                or re.search(rf"\b0?{m}[/-]0?{dd}\b", pool) or f"{y}-{mo}-{d}" in pool)


def _unsupported_specifics(text: str, pool: str) -> bool:
    """A number, or a capitalised name mid-sentence, that no fact line has."""
    # A date the writer normalised (010198's "Sunday, August 16" written as
    # "Requested Start Date: 2026-08-16") is the fact's own date.
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b",
                  lambda m: "" if _date_stated(m.group(1), m.group(2), m.group(3), pool) else m.group(0), text)
    # A count is its value, however it is written: "forty-eight (48)" is the
    # 48 a fact gives (010373 pass U: the summary's lead install sentence and
    # the counted equipment assumption went over "forty" and "eight").
    text = _digits(re.sub(r"\bone\s+(?=(of|or|end|side|another)\b)", "", text, flags=re.I))
    pool_nums = set(re.findall(r"\d+", _NUM_RUN.sub(lambda m: f" {_words_value(m.group(0))} ", pool)))
    for m in _NUM.finditer(text):
        tok = m.group(1).lower().rstrip(".,")
        if not tok or tok in pool:
            continue
        if tok.isdigit() and tok.lstrip("0") in pool_nums | {n.lstrip("0") for n in pool_nums}:
            continue
        return True
    words = re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][A-Za-z0-9&\-]{2,})\b", text.strip())
    for w in words:
        if w.lower() in _STOCK:
            continue
        if w.lower() not in pool:
            return True
    return False


def _verified(items: list[dict[str, Any]], lines: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start in range(0, len(items), 40):
        batch = items[start:start + 40]
        body = "\n".join(f"{i}. {x['text']}\n" + "\n".join(f"   [{c}] {lines.get(c, '')[:300]}" for c in x["cites"])
                         for i, x in enumerate(batch))
        got = _json_of(_complete(_VERIFY_PROMPT % body, sow_model(), 1200)) or {}
        okn = {int(i) for i in got.get("supported") or [] if str(i).isdigit()}
        for i, x in enumerate(batch):
            # A bullet whose content words its own cited lines state is kept even
            # when the judge says no: it deleted 010198's router install ("Install
            # a Ubiquiti router with cellular backup", cited to the request that
            # asks for it) and its service date, and 000123's 161-store count.
            cited = " ".join(lines.get(c, "") for c in x["cites"])
            explicit_absence = re.match(r"(?i)\s*[\w ]{3,30}:\s*not (provided|stated|specified)\b", x["text"])
            # An equipment-list row ("Printer", "Credit card reader (pin pad)")
            # is the thing to install: "Install the printer" states it (000123).
            short = [lines.get(c, "") for c in x["cites"] if 0 < len(lines.get(c, "").split()) <= 6]
            install_row = bool(short) and re.match(r"(?i)\s*(install|mount|connect|set up|place)\b", x["text"]) and all(
                {w for w in _toks(r) if len(w) > 2} <= _toks(x["text"]) for r in short) and len(short) == len(x["cites"]) \
                and set(re.findall(r"\d+", x["text"])) <= set(re.findall(r"\d+", cited))
            # "about 160" stated, 161 resolved from the roster (000123's count).
            approx = re.search(r"(?i)\b(about|approximately|approx\.?|around|roughly|~)\s*\d", cited) and _stated_by(
                re.sub(r"\d+", lambda m: next((n for n in re.findall(r"\d+", cited) if abs(int(n) - int(m.group(0))) <= 2), m.group(0)), x["text"]), cited)
            if i in okn or explicit_absence or install_row or approx or _stated_by(x["text"], cited):
                out.append(x)
            else:
                # Cited to the wrong line but stated by another: re-cite, do not
                # drop (010120's MS120 switch line cited the cleanup sentence).
                other = next((c for c, t in lines.items() if c not in x["cites"] and len(t) < 400 and _stated_by(x["text"], t)), None)
                if other:
                    x["cites"] = [other]
                    out.append(x)
    return out


def _stated_by(text: str, cited: str) -> bool:
    """Most of a bullet's content words, and every number in it, appear in its cited lines."""
    want = {w for w in _toks(text) if len(w) > 2} - {"provider", "will", "customer"}  # framing, not content
    have = _toks(cited)
    nums = set(re.findall(r"\d+", text))
    parts = set(re.findall(r"\b[A-Z]{1,6}\d{1,5}(?:-[A-Z0-9]{1,6})?\b", text))
    if parts and parts <= set(re.findall(r"\b[A-Z]{1,6}\d{1,5}(?:-[A-Z0-9]{1,6})?\b", cited)):
        return True  # a line built on the cited order line's part number (010043's CF83-E)
    return bool(want) and len(want & have) >= 0.7 * len(want) and nums <= set(re.findall(r"\d+", cited))


def enabled() -> bool:
    """``SOWSMITH_C3_SOW`` is on (dev): the compile writes the SOW content."""
    return os.environ.get("SOWSMITH_C3_SOW", "").strip().lower() in ("1", "true", "yes", "on")


_KV = re.compile(r"\s*([A-Za-z][A-Za-z #/]{1,24}?)\s*:\s*([^|]*)")


def location_table(envelope: dict[str, Any]) -> list[dict[str, str]]:
    """The deal's own location list, read off a table whose rows carry a City
    and a State: 010279's store list ("Store: 3 | Region: Southern | City:
    Charlotte | State: NC") and 000123's tab-separated BevMo roster. The type
    head is unsure of such rows one at a time (keep 0.27-0.69); as a table they
    are the job's sites, and the SOW's location count and exhibit come from
    them, not from a count some other conversation mentions."""
    by_doc: dict[str, list[dict[str, str]]] = {}
    headers: dict[str, list[str]] = {}
    for a in envelope.get("atoms") or []:
        t = str(a.get("text") or "")
        art = str(a.get("artifact_id") or "")
        row: dict[str, str] = {}
        if "|" in t and re.search(r"(?i)\bcity\s*:", t) and re.search(r"(?i)\bstate\s*:", t):
            for cell in t.split("|"):
                m = _KV.match(cell)
                if m:
                    row[m.group(1).strip().lower()] = m.group(2).strip()
        elif "\t" in t:
            cells = [c.strip() for c in t.split("\t")]
            low = [c.lower() for c in cells]
            if "city" in low and "state" in low:
                headers[art] = low
                # The header row can carry the first data row after it.
                continue
            if art in headers and len(cells) >= 4:
                # A header row can run on into the first data row ("... Zip Code
                # Counties 1 406 OAK_Walnut-Creek_406"): its leading names fit.
                head = headers[art][:len(cells)]
                if "city" in head and "state" in head:
                    row = dict(zip(head, cells))
        if row.get("city") and row.get("state"):
            by_doc.setdefault(art, []).append(row)
    # The document's own table, whole: the row atoms lose rows whose cells
    # repeat another row's (010279 pass T: stores 310 and 419 share Ashburn
    # and Virginia Beach with other stores; their row atoms were cut to
    # "Store: 419 |" and 111 of 113 stores reached the exhibit).
    for d in envelope.get("documents") or []:
        for pg in ((d.get("structured") or {}).get("pages") or []):
            for sec in pg.get("sections") or []:
                for b in sec.get("blocks") or []:
                    cols = [str(c).strip().lower() for c in b.get("columns") or []] if isinstance(b, dict) else []
                    if b.get("kind") != "table" or "city" not in cols or "state" not in cols:
                        continue
                    rows = [{str(k).strip().lower(): str(v or "").strip() for k, v in r.items()}
                            for r in b.get("rows") or [] if isinstance(r, dict)]
                    rows = [r for r in rows if r.get("city") and r.get("state")]
                    key = "doc:" + str(d.get("artifact_id") or d.get("filename") or "")
                    by_doc[key] = by_doc.get(key, []) + rows
    best = max(by_doc.values(), key=len, default=[])
    if len(best) < 3:
        return []
    out, seen = [], set()
    for r in best:
        name = r.get("store") or r.get("locationname") or r.get("location name") or r.get("site") or r.get("name") or ""
        street = r.get("address") or r.get("street") or ""
        key = (r.get("locid") or r.get("store #") or r.get("store number") or name).lower(), street.lower(), r["city"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": (f"Store {name}" if name.isdigit() else name), "address": street, "city": r["city"],
                    "state": r.get("state", ""), "postal_code": r.get("zip code") or r.get("zip") or ""})
    return out


_KEYCELL = re.compile(r"^\s*[A-Za-z][\w /#&-]{0,40}:\s*(.+)$")
_ADDR_TAIL = re.compile(r"^(?P<pre>.+?),?\s+(?P<st>[A-Z]{2})\s*(?P<zip>\d{5}(?:-\d{4})?)?\s*$")


def count_table(envelope: dict[str, Any], scope_text: str = "") -> list[dict[str, Any]]:
    """The deal's per-site device count list, read as a table: rows of "<site>
    [(address)] | <count>" or "<site> | <address> | <count>" in one document,
    with a TOTAL row the counts add up to (or a total the task list states).
    010326's "Physical Site / location | Phone Count" list is 30 rows totalling
    317; the location table and its exhibit came from a lossy site resolve
    (33 rows, duplicate spellings, two sites outside the list, no counts)."""
    by_doc: dict[str, dict[str, Any]] = {}
    for a in envelope.get("atoms") or []:
        t = str(a.get("text") or "")
        if "|" not in t or len(t) > 300:
            continue
        cells = [c.strip() for c in t.split("|")]
        cells = [(_KEYCELL.match(c).group(1).strip() if _KEYCELL.match(c) and not re.match(r"^\d", c) else c) for c in cells]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        doc = by_doc.setdefault(str(a.get("artifact_id") or ""), {"rows": {}, "total": None, "unit": ""})
        last, first = cells[-1], cells[0]
        if not re.fullmatch(r"\d{1,5}", last):
            head = re.search(r"(?i)\b([a-z]+)\s*(?:count|qty|quantity|#)\s*$", last)
            if head and len(cells) >= 2:
                doc["unit"] = head.group(1).lower()
                doc["site_header"] = doc.get("site_header") or bool(re.search(r"(?i)site|location|store|branch|facility|address", first))
            continue
        if re.match(r"(?i)\s*(grand )?total\b", first):
            doc["total"] = int(last)
            continue
        if not re.search(r"[A-Za-z]{3}", first) or re.fullmatch(r"\d+", first):
            continue
        key = re.sub(r"\W+", " ", re.sub(r"\(.*", "", first).lower()).strip()
        if m0 := re.match(r"(?i)\s*(?:physical site|site|location)\s*/?\s*(?:location)?\s*:\s*(.+)", t.split("|")[0]):
            unit_cell = t.split("|")[-1]
            hk = re.match(r"\s*([A-Za-z]+)\s*(?:count|qty|quantity)\s*:", unit_cell, re.I)
            if hk:
                doc["unit"] = hk.group(1).lower()
                doc["site_header"] = True
        doc["rows"].setdefault(key, (first, cells[1:-1], int(last)))
    best: list[tuple[str, list[str], int]] = []
    unit = ""
    for doc in by_doc.values():
        rows = list(doc["rows"].values())
        total = sum(r[2] for r in rows)
        # Rows of places: a site/location header, or addresses on most rows
        # (010264's clock status summary also adds up to its total).
        addressed = sum(1 for r in rows if re.search(r"\d{2,6}\s+[A-Za-z]", " ".join([r[0]] + list(r[1]))))
        if len(rows) < 6 or not (doc.get("site_header") or addressed >= 0.6 * len(rows)):
            continue
        ok = (doc["total"] is not None and doc["total"] == total) or (
            doc["total"] is None and re.search(r"\b%d\b" % total, scope_text))
        if ok and len(rows) > len(best):
            best, unit = rows, doc["unit"]
    if not best:
        return []
    def parse(addr: str) -> dict[str, str]:
        m1 = _ADDR_TAIL.match(addr.strip())
        if not m1:
            return {"street": addr.strip(), "city": "", "state": "", "postal_code": ""}
        pre = m1.group("pre")
        street, _, city = pre.rpartition(",")
        return {"street": street.strip() or pre.strip(), "city": city.strip() if street else "", "state": m1.group("st"),
                "postal_code": m1.group("zip") or ""}
    out = []
    for first, mid, n in best:
        mm = re.match(r"^(.*?)\s*\((.+)\)\s*$", first)
        name, addr = (mm.group(1).strip(), mm.group(2)) if mm else (first.strip(), "")
        if not addr and mid and re.search(r"\d", mid[0]) and not re.search(r"(?i)not provided|unknown|tbd", mid[0]):
            addr = mid[0]
        out.append({"name": name, **(parse(addr) if addr else {"street": "", "city": "", "state": "", "postal_code": ""}), "count": n})
    # A street written without the comma before its city ("911 Franklin Street
    # Manitowoc, WI") takes the city the other rows name.
    cities = {r["city"] for r in out if r["city"]}
    for r in out:
        if not r["city"] and r["street"]:
            for ci in sorted(cities, key=len, reverse=True):
                if r["street"].lower().endswith(" " + ci.lower()):
                    r["street"], r["city"] = r["street"][:-len(ci)].strip(" ,"), ci
                    break
    label = (unit.rstrip("s") + "s") if unit else "devices"
    for r in out:
        tail = ", ".join(x for x in (r["city"], " ".join(x for x in (r["state"], r["postal_code"]) if x)) if x)
        r["address"] = ", ".join(x for x in (r["street"], tail) if x)
        r["unit"] = label
        r["extra"] = {label: str(r["count"])}
    return out


def _reseller_streets(envelope: dict[str, Any]) -> set[str]:
    """Street addresses the account's own people sign with: a contact line or
    signature from the reseller's domain. 010083's only "site" was CDW's sales
    office, 310 Seven Springs Way, off the rep's contact card and signatures."""
    account = str((envelope.get("crm") or {}).get("account_name") or "").lower().split()
    dom = account[0] if account else ""
    out: set[str] = set()
    if not dom:
        return out
    for a in envelope.get("atoms") or []:
        v = a.get("structured") or {}
        t = str(a.get("text") or "")
        said = (v.get("said_by") or {}).get("company") or (a.get("locator") or {}).get("sender") or ""
        contact_line = bool(re.search(r"[\w.+-]+@([\w-]+)\.", t)) and dom in t.lower()
        # A signature carries contact detail; a bare quoted address said by a rep
        # is not one (010373: "530 Fifth Avenue" typed stakeholder filtered the site).
        signature = (a.get("atom_type") == "stakeholder" and dom in str(said).lower()
                     and bool(re.search(r"@|\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}|\b(manager|engineer|director|"
                                        r"account|specialist|executive|consultant)\b", t, re.I)))
        if contact_line or signature:
            m = re.search(r"\b\d{2,6}\s+[A-Za-z0-9 .'-]+?\b(?:way|st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|"
                          r"ln|lane|pkwy|parkway|ct|court|pl|place|hwy|highway)\b", t, re.I)
            if m:
                out.add(re.sub(r"\W+", " ", m.group(0).lower()).strip())
    # A street repeated in the suppressed signature chrome, or on a line with a
    # phone number, is the sender's card (010373: CDW's Rosemont office typed
    # physical_site 6 times, 18 copies in chrome, printed as the location).
    # Only the reseller's own cards: the customer signs with its site address
    # too (010043: Lane's Tyvola office, 47 signature copies, is the site).
    by_dom = lambda r: dom in str((r.get("locator") or {}).get("sender") if isinstance(r.get("locator"), dict) else r.get("locator") or "").lower()
    chrome = " ".join(str(c.get("text") or "") for c in envelope.get("suppressed_chrome") or [] if isinstance(c, dict) and by_dom(c)).lower()
    for a in envelope.get("atoms") or []:
        t = str(a.get("text") or "")
        m = re.search(r"\b\d{2,6}\s+[A-Za-z0-9 .'-]+?\b(?:way|st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|"
                      r"ln|lane|pkwy|parkway|ct|court|pl|place|hwy|highway)\b", t, re.I)
        if not m:
            continue
        street = m.group(0)
        phone = re.search(r"\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}", t)
        if (phone and by_dom(a)) or (chrome and chrome.count(street.lower()) >= 2):
            out.add(re.sub(r"\W+", " ", street.lower()).strip())
            # The glued form ("9754 5505 Pearl St") the address parser made of it.
            m2 = re.search(r"\d{2,6}\s+[A-Za-z].*", street)
            if m2:
                out.add(re.sub(r"\W+", " ", m2.group(0).lower()).strip())
    # Never an office: a street the sources call the customer's site.
    for a in envelope.get("atoms") or []:
        t = str(a.get("text") or "")
        if re.search(r"(?i)\b(customer|end[- ]user|client)'?s?\b.{0,25}\b(site|address|location)\b", t):
            low = re.sub(r"\W+", " ", t.lower())
            out = {o for o in out if o not in low}
    return out


_SHEET_SITES = re.compile(r"This sheet can supply ([\d,]+) distinct site addresses")


def sheet_site_count(envelope: dict[str, Any]) -> int | None:
    """How many places an inventory workbook puts the job at: the parser's
    "This sheet can supply N distinct site addresses" finding, from the sheet
    that matches its workbook. 010264's clock inventory has two: the 9000-series
    device list (797 addresses, the PM's "up to 795") and Sodexo's whole cost
    centre directory (65,902). The sheet sharing the workbook's model token
    ("9000_Clocks_Inventory" / "DeviceList-08-06_9000") is the one in scope; with
    no shared token, the smallest list is, since a directory is the largest."""
    names = {str(d.get("artifact_id") or d.get("id") or ""): str(d.get("filename") or "")
             for d in envelope.get("documents") or []}
    found: list[tuple[int, int, str]] = []
    for a in envelope.get("atoms") or []:
        st = a.get("structured") or {}
        m = _SHEET_SITES.search(str(a.get("text") or ""))
        if not (st.get("finding") == "supplies:site_address" or m):
            continue
        try:
            n = int(str(st.get("value") or (m.group(1) if m else "")).replace(",", ""))
        except ValueError:
            continue
        loc = ((a.get("source_refs") or [{}])[0] or {}).get("locator") or {}
        sheet = str(st.get("sheet") or loc.get("sheet") or a.get("sheet") or "")
        book = names.get(str(a.get("artifact_id") or ""), "")
        tokens = lambda x: {t for t in re.findall(r"[a-z]*\d+[a-z0-9]*|[a-z]{4,}", x.lower()) if not t.isdigit() or len(t) >= 3}
        shared = len(tokens(sheet) & tokens(re.sub(r"\.\w+$", "", book)) - {"list", "sheet", "review"})
        if n >= 3:
            found.append((shared, n, sheet))
    if not found:
        return None
    best = max(found, key=lambda f: (f[0], -f[1]))
    return best[1]


WRITER_RUNS = int(os.environ.get("SOWSMITH_C3_SOW_RUNS", "3"))


def _coverage_score(content: dict[str, Any], fs: list[dict[str, Any]]) -> int:
    """How many of the brief's kept facts a draft cites (also-mentioned, unsure
    and typed-only lines count for nothing), less what its checks dropped."""
    body = {f["id"] for f in fs if f["section"] not in ("also_kept", "unsure", "typed_only", "figure")}
    cited = {c for k in LISTS for x in content.get(k) or [] if isinstance(x, dict) for c in x.get("cites") or []}
    dropped = sum(int(v or 0) for v in (content.get("checks") or {}).values() if isinstance(v, (int, float)))
    # A draft missing a section a PM always reads loses outright to one that has it.
    missing = sum(1 for k in ("overview", "scope", "customer_responsibilities", "assumptions", "out_of_scope")
                  if not content.get(k))
    return len(cited & body) - dropped - 25 * missing


def _best_of(envelope: dict[str, Any], pm_handoff: dict[str, Any]) -> dict[str, Any]:
    """The writer drafts more than once and the draft that carries more of the
    brief wins. One run of two dropped 010043's camera install line, its
    under-200-ft assumption and its licensing exclusion that the other kept
    (grade B+ on one, B- on the next)."""
    from concurrent.futures import ThreadPoolExecutor

    if WRITER_RUNS <= 1:
        return write(envelope, pm_handoff)
    fs = facts(envelope)
    with ThreadPoolExecutor(max_workers=WRITER_RUNS) as pool:
        drafts = [f.result() for f in [pool.submit(write, envelope, pm_handoff) for _ in range(WRITER_RUNS)]]
    scored = sorted(((_coverage_score(d, fs), i) for i, d in enumerate(drafts)), reverse=True)
    best = drafts[scored[0][1]]
    best["drafts_scored"] = [sc for sc, _ in scored]
    return best


def _no_site_multiples(c: dict[str, Any], n_sites: int, atom_text: dict[str, str]) -> None:
    """A device count that is the site count is a site count multiplied into
    devices, unless a line states it (000123 pass T: "160 printers installed
    and tested" on 161 stores with several register lanes each): the line
    says the devices are done at each location instead."""
    pool = " ".join(atom_text.values()).lower()
    site_word = re.compile(r"(?i)^(customer |total |store |site )?(locations?|stores?|sites?|restaurants?|branches|facilities)\b")
    def fix(t: str) -> str:
        mm = re.match(r"^\s*(?:all\s+)?(\d{2,4})\s*(?:\([a-z -]+\)\s*)?(?:x\s+)?(.+)$", t, re.I) or \
            re.match(r"^\s*(?:[a-z -]+)\s*\((\d{2,4})\)\s+(.+)$", t, re.I)
        if not mm:
            return t
        n, rest = int(mm.group(1)), mm.group(2).strip()
        if abs(n - n_sites) > max(2, 0.1 * n_sites) or site_word.match(rest):
            return t
        noun = (re.findall(r"[a-z]+", rest.lower()) or [""])[0]
        if noun and re.search(r"\b%d\s+(?:[a-z-]+\s+){0,2}%s" % (n, re.escape(noun[:5])), pool):
            return t  # the deal states that device count
        rest = rest.rstrip(".")
        return rest[:1].upper() + rest[1:] + " at each location."
    for d in c.get("deliverables") or []:
        if isinstance(d, dict):
            d["description"] = [fix(str(t)) for t in d.get("description") or []]


def for_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    """The SOW content as the envelope carries it: the writer's lists, the
    partner's requester, the CRM record and the sites."""
    sites = [{"address": s.get("address") or s.get("display_name") or "", "city": s.get("city") or "",
              "state": s.get("state") or "", "postal_code": s.get("postal_code") or s.get("zip") or "",
              "name": s.get("display_name") or ""}
             for s in (envelope.get("site_readiness") or {}).get("sites") or []]
    offices = _reseller_streets(envelope)
    sites = [x for x in sites if str(x.get("address") or "").strip() or str(x.get("name") or "").strip()]  # 010116: one blank roster row
    sites = [x for x in sites if not any(o in re.sub(r"\W+", " ", x["address"].lower()) for o in offices)]
    if not sites:
        # Every resolved site was an office: the address a message names as the
        # customer's site is the site (010373: "The customer's site address will
        # be 530 Fifth Avenue, New York, NY 10036").
        for a in envelope.get("atoms") or []:
            m = re.search(r"(?i)\b(?:customer|end[- ]user|client)'?s?\b.{0,25}\b(?:site|address|location)\b.{0,20}?"
                          r"(\d{2,6}\s+[^,\n]+),\s*([A-Za-z .]+),\s*([A-Z]{2})\s+(\d{5})", str(a.get("text") or ""))
            if m:
                street, city, st, z = (g.strip() for g in m.groups())
                sites = [{"address": street, "city": city, "state": st, "postal_code": z,
                          "name": f"{street}, {city}, {st} {z}"}]
                break
    if not sites:
        # A form or sheet row naming the location (010116: "Location | 1221
        # Brickell Ave, Miami, Florida", no ZIP, no site row).
        doc_lines = [ln for d in envelope.get("documents") or [] for pg in ((d.get("structured") or {}).get("pages") or [])
                     for sec in pg.get("sections") or [] for b in sec.get("blocks") or [] for ln in str(b.get("text") or "").splitlines()]
        for a in list(envelope.get("atoms") or []) + [{"text": ln} for ln in doc_lines]:
            m = re.match(r"(?i)\s*(?:location|site address|address|site location|install location)\s*[|:]\s*"
                         r"(\d{2,6}\s+[^,|\n]+),\s*([A-Za-z .]+?),\s*([A-Za-z .]+?)(?:\s+(\d{5}))?\s*$", str(a.get("text") or ""))
            if m:
                street, city, st, z = (str(g or "").strip() for g in m.groups())
                sites = [{"address": street, "city": city, "state": st, "postal_code": z,
                          "name": ", ".join(p for p in (street, city, (st + " " + z).strip()) if p)}]
                break
    # Roster rows named by a trunk or circuit id ("SIP-6522-0134...") are not
    # sites; the writer's own location list with names and addresses is
    # (010326: 18 garbled rows against 30 clean locations).
    if any(re.match(r"(?i)\s*(sip|ckt|circuit|trunk)[-_ ]?\d", str(x.get("name") or "")) for x in sites):
        sites = [x for x in sites if not re.match(r"(?i)\s*(sip|ckt|circuit|trunk)[-_ ]?\d", str(x.get("name") or ""))]
    # On a many-site deal a signature address is a head office, not a site
    # (010209: Ralph Lauren's Nutley, NJ signature beside "30 locations").
    stated0 = int(((envelope.get("site_count_reconciliation") or {}).get("stated")) or 0)
    if stated0 >= 5 and sites:
        chrome_all = " ".join(str(c.get("text") or "") for c in envelope.get("suppressed_chrome") or [] if isinstance(c, dict)).lower()
        sites = [x for x in sites if not (str(x.get("address") or "").strip()
                 and chrome_all.count(re.sub(r"\s+", " ", str(x.get("address")).split(",")[0].lower()).strip()) >= 2)]
    table = location_table(envelope)
    if len(table) > len(sites):
        sites = table
    from_sheet = None if table else sheet_site_count(envelope)
    c = _best_of(envelope, {"sites": sites, "site_count": from_sheet})
    rec = envelope.get("site_count_reconciliation") or {}
    stated = rec.get("stated")
    if from_sheet and from_sheet > len(sites):
        stated = from_sheet
    # A roster that resolves more sites than a count some message stated is the
    # count (000123: the note said 160, the store list holds 161).
    # Only addressed rows count (010082: twelve address-less placeholders from
    # a set-aside SD-WAN conversation raised a one-site job to 13).
    addressed = [x for x in sites if re.search(r"\d", str(x.get("address") or ""))]
    if len(addressed) >= 10 and len(addressed) > int(stated or 0):
        stated = len(addressed)
    # Not the rows of the deal's own location table, which carry a store and a
    # city but no street: 010279's 111 store rows were cut to the six sites a
    # set-aside SmartHands call mentioned.
    if len(sites) >= 10 and len(addressed) < 0.5 * len(sites) and not (table and sites is table):
        sites = addressed or sites[:max(1, int(stated or 1))]
    if table:
        # A count the documents state that the table nearly bears out (010279:
        # "113" stores, 98 rows read whole) beats both the rows and a count
        # some other conversation gave.
        seen = [int(k) for k in (rec.get("all_counts_seen") or {}) if str(k).isdigit()]
        near = [n for n in seen if len(table) <= n <= len(table) * 1.4]
        stated = max(near) if near else len(table)
    c["site_summary"] = {"stated": stated, "resolved": len(sites), "from_table": bool(table)}
    c["requested_by"] = requested_by(envelope)
    # The end customer the documents name outranks the CRM account (010171:
    # "A different customer, APWireless Infrastructure Partners LLC"; the SOW said SHI).
    for a in envelope.get("atoms") or []:
        m = re.search(r"(?i)\b(?:end[- ]customer|different customer|the customer is|for (?:our|the) (?:end )?customer)[,:]?\s+"
                      r"([A-Z][\w&.,' -]{2,60}?(?:LLC|L\.L\.C\.|Inc\.?|Corp(?:oration)?\.?|Partners|Company|Co\.|Group|Holdings))\b",
                      str(a.get("text") or ""))
        if m:
            c["end_customer"] = m.group(1).strip(" ,")
            break
    else:
        # A reseller's deal names its end customer as "the X Team" / "<person>
        # from X" (010171: "answers from the APWireless Team", "Ben from APWireless").
        acct = str((envelope.get("crm") or {}).get("account_name") or "")
        ec = str(c.get("end_customer") or "").strip()
        plain = " ".join(re.sub(r"\S+@\S+|https?://\S+", " ", str(a.get("text") or "")) for a in envelope.get("atoms") or []).lower()
        only_domain = bool(ec) and ec.lower() not in plain  # 010171: "Radius Global" only as radiusglobal.com
        if re.search(r"(?i)\b(cdw|shi|insight|presidio|connection|computacenter|paragon)\b", acct) and \
                (not ec or ec.lower() == acct.strip().lower() or only_domain):
            seen: dict[str, int] = {}
            for a in envelope.get("atoms") or []:
                for m2 in re.finditer(r"\b(?:the\s+)?([A-Z][A-Za-z0-9&]{2,30})\s+[Tt]eam\b|\b[A-Z][a-z]+ from ([A-Z][A-Za-z0-9&]{2,30})\b",
                                      str(a.get("text") or "")):
                    org = m2.group(1) or m2.group(2)
                    if not re.search(r"(?i)^(cdw|shi|purtera|insight|presidio|the|our|your|this|sales|delivery|project|support|it)$", org):
                        seen[org] = seen.get(org, 0) + 1
            if seen:
                c["end_customer"] = max(seen, key=seen.get)
    # A count near the resolved site count is that count (000123: the roster
    # holds 161 stores; a note said "about 160" and the scope printed 160).
    n_sites = int(c["site_summary"].get("stated") or 0) or len(sites)
    if n_sites >= 10:
        # Never a count the line's own cited facts state (010083: "Apx 83
        # sites will be singles" snapped to the 92-site total).
        atom_text = {str(a.get("id")): str(a.get("text") or "") for a in envelope.get("atoms") or []}
        stated_here: set[int] = set()
        def _snap(m: re.Match) -> str:
            n = int(m.group(1))
            return (str(n_sites) if n != n_sites and n not in stated_here and abs(n - n_sites) <= max(2, 0.1 * n_sites) else m.group(1)) + m.group(2)
        for k in ("overview", "scope", "schedule", "assumptions", "customer_responsibilities"):
            for x in c.get(k) or []:
                if isinstance(x, dict):
                    stated_here.clear()
                    stated_here.update(int(n) for cc in x.get("cites") or [] for n in re.findall(r"(?<![\w.])(\d{2,4})(?![\w.])", atom_text.get(str(cc), "")))
                    if x.get("cites") and re.search(r"\(\d{2,4}\)", str(x.get("text"))) and any(
                            int(n) in stated_here and int(n) != n_sites for n in re.findall(r"\((\d{2,4})\)", str(x.get("text")))):
                        continue
                    x["text"] = re.sub(r"\b(\d{2,4})(\)?[\s-]+(?:\w+\s+)?(?:locations?|stores?|sites?|restaurants?|branches))", _snap, str(x["text"]))
                    # "160 (one hundred sixty) Customer locations" (000123).
                    x["text"] = re.sub(r"(?i)\b(\d{2,4})\s*\((?:[a-z -]+)\)(\s+(?:\w+\s+){0,2}(?:locations?|stores?|sites?|restaurants?|branches))",
                                       lambda m: (f"{n_sites} ({_words(n_sites)})" if int(m.group(1)) != n_sites and abs(int(m.group(1)) - n_sites) <= max(2, 0.1 * n_sites)
                                                  else m.group(0)[:m.start(2) - m.start(0)]) + m.group(2), x["text"])
                    # The spelled number before "(161)" says the same count.
                    x["text"] = re.sub(r"(?i)\b(" + _NUMW + r"(?:[\s-]+" + _NUMW + r"){0,4})\s*\((\d{2,4})\)",
                                       lambda m: (_words(int(m.group(2))) + f" ({m.group(2)})")
                                       if int(m.group(2)) == n_sites and _words_value(m.group(1)) not in (None, n_sites) else m.group(0),
                                       x["text"])
        for d in c.get("deliverables") or []:
            if isinstance(d, dict):
                d["description"] = [re.sub(r"(?i)\b(\d{2,4})\s*\((?:[a-z -]+)\)(\s+(?:\w+\s+){0,2}(?:locations?|stores?|sites?))",
                                           lambda m: (f"{n_sites} ({_words(n_sites)})" if int(m.group(1)) != n_sites and abs(int(m.group(1)) - n_sites) <= max(2, 0.1 * n_sites)
                                                      else m.group(0)[:m.start(2) - m.start(0)]) + m.group(2),
                                           re.sub(r"\b(\d{2,4})([\s-]+(?:\w+\s+)?(?:locations?|stores?|sites?))", _snap, str(t))) for t in d.get("description") or []]
        _no_site_multiples(c, n_sites, atom_text)
        # A many-site job states its site count in the summary (010279: no
        # "113" anywhere in the SOW, which read as a six-store job).
        ov = c.setdefault("overview", [])
        # Not beside a summary that already counts the locations otherwise,
        # nor pointing at a table the SOW does not have (010264 pass U: "1,400
        # U.S. locations" and "797 Customer locations, listed in the location
        # table").
        counted_elsewhere = any(re.search(r"(?i)\b\d[\d,]*\)?\s+(?:[\w.]+\s+){0,2}(locations|sites|stores)\b",
                                          _digits(str(x.get("text") if isinstance(x, dict) else x))) for x in ov)
        has_table = bool(table) or len(c.get("sites") or []) >= n_sites * 0.5 or bool(
            count_table(envelope, " ".join(str(x.get("text")) for x in c.get("scope") or [] if isinstance(x, dict))))
        if not counted_elsewhere and has_table and \
                not any(re.search(r"\b%d\b" % n_sites, str(x.get("text") if isinstance(x, dict) else x)) for x in ov):
            ov.insert(min(1, len(ov)), {"text": f"The work covers {_words(n_sites)} ({n_sites}) Customer locations, listed in the "
                                                f"location table.", "cites": [], "kind": "fact"})
    c["crm"] = {k: (envelope.get("crm") or {}).get(k) for k in ("deal_name", "account_name")}
    c["sites"] = [x for x in _sites(envelope, {"sites": sites}) if (x.get("address") or x.get("name"))]
    # The deal's own per-site count list, when one adds up, is the site table
    # and the site count, with each site's device count (010326: 30 rows, 317).
    counted = count_table(envelope, " ".join(str(x.get("text")) for x in c.get("scope") or [] if isinstance(x, dict)))
    if counted:
        c["sites"] = counted
        c["site_summary"] = {**c.get("site_summary", {}), "stated": len(counted), "resolved": len(counted), "from_table": True,
                             "unit": counted[0]["unit"], "unit_total": sum(r["count"] for r in counted)}
    # On a job of a few sites the writer's own cited locations are the table:
    # the roster kept SHI's Austin office (a signature) on 010376 and lost
    # 010147's Denver origin, while the writer's locations had both right.
    # A writer location list of "Name (street, City, ST 12345) - N phones" rows
    # longer than the roster is the site table (010326: 30 clean rows, 18
    # roster rows with trunk ids for names).
    parsed = []
    for x in c.get("locations") or []:
        t = str(x.get("text") if isinstance(x, dict) else x)
        m = re.match(r"(?i)\s*(.+?)\s*\((\d{1,6}\s+[^,()]+?\b(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|blvd|boulevard|way|court|ct|place|pl|"
                     r"parkway|pkwy|highway|hwy|circle|cir|trail|terrace)\.?)\s*,?\s+([A-Za-z .]+?),\s*([A-Z]{2})\s+(\d{5})\)", t) or \
            re.match(r"\s*(.+?)\s*\((\d{1,6}\s+[^,()]+?),\s*([A-Za-z .]+?),\s*([A-Z]{2})\s+(\d{5})\)", t)
        if m:
            name, street, city, st, z = (g.strip() for g in m.groups())
            parsed.append({"name": name, "address": f"{street}, {city}, {st} {z}", "street": street, "city": city, "state": st, "postal_code": z})
    if not counted and len(parsed) >= 6 and len(parsed) > len(c["sites"]) and len(parsed) >= 0.4 * len(c.get("locations") or []):
        # Name-only rows ("Lakefront Substation.") are sites too (010326).
        named = [{"name": str(x.get("text") if isinstance(x, dict) else x).strip().rstrip(".").split(" — ")[0], "address": "",
                  "street": "", "city": "", "state": "", "postal_code": ""}
                 for x in c.get("locations") or []
                 if not re.search(r"\d{5}", str(x.get("text") if isinstance(x, dict) else x))
                 and 1 <= len(str(x.get("text") if isinstance(x, dict) else x).split()) <= 6]
        c["sites"] = parsed + [n for n in named if n["name"] and not any(n["name"].lower() == p_["name"].lower() for p_ in parsed)]
    locs = [str(x.get("text") or "") for x in c.get("locations") or [] if isinstance(x, dict) and x.get("cites")]
    if locs and len(c["sites"]) <= 5 and not table:
        rows = []
        roster = list(c["sites"])
        for t in locs:
            place, _, street = t.partition("|")
            city, _, st = place.strip().partition(",")
            full = street.strip() if place.strip().lower() in street.lower() else ", ".join(
                x for x in (street.strip(), place.strip()) if x)
            # The roster's record of the same street keeps its name and ZIP: the
            # cited line alone blanked both (010115 "Delta Admin", 30309).
            place = re.sub(r"(?i)^\s*specific location\(?s?\)?\s*[:|]\s*", "", place).strip(" .")
            if not re.search(r"\d", f"{place} {street}") and len(roster) == 1 and roster[0].get("address"):
                # A named place inside the one addressed site (010115 "Sodexo Dining CAFE"):
                # it names the site; the address stays the roster's.
                r0 = roster[0]
                rows.append({"name": place, "city": r0.get("city") or "", "address": r0.get("address") or "",
                             "street": r0.get("street") or r0.get("address") or "", "state": r0.get("state") or "",
                             "postal_code": r0.get("postal_code") or "", "_named": True})
                continue
            zipped = [r for r in roster if re.search(r"\b\d{5}\b", f"{r.get('address')} {r.get('postal_code')}")]
            # Not a "City, ST" line: that is a real place (010147 pass U: the
            # Denver origin "Denver, CO | 1875 Lawrence Street, Suite 750" became
            # a second copy of the Lafayette destination).
            if len(roster) == 1 and zipped and not re.search(r"\b\d{5}\b", t) and not re.search(r"\b[A-Za-z .]+,\s*[A-Z]{2}\b", t) and \
                    not any(w in str(zipped[0].get("address") or "").lower() for w in re.findall(r"[a-z]{4,}", (street or place).lower())):
                # A cited place with no ZIP that the one fully addressed roster site does
                # not name is a caption, not the site (010012: "1 Landmark Square, office
                # space" from a leasing brochure over 1 Lincoln St, Boston 02111).
                rows.append(dict(zipped[0]))
                continue
            key = " ".join(re.sub(r"\W+", " ", (street or place).lower()).split()[:3])
            same = next((r for r in roster if key and key in re.sub(r"\W+", " ", str(r.get("address") or "").lower())), {})
            # The roster's display name carries the address after a dash ("Delta Admin — 1030 Delta
            # Blvd ..."), which printed the address twice in the Location cell.
            same = dict(same, name=re.split(r"\s+[—–-]\s+", str(same.get("name") or ""), maxsplit=1)[0]) if same else {}
            zp = re.search(r"\b\d{5}(?:-\d{4})?\b", full or "")
            rows.append({"name": same.get("name") or "", "city": city.strip(), "address": full or place.strip(),
                         "street": street.strip(), "state": st.strip(),
                         "postal_code": same.get("postal_code") or (zp.group(0) if zp else "")})
        # A roster site the writer did not cite is still a site (000132: Delphos,
        # OH was parsed with its site id and left out of the six-site list).
        # Only on a multi-site list: beside a single cited site, the one extra
        # roster row was a reseller rep's signature (010373: Rosemont, IL).
        for r0 in (roster if len(rows) >= 2 or len(roster) >= 3 else []):
            key = " ".join(re.sub(r"\W+", " ", str(r0.get("address") or r0.get("name") or "").lower()).split()[:3])
            if key and not any(key in re.sub(r"\W+", " ", f"{r.get('address')} {r.get('name')}".lower()) for r in rows):
                rows.append(dict(r0))
        # One row per address: a named place inside the site and the roster row
        # for the same street are one location (010115 printed both).
        by_addr: dict[str, dict] = {}
        for r in rows:
            k = " ".join(re.sub(r"\W+", " ", str(r.get("address") or r.get("name") or "").lower()).split()[:4])
            if k in by_addr:
                if r.get("_named") and not by_addr[k].get("_named"):
                    by_addr[k] = r
            else:
                by_addr[k] = r
        rows = [{kk: vv for kk, vv in r.items() if kk != "_named"} for r in by_addr.values()]
        # A move's two sites say which is which ("from Denver to Lafayette").
        ov = " ".join(str(x.get("text") if isinstance(x, dict) else x) for x in c.get("overview") or [])
        mv = re.search(r"\bfrom\s+([A-Z][A-Za-z .]+?)\s+to\s+([A-Z][A-Za-z .]+?)\b(?:[.,;]|$|\s)", ov)
        if mv and len(rows) == 2:
            for r in rows:
                city = str(r.get("city") or "").strip().lower()
                if city and city == mv.group(1).strip().lower():
                    r["name"] = "Origin"
                elif city and city == mv.group(2).strip().lower():
                    r["name"] = "Destination"
        c["sites"] = rows
    # The project is named as the deal is: PurTera's v1 titles are the CRM deal
    # name without its number and the reseller ("010115 - CDW Sodexo Decomm
    # Delta Atlanta" -> "Sodexo Decomm Delta Atlanta"); the writer coined its own.
    dn = str((envelope.get("crm") or {}).get("deal_name") or "").strip()
    dn = re.sub(r"^\s*\d{5,6}\s*[-–:]?\s*", "", dn)
    dn = dn.split("|")[0].strip()  # 010419: "... | IT Transition & Onboarding Support | Wilmington, NC"
    dn = re.sub(r"(?i)\s*[-–]?\s*\b(cdw|shi|insight|presidio|computacenter)\s*$", "", dn).strip()  # 010147 "... Move CDW"
    dn = re.sub(r"(?i)^(cdw|shi|insight|presidio|computacenter|paragon micro)\s*[-–:]?\s*", "", dn).strip(" -–")
    dn = re.sub(r"(?i)\s+w/\s*\w+.*$|\s+with\s+[A-Z][a-z]+$", "", dn).strip()  # "161 Site POS install w/ Ahmed"
    dn = re.sub(r"\s+-(\d+)\b", r" \1", dn)  # "NYC -48 monitor mounting"
    rq = str((c.get("requested_by") or {}).get("name") or "").strip()
    # The rep named at the end of the deal name, after the reseller, is who asked
    # (010012: "CFGI - Conference Room Install CDW Frank Lapp", Requested By TBD).
    m_rep = re.search(r"\s+(?i:cdw|shi|insight|presidio|computacenter)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z'-]+){1,2})\s*$", dn)
    if m_rep:
        if not rq:
            c["requested_by"] = {**(c.get("requested_by") or {}), "name": m_rep.group(1)}
        dn = dn[:m_rep.start()].strip(" -–")
    # Any deal person's name trailing the deal name (010209: "Ralph Lauren
    # Switch Replacement Dan Tighe").
    for pp in (envelope.get("deal_roster") or {}).get("people") or []:
        nm = str(pp.get("name") or "").strip()
        parts = nm.split()
        if len(parts) >= 2:
            short = f"{parts[0]} {parts[-1]}"
            for cand in {nm, short}:
                if re.search(r"\s+" + re.escape(cand) + r"\s*$", dn, re.I):
                    dn = re.sub(r"\s*[-–]?\s*" + re.escape(cand) + r"\s*$", "", dn, flags=re.I).strip(" -–")
    if rq and rq.lower() in dn.lower():
        dn = re.sub(r"\s*[-–]?\s*" + re.escape(rq) + r"\s*", " ", dn, flags=re.I).strip(" -–")  # "... Dan Tighe"
    # Unless the CRM title names work this SOW gives the customer or excludes:
    # 010326's deal was titled "Racking/Wall mount" after the call that gave
    # racking to the customer; the job is a phone deployment, as the writer said.
    writer_name = str(c.get("project_name") or "").strip()
    if len(dn.split()) >= 2 and not (_name_contradicted(dn, c) and writer_name and not _name_contradicted(writer_name, c)):
        c["project_name"] = dn
    # A person's name is never part of the project name, in any form
    # (010209: requester "Daniel Tighe" written "Dan Tighe"; 000123 "w/ Ahmed").
    c["project_name"] = _strip_people(str(c.get("project_name") or ""),
                                      [str((c.get("requested_by") or {}).get("name") or "")]
                                      + [str(pp.get("name") or "") for pp in (envelope.get("deal_roster") or {}).get("people") or []])
    # On a one-address job, a place the request names inside it ("Specific
    # Location(s) | Sodexo Dining CAFE") is where the work is: it names the site.
    if len(c["sites"]) == 1:
        named = []
        for a in envelope.get("atoms") or []:
            t = str(a.get("raw_text") or a.get("text") or "")
            if str(a.get("atom_type") or "") == "physical_site" and "|" in t and not re.search(r"\d", t):
                place = t.rsplit("|", 1)[1].strip()
                if 2 <= len(place.split()) <= 6 and place not in named:
                    named.append(place)
        if len(named) == 1:
            c["sites"][0]["name"] = named[0]
    # A roster display name carries the address after a dash ("SAN_La-Jolla_1016 —
    # 844 Pearl Street, ..."), which printed it twice in every location cell.
    for x in c["sites"]:
        nm = str(x.get("name") or "")
        head, sep, rest = nm.partition(" — ")
        if sep and rest and (rest.split(",")[0].strip().lower() in str(x.get("address") or "").lower()):
            x["name"] = head.strip()
    # A line naming most of the sites names them all (000132: weekly visits "at
    # Plymouth, Hudson, Wilmington, Troy and Tupelo", Delphos left out).
    cities = [(str(x.get("city") or "").strip(), str(x.get("state") or "").strip()) for x in c["sites"]]
    cities = [(ci, st) for ci, st in cities if ci]
    if len(cities) >= 3:
        for k in ("overview", "scope", "schedule"):
            for x in c.get(k) or []:
                if not isinstance(x, dict):
                    continue
                t = str(x.get("text") or "")
                hits = [m for ci, st in cities for m in re.finditer(r"\b" + re.escape(ci) + r"\b(?:,\s*" + re.escape(st) + r"\b)?", t)] if True else []
                named = {m.group(0).split(",")[0] for m in hits}
                if 2 <= len(named) < len(cities):
                    a, b = min(m.start() for m in hits), max(m.end() for m in hits)
                    full = ", ".join(f"{ci}, {st}" if st else ci for ci, st in cities[:-1]) + ", and " + \
                        (f"{cities[-1][0]}, {cities[-1][1]}" if cities[-1][1] else cities[-1][0])
                    x["text"] = t[:a] + full + t[b:]
    _traced(c, "per_site_lines", _per_site_lines, c, envelope.get("atoms") or [])
    _summary_names_job(c)
    _traced(c, "summary_services_tasked", _summary_services_tasked, c, envelope.get("atoms") or [])
    # The site-count summary line the fact check set aside comes back when its
    # count is the resolved roster (000123: "161 customer locations" dropped
    # against a note's "about 160").
    n_sites = len(c["sites"])
    if n_sites >= 10 and not any(str(n_sites) in str(x.get("text") if isinstance(x, dict) else x) for x in c.get("overview") or []):
        back = [x for x in c.get("dropped") or [] if isinstance(x, dict) and x.get("list") == "overview"
                and re.search(r"\b%d\b" % n_sites, str(x.get("text")))]
        if back:
            c.setdefault("overview", []).insert(0, {k: v for k, v in back[0].items() if k not in ("list", "reason")})
    if len(c["sites"]) == 1:  # 010085: multi-site standards wording in a one-site SOW
        for k in LISTS:
            for x in c.get(k) or []:
                if isinstance(x, dict) and isinstance(x.get("text"), str):
                    x["text"] = re.sub(r"(?i)\b(at |to |for )?each (customer |store |site )?location\b", lambda m: (m.group(1) or "") + "the site", x["text"])
        for x in c.get("provider_tools") or []:
            if isinstance(x, dict) and re.search(r"(?i)rack hardware|all materials", " ".join(str(y.get("text")) for y in c.get("customer_responsibilities") or [] if isinstance(y, dict))):
                x["text"] = re.sub(r"(?i)\s+and hardware\b", "", str(x.get("text")))
    # A parenthetical the writer added to a site name that no source line
    # gives is not the site's name (000123 pass T: "SAN_Point-Loma_446 (Diego)").
    pool_all = " ".join(str(a.get("text") or "") for a in envelope.get("atoms") or [])
    for x in c["sites"]:
        nm = str(x.get("name") or "")
        mm = re.match(r"^(.*\S)\s*\(([^()]{1,40})\)\s*$", nm)
        if mm and f"({mm.group(2)})" not in pool_all and f"{mm.group(1)} {mm.group(2)}" not in pool_all:
            x["name"] = mm.group(1)
    for x in c["sites"]:  # 010198: a blank name left the Location cell empty
        if not str(x.get("name") or "").strip():
            x["name"] = str(x.get("address") or "")
    if len(c["sites"]) == 1:
        site_words = _toks(" ".join(str(c["sites"][0].get(k) or "") for k in ("name", "address")))
        c["scope"] = [x for x in c.get("scope") or [] if not (isinstance(x, dict) and x.get("kind") == "heading"
                      and _toks(x.get("text")) and _toks(x.get("text")) <= site_words)]
    # Weekend work, checked last: the schedule's date is only final here
    # (010198: "Sunday, August 16" was filled after the grounding pass ran).
    sched = " ".join(str(x.get("text") if isinstance(x, dict) else x) for x in c.get("schedule") or [])
    from datetime import date as _d
    for mo, dy, yr_ in re.findall(r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(20\d\d)", sched):
        try:
            wd = _d(int(yr_), ["january", "february", "march", "april", "may", "june", "july", "august", "september",
                               "october", "november", "december"].index(mo.lower()) + 1, int(dy)).weekday()
            sched += " saturday" if wd == 5 else (" sunday" if wd == 6 else "")
        except ValueError:
            pass
    day = re.search(r"(?i)\b(saturday|sunday)\b", sched)
    all_txt = " ".join(str(a.get("text") or "") for a in envelope.get("atoms") or [])
    asm = c.setdefault("assumptions", [])
    if day and not re.search(r"(?i)\b(no|not|never|excluding|except|without)\b[^.]{0,40}\b(saturday|sunday|weekend)", all_txt) \
            and not any(re.search(r"(?i)weekend|performed on (saturday|sunday)", str(x.get("text") if isinstance(x, dict) else x)) for x in asm):
        asm.append({"text": f"Work is performed on {day.group(1).capitalize()} as scheduled; site access is available that day.",
                    "cites": [], "kind": "method"})
        c["assumptions"] = [x for x in asm if not re.search(r"(?i)(regular|normal|standard) business hours", str(x.get("text") if isinstance(x, dict) else x))]
    # A caption or sentence is never a location (010120 pass U: "Front wall
    # of the conference room includes two large TVs mounted in place." as a
    # second site, glued onto the address).
    caption = lambda t: len(str(t).split()) >= 6 and not re.search(r"\d", str(t)) and bool(
        re.search(r"(?i)\b(includes?|is|are|has|have|shows?|contains?|mounted|located)\b", str(t)))
    c["locations"] = [x for x in c.get("locations") or [] if not caption(x.get("text") if isinstance(x, dict) else x)]
    c["sites"] = [x for x in c.get("sites") or [] if not caption(x.get("name") or x.get("address"))] or c.get("sites")
    # A location line the check dropped as uncited is the site the roster
    # holds (010116 pass U: "Miami, FL | 1221 Brickell Ave").
    if not c.get("locations"):
        for x in c.get("dropped_unsupported") or []:
            if isinstance(x, dict) and x.get("list") == "locations" and any(
                    str(s_.get("street") or s_.get("address") or "").split(",")[0].lower() in str(x.get("text")).lower()
                    for s_ in c.get("sites") or [] if str(s_.get("street") or s_.get("address") or "").strip()):
                c.setdefault("locations", []).append({"text": x["text"], "cites": [], "kind": "fact", "from": "sites"})
    _traced(c, "standard_exclusion_survives", _standard_exclusion_survives, c)
    _traced(c, "materials_counted", _materials_counted, c)
    _locations_cover_sites(c)
    # Lines an outside sender wrote, or a pasted copy of one (010209: the
    # RFQ pasted into a HubSpot note and also in Ralph Lauren's own email).
    sender = lambda a: str((a.get("locator") or {}).get("sender") or "")
    norm = lambda t: re.sub(r"\W+", " ", str(t).lower()).strip()
    outside = [a for a in envelope.get("atoms") or [] if sender(a) and "purtera" not in sender(a).lower()]
    outside_text = {norm(a.get("text")) for a in outside}
    theirs = {str(a.get("id")) for a in outside} | {str(a.get("id")) for a in envelope.get("atoms") or []
                                                   if not sender(a) and norm(a.get("text")) in outside_text}
    _traced(c, "schedule_voice", _schedule_voice, c, str((envelope.get("run_scope") or {}).get("as_of") or ""), theirs)
    _traced(c, "materials_onsite_only", _materials_onsite_only, c)
    _traced(c, "schedule_not_estimates", _schedule_not_estimates, c)
    _traced(c, "window_rollover", _window_rollover, c)
    _traced(c, "qualified_over_universal", _qualified_over_universal, c)
    _traced(c, "field_sequence", _field_sequence, c)
    _traced(c, "no_install_wording", _no_install_wording, c, envelope.get("atoms") or [])
    _traced(c, "followup_not_promised", _followup_not_promised, c, envelope.get("atoms") or [])
    _traced(c, "sentence_integrity", _sentence_integrity, c)
    _std_gone(c, "for_envelope")
    c.pop("_std_objs", None)
    _decisions_match_output(c)
    c["as_of"] = (envelope.get("run_scope") or {}).get("as_of")
    c["version"] = "c3_sow_v1"
    return c


def _name_contradicted(name: str, c: dict[str, Any]) -> bool:
    """A project title naming work that a cited customer duty holds and no
    Provider task or summary line does."""
    tasks = set().union(*[_object_toks(str(x.get("text"))) for k in ("scope", "overview") for x in c.get(k) or []
                          if isinstance(x, dict)] or [set()])
    # The SOW's exclusions hold work too (010326 pass U: "Racking/Wall mount"
    # beside "Rack hardware, gateways, or rack-and-stack work" out of scope).
    other = set().union(*[_object_toks(str(x.get("text"))) for x in c.get("customer_responsibilities") or []
                          if isinstance(x, dict) and x.get("cites")] or [set()])
    other |= set().union(*[_object_toks(str(x.get("text"))) for x in c.get("out_of_scope") or []
                           if isinstance(x, dict) and not x.get("standard")] or [set()])
    return any(w in other and w not in tasks for w in _object_toks(name))


def _when_sent(raw: str) -> float | None:
    """A header date ("Thu, 08 Oct 2026 14:09:23 +0000") or a quoted one
    ("Wednesday, October 7, 2026 10:07 PM"), as a timestamp."""
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(raw).timestamp()
    except Exception:
        pass
    for fmt in ("%A, %B %d, %Y %I:%M %p", "%B %d, %Y %I:%M %p", "%A, %B %d, %Y at %I:%M %p"):
        try:
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            continue
    return None


def requested_by(envelope: dict[str, Any]) -> dict[str, str]:
    """The partner's salesperson who asked PurTera for the work: the account-side
    sender of the earliest email that went to PurTera (010419's request came from
    Tim George with Juli Bonner on copy; the most prolific writer was the copy)."""
    account = str((envelope.get("crm") or {}).get("account_name") or "").lower().split()
    dom = account[0] if account else ""
    roster = (envelope.get("deal_roster") or {}).get("people") or []
    cards = {str(c.get("name") or "").lower(): c for c in (envelope.get("c3_brief") or {}).get("contacts") or []}
    first: tuple[float, str] | None = None
    last_quoted: tuple[float, str] | None = None
    for a in envelope.get("atoms") or []:
        loc = a.get("locator") or {}
        raw_sender = str(loc.get("sender") or "").lower()
        m_addr = re.search(r"[\w.+-]+@[\w.-]+", raw_sender)
        sender = m_addr.group(0) if m_addr else ""
        # A quoted message counts: the request often reaches PurTera only as the
        # quote under a forward (010419: Tim George at cdwg.com).
        if not sender or not dom or dom not in sender.split("@")[-1]:
            continue
        thread = (a.get("structured") or {}).get("email_thread") or {}
        to = " ".join(thread.get("to") or []).lower()
        # A direct message must have gone to PurTera; a quoted one went to
        # whoever forwarded it (010419: Tim George's request, quoted under
        # the forward, lists only his own colleagues).
        if to and "purtera" not in to and not loc.get("quoted"):
            continue
        t = _when_sent(str(loc.get("sent_at") or ""))
        if t is None:
            continue
        if loc.get("quoted"):
            # Quoted history reaches back to earlier contacts (010043's May
            # thread with a former account manager): only the latest counts,
            # and only when no one wrote directly.
            if last_quoted is None or t > last_quoted[0]:
                last_quoted = (t, sender)
        elif first is None or t < first[0]:
            first = (t, sender)
    first = first or last_quoted
    if first is None:
        people = [p for p in roster if p.get("side") == "theirs" and dom and dom in str(p.get("domain") or "")]
        if not people:
            return {"name": "", "phone": "", "email": ""}
        best = max(people, key=lambda p: int(p.get("messages_sent") or 0))
        email = str(best.get("email") or "")
    else:
        email = first[1]
    person = next((p for p in roster if str(p.get("email") or "").lower() == email), {})
    name = str(person.get("name") or "")
    local = email.split("@")[0]
    if not name:
        for k, c in cards.items():
            parts = k.split()
            if parts and parts[-1] in local and (parts[0][:3] in local or parts[0][0] == local[0]):
                name = str(c.get("name") or "")
                break
    card = cards.get(name.lower(), {})
    phone = str(person.get("phone") or card.get("phone") or "")
    if not phone and email:
        # The requester's own signature line carries it; signatures are set
        # aside as chrome, so read them there (010115: Quinton James's
        # "Office: 312.705.1872 | ... | Email: quinton.james@cdw.com").
        for r in (envelope.get("suppressed_chrome") or []) + (envelope.get("atoms") or []):
            t = str(r.get("text") or r.get("raw_text") or "")
            if email in t.lower():
                m = re.search(r"(?:office|direct|o|p|phone|tel)?\s*[:.]?\s*(\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4})", t, re.I)
                if m:
                    phone = m.group(1)
                    break
    return {"name": name, "phone": phone, "email": email}


def as_markdown(content: dict[str, Any]) -> tuple[str, str]:
    """The scope region as the CDW template's two rich-text slots: the lead
    (overview, schedule) and the task list with its supply lists."""
    lead = [x["text"] for x in content.get("overview") or []]
    if content.get("schedule"):
        lead += ["## Schedule"] + ["- " + x["text"] for x in content["schedule"]]
    tasks = []
    if content.get("scope_lead"):
        tasks.append(strip_cites(str(content["scope_lead"])))
    tasks += ["- " + x["text"] for x in content.get("scope") or []]
    if content.get("provider_materials"):
        tasks += ["## Provider Supplied Materials and Tools"] + ["- " + x["text"] for x in content["provider_materials"]]
    if content.get("customer_materials"):
        tasks += ["## Customer Provided Equipment and Materials"] + ["- " + x["text"] for x in content["customer_materials"]]
    return "\n".join(lead), "\n".join(tasks)


__all__ = ["enabled", "for_envelope", "facts", "write", "check", "requested_by", "as_markdown"]
