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
    "(\"Customer notes:\"), \"we\" is the customer (\"We will <do X>\" = the customer does X). A request "
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
        "done, how many, where, for whom; on a many-site job the site count: SITES count_in_site_lists when it is larger than a count a message states], \"schedule\": [labeled "
        "lines only when a fact gives them: \"Requested Start Date: ...\", \"Requested Completion Date: ...\", "
        "\"Arrival Time: ...\", \"Work Window: ...\" (after-hours, weekend or overnight), "
        "\"Number of Resources: ...\", \"Hardware Quantity: ...\" (\"Hardware Quantity: Not provided\" when a fact says the count is not known); a date stated anywhere in the facts for the work itself belongs here], \"arrival_protocol\": [only "
        "check-in, badge, parking, escort and notice rules a fact states, saying whose duty each is (the technician's "
        "name is sent ahead by Provider and the site adds it to its security list, so a name-in-advance rule reads "
        "\"Provider submits the technician's name <when> before arrival.\"; the site issues badges and passes; a greeting happens on arrival, credentials come in advance: one rule per line); "
        "empty when no fact gives one]}"),
    "scope": (
        "STANDARD PURTERA ITEMS FOR THIS KIND OF JOB (a purtera_standard_* list is PurTera's fixed handling of this job "
        "type: write every one of its items, reworded to this job, unless a fact contradicts it; the other lists: method steps, tools, deliverables PurTera writes on such jobs; "
        "use, as kind method, every one that fits THIS job's facts, reworded to them, never one the facts contradict "
        "or one about work this job does not include): %s\n"
        "TASK: the task list. Return {\"scope_lead\": one line introducing the tasks (\"Provider will dispatch <N> (<n>) <trade> "
        "technicians to perform the following:\"; a count only when a fact states the staffing, never "
        "from meeting attendees), \"scope\": [group the tasks under short headings, one per area, system or room (an item "
        "{\"text\": \"<Heading>\", \"kind\": \"heading\", \"cites\": []} of at most six words, then that group's tasks); "
        "the tasks in the order the work is done: arrive and check in; each "
        "piece of work with its counts, lengths and locations kept together (\"<location>: approximately <N> ft\"); "
        "every device to be installed gets an install or mount task with its count, model and location; every count, "
        "length and model the job lines give appears in a task; routing, termination, mounting and testing; when runs end at an IDF or rack, terminate and connect them "
        "there; cleanup; completion confirmed with the onsite contact. When the request says assist the customer's "
        "staff, the tasks assist as directed. A manufacturer's install or setup guide tells how, not what: write the "
        "SOW-level work it supports (mount, connect, power, enroll or test the device, as the job needs), never its "
        "numbered steps, menus, settings or values. On a replacement job, removing the old unit is a task and where it "
        "goes (left onsite, boxed, returned) follows the facts. A request the customer made to the reseller or to its own team (receiving, "
        "staging, shipping, procurement, configuration the reseller does) is not Provider work. Never claim a route or a device-to-location pairing no fact gives], "
        "\"deliverables\": [{\"item\": \"Work Completion Report\", \"description\": [the finished outcomes with their "
        "counts], \"format\": \"PDF\"}]}"),
    "supply": (
        "STANDARD PURTERA ITEMS FOR THIS KIND OF JOB (a purtera_standard_* list is PurTera's fixed handling of this job "
        "type: write every one of its items, reworded to this job, unless a fact contradicts it; the other lists: tools Provider brings; customer duties PurTera writes on such "
        "jobs; use those that fit THIS job's facts, never one the facts contradict): %s\n"
        "TASK: who supplies what. Tools must agree with the job's facts: no lift or scissor lift in the tools when a "
        "ladder is confirmed. Return {\"provider_materials\": [materials Provider supplies, one item each, each "
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
        "of the customer belongs in customer responsibilities, not here. A fact from a "
        "figure or photo is a site condition only when it bears on the work (a mounting surface, a height); never "
        "describe what a photo shows. The template "
        "already prices after-hours, weekend and holiday work and travel, and already states business hours, "
        "billing and cancellation terms: never repeat or exclude those. Return {\"assumptions\": [5-10 conditions "
        "that bound the price: each one a fact settles (a ladder confirmed -> \"No lift is required.\" and "
        "\"Standard ladders reach every work location.\"; the longest run -> \"All cable runs are under N feet.\" with "
        "N the longest rounded up to 50; a PurTera person's own stated assumption, reworded as a SOW assumption) and "
        "the standard ones for THIS trade only (cabling or devices: pathways accessible, PoE ports available on the "
        "IDF switch, equipment on site before arrival; decommission or packing: equipment powered down and released, "
        "the customer directs what is packed; any job: permits by the customer)], \"out_of_scope\": [5-9 exclusions as "
        "noun phrases: those a fact states and the standard ones for THIS trade (cabling or devices: lift rental, "
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
    "is about existing equipment that is not part of the new work or another job. A fact in section answered_question "
    "was answered: a request in it (\"I would also like to add <X> - can you help?\") is scope to write. Same "
    "WHO rules: \"we\" under a customer heading is the customer; a reseller asked to provide materials means Provider "
    "supplies them; \"tech to provide <X>\" means Provider supplies <X>. Return JSON only: {\"add\": [{\"list\": one "
    "of %s, \"text\": str, \"cites\": [ids], \"kind\": \"fact\"}]}.\nDRAFT: %s\nFACTS:\n%s"
)

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
    fam = data.get(family) or {}
    return {k: [str(t) for t in v] for k, v in fam.items() if not k.startswith("_") and isinstance(v, list)}


_WHEN = re.compile(r"^\{when:(.*?)\}\s+(?=[A-Z])")  # the regex may hold {n,m}: end at "} " before the item


def _standard_applies(t: str, facts_text: list[str]) -> tuple[bool, str]:
    """A standard item may carry "{when:<regex>}": written only when a deal fact matches."""
    # "{unless:<regex>}": skipped when a deal fact matches (010198: config
    # exclusions beside the router setup the job does).
    u = re.match(r"^\{unless:(.*?)\}\s+(?=[A-Z{])", t)
    if u:
        if any(re.search(u.group(1), f, re.I) for f in facts_text):
            return False, t[u.end():]
        t = t[u.end():]
    m = _WHEN.match(t)
    if not m:
        return True, t
    pat = re.compile(m.group(1), re.I)
    keep_words = re.compile(r"(?i)\b(stay|stays|remain|remains|no change|already|left in place|existing)\b")
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
TYPED_MAX = int(os.environ.get("SOWSMITH_C3_SOW_TYPED_MAX", "25"))


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
                or (abstained and len(txt) < 25) or "fireflies" in docs.get(str(a.get("artifact_id") or ""), "")
                or "transcript" in docs.get(str(a.get("artifact_id") or ""), "").lower()):
            continue
        typed.append((float(lab.get("p") or 0), {"id": a["id"], "text": txt}))
    typed = [x for _, x in sorted(typed, key=lambda t: -t[0])[:TYPED_MAX]]
    # A written instruction about the work is the job whatever its type label:
    # 010120's survey slides ("this should be moved behind wall to not be
    # visible", "No cables should be visible after mount") never reached the
    # writer (types _keep / assumption, below every bar).
    directive = re.compile(r"(?i)\b(should|will need to|needs? to|must|has to|have to)\s+(be\s+)?(moved|mounted|installed|run|placed|hidden|"
                           r"routed|relocated|replaced|removed|visible|concealed|reused)\b|\bno cables? (should|will) be visible|"
                           r"\b(stay|stays|remain|remains) (in place|as is)\b")
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
        if f["under"] or f["id"] not in where:
            continue
        art, ln = where[f["id"]]
        # The two nearest lead-ins: "Customer notes:" above "...we'd ship
        # out:" says whose list it is.
        heads = [t.strip() for _, t in by_art[art] if ln - 12 <= _ < ln and _LEAD_IN.search(t.strip())
                 and len(t.strip()) < 200]
        if heads:
            f["under"] = " > ".join(h[:120] for h in heads[-2:])


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
    family = str(got.get("job_family") or "other")
    family = family if family in FAMILIES else "other"
    # The writer's family call wobbles run to run (010082: low_voltage_cabling,
    # then smart_hands_general, losing the cabling standards). A generic call
    # yields to what the facts plainly are.
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
    content["added"] = _cover(content, fs, model)
    content["added"] += _must_include(content, fs, model)
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
    content["schedule"] = [x for x in content.get("schedule") or []
                           if not (x.get("cites") and all(c in quoted for c in x["cites"])
                                   and not re.match(r"(?i)\s*(number of resources|hardware quantity)", str(x.get("text"))))]
    sched = content.setdefault("schedule", [])
    if not any("completion date" in str(x.get("text", "")).lower() for x in sched if isinstance(x, dict)):
        for f in fs:
            m = re.search(r"(?i)\b(end|completion|finish)\s*date\s*[:|]\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})", f["text"])
            if m:
                sched.append({"text": f"Requested Completion Date: {m.group(2)}", "cites": [f["id"]], "kind": "fact"})
                break
    _reseller_supplies(content, fs)
    # Every member of the customer's ship-out list is customer-provided
    # material (000123: "we'd ship out: ... Printer" and the printer vanished).
    cm = content.setdefault("customer_materials", [])
    # Short item lines right beside a cited customer-provided item, in the
    # same document, are its list siblings (000123: "Printer", "Credit card
    # reader (pin pad)" beside "Samsung Tablet"; the keep head left them out).
    atoms = envelope.get("atoms") or []
    pos = {str(a.get("id")): (str(a.get("artifact_id") or ""), a.get("order")) for a in atoms}
    anchors = [pos.get(c) for x in cm if isinstance(x, dict) for c in x.get("cites") or [] if pos.get(c)]
    for a in atoms:
        t = str(a.get("text") or "").strip()
        art, order = str(a.get("artifact_id") or ""), a.get("order")
        if not t or len(t) > 60 or re.search(r"[.!?]\s|\b(will|we|you|please)\b", t, re.I):
            continue
        if any(art == aa for aa, _oo in anchors) and not _said(t, cm, 0.6) \
                and re.fullmatch(r"(?i)[\w\s()/+&.-]{3,60}", t) \
                and re.search(r"(?i)\b(printer|reader|pin ?pad|scanner|cash drawer|tablet|terminal|stand|hub|cords? for|cables?)\b", t):
            cm.append({"text": t, "cites": [str(a.get("id"))], "kind": "fact"})
    for f in fs:
        if re.search(r"(?i)we'?d ship( out)?|we (will|'ll) (ship|provide)|customer (will )?provide|provided by (the )?customer|ship(ping)? out", f.get("under") or "") \
                and len(f["text"]) <= 120 and not _said(f["text"], cm, 0.6):
            cm.append({"text": f["text"].strip().rstrip(".") , "cites": [f["id"]], "kind": "fact"})
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
    _dedupe_items(content)
    content["withheld"] = _scrub(content)
    # A number that names a document ("785 POS Termainal Refresh RFP.doc") is a
    # model or project number: an item using it as a count is wrong (010279
    # "Remove the existing seven hundred eighty-five (785) POS terminals").
    titled = {n for d in envelope.get("documents") or [] for n in re.findall(r"\b(\d{3,4})\b", str(d.get("filename") or ""))
              if not re.fullmatch(r"(19|20)\d\d", n)}
    if titled:
        count_of = re.compile(r"\((%s)\)|\b(%s)\s+(existing\s+)?[A-Za-z]" % ("|".join(titled), "|".join(titled)))
        for k in LISTS:
            content[k] = [x for x in content.get(k) or []
                          if not (isinstance(x, dict) and count_of.search(str(x.get("text") or "")))]
        for d in content.get("deliverables") or []:  # 010279: "the 785 replaced POS terminals" in the report
            if isinstance(d, dict):
                d["description"] = [t for t in d.get("description") or [] if not count_of.search(str(t))]
    # A heading whose tasks the checks dropped is an empty promise in the SOW
    # (010043: "Camera mounting" with nothing under it).
    sc = content.get("scope") or []
    content["scope"] = [x for i, x in enumerate(sc)
                        if not (isinstance(x, dict) and x.get("kind") == "heading")
                        or (i + 1 < len(sc) and not (isinstance(sc[i + 1], dict) and sc[i + 1].get("kind") == "heading"))]
    # Picture captions never trigger standards (010120: "Two large TVs are
    # mounted on the front wall" fired the display-mounting block).
    _apply_standards(content, family, [f["text"] for f in fs if not f.get("pictured")],
                     [str(a.get("text") or "") for a in envelope.get("atoms") or [] if not (a.get("structured") or {}).get("quoted")])
    _tidy(content, facts_say_permit=bool(re.search(r"(?i)\bpermit", pool_text)))
    _arrival_name_rule(content, fs)
    _grounding(content, fs)
    _materials_hygiene(content, family, fs)
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
    asked = {f["id"] for f in fs if _unanswered(f) and not requests_work(f["text"])}
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
    if re.search(r"(?i)\btransition|onboarding|cut-?over\b", pool_text + " " + crm_name.lower()) and not re.search(
            r"(?i)once (a|per) week|weekly (on-?site|visit)|recurring|staff aug|ongoing", pool_text):
        content["services"] = svc = ["assessment", "implementation"]
    text_by = {f["id"]: f["text"] for f in fs}
    neg = re.compile(r"(?i)\b(not|no|exclud|outside|except|without)\b")
    for x in list(content.get("assumptions") or []):
        t = str(x.get("text")) if isinstance(x, dict) else str(x)
        if isinstance(x, dict) and x.get("cites") and re.search(r"(?i)excluded from|outside (of )?the (work )?scope|not (included|in scope)", t) \
                and not any(neg.search(text_by.get(c, "")) for c in x["cites"]):
            content["assumptions"].remove(x)  # a negation the source lacks (000132 excluded end-user support)
            content.setdefault("scope", []).append({"text": text_by.get(x["cites"][0], t).strip().rstrip(".") + ".", "cites": x["cites"], "kind": "fact"})
    prov = [x for x in content.get("customer_responsibilities") or [] if isinstance(x, dict) and not x.get("standard")
            and re.match(r"(?i)\s*(provide\s+)?(troubleshoot|incident response|end-user support|support for|patch|monitor(ing)? and)", str(x.get("text")))]
    if prov and family in ("smart_hands_general", "other"):
        content["customer_responsibilities"] = [x for x in content["customer_responsibilities"] if x not in prov]
        content.setdefault("scope", []).extend(prov)
    if "implementation" in svc and "custom_work" in svc and family not in ("decommission_packing", "relocation_move"):
        content["services"] = svc = [v for v in svc if v != "custom_work"]  # 010373: haul-away read as packing
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
    if re.search(r"(?i)\bladders?\b", pool_text) and not re.search(r"(?i)\b(lift|boom|scissor)\b(?!\s*(is|are)?\s*not)", " ".join(
            str(x.get("text")) for k in ("scope", "provider_tools", "provider_materials") for x in content.get(k) or [] if isinstance(x, dict))):
        asm = content.setdefault("assumptions", [])
        for t in ("No lift is required.", "Standard ladders reach every work location."):
            if not _said(t, asm, 0.6):
                asm.append({"text": t, "cites": [], "kind": "method"})
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
    content.update(library_counts)
    content["facts_given"] = len(fs)
    return content


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
        if f["section"] not in MUST_SECTIONS or f["id"] in used or f.get("pictured"):
            continue
        toks = {m.group(1).lower().strip() for m in _SALIENT.finditer(f["text"])}
        toks |= {m.group(1).lower() for m in _PRODUCT.finditer(f["text"]) if m.start() > 0}
        toks = {t for t in toks if len(t) >= 2 and t not in ("the", "we", "you", "and")}
        if toks and not any(t in text for t in toks):
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
    words = [w for w in re.findall(r"[a-z0-9]{3,}", x["text"].lower()) if w not in _STOPWORDS]
    if not words:
        return True
    cited = " ".join(lines.get(c, "") for c in x["cites"]).lower()
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


def _dedupe_items(content: dict[str, Any]) -> None:
    """One line per thing in each list: the writer and the coverage pass both
    wrote 010082's drop-off arrangement."""
    for k in LISTS:
        seen, keep = set(), []
        for x in content.get(k) or []:
            key = re.sub(r"\W+", " ", x["text"].lower()).strip()
            short = " ".join(key.split()[:8])
            if key in seen or short in seen:
                continue
            seen.update({key, short})
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


def _said(text: str, items: list[Any], share: float = 0.5) -> bool:
    """Does some item already say ``text`` (most of its words)?"""
    want = _toks(text)
    return bool(want) and any(len(want & _toks(x.get("text") if isinstance(x, dict) else x)) >= share * len(want)
                              for x in items)


def _apply_standards(content: dict[str, Any], family: str, facts_text: list[str] | None = None,
                     context_text: list[str] | None = None) -> int:
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
    stays = " ".join(str(f) for f in facts_text if re.search(r"(?i)\b(stay|stays|remain|remains|no change|left in place|already (mounted|installed|on the wall|in place)|"
                                                                r"keep (the )?existing|not (be )?(moved|replaced|removed))\b", str(f))).lower()
    def contradicted(t: str) -> bool:
        if not stays:
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
    for sec, lst in _STD_LIST.items():
        std = []
        for t0 in fixed.get(sec) or []:
            ok, t = _standard_applies(t0, facts_text or [])
            if self_test and re.search(r"(?i)\btest|confirm .{0,90}\bfunction|verify (basic )?operation", t):
                continue
            if ok and not contradicted(t):
                std.append({"text": t, "cites": [], "kind": "method", "standard": True})
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
            return (lst != "scope" or not re.search(r"\d", t)) and _said(t, std, share)
        items[:] = [x for x in items if not covered(x) and not (isinstance(x, dict) and x.get("standard"))]
        if lst == "scope":
            # A counted deal step is the standard step with the deal's numbers:
            # keep it, not both (010373: install/mount/cable/trash written twice).
            counted = [x for x in items if isinstance(x, dict) and x.get("kind") != "heading" and re.search(r"\d", str(x.get("text")))]
            std = [it for it in std if not _said(it["text"], counted, 0.6)]
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
                                                                     "remove all debris", "dispose", "haul"))]
        # Test and validation steps follow the work they test (000123: tests
        # and the test print ahead of the device installs).
        checks = [it for it in std if it not in tail and re.match(
            r"(?i)(perform (functional |basic )?test|test |verify|validate|confirm (the installed|display|each)|power on each)", it["text"])
            and not re.search(r"(?i)\b(required|present|onsite|on site|available|undamaged|before installation)\b", it["text"])]
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
        pre = [it for it in lead if re.match(r"(?i)(review|confirm (the |each )?(install|mount|location)|verify|unbox|unpack|inspect|check in|meet)", it["text"])]
        post = [it for it in lead if it not in pre]
        deal_work = work and sum(1 for x in items if isinstance(x, dict) and x.get("kind") != "heading" and not x.get("standard")
                                    and re.match(r"(?i)\s*(install|mount|replace|connect|set up|deploy|place)\b", str(x.get("text")))) >= 1
        if deal_work and post and family in ("pos_install_refresh", "av_conference_room", "network_switch_refresh"):
            items[work[0] + 1:work[0] + 1] = pre
            close0 = next((i for i, x in enumerate(items) if isinstance(x, dict) and x.get("kind") == "heading"
                           and re.search(r"(?i)clos|clean|wrap|complet", str(x.get("text")))), len(items))
            items[close0:close0] = post
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
        items[close:close] = [it for it in checks if not _said(it["text"], items, 0.6)]
        close += len([it for it in checks if any(y is it for y in items)])
        items[close + 1:close + 1] = [it for it in tail if not _said(it["text"], items[close + 1:], 0.5)]
        # One closing group: "Cleanup" and "Closeout" both held closing steps.
        items[close]["text"] = "Closeout"
        items[:] = [x for i, x in enumerate(items) if not (i > close and isinstance(x, dict) and x.get("kind") == "heading"
                    and re.search(r"(?i)clos|clean|wrap|complet", str(x.get("text"))))]
    outcomes = [t for ok, t in (_standard_applies(o, facts_text or []) for o in fixed.get("deliverable_outcomes") or [])
                if ok and not contradicted(t) and not (self_test and re.search(r"(?i)\btest", t))]
    if outcomes:
        dl = content.setdefault("deliverables", [])
        if not dl:
            dl.append({"item": "Work Completion Report", "description": [], "format": "PDF"})
        d0 = dl[0] if isinstance(dl[0], dict) else {"item": "Work Completion Report", "description": [], "format": "PDF"}
        own = [str(t) for t in d0.get("description") or [] if str(t).strip()]
        d0["description"] = own + [o for o in outcomes if not _said(o, own, 0.5)]
        # Photographs ship as images beside the report (010116, 010209: "PDF, Jpeg").
        if any(re.search(r"(?i)photo", str(t)) for t in d0["description"]) and "JPEG" not in str(d0.get("format") or "").upper():
            d0["format"] = (str(d0.get("format") or "PDF").strip() or "PDF") + ", JPEG"
        dl[0] = d0
    content["standards_added"] = n
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
            keep = [{k: v for k, v in x.items() if k != "list"} for x in content.get("dropped") or []
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
    neg = re.compile(r"(?i)\b(not|no|exclud\w*|except|outside|without|won'?t|n't|never|customer'?s? (team|staff|it)|by (the )?customer|handled by|responsib)")
    content["out_of_scope"] = [x for x in content.get("out_of_scope") or []
                               if not (isinstance(x, dict) and x.get("cites") and all(c in by for c in x["cites"])
                                       and not any(neg.search(by[c]["text"]) for c in x["cites"]))]
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
        keep = [{k: v for k, v in x.items() if k != "list"} for x in content.get("dropped") or []
                if isinstance(x, dict) and x.get("list") == "overview" and x.get("text") and not greet.search(str(x["text"]))][:2]
    if keep:
        content["overview"] = keep
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
        for x in content.get("scope") or []:
            if isinstance(x, dict) and re.match(r"(?i)\s*provide and install\b", str(x.get("text"))):
                x["text"] = re.sub(r"(?i)^\s*provide and install", "Install", str(x["text"]))
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
        if _said(t, keep_cr, 0.7):
            continue
        keep_cr.append(x)
    content["customer_responsibilities"] = keep_cr
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
                                  if not re.search(r"(?i)\bladders?\b|\bno lift\b", str(x.get("text") if isinstance(x, dict) else x))]
    # A hedged ask is not a promise (000132: "24/7 on-call can be tough ...
    # depending on volume" became "24/7 on-call availability" in the summary,
    # beside its own no-SLO and business-hours assumptions).
    promise = re.compile(r"(?i)24\s*/\s*7|24x7|on-?call|guaranteed response|response time|\bsla\b")
    hedged = any(promise.search(f["text"]) and re.search(r"(?i)tough|depend|might|may |not sure|tbd|hard to|difficult|\?", f["text"]) for f in fs)
    no_slo = any(re.search(r"(?i)no (service[- ]level|slo|sla)|business hours", str(x.get("text") if isinstance(x, dict) else x))
                 for x in content.get("assumptions") or [])
    if hedged or no_slo:
        for k in ("overview", "scope", "schedule"):
            kept = []
            for x in content.get(k) or []:
                t = str(x.get("text") if isinstance(x, dict) else x)
                if isinstance(x, dict) and promise.search(t):
                    t2 = re.sub(r"(?i),?\s*(including |and |with |provide |offer |providing )*(24\s*/\s*7|24x7)?\s*on-?call( availability| support| coverage)?"
                                r"|,?\s*(with |and |provide |providing )*24\s*/\s*7( availability| support| coverage)?", "", t).strip()
                    t2 = re.sub(r"\s+,", ",", re.sub(r",\s*,", ",", t2))
                    if len(t2) < 25 or promise.search(t2):
                        continue
                    x["text"] = t2
                kept.append(x)
            if kept or not content.get(k):
                content[k] = kept
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
        content["scope"] = [x for x in content.get("scope") or [] if not (isinstance(x, dict) and x.get("kind") != "heading"
                            and re.search(r"(?i)for shipment|ship (them |it )?to|shipping to|for disposal", str(x.get("text"))))]
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
        heads_i = [i for i, x in enumerate(sc) if isinstance(x, dict) and x.get("kind") == "heading"]
        rest = [x for x in sc if not any(x is y for y in pre_ph)]
        at = 1 if rest and isinstance(rest[0], dict) and rest[0].get("kind") == "heading" else 0
        if len(heads_i) > 1:
            first_work = next((i for i, x in enumerate(rest) if isinstance(x, dict) and x.get("kind") == "heading" and i > 0), at)
            at = first_work + 1
        content["scope"] = rest[:at] + pre_ph + rest[at:]
    # A task the SOW also excludes goes (010326: "Provide help with racking/
    # stacking/wall mounting" beside rack-and-stack in Out of Scope).
    oos_items = [x for x in content.get("out_of_scope") or [] if isinstance(x, dict)]
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
        logi = re.compile(r"(?i)accept (the )?deliver|receiv(e|ing) (the )?(equipment|hardware|shipments?)|\bwarehous|\bstor(e|ing|age) (the )?(equipment|hardware)|"
                          r"asset[- ]tag|transport|\bkitting\b|\bstaging (area|facility|center|warehouse)|proof of delivery|inventory recap")
        for k in ("scope", "customer_responsibilities", "deliverables_text"):
            content[k] = [x for x in content.get(k) or [] if not (isinstance(x, dict) and x.get("kind") != "heading" and logi.search(str(x.get("text"))))]
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
        and re.match(r"(?i)\s*(services|work|provider'?s? (work|scope|services)) (are|is) limited to (basic|general|light)\b", str(x.get("text"))))]
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
            if skip or (isinstance(x, dict) and re.match(r"(?i)\s*(unpack|reconnect|re-?install|place (the )?equipment at|set up)\b", str(x.get("text")))):
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
    for k in LISTS:
        kept = [x for x in content.get(k) or [] if not (isinstance(x, dict) and ((x.get("cites") and all(c in bro for c in x["cites"]))
                                                                                 or brochure.search(str(x.get("text")))))]
        if kept or k != "overview":
            content[k] = kept
    if not content.get("overview") or all(isinstance(x, dict) and ((x.get("cites") and all(c in bro for c in x["cites"])) or brochure.search(str(x.get("text"))))
                                          for x in content.get("overview") or []):
        alt = [{k2: v for k2, v in x.items() if k2 != "list"} for x in content.get("dropped") or []
               if isinstance(x, dict) and x.get("list") == "overview" and not all(c in bro for c in x.get("cites") or ["-"])
               and not brochure.search(str(x.get("text")))][:2]
        if alt:
            content["overview"] = alt
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
            asm.append({"text": f"Work is performed on {day} as scheduled; site access is available that day.", "cites": [], "kind": "method"})
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
    # Last: a heading left with nothing under it goes (010209 "Switch replacement").
    sc = content.get("scope") or []
    content["scope"] = [x for i, x in enumerate(sc)
                        if not (isinstance(x, dict) and x.get("kind") == "heading")
                        or (i + 1 < len(sc) and not (isinstance(sc[i + 1], dict) and sc[i + 1].get("kind") == "heading"))]


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
        if not t or _LEDGER.search(t) or (not std and _said(t, cr, 0.5)) or _said(t, asm, 0.6):
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
              "number of resources", "hardware quantity")
    sched = content.get("schedule") or []
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
        first = next(iter(re.findall(r"[a-z]+", t.lower())), "")
        if first and _stem(first) in std_verbs and len(_toks(t) & std_words) >= 0.35 * max(1, len(_toks(t))):
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
    photo = [c for c in cr if re.search(r"(?i)\b(photo|picture|serial)", tx(c)) and not (isinstance(c, dict) and c.get("standard"))]
    if photo:
        cr[:] = [c for c in cr if c not in photo]
        dl = content.setdefault("deliverables", [])
        if not dl:
            dl.append({"item": "Work Completion Report", "description": [], "format": "PDF"})
        if isinstance(dl[0], dict):
            dl[0].setdefault("description", []).extend(re.sub(r"(?i)^provide\s+", "", tx(c)).rstrip(".") + "." for c in photo)
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
    # Standard 8-5 site hours repeat the template's Business Hours line.
    content["assumptions"] = [a for a in content.get("assumptions") or [] if (isinstance(a, dict) and a.get("standard"))
                              or not re.search(r"(?i)monday.{0,15}friday.{0,30}8:?0?0?\s*(am|a\.m\.).{0,10}5:?0?0?\s*(pm|p\.m\.)", tx(a))]
    # The template already asks for a point of contact.
    content["customer_responsibilities"] = [c for c in cr if not re.search(r"(?i)point of contact|onsite contact for", tx(c))
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
    for k in LISTS:
        keep = []
        for i, x in enumerate(content.get(k) or []):
            if x["kind"] == "heading":
                if k == "scope" and len(x["text"].split()) <= 7:
                    keep.append(x)
                continue
            x["cites"] = [c for c in x["cites"] if c in lines]
            if x["kind"] == "fact" and not x["cites"]:
                # A deal claim with nothing behind it reads as method only if
                # it carries nothing deal-specific.
                if _unsupported_specifics(x["text"], pool):
                    dropped["uncited_fact"] += 1
                    continue
                x["kind"] = "method"
            if x["kind"] == "method" and (k in ("customer_materials", "provider_materials", "locations")
                                          or (k == "provider_tools" and _unsupported_specifics(x["text"], pool))):
                # Who supplies what is a deal fact, never a default.
                dropped["method"] += 1
                continue
            if x["kind"] == "method" and _unsupported_specifics(x["text"], pool):
                dropped["method"] += 1
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
    gone = content.setdefault("dropped", [])
    for k in LISTS:
        before = len(content[k])
        gone += [{"list": k, **x} for x in content[k] if id(x) in bad]
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
    for m in _NUM.finditer(text):
        tok = m.group(1).lower().rstrip(".,")
        if tok in ("one",) and re.search(r"\bone\s+(of|or|end|side|another)\b", text, re.I):
            continue
        if tok and tok not in pool:
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
    if len(sites) >= 10 and len(addressed) < 0.5 * len(sites):
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
        def _snap(m: re.Match) -> str:
            n = int(m.group(1))
            return (str(n_sites) if n != n_sites and abs(n - n_sites) <= max(2, 0.1 * n_sites) else m.group(1)) + m.group(2)
        for k in ("overview", "scope", "schedule"):
            for x in c.get(k) or []:
                if isinstance(x, dict):
                    x["text"] = re.sub(r"\b(\d{2,4})(\)?\s+(?:\w+\s+)?(?:locations?|stores?|sites?|restaurants?|branches))", _snap, str(x["text"]))
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
                                           re.sub(r"\b(\d{2,4})(\s+(?:\w+\s+)?(?:locations?|stores?|sites?))", _snap, str(t))) for t in d.get("description") or []]
    c["crm"] = {k: (envelope.get("crm") or {}).get(k) for k in ("deal_name", "account_name")}
    c["sites"] = [x for x in _sites(envelope, {"sites": sites}) if (x.get("address") or x.get("name"))]
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
    if len(parsed) >= 6 and len(parsed) > len(c["sites"]) and len(parsed) >= 0.8 * len(c.get("locations") or []):
        c["sites"] = parsed
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
            if len(roster) == 1 and zipped and not re.search(r"\b\d{5}\b", t) and \
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
        c["sites"] = rows
    # The project is named as the deal is: PurTera's v1 titles are the CRM deal
    # name without its number and the reseller ("010115 - CDW Sodexo Decomm
    # Delta Atlanta" -> "Sodexo Decomm Delta Atlanta"); the writer coined its own.
    dn = str((envelope.get("crm") or {}).get("deal_name") or "").strip()
    dn = re.sub(r"^\s*\d{5,6}\s*[-–:]?\s*", "", dn)
    dn = dn.split("|")[0].strip()  # 010419: "... | IT Transition & Onboarding Support | Wilmington, NC"
    dn = re.sub(r"(?i)^(cdw|shi|insight|presidio|computacenter|paragon micro)\s*[-–:]?\s+", "", dn).strip(" -–")
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
    if len(dn.split()) >= 2:
        c["project_name"] = dn
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
    # The site-count summary line the fact check set aside comes back when its
    # count is the resolved roster (000123: "161 customer locations" dropped
    # against a note's "about 160").
    n_sites = len(c["sites"])
    if n_sites >= 10 and not any(str(n_sites) in str(x.get("text") if isinstance(x, dict) else x) for x in c.get("overview") or []):
        back = [x for x in c.get("dropped") or [] if isinstance(x, dict) and x.get("list") == "overview"
                and re.search(r"\b%d\b" % n_sites, str(x.get("text")))]
        if back:
            c.setdefault("overview", []).insert(0, {k: v for k, v in back[0].items() if k != "list"})
    if len(c["sites"]) == 1:  # 010085: multi-site standards wording in a one-site SOW
        for k in LISTS:
            for x in c.get(k) or []:
                if isinstance(x, dict) and isinstance(x.get("text"), str):
                    x["text"] = re.sub(r"(?i)\b(at |to |for )?each (customer |store |site )?location\b", lambda m: (m.group(1) or "") + "the site", x["text"])
        for x in c.get("provider_tools") or []:
            if isinstance(x, dict) and re.search(r"(?i)rack hardware|all materials", " ".join(str(y.get("text")) for y in c.get("customer_responsibilities") or [] if isinstance(y, dict))):
                x["text"] = re.sub(r"(?i)\s+and hardware\b", "", str(x.get("text")))
    for x in c["sites"]:  # 010198: a blank name left the Location cell empty
        if not str(x.get("name") or "").strip():
            x["name"] = str(x.get("address") or "")
    if len(c["sites"]) == 1:
        site_words = _toks(" ".join(str(c["sites"][0].get(k) or "") for k in ("name", "address")))
        c["scope"] = [x for x in c.get("scope") or [] if not (isinstance(x, dict) and x.get("kind") == "heading"
                      and _toks(x.get("text")) and _toks(x.get("text")) <= site_words)]
    c["as_of"] = (envelope.get("run_scope") or {}).get("as_of")
    c["version"] = "c3_sow_v1"
    return c


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
