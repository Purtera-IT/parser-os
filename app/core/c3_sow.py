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
    "A fact marked (type head: a prerequisite or condition) is a condition the work depends on; whose duty it is follows its wording and its heading (a customer's 'we will unrack' is the customer's; 'drilling rules must be followed' binds whoever drills). A question is never scope by itself: only its answer is, and a later decision overrides the question ('Do we need a site survey?' answered 'probably not' is not a task). A fact marked (quoted earlier email) is older thread context: time slots offered there for a call or meeting are never install dates. A hedge or a habit ('24/7 can be tough', 'it usually needs a follow-up') is never a commitment: never promise return visits, on-call or follow-ups the facts do not settle. The reseller's own work (receiving, shipping, configuration, procurement it does) is never Provider work, in the overview too. A 'remote resource' is remote, never the onsite contact. A model or series number ('785 POS terminals', 'InTouch 9000', 'C9350') is never a count. A license, subscription or cloud/portal setup is never an install task or onsite material: it is the customer's, or out of scope.  A fact in section typed_only is a line the type head read as scope that the brief left out: use it when it is this job's own work, a decision about it, or who does it. A fact marked (unsure) is unconfirmed: use it only when it fits the confirmed facts. Ignore "
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
        "office in a signature; empty on a job with many sites], \"overview\": [2-4 sentences: what is installed or "
        "done, how many, where, for whom; on a many-site job the site count: SITES count_in_site_lists when it is larger than a count a message states], \"schedule\": [labeled "
        "lines only when a fact gives them: \"Requested Start Date: ...\", \"Requested Completion Date: ...\", "
        "\"Arrival Time: ...\", \"Work Window: ...\" (after-hours, weekend or overnight), "
        "\"Number of Resources: ...\", \"Hardware Quantity: ...\"; a date stated anywhere in the facts for the work itself belongs here], \"arrival_protocol\": [only "
        "check-in, badge, parking, escort and notice rules a fact states, saying whose duty each is; empty when no "
        "fact gives one]}"),
    "scope": (
        "STANDARD PURTERA ITEMS FOR THIS KIND OF JOB (method steps, tools, deliverables PurTera writes on such jobs; "
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
        "STANDARD PURTERA ITEMS FOR THIS KIND OF JOB (tools Provider brings; customer duties PurTera writes on such "
        "jobs; use those that fit THIS job's facts, never one the facts contradict): %s\n"
        "TASK: who supplies what. Tools must agree with the job's facts: no lift or scissor lift in the tools when a "
        "ladder is confirmed. Return {\"provider_materials\": [materials Provider supplies, one item each, each "
        "citing the fact that assigns it], \"provider_tools\": [tools Provider brings; standard hand tools may be "
        "uncited], \"customer_materials\": [equipment and materials the customer or end customer supplies, each "
        "citing the fact that assigns it; an item on the customer's purchase order is customer-supplied], "
        "\"customer_responsibilities\": [duties a fact gives the customer or end customer, e.g. providing an "
        "escort, a staging area, packing materials, directions; never power or PoE, which is an assumption]}"),
    "bounds": (
        "STANDARD PURTERA ITEMS FOR THIS KIND OF JOB (assumptions and exclusions PurTera writes on such jobs; use "
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
        "disposition, configuration); never anything the draft's scope includes, any equipment it lists as in scope, or what the customer is said to supply]}\n"
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
        if (a.get("id") in seen or lab.get("type") not in _SCOPE_TYPES or float(lab.get("p") or 0) < TYPED_BAR
                or not 15 < len(txt) < 600 or _CHATTER.search(txt.strip()) or txt.count("|") >= 2
                or re.search(r"(?i)unsubscribe|book time|my calendar|confidential", txt) or "fireflies" in docs.get(str(a.get("artifact_id") or ""), "")
                or "transcript" in docs.get(str(a.get("artifact_id") or ""), "").lower()):
            continue
        typed.append((float(lab.get("p") or 0), {"id": a["id"], "text": txt}))
    typed = [x for _, x in sorted(typed, key=lambda t: -t[0])[:TYPED_MAX]]
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
              "pricing_assumption": "type head: a pricing assumption"}


def _fact_line(f: dict[str, Any]) -> str:
    sup = f.get("supplier")
    who = (f" supplied-by:{_SUPPLIER.get(sup[0], sup[0])}" if sup and float(sup[1] or 0) >= SUPPLIER_BAR
           and sup[0] in _SUPPLIER and f.get("type") in _ITEM_TYPES else "")
    role = _TYPE_ROLE.get(f.get("type") or "")
    return (f"[{f['id']}] ({f['section']}" + (f", {role}" if role else "") + (", from a photo" if f.get("pictured") else "")
            + (", quoted earlier email" if f.get("quoted") else "") + (", a question" if f.get("asked") else "")
            + (f", {f['when']}" if f["when"] else "") + who + ") "
            + (f"[under: {f['under']}] " if f["under"] else "") + _dated(f["text"], f["when"] or None)[:FACT_CHARS])


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
    deal_no = (re.match(r"\s*(\d{6})", str(crm_rec.get("deal_name") or "")) or [None, ""])[1]
    std = standard_items(family, deal_no, [f["text"][:300] for f in fs
                                            if f["section"] in ("scope_of_work", "customer_responsibilities",
                                                                "assumptions", "out_of_scope")][:40])
    pick = lambda *secs: json.dumps({k: std.get(k, []) for k in secs})
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
    content["moved_by_supplier_head"] = _follow_supplier(content, fs)
    content["unassigned"] = _party_check(content, lines)
    _dedupe_items(content)
    content["withheld"] = _scrub(content)
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


def _normalise(got: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: got.get(k) or "" for k in ("project_name", "end_customer", "scope_lead",
                                                          "pricing_model", "job_family")}
    # The writer sometimes returns services as items ({"text": "implementation", ...}); the
    # checkbox row matches on the bare name, so every box printed empty.
    svc = got.get("services") or []
    svc = [svc] if isinstance(svc, (str, dict)) else svc
    out["services"] = [t for t in (str(s.get("text") if isinstance(s, dict) else s).strip().lower().replace(" ", "_")
                                   for s in svc) if t]
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
        out += [x for i, x in enumerate(batch) if i in okn]
    return out


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
        signature = a.get("atom_type") == "stakeholder" and dom in str(said).lower()
        if contact_line or signature:
            m = re.search(r"\b\d{2,6}\s+[A-Za-z0-9 .'-]+?\b(?:way|st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|"
                          r"ln|lane|pkwy|parkway|ct|court|pl|place|hwy|highway)\b", t, re.I)
            if m:
                out.add(re.sub(r"\W+", " ", m.group(0).lower()).strip())
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


WRITER_RUNS = int(os.environ.get("SOWSMITH_C3_SOW_RUNS", "2"))


def _coverage_score(content: dict[str, Any], fs: list[dict[str, Any]]) -> int:
    """How many of the brief's kept facts a draft cites (also-mentioned, unsure
    and typed-only lines count for nothing), less what its checks dropped."""
    body = {f["id"] for f in fs if f["section"] not in ("also_kept", "unsure", "typed_only", "figure")}
    cited = {c for k in LISTS for x in content.get(k) or [] if isinstance(x, dict) for c in x.get("cites") or []}
    dropped = sum(int(v or 0) for v in (content.get("checks") or {}).values() if isinstance(v, (int, float)))
    return len(cited & body) - dropped


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
              "state": s.get("state") or "", "postal_code": s.get("postal_code") or "",
              "name": s.get("display_name") or ""}
             for s in (envelope.get("site_readiness") or {}).get("sites") or []]
    offices = _reseller_streets(envelope)
    sites = [x for x in sites if not any(o in re.sub(r"\W+", " ", x["address"].lower()) for o in offices)]
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
    if len(sites) >= 10 and len(sites) > int(stated or 0):
        stated = len(sites)
    if table:
        # A count the documents state that the table nearly bears out (010279:
        # "113" stores, 98 rows read whole) beats both the rows and a count
        # some other conversation gave.
        seen = [int(k) for k in (rec.get("all_counts_seen") or {}) if str(k).isdigit()]
        near = [n for n in seen if len(table) <= n <= len(table) * 1.4]
        stated = max(near) if near else len(table)
    c["site_summary"] = {"stated": stated, "resolved": len(sites), "from_table": bool(table)}
    c["requested_by"] = requested_by(envelope)
    c["crm"] = {k: (envelope.get("crm") or {}).get(k) for k in ("deal_name", "account_name")}
    c["sites"] = [x for x in _sites(envelope, {"sites": sites}) if (x.get("address") or x.get("name"))]
    # On a job of a few sites the writer's own cited locations are the table:
    # the roster kept SHI's Austin office (a signature) on 010376 and lost
    # 010147's Denver origin, while the writer's locations had both right.
    locs = [str(x.get("text") or "") for x in c.get("locations") or [] if isinstance(x, dict) and x.get("cites")]
    if locs and len(c["sites"]) <= 5 and not table:
        rows = []
        for t in locs:
            place, _, street = t.partition("|")
            city, _, st = place.strip().partition(",")
            rows.append({"name": "", "city": city.strip(), "address": (street.strip() or place.strip()),
                         "street": street.strip(), "state": st.strip(), "postal_code": ""})
        c["sites"] = rows
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
    return {"name": name, "phone": str(person.get("phone") or card.get("phone") or ""), "email": email}


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
