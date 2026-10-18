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
                 "job_details", "site", "acceptance", "terms", "change_control", "figure", "also_kept", "unsure")
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
    "supplies them. A fact tagged supplied-by:X is the supplier head's decision of who furnishes that item: "
    "follow it (supplied-by:reseller means Provider supplies it, since Provider delivers the reseller's work). "
    "A fact marked (unsure) is unconfirmed: use it only when it fits the confirmed facts. Ignore "
    "email banners and signature blocks. Never write a price, rate or dollar amount, never mention the CRM, ids or "
    "these instructions, and leave a list empty rather than guess. Return JSON only.\n"
    "CRM RECORD: %s\nSITES: %s\nCONTACTS: %s\nFACTS (id, section the heads placed it in, date, text):\n%s\n\n"
)

_PARTS = {
    "header": (
        "TASK: the SOW header. Return {\"project_name\": PurTera's title (end customer + work + place, e.g. \"<End Customer> "
        "<Work> <Place>\"), \"end_customer\": who the work is for, from a request form, a site sheet header, a "
        "signature or the CRM deal name, or \"\", \"services\": [implementation for installs; custom_work for "
        "decommission, packing, smart-hands or labor-only support], \"pricing_model\": \"fixed\"|\"tm\"|\"\", "
        "\"locations\": [each job site as \"City, ST | street address\", fact items citing where it is stated; never an "
        "office in a signature; empty on a job with many sites], \"overview\": [2-4 sentences: what is installed or "
        "done, how many, where, for whom; on a many-site job the site count the facts state], \"schedule\": [labeled "
        "lines only when a fact gives them: \"Requested Start Date: ...\", \"Requested Completion Date: ...\", "
        "\"Arrival Time: ...\", \"Number of Resources: ...\", \"Hardware Quantity: ...\"], \"arrival_protocol\": [only "
        "check-in, badge, parking, escort and notice rules a fact states, saying whose duty each is; empty when no "
        "fact gives one]}"),
    "scope": (
        "TASK: the task list. Return {\"scope_lead\": one line introducing the tasks (\"Provider will dispatch <N> (<n>) <trade> "
        "technicians to perform the following:\"; a count only when a fact states the staffing, never "
        "from meeting attendees), \"scope\": [the tasks in the order the work is done: arrive and check in; each "
        "piece of work with its counts, lengths and locations kept together (\"<location>: approximately <N> ft\"); "
        "routing, termination, mounting and testing; when runs end at an IDF or rack, terminate and connect them "
        "there; cleanup; completion confirmed with the onsite contact. When the request says assist the customer's "
        "staff, the tasks assist as directed. Never claim a route or a device-to-location pairing no fact gives], "
        "\"deliverables\": [{\"item\": \"Work Completion Report\", \"description\": [the finished outcomes with their "
        "counts], \"format\": \"PDF\"}]}"),
    "supply": (
        "TASK: who supplies what. Return {\"provider_materials\": [materials Provider supplies, one item each, each "
        "citing the fact that assigns it], \"provider_tools\": [tools Provider brings; standard hand tools may be "
        "uncited], \"customer_materials\": [equipment and materials the customer or end customer supplies, each "
        "citing the fact that assigns it; an item on the customer's purchase order is customer-supplied], "
        "\"customer_responsibilities\": [duties a fact gives the customer or end customer, e.g. providing an "
        "escort, a staging area, packing materials, directions; never power or PoE, which is an assumption]}"),
    "bounds": (
        "TASK: assumptions and exclusions, written against the DRAFT below so they never contradict it. The template "
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
        "disposition, configuration); never anything the draft's scope includes or the customer is said to supply]}\n"
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
    "Below is a PurTera SOW draft (JSON) and deal facts it does not use yet. For each unused fact that a PM would "
    "put in the SOW (a count, length, location, model, access condition, date, staffing, who supplies what, a "
    "stated assumption or exclusion, a requested task), add ONE bullet to the list it belongs in, citing it. Skip "
    "facts that are chatter, already covered, about another job, or pricing. Same voice and WHO rules as the draft "
    "(\"we\" under a customer heading is the customer; a reseller asked to provide materials means Provider "
    "supplies them). Return JSON only: {\"add\": [{\"list\": one of %s, \"text\": str, \"cites\": [ids], "
    "\"kind\": \"fact\"}]}. Return {\"add\": []} if nothing is missing.\nDRAFT: %s\nUNUSED FACTS:\n%s"
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
    for sec, items in (("also_kept", brief.get("also_mentioned") or []), ("figure", brief.get("figures") or []),
                       ("unsure", unsure)):
        for x in items:
            if x["id"] in seen:
                continue
            seen.add(x["id"])
            out.append({"id": x["id"], "section": sec, "when": x.get("when") or "", "under": "",
                        "text": str(x["text"]), "supplier": x.get("supplier")})
    # The brief keeps a short copy of each line; the writer reads the whole one.
    full = {a.get("id"): str(a.get("text") or "") for a in envelope.get("atoms") or []}
    for f in out:
        if len(full.get(f["id"], "")) > len(f["text"]):
            f["text"] = full[f["id"]]
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


def _fact_line(f: dict[str, Any]) -> str:
    sup = f.get("supplier")
    who = (f" supplied-by:{_SUPPLIER.get(sup[0], sup[0])}" if sup and float(sup[1] or 0) >= SUPPLIER_BAR
           and sup[0] in _SUPPLIER and f.get("type") in _ITEM_TYPES else "")
    return (f"[{f['id']}] ({f['section']}" + (f", {f['when']}" if f["when"] else "") + who + ") "
            + (f"[under: {f['under']}] " if f["under"] else "") + _dated(f["text"], f["when"] or None)[:FACT_CHARS])


def _sites(envelope: dict[str, Any], pm_handoff: dict[str, Any] | None) -> list[dict[str, str]]:
    rows = (pm_handoff or {}).get("sites") or []
    out = []
    for s in rows:
        street, city, st, zp = (str(s.get(k) or "").strip() for k in ("address", "city", "state", "postal_code"))
        tail = ", ".join(x for x in (city, " ".join(x for x in (st, zp) if x)) if x)
        full = ", ".join(x for x in (street, tail) if x) if street and street.lower() not in tail.lower() else tail
        out.append({"name": str(s.get("name") or ""), "city": city, "address": full or str(s.get("name") or "")})
    return out


def write(envelope: dict[str, Any], pm_handoff: dict[str, Any] | None = None, *,
          model: str | None = None) -> dict[str, Any]:
    """The SOW content for one deal: four focused calls (header, tasks, who
    supplies what, then the assumptions and exclusions against that draft),
    a pass over the facts still unused, and every item checked. One call
    asked for all of it dropped a different rule each run."""
    from concurrent.futures import ThreadPoolExecutor

    fs = facts(envelope)
    brief = envelope.get("c3_brief") or {}
    crm = envelope.get("crm") or {}
    crm_rec = {"deal_name": crm.get("deal_name") or (brief.get("deal") or {}).get("deal_name"),
               "account_name": crm.get("account_name") or (brief.get("deal") or {}).get("account_name")}
    contacts = [c.get("line") for c in brief.get("contacts") or [] if c.get("line")][:12]
    sites = _sites(envelope, pm_handoff)
    rows = len([x for x in (brief.get("sections") or {}).get("contacts") or [] if x.get("type") == "physical_site"])
    site_info = {"count_in_site_lists": max(rows, len(sites)), "sites": sites[:15]}
    base = _BASE % (json.dumps(crm_rec), json.dumps(site_info), json.dumps(contacts),
                    "\n".join(_fact_line(f) for f in fs))
    got: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {k: pool.submit(_complete, base + _PARTS[k], model, 3500) for k in ("header", "scope", "supply")}
        for k, fut in futs.items():
            got.update(_json_of(fut.result()) or {})
    draft = {k: [str((x or {}).get("text") if isinstance(x, dict) else x) for x in got.get(k) or []]
             for k in ("overview", "scope", "provider_materials", "customer_materials", "customer_responsibilities")}
    got.update(_json_of(_complete(base + _PARTS["bounds"] % json.dumps(draft)[:10000], model, 2500)) or {})
    content = _normalise(got)
    content["overview"] = content["overview"][:4]
    pool_text = " ".join(f["text"] for f in fs).lower()
    content["scope_lead"] = _lead(str(content.get("scope_lead") or ""), pool_text)
    lines = {f["id"]: (f"[under: {f['under']}] " if f["under"] else "") + f["text"]
             + (f" [supplier head: {_SUPPLIER.get(f['supplier'][0], f['supplier'][0])}]"
                if f.get("supplier") and float(f["supplier"][1] or 0) >= SUPPLIER_BAR
                and f.get("type") in _ITEM_TYPES else "") for f in fs}
    content["added"] = _cover(content, fs, model)
    content["checks"] = check(content, lines)
    content["moved_by_supplier_head"] = _follow_supplier(content, fs)
    content["unassigned"] = _party_check(content, lines)
    _dedupe_items(content)
    content["facts_given"] = len(fs)
    return content


COVER_FACTS = int(os.environ.get("SOWSMITH_C3_SOW_COVER_FACTS", "120"))


def _cover(content: dict[str, Any], fs: list[dict[str, Any]], model: str | None) -> int:
    """Offer the writer every job fact it left uncited and add the ones a SOW
    needs: on 010043 the same "a ladder would do the trick" line made "no lift
    is required" one run and was dropped the next."""
    used = {c for k in LISTS for x in content.get(k) or [] for c in x["cites"]}
    unused = [f for f in fs if f["id"] not in used and f["section"] not in ("acceptance", "terms", "change_control")]
    if not unused:
        return 0
    draft = {k: [x["text"] for x in content.get(k) or []] for k in LISTS}
    raw = _complete(_COVER_PROMPT % (json.dumps(list(LISTS)), json.dumps(draft)[:12000],
                                     "\n".join(_fact_line(f) for f in unused[:COVER_FACTS])), model, 2500)
    got = _json_of(raw) or {}
    n = 0
    for a in got.get("add") or []:
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
    got = _json_of(_complete(_PARTY_PROMPT % body, None, 300)) or {}
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
    return {"text": re.sub(r"\s+", " ", text).strip(), "cites": cites,
            "kind": "fact" if str(x.get("kind")) == "fact" else "method"}


def _normalise(got: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: got.get(k) or "" for k in ("project_name", "end_customer", "scope_lead",
                                                          "pricing_model")}
    out["services"] = [str(s).strip().lower() for s in got.get("services") or [] if str(s).strip()]
    for k in LISTS:
        out[k] = [i for i in (_item(x) for x in got.get(k) or []) if i]
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


def _unsupported_specifics(text: str, pool: str) -> bool:
    """A number, or a capitalised name mid-sentence, that no fact line has."""
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
        got = _json_of(_complete(_VERIFY_PROMPT % body, None, 400)) or {}
        okn = {int(i) for i in got.get("supported") or [] if str(i).isdigit()}
        out += [x for i, x in enumerate(batch) if i in okn]
    return out


def enabled() -> bool:
    """``SOWSMITH_C3_SOW`` is on (dev): the compile writes the SOW content."""
    return os.environ.get("SOWSMITH_C3_SOW", "").strip().lower() in ("1", "true", "yes", "on")


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


def for_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    """The SOW content as the envelope carries it: the writer's lists, the
    partner's requester, the CRM record and the sites."""
    sites = [{"address": s.get("address") or s.get("display_name") or "", "city": s.get("city") or "",
              "state": s.get("state") or "", "postal_code": s.get("postal_code") or "",
              "name": s.get("display_name") or ""}
             for s in (envelope.get("site_readiness") or {}).get("sites") or []]
    offices = _reseller_streets(envelope)
    sites = [x for x in sites if not any(o in re.sub(r"\W+", " ", x["address"].lower()) for o in offices)]
    c = write(envelope, {"sites": sites})
    rec = envelope.get("site_count_reconciliation") or {}
    c["site_summary"] = {"stated": rec.get("stated"), "resolved": len(sites)}
    c["requested_by"] = requested_by(envelope)
    c["crm"] = {k: (envelope.get("crm") or {}).get(k) for k in ("deal_name", "account_name")}
    c["sites"] = _sites(envelope, {"sites": sites})
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
        if to and "purtera" not in to:
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
