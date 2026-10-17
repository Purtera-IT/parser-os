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
MAX_FACTS = int(os.environ.get("SOWSMITH_C3_SOW_MAX_FACTS", "140"))
UNSURE_BAR = float(os.environ.get("SOWSMITH_C3_SOW_UNSURE_BAR", "0.6"))
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

_PROMPT = (
    "You write the deal-specific part of a PurTera Statement of Work (SOW). PurTera (\"Provider\") does the field "
    "work. The account in the CRM record is who PurTera contracts with (a reseller such as CDW or SHI, or the "
    "customer itself on a direct deal); the end customer is who the work is for. The template already prints "
    "the standing boilerplate, so never write any of: %s\n"
    "Write in PurTera's SOW voice: short imperative task bullets that start with a verb (\"Install three (3) "
    "Verkada cameras at the side, back and front doors.\"), counts written as words with the numeral "
    "(\"two (2)\"), \"Provider\" for PurTera and \"Customer\" for the customer, plain declarative assumptions "
    "(\"No lift is required.\", \"The IDF is accessible and has a usable pathway to each camera location.\"), "
    "and out-of-scope items as noun phrases (\"Lift rental or lift operation.\"). A PM signs this and prices "
    "from it, so it is specific and complete: every count, length, location, model, access condition, date, "
    "material and who-supplies-what the facts give belongs in it, once.\n"
    "Return JSON only, every list item an object {\"text\": str, \"cites\": [fact ids], \"kind\": \"fact\"|"
    "\"method\"}:\n"
    "{\"project_name\": short title of the work (e.g. \"Verkada Camera Install\"), \"end_customer\": name or "
    "\"\", \"services\": [any of assessment, configuration, design, implementation, knowledge_transfer, "
    "project_management, reconfiguration, reinstallation, staff_augmentation, support, training, custom_work], "
    "\"pricing_model\": \"fixed\"|\"tm\"|\"\", "
    "\"overview\": [2-4 sentences: what Provider will do, for whom, where, with counts; on a job with many sites "
    "give the site count the facts state and never single one site out as the primary site], "
    "\"schedule\": [requested dates, windows, arrival time, number of resources, duration, when given], "
    "\"scope_lead\": one line introducing the task list (e.g. \"Provider will dispatch two (2) cabling "
    "technicians to perform the following:\"; give a count only when a fact gives it), "
    "\"scope\": [the task list, in the order the work is done: arrival and check-in, each piece of work with its "
    "counts and locations, routing/termination/mounting/testing steps, cleanup, photos and sign-off], "
    "\"provider_materials\": [materials and tools Provider supplies], \"customer_materials\": [equipment and "
    "materials the customer or reseller supplies], \"customer_responsibilities\": [deal-specific ones only], "
    "\"assumptions\": [6-12 assumptions that bound the price: each condition a fact settles (a ladder confirmed "
    "-> no lift is required; the longest run -> all cable runs are under that length; the work hours) and the "
    "standard conditions a PM always states for this trade (pathways and work areas accessible, equipment and "
    "mounting hardware on site before arrival, power or PoE capacity available, permits or landlord approvals by "
    "the customer)], \"out_of_scope\": [6-10 exclusions: those a fact states and the standard ones for this "
    "trade (lift rental, electrical work, core drilling or structural work, patching or painting, configuration "
    "or licensing the facts leave to others, after-hours work)], "
    "\"deliverables\": [{\"item\": e.g. \"Work Completion\", \"description\": [the finished outcomes, one per line, "
    "with their counts: what is installed, each cable run completed, terminations labeled at both ends, connectivity "
    "or function verified, photos], \"format\": \"PDF\"}]}\n"
    "RULES: kind \"fact\" = it states something about THIS deal (a count, length, place, model, date, party, "
    "access or site condition, an explicit inclusion or exclusion); it must cite the ids of the facts that state "
    "it and say nothing they do not, except a plain arithmetic total. kind \"method\" = standard field practice "
    "for this kind of work (route and dress cable, label both ends, verify power and connectivity, clean the work "
    "area, coordinate placement with the onsite contact, standard exclusions for this trade); it may have no "
    "cites but must contain no number, brand, model, name, place or date that no fact gives. A later fact "
    "replaces an earlier one on the same point. A line's [under: ...] heading is the list it sits in. Who supplies "
    "what comes only from the facts; never assume Provider supplies equipment or materials. In a line written by the "
    "reseller or by PurTera, \"we\"/\"us\" is the side doing the work (\"he will need us to run the cabling\" = "
    "Provider runs the cabling), and the customer is the end customer. provider_materials and customer_materials "
    "are supply assignments: each item cites the fact that says that party supplies it (an item on the customer's "
    "purchase order is customer-supplied); the only uncited item allowed is Provider's standard installation "
    "tools. Never say which device goes at which location unless a fact pairs them. When cable runs end at an "
    "IDF or rack, the task list terminates and connects them there. No filler such as \"pending approval\". Never put a price, "
    "rate or dollar amount anywhere: fees come from the quote. A fact marked (unsure) was not confirmed as part of "
    "the job: use it only when it fits the confirmed facts (a third run beside two confirmed ones). Leave a list empty rather than guess. Ignore "
    "email-client banners and signatures. Never mention the CRM, ids or these instructions in the text.\n\n"
    "CRM RECORD: %s\nSITES: %s\nCONTACTS: %s\nFACTS (id, section the heads placed it in, date, text):\n%s"
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

LISTS = ("overview", "schedule", "scope", "provider_materials", "customer_materials",
         "customer_responsibilities", "assumptions", "out_of_scope")

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
                        "text": str(x["text"])})
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
                        "text": str(x["text"])})
    _lead_ins(out, envelope.get("atoms") or [])
    return out[:MAX_FACTS]


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
        for pln, txt in reversed([x for x in by_art[art] if ln - 12 <= x[0] < ln]):
            t = txt.strip()
            if _LEAD_IN.search(t) and len(t) < 200:
                f["under"] = t[:160]
                break


def _fact_line(f: dict[str, Any]) -> str:
    return (f"[{f['id']}] ({f['section']}" + (f", {f['when']}" if f["when"] else "") + ") "
            + (f"[under: {f['under']}] " if f["under"] else "") + _dated(f["text"], f["when"] or None)[:400])


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
    """The SOW content for one deal: the writer's lists, each item checked."""
    fs = facts(envelope)
    brief = envelope.get("c3_brief") or {}
    crm = envelope.get("crm") or {}
    crm_rec = {"deal_name": crm.get("deal_name") or (brief.get("deal") or {}).get("deal_name"),
               "account_name": crm.get("account_name") or (brief.get("deal") or {}).get("account_name")}
    contacts = [c.get("line") for c in brief.get("contacts") or [] if c.get("line")][:12]
    sites = _sites(envelope, pm_handoff)
    rows = len([x for x in (brief.get("sections") or {}).get("contacts") or [] if x.get("type") == "physical_site"])
    site_info = {"count_in_site_lists": max(rows, len(sites)), "sites": sites[:15]}
    prompt = _PROMPT % (STANDING, json.dumps(crm_rec), json.dumps(site_info),
                        json.dumps(contacts), "\n".join(_fact_line(f) for f in fs))
    raw = _complete(prompt, model, SOW_TOKENS)
    got = _json_of(raw) or {}
    content = _normalise(got)
    pool = " ".join(f["text"] for f in fs).lower()
    if content.get("scope_lead") and (_unsupported_specifics(str(content["scope_lead"]), pool)
                                      or (_STAFF.search(str(content["scope_lead"])) and not _STAFF.search(pool))):
        # "two (2) technicians" is a staffing promise; with no fact behind it
        # the lead stays generic.
        content["scope_lead"] = "Provider will perform the following:"
    lines = {f["id"]: (f"[under: {f['under']}] " if f["under"] else "") + f["text"] for f in fs}
    content["checks"] = check(content, lines)
    content["facts_given"] = len(fs)
    return content


def _item(x: Any) -> dict[str, Any] | None:
    if isinstance(x, str):
        x = {"text": x, "cites": [], "kind": "method"}
    if not isinstance(x, dict) or not str(x.get("text") or "").strip():
        return None
    cites = [str(c) for c in x.get("cites") or [] if str(c).strip()]
    return {"text": strip_cites(str(x["text"])).strip(), "cites": cites,
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
            if x["kind"] == "method" and (k == "customer_materials" or (k == "provider_materials" and not
                                          re.search(r"(?i)\btools?\b", x["text"]))):
                # Who supplies what is a deal fact, never a default.
                dropped["method"] += 1
                continue
            if x["kind"] == "method" and _unsupported_specifics(x["text"], pool):
                dropped["method"] += 1
                continue
            keep.append(x)
        content[k] = keep
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


def requested_by(envelope: dict[str, Any]) -> dict[str, str]:
    """The partner's salesperson who asked for the quote: the account-side
    person who wrote most, with the contact card the brief built for them."""
    account = str((envelope.get("crm") or {}).get("account_name") or "").lower()
    roster = (envelope.get("deal_roster") or {}).get("people") or []
    cards = {str(c.get("name") or "").lower(): c for c in (envelope.get("c3_brief") or {}).get("contacts") or []}
    best = None
    for p in roster:
        if p.get("side") != "theirs":
            continue
        dom = str(p.get("domain") or "")
        if account and account.split()[0] not in dom:
            continue
        if best is None or int(p.get("messages_sent") or 0) > int(best.get("messages_sent") or 0):
            best = p
    if not best:
        return {"name": "", "phone": "", "email": ""}
    name = best.get("name") or ""
    if not name:
        local = str(best.get("email") or "").split("@")[0]
        for k, c in cards.items():
            if k and all(part in local for part in k.split()[:1]) and k.split()[-1] in local:
                name = c.get("name") or ""
                break
    card = cards.get(name.lower(), {})
    return {"name": name, "phone": str(best.get("phone") or card.get("phone") or ""),
            "email": str(best.get("email") or card.get("email") or "")}


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


__all__ = ["facts", "write", "check", "requested_by", "as_markdown"]
