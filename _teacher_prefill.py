"""DeepSeek TEACHER pre-fill for the gold labeler. For every atom it proposes
admission + fine type + 5 facets + a parser flag, so the intern does ACCEPT/FIX
(weight-0.7 teacher labels) instead of labeling from scratch. Cached per deal.

Key is read from env DEEPSEEK_API_KEY or the sk- token in Downloads/key.txt;
never logged, never written to disk.

Run:  python _teacher_prefill.py optbot
"""
import os, re, json, glob, sys, time
import urllib.request
from pathlib import Path
from collections import defaultdict
from _labeler_core import parse_deal, recommend, facet_prefill, COARSE, FACETS, ADMISSION
from _context_v2 import build_index, context

FINE = [t for fam in COARSE.values() for t in fam]
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
BATCH = 18
CACHE = Path("_teacher_cache"); CACHE.mkdir(exist_ok=True)


def _key():
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k.strip()
    p = Path("C:/Users/lilli/Downloads/key.txt")
    if p.exists():
        m = re.search(r"sk-[A-Za-z0-9_\-]{20,}", p.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return m.group(0)
    raise SystemExit("no DeepSeek key (env DEEPSEEK_API_KEY or Downloads/key.txt)")


# Type definitions: lifted from the classifier's TYPES table (desc strings),
# plus the v2 additions that are not in it yet.
_tac = open("app/core/typed_atom_classifier.py", encoding="utf-8").read()
DEFS = {m.group(1): re.sub(r"\s+", " ", m.group(2)).strip()
        for m in re.finditer(r'"([a-z_]+)"\s*:\s*\{[^}]*?"desc"\s*:\s*"([^"]+)"', _tac, re.S)}
DEFS.setdefault("rate_card", "A reference table of unit rates (role/skill -> hourly or unit rate); NOT an actual priced line in this deal (that is service_line).")
DEFS.setdefault("work_scope_item", "A statement of work to be performed (a scope bullet/clause), with NO price; describes WHAT is done, not a priced row.")
DEFS.setdefault("risk", "An identified risk/threat to the project (schedule, budget, technical, dependency) \u2014 separate from its mitigation.")
DEFS.setdefault("data_flow_step", "A step in a data/integration flow: source -> transform -> destination.")
DEFS.setdefault("system_mapping", "A mapping between two systems/fields (field A in system X -> field B in system Y).")

DISAMBIG = [
    "service_line vs bom_line: service_line = LABOR/SERVICE priced by hour/day/unit (technician, install, PM, training). bom_line = PHYSICAL HARDWARE/equipment/materials. A row priced 'Per Hour' or naming a role/service is service_line, NEVER bom_line.",
    "service_line vs rate_card: service_line = an ACTUAL priced line in THIS deal (qty + extended total). rate_card = a generic rate REFERENCE table, no deal-specific qty/extended.",
    "contract_term vs payment_term vs deal_metadata: contract_term = a CONTRACTUAL clause (term length, renewal, warranty, liability, governing law, termination). payment_term = a billing/payment SCHEDULE tier ('30% at acceptance'). deal_metadata = an identifier/date/ref/customer-name, no obligation.",
    "pricing_assumption vs work_scope_item: pricing_assumption = an assumption/exclusion/caveat affecting PRICE ('taxes excluded', 'substitutions need approval'). work_scope_item = work to perform, no price caveat.",
    "stakeholder vs deal_metadata: stakeholder = a NAMED PERSON/role/contact. deal_metadata = non-person identifiers/dates/refs.",
    "site_attribute vs site_access_window: site_attribute = a scalar count (users, rooms, sqft, floors). site_access_window = WHEN a site is accessible (days/hours/escort).",
    "service_line vs rate_card vs bom_line: a priced row with a deal-specific Site/Unit/Sell-Quantity or billing-type feeding THIS deal's price = service_line; a generic country/tier rate lookup with NO deal qty = rate_card; a physical material with qty/SKU = bom_line; a tool/PPE name without a sell qty is NOT bom_line.",
    "requirement vs work_scope_item: requirement = a binding 'shall/must' obligation; an atom containing '?' or a spec/component-table row = work_scope_item.",
    "deal_metadata cues: 'Customer:/OEM:/Division:/Bidder:' prefixed = deal_metadata; 'Billing Type:' = payment_term.",
]

SYS = (
    "You are a labeling teacher for a B2B deal-document parser. For each atom (one fact) return STRICT JSON. "
    "Choose the SINGLE best fine type using the TYPE DEFINITIONS below \u2014 match the atom to the definition it fits best, "
    "not to the most similar-sounding name. When two types seem close, apply the DISAMBIGUATION rules. "
    "Facets: scope_polarity is ONLY for work statements (else not_applicable); amount_kind is not_amount unless the atom "
    "states money; metadata_kind applies ONLY to deal_metadata. "
    "flag='glued' if the atom packs several distinct facts, 'split' if it is a fragment of a larger fact, "
    "'needs_extractor' if it is an unextracted image/table reference, else 'none'."
)


def _prompt(atoms):
    defs = "\n".join(f"  {t}: {DEFS.get(t, '')}" for t in FINE)
    dis = "\n".join(f"  - {d}" for d in DISAMBIG)
    lines = [
        "TYPE DEFINITIONS (choose exactly one fine type):",
        defs,
        "",
        "DISAMBIGUATION (commonly-confused pairs):",
        dis,
        "",
        f"admission options: {ADMISSION}",
        f"facets: scope_polarity{FACETS['scope_polarity']} responsibility{FACETS['responsibility']} amount_kind"
        f"{FACETS['amount_kind']} obligation_modality{FACETS['obligation_modality']} metadata_kind"
        f"{FACETS['metadata_kind']}",
        "",
        'Return JSON: {"labels":[{"i":int,"admission":str,"fine":str,"scope_polarity":str,'
        '"responsibility":str,"amount_kind":str,"obligation_modality":str,"metadata_kind":str,'
        '"flag":str,"note":str}]}. One entry per atom, same i.',
        "",
        "ATOMS:",
    ]
    for a in atoms:
        lines.append(f"[{a['i']}] doc={a['doc']} section=\"{a['section'][:60]}\" :: {a['body'][:300]}")
    return "\n".join(lines)


def _call(key, atoms):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": _prompt(atoms)}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                j = json.loads(r.read())
            out = json.loads(j["choices"][0]["message"]["content"])
            return out.get("labels", out if isinstance(out, list) else [])
        except Exception as e:
            if attempt == 2:
                print(f"   batch failed: {type(e).__name__}: {str(e)[:80]}")
                return []
            time.sleep(2 * (attempt + 1))


def label_corpus(key, corpus):
    """Atom teacher over an already-parsed corpus (no source files needed)."""
    atoms = [{"i": i, "doc": a["doc"][:22], "section": a.get("section", ""), "body": a["body"]}
             for i, a in enumerate(corpus)]
    labels = {}
    for s in range(0, len(atoms), BATCH):
        for L in _call(key, atoms[s:s + BATCH]):
            if isinstance(L, dict) and "i" in L:
                labels[int(L["i"])] = L
    return labels


def prefill(slug):
    # optbot lives in _optbot_inputs; blob deals in _blob_pool/<slug> or
    # _blob_deals/<slug>.
    fs = []
    for base in (("_optbot_inputs",) if slug == "optbot" else (f"_blob_pool/{slug}", f"_blob_deals/{slug}")):
        if os.path.isdir(base):
            fs = sorted(glob.glob(base + "/*.pdf") + glob.glob(base + "/*.docx") + glob.glob(base + "/*.xlsx"))
            if fs:
                break
    if not fs:
        print(f"   ! no source files for {slug}"); return {}
    corpus, _, _ = parse_deal(fs)
    atoms = [{"i": i, "doc": a["doc"][:22], "section": a.get("section", ""), "body": a["body"]}
             for i, a in enumerate(corpus)]
    key = _key()
    labels = {}
    for s in range(0, len(atoms), BATCH):
        chunk = atoms[s:s + BATCH]
        for L in _call(key, chunk):
            if isinstance(L, dict) and "i" in L:
                labels[int(L["i"])] = L
        print(f"   {min(s + BATCH, len(atoms))}/{len(atoms)} atoms labeled")
    out = CACHE / f"{slug}.json"
    out.write_text(json.dumps({"model": MODEL, "labels": labels}, ensure_ascii=False, indent=1), encoding="utf-8")
    # agreement with the heuristic recommender, for a quick sanity read
    agree = sum(1 for i, a in enumerate(corpus)
                if str(labels.get(i, {}).get("fine", "")) == recommend(a["body"], a.get("section", ""), a["type"]))
    print(f"WROTE {out} \u00b7 {len(labels)}/{len(corpus)} labeled \u00b7 {agree} agree with heuristic ({100 * agree // max(1, len(corpus))}%)")
    return labels


if __name__ == "__main__":
    for slug in sys.argv[1:] or ["optbot"]:
        print(f"\n=== teacher prefill: {slug} ({MODEL}) ===")
        prefill(slug)
