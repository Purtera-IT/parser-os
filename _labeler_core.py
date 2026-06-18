"""Shared core for the gold labeler: parse a deal's docs into a clean atom
corpus, my recommended label, cross-doc context, coverage-aware sampling, and
the self-contained HTML+zip builder. Imported by _build_html.py (default 11
deals) and _corpus_select.py (gap-closing selection over the full blob pool)."""
import os, re, json, shutil, zipfile
os.environ.setdefault("SOWSMITH_DISABLE_LLM", "1")
os.environ.setdefault("SOWSMITH_DROP_QUANTITY_ATOM", "1")
os.environ.setdefault("SOWSMITH_DROP_DERIVED_SUBATOMS", "1")
from pathlib import Path
from collections import defaultdict, Counter
from app.parsers.parser_router import parse_artifact
from app.core.entity_extraction import _enrich_table_atoms
from app.core.semantic_dedup import cross_type_dedup_atoms
from app.core.typed_atom_classifier import _atom_decide_text

TAX = json.load(open("_label_taxonomy.json", encoding="utf-8"))
TAX.setdefault("work_scope_item", {"desc": "A statement of work / exclusion / included or optional activity / customer or vendor responsibility / site-specific implementation instruction. MUST carry scope_polarity + responsibility facets."})
TAX.setdefault("rate_card", {"desc": "A unit rate / pricebook entry (role or labor category rate, day rate, $/hr) that defines pricing basis but is NOT a purchased quantity. Distinct from service_line (has qty x price)."})
import yaml as _yaml
_V2 = _yaml.safe_load(open("_taxonomy_v2.yaml", encoding="utf-8"))
COARSE = _V2["coarse"]                         # family -> [fine,...]
ALIASES = _V2["aliases"]                        # retired/legacy -> v2 fine
ADMISSION = _V2["admission"]                    # keep / boilerplate / needs_extractor
FACETS = _V2["facets"]                          # dim -> [values]
FINE = [f for fs in COARSE.values() for f in fs]
FINE2COARSE = {f: c for c, fs in COARSE.items() for f in fs}
TYPES = {k: TAX.get(k, {}).get("desc", "")[:150] for k in FINE}

def to_v2(label):
    """Map any legacy/parser label to a v2 fine type (or admission exit)."""
    l = ALIASES.get(label, label)
    return l if (l in FINE or l in ADMISSION) else "work_scope_item"

def facet_prefill(body, fine):
    """Rule-based facet guesses; human corrects only when wrong."""
    b = body.lower()
    # amount_kind
    if "grand total" in b: ak = "grand_total"
    elif "subtotal" in b: ak = "subtotal"
    elif fine == "rate_card" or "/hr" in b or "per hour" in b or "day rate" in b: ak = "rate"
    elif fine == "bom_line": ak = "extended_price"
    elif fine == "service_line": ak = "line_item_total"
    elif fine == "payment_term": ak = "payment_milestone"
    elif fine == "site_budget": ak = "budget"
    elif "tax" in b: ak = "tax"
    elif "freight" in b or "shipping" in b: ak = "freight_shipping"
    elif fine == "commercial_total": ak = "subtotal"
    else: ak = "not_amount"
    # scope_polarity
    if "out of scope" in b or "exclud" in b: sp = "out_of_scope"
    elif "optional" in b: sp = "optional"
    elif "assumption" in b or "assume" in b: sp = "assumption"
    elif "customer to provide" in b or "customer provides" in b or "depend" in b: sp = "dependency"
    elif fine in ("work_scope_item", "deliverable", "task", "service_line"): sp = "in_scope"
    else: sp = "not_applicable"
    # responsibility
    if "customer" in b or "client" in b: rsp = "customer"
    elif "vendor" in b or "contractor" in b or "installer" in b or "thinktls" in b: rsp = "vendor"
    elif "bidder" in b or "offeror" in b: rsp = "bidder"
    else: rsp = "unknown"
    # obligation_modality
    if "shall" in b or "must" in b or "required" in b or fine in ("requirement", "contract_term", "submission_req"): om = "mandatory"
    elif "may " in b or "optional" in b: om = "optional"
    elif "prohibit" in b or "not permitted" in b or "no weekend" in b: om = "prohibited"
    elif fine in ("risk", "stakeholder", "deal_metadata", "physical_site"): om = "informational"
    else: om = "unknown"
    # metadata_kind — sorts the deal_metadata pile (only meaningful there)
    if fine == "deal_metadata":
        if re.search(r"\bHS-DEAL|deal id|opportunity id|hubspot deal|quote\b|Q-[A-Z]", b, re.I): mk = "identifier"
        elif re.search(r"customer|company|client|end user|vendor|account", b, re.I): mk = "party"
        elif re.search(r"\d{4}-\d{2}-\d{2}|close date|effective date", b, re.I): mk = "date"
        elif re.search(r"\bPO\b|PO-|MSA|purchase order|contract\b|msa-", b, re.I): mk = "commercial_ref"
        elif re.search(r"azure|container|workspace|parser batch|orbitbrief|blob", b, re.I): mk = "system_ref"
        elif re.search(r"stage|status", b, re.I): mk = "stage"
        elif re.search(r"fictional|mock data|disclaimer|do not", b, re.I): mk = "disclaimer"
        else: mk = "unknown"
    else:
        mk = "not_applicable"
    return {"scope_polarity": sp, "responsibility": rsp, "amount_kind": ak,
            "obligation_modality": om, "metadata_kind": mk}

DATE_RE  = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MONEY_RE = re.compile(r"\$\s?[\d,]{3,}(?:\.\d{2})?")
REF_RE   = re.compile(r"\b(?:HS-DEAL|Q-[A-Z]|PO-|MSA|REQ-|HW-|SV-|R-\d|T-\d|\d{6})[A-Z0-9-]{2,}\b")
SITE_RE  = re.compile(r"\b[A-Z]{2,5}-[A-Z]{2,}(?:-\d+)?\b")
# id namespaces that share the LETTERS-LETTERS shape of a site code but are
# refs/quotes/POs/skus, NOT sites. Without this, "HS-DEAL", "PO-MOCK",
# "Q-DEV", "PS-PROJMGMT" get scooped up as bogus site links.
_NONSITE_PREFIX = {"HS", "PO", "Q", "MSA", "REQ", "HW", "SV", "R", "T",
                   "DEV", "MOCK", "DEAL", "PS", "SKU", "PN", "QTE", "INV",
                   "MDF", "IDF", "INTUNE", "VLAN", "PDU",          # infra/network labels
                   "IC", "AM", "IT", "SVC", "WAN", "DNS", "DHCP", "FAT", "ATP"}  # checkpoint/item/acronym codes
NUM_RE   = re.compile(r"\b\d{5,}\b")
STRENGTH = {"ref": 4, "date": 3, "$": 3, "site": 2, "#": 1}

def _t(a): at=getattr(a,"atom_type",None); return getattr(at,"value",str(at))
_EMIT_SITES = None
def _prod_sites(t):
    """Use PROD's site extractor (app.core.entity_extraction) so offline labeling
    has no train/serve skew — it catches numeric site codes (ATL-047-04) the toy
    SITE_RE missed and denylists system-token fakes (ATL-REFRESH). Keeps only
    code-shaped site keys; facility-name variants are merged by the neural
    resolution head, not counted as separate sites here."""
    global _EMIT_SITES
    if _EMIT_SITES is None:
        try:
            from app.core.entity_extraction import _emit_sites
            _EMIT_SITES = _emit_sites
        except Exception:
            _EMIT_SITES = False
    if not _EMIT_SITES:
        return None
    out = set()
    for k in _EMIT_SITES(t):
        if not k.startswith("site:"):
            continue
        code = k[5:].upper().replace("_", "-")
        if code.split("-", 1)[0] in _NONSITE_PREFIX:
            continue                    # BOM/service/checkpoint/infra item codes, not sites
        if re.search(r"\d", code) or re.match(r"^[A-Z]{2,5}-[A-Z]{2,5}$", code):
            out.add(code)               # code-shaped; drop facility-name keys
    return out

def _ents(t):
    ps = _prod_sites(t)
    site = ps if ps is not None else {s for s in SITE_RE.findall(t) if s.split("-",1)[0] not in _NONSITE_PREFIX}
    return {"ref":set(REF_RE.findall(t)),"date":set(DATE_RE.findall(t)),
            "$":set(m.replace(' ','') for m in MONEY_RE.findall(t)),
            "site":site,"#":set(NUM_RE.findall(t))}

def _rec_raw(body, section, guess):
    b, s = body, (section or "").lower()
    def has(*xs): return any(x in body for x in xs)
    if body.startswith("[dropped sheet") or re.match(r"^(Subject:|Table \d)", body): return "boilerplate"
    if has("awaiting OCR", "not fully extracted", "vision or embedded-object", "Drawing /",
           "OCR chain", "scanned image", "low-text page", "OLE extraction", "vision-LLM",
           "Image artifact", "Image awaiting"): return "needs_extractor"
    if re.match(r"Mitigation:", b): return "mitigation"
    if re.match(r"Risk ID:", b) or "risk" in s or "watch item" in s: return "risk"
    if re.match(r"Requirement ID:|REQ-\d", b): return "requirement"
    if re.match(r"Acceptance Area:", b): return "acceptance_criterion"
    if re.match(r"Phase:\s*\d", b): return "milestone_phase"
    if re.match(r"Task ID:|T-\d{2,}", b): return "task"
    if re.match(r"Item ID:\s*HW", b): return "bom_line"
    if re.match(r"Service ID:\s*SV", b): return "service_line"
    if re.match(r"Room Type:", b): return "site_room_mix"
    if re.match(r"Tag Prefix:", b): return "site_implementation_note"
    if re.match(r"site_id:", b): return "physical_site"
    if re.match(r"Site Code:|Address:.*City:", b): return "physical_site"
    if re.match(r"Category:\s*.*(Subtotal|Total|Contingency|Taxes|Freight)", b) or "Grand Total" in b: return "commercial_total"
    if re.search(r"\blead time\b", b, re.I) or "ARO" in b: return "lead_time_constraint"
    if re.match(r"Field:\s*Address", b): return "site_attribute"
    if re.match(r"Field:\s*(Users|Rooms|Square Feet|Priority)", b): return "site_attribute"
    if re.match(r"Field:\s*Access", b): return "site_access_window"
    if re.search(r"HS-DEAL|PO-MOCK-|Q-DEV-|MOCK-MSA|HubSpot \d", b): return "deal_metadata"
    if re.match(r"Field:", b) or (re.match(r"[A-Za-z ]+:\s*[A-Z0-9]", b) and ("Deal" in b or "Opportunity" in b or "Customer" in b)): return "deal_metadata"
    if "out of scope" in s or "exclud" in b.lower(): return "exclusion" if "exclusion" in TYPES else guess
    if "assumption" in s: return "pricing_assumption" if "pric" in b.lower() or "tax" in b.lower() else "dependency"
    if "force majeure" in b.lower() or "additional insured" in b.lower(): return "contract_term"
    if "signature" in s or re.search(r":\s*_{3,}.*Date", b): return "signatory"
    if re.search(r"\$\s?\d[\d,]*\s*(?:/|per\s)\s*(?:hr|hour|day)\b", b, re.I) or re.search(r"\bday rate\b", b, re.I): return "rate_card"
    if guess in TYPES: return guess
    return "deal_metadata" if re.match(r"[A-Za-z ]+:", b) else "scope_item"

def recommend(body, section, guess):
    """v2 fine type: raw rule result mapped through the alias table."""
    return to_v2(_rec_raw(body, section, guess))

# CAD drawing-set / site-schematic filenames — the labeler discards these docs
# downstream ("not supported yet"), but fully parsing one is the single biggest
# cost in a deal (a graphics-heavy schematic PDF takes ~20s of vector/raster
# analysis). Skip the parse entirely for files the renderer will throw away —
# same output, ~20s/deal faster. (Labeler-only; the real pipeline still parses
# schematics through its own path.)
_SCHEM_NAME = re.compile(
    r"CONSOLIDATED SET|\bIFC\b|SYMBOLS?\s*&?\s*LEGEND|RISER| - T | - TA |\bT0\d\d\b|\.dwg|DRAWING|SCHEMATIC",
    re.I,
)


def parse_deal(paths):
    corpus, srcdump, pathmap = [], defaultdict(list), {}
    for p in paths:
        doc = Path(p).stem
        pathmap[doc] = p.replace("\\", "/")
        if _SCHEM_NAME.search(Path(p).name):
            print(f"  ~ skip schematic (labeler discards): {Path(p).name}")
            continue
        # Author-HIDDEN xlsx rows are still parsed (no silent drops) but look
        # identical to visible rows in the review list — so a reviewer can't tell a
        # collapsed/0-hour row from the live estimate. Compute their distinctive
        # text here (content survives enrichment, unlike the atom's review_flags)
        # so the body can be tagged below.
        hidden_sigs: set[str] = set()
        if str(p).lower().endswith(".xlsx"):
            try:
                from app.parsers.xlsx_parser import XlsxParser
                import openpyxl
                hd = XlsxParser._hidden_dims(Path(p))
                wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
                for ws in wb.worksheets:
                    _hc, hr = hd.get(ws.title, (set(), set()))
                    if not hr:
                        continue
                    try:
                        ws.reset_dimensions()
                    except Exception:
                        pass
                    rws = [list(r) for r in ws.iter_rows(values_only=True)]
                    for i in hr:
                        if 0 <= i < len(rws):
                            texts = [str(c).strip() for c in rws[i] if c is not None and str(c).strip()]
                            longest = max(texts, key=len) if texts else ""
                            if len(longest) >= 8:
                                hidden_sigs.add(longest.lower())
                wb.close()
            except Exception:
                hidden_sigs = set()
        try:
            out = parse_artifact("pkg","a",Path(p),None)
            atoms=list(getattr(out,"atoms",out) or [])
            # READING ORDER: the parser returns atoms in document order. The
            # schema-enriched table atoms (contacts, LOE rows, ...) are NEW objects
            # so we must put them back at their SOURCE cell's position, else they
            # all sink to the bottom of the list (contacts appeared after every
            # paragraph instead of right where the table sits). Map by provenance.
            def _prov(a):
                loc=a.source_refs[0].locator if getattr(a,"source_refs",None) else {}
                return (loc.get("sheet"),loc.get("table_index"),loc.get("row"),loc.get("paragraph_index"))
            pos={id(a):i for i,a in enumerate(atoms)}
            prov2pos={}
            for i,a in enumerate(atoms):
                prov2pos.setdefault(_prov(a),i)
            typed=_enrich_table_atoms(list(atoms),project_id="pkg")
            for x in typed:
                pos[id(x)]=prov2pos.get(_prov(x),len(atoms)+1)
            keep=[a for a in atoms if _t(a)!="raw_table_row"]+[x for x in typed if _t(x)!="raw_table_row"]
            keep=[a for a in cross_type_dedup_atoms(keep) if _t(a)!="raw_table_row"]
            keep.sort(key=lambda a: pos.get(id(a),10**9))   # restore document order
        except Exception as e:
            print("  ! fail",doc,str(e)[:50]); continue
        for a in keep:
            # The rollup SUMMARY atom ("N pricing lines, $lo-$hi") is a
            # PM-packet aggregate, not a parse fact to review — and now that
            # every catalog / rate-card ROW is its own real atom, the summary
            # would just be a confusing duplicate banner. Skip it in the review
            # tool; the per-row atoms render on their own.
            _v = a.value if isinstance(getattr(a,"value",None),dict) else {}
            if _v.get("is_summary"):
                continue
            dt=_atom_decide_text(a) or ""
            m=re.search(r"\[section: ([^\]]+)\]",dt); sec=m.group(1) if m else ""
            mi=re.search(r"\[intro: ([^\]]+)\]",dt); intro=mi.group(1) if mi else ""
            body=re.sub(r"\s*\[(table|section|intro)[^\]]*\]","",dt).strip()
            if len(body)<4: continue
            loc=a.source_refs[0].locator if getattr(a,"source_refs",None) else {}
            page=loc.get("page"); page=(int(page)+1) if isinstance(page,int) else None
            fl=set(getattr(a,"review_flags",None) or [])
            is_hidden = ("xlsx_parser:hidden_in_source" in fl
                         or (hidden_sigs and any(s in body.lower() for s in hidden_sigs)))
            if is_hidden and "[hidden row in source sheet]" not in body:
                body = f"{body}  [hidden row in source sheet]"
            flag=("hidden_in_source" if is_hidden else
                  "weak_label" if "weak_label" in fl else
                  ("truncated_cell" if "truncated_cell" in fl else ""))
            corpus.append({"doc":doc,"type":_t(a),"body":body[:2000],"section":sec,"intro":intro,"flag":flag,
                           "page":page,"ents":_ents(body+" "+sec),"vkind":_v.get("kind","")})
            srcdump[doc].append(body[:160])
    return corpus, dict(srcdump), pathmap

def cross(a, corpus):
    out=[]
    for b in corpus:
        if b is a or b["doc"]==a["doc"]: continue
        sh, score, strong = [], 0, False
        for k in ("ref","date","$","site","#"):
            ov=a["ents"][k]&b["ents"][k]
            if ov:
                sh.append(k+" "+",".join(list(ov)[:2])); score=max(score,STRENGTH[k])
                if k in ("ref","site","date"): strong=True
        if not sh: continue
        # conflict heuristic: anchored on a shared ref/site/date, but a $ or qty DIFFERS
        cf=""
        if strong:
            for k,lab in (("$","$"),("#","qty")):
                if a["ents"][k] and b["ents"][k] and (a["ents"][k]-b["ents"][k] or b["ents"][k]-a["ents"][k]):
                    cf=f"{lab} {', '.join(list(a['ents'][k])[:2])} vs {', '.join(list(b['ents'][k])[:2])}"
                    break
        rec={"d":b["doc"][:18],"t":b["type"],"b":b["body"][:90],"sh":"; ".join(sh),"_sc":score}
        if cf: rec["cf"]=cf; rec["_sc"]=score+10          # surface conflicts first
        out.append(rec)
    out.sort(key=lambda x:-x["_sc"])
    for o in out: o.pop("_sc",None)
    return out[:6]

def sample(corpus, n=16, rare=None):
    """Coverage-aware: round-robin across (doc, section-leaf) groups, but
    prioritise atoms whose recommended class is in `rare` (starved classes)."""
    rare = rare or set()
    groups=defaultdict(list)
    for a in corpus:
        leaf=a["section"].split(" > ")[-1] if a["section"] else "(none)"
        groups[(a["doc"],leaf)].append(a)
    def keyf(a):
        r=recommend(a["body"],a["section"],a["type"])
        has_cf=any(x.get("cf") for x in a.get("xc",[]))   # surface conflict atoms first
        return (0 if has_cf else 1, 0 if r in rare else 1, -len(a.get("xc",[])), len(a["body"]))
    gl=[sorted(g,key=keyf) for g in groups.values()]
    pick,sb,st=[],set(),Counter()
    while len(pick)<n and any(gl):
        prog=False
        for g in gl:
            if not g: continue
            a=g.pop(0); k=a["body"][:40].lower()
            if k in sb or st[a["type"]]>=4: continue
            pick.append(a); sb.add(k); st[a["type"]]+=1; prog=True
            if len(pick)>=n: break
        if not prog: break
    return pick

GUIDE = """
<b>How to label — one fact, one type.</b> The picker is the deal taxonomy. ★ = my recommendation (pre-filled). ● = the parser's raw guess. Click the correct type; add a note if anything's off.
<br><br><b>Decision order:</b>
<br>1. <b>Is it furniture?</b> page bands, ToC lines, dropped helper sheets, doc subtitles → <code>boilerplate</code>.
<br>2. <b>Did we fail to read it?</b> "awaiting OCR", embedded drawing, "page N not fully extracted" → <code>needs_extractor</code> (a coverage gap, not a type).
<br>3. <b>Otherwise pick the most specific fact type.</b> A header field ("Field: X | Value: Y", deal id, close date, contract type) → <code>deal_metadata</code>. A money aggregate (subtotal/grand total) → <code>commercial_total</code>. A billing % tier → <code>payment_term</code>. A real building → <code>physical_site</code>; a site's number (users/rooms/sqft) → <code>site_attribute</code>. A BOM hardware row → <code>bom_line</code>; a services row → <code>service_line</code>. A phase → <code>milestone_phase</code>; a task row → <code>task</code>. A risk → <code>risk</code>; its fix → <code>mitigation</code>. An electrical test → <code>electrical_acceptance_test</code>; a pass-threshold row → <code>acceptance_criterion</code>; a numbered REQ → <code>requirement</code>. A "customer provides X" → <code>dependency</code>; a "taxes excluded" → <code>pricing_assumption</code>; a signature line → <code>signatory</code>.
<br><br><b>🔧 Parser-wrong flags (orange row)</b> — these critique <i>me</i>, separate from the type. Toggle any that apply:
<br>• <b>mis-typed</b> — my ● guess is just wrong (still set the right type above).
<br>• <b>mis-split</b> — this atom is really ≥2 facts glued together.
<br>• <b>dropped-neighbor</b> — a fact next to this one is missing from the list.
<br>• <b>wrong-section</b> — the [section] path is wrong.
<br>• <b>scrambled-fields</b> — the columns/values got cross-contaminated.
<br>• <b>should-drop</b> — this shouldn't be an atom at all (it's furniture).
<br><br><b>👎 Context critique</b> — each cross-doc link has a 👎. Click it if the join is wrong or irrelevant (e.g. matched only on a shared year, or links two unrelated facts). That teaches the context builder what's a real corroboration vs noise.
<br><br><b>Conflicts / coverage gaps are NOT a label.</b> If a fact disagrees with another doc, still label its <i>type</i> and write the conflict in the <b>note</b> — the graph head handles contradictions. "3 of 5 sites have budgets" → label normally; note the gap.
<br><b>Use the note box for the WHY</b> — single-source, conflicts-with-X, ambiguous. Notes train the head too.
"""

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>parser-os gold labeler</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--card:#1e222b;--bd:#2a2f3a;--txt:#e6e8ec;--mut:#9aa3b2;--ok:#3fb950;--info:#58a6ff;--warn:#d29922;--rec:#bc8cff;--bad:#f85149;--mono:ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 system-ui,sans-serif}
.top{display:flex;align-items:center;gap:9px;padding:9px 14px;border-bottom:1px solid var(--bd);flex-wrap:wrap}
.tab{font-size:12px;padding:4px 11px;border-radius:8px;border:1px solid var(--bd);background:var(--panel);color:var(--mut);cursor:pointer}
.tab.on{border-color:var(--info);color:var(--info);background:#10243f;font-weight:600}
.btn{font-size:12px;padding:5px 11px;border-radius:8px;border:1px solid var(--bd);background:var(--panel);color:var(--txt);cursor:pointer;text-decoration:none}
.btn.save{border-color:var(--ok);color:var(--ok)} .btn.help{border-color:var(--info);color:var(--info)}
.guide{display:none;padding:12px 16px;border-bottom:1px solid var(--bd);background:#10131a;font-size:12.5px;line-height:1.6;color:var(--mut)}
.guide code{background:var(--card);padding:1px 5px;border-radius:4px;color:var(--rec);font-family:var(--mono)}
.wrap{display:flex;height:calc(100vh - 52px)}
.lcol{flex:1.1;border-right:1px solid var(--bd);display:flex;flex-direction:column;min-width:0}
.srcnav{display:flex;align-items:center;gap:7px;padding:7px 10px;border-bottom:1px solid var(--bd);background:var(--panel);flex-wrap:wrap}
.src{flex:1;overflow:auto;background:#0b0d11}
.src iframe{width:100%;height:100%;border:0;background:#fff}
.srcdump{padding:13px;font-family:var(--mono);font-size:12px}
.srcline{padding:5px 8px;margin-bottom:4px;border-left:2px solid var(--bd);background:var(--card);border-radius:4px;color:var(--mut)}
.srcline.cur{border-left-color:var(--info);color:var(--txt);background:#10243f}
.lab{flex:1;overflow:auto;padding:13px 17px}
.nav{display:flex;gap:8px;align-items:center;margin-bottom:9px}
.nav input[type=range]{flex:1}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:11px 13px;margin-bottom:9px}
.body{font-family:var(--mono);font-size:13px;white-space:pre-wrap;word-break:break-word;margin:6px 0}
.muted{color:var(--mut);font-size:11px} .abs{font-size:10px;color:var(--mut);font-family:var(--mono);word-break:break-all}
.pill{font-size:10px;padding:2px 7px;border-radius:6px;background:var(--panel);color:var(--mut)}
.pill.w{background:#3a2d10;color:var(--warn)}
.ctx{font-size:11px;padding:5px 8px;margin-bottom:4px;border-left:2px solid var(--info);background:var(--panel);border-radius:4px;display:flex;gap:7px;align-items:flex-start}
.ctx.bad{border-left-color:var(--bad);opacity:.55;text-decoration:line-through}
.ctx.conflict{border-left-color:var(--warn);background:#2a2210}
.cflag{color:var(--warn);font-weight:600;font-size:10.5px}
.ctx .tdn{cursor:pointer;font-size:13px;user-select:none;flex-shrink:0}
.grid{display:flex;flex-wrap:wrap;gap:4px;max-height:200px;overflow:auto}
.ty{font-size:10.5px;padding:3px 8px;border-radius:6px;font-family:var(--mono);cursor:pointer;border:1px solid var(--bd);background:var(--panel);color:var(--mut)}
.ty.sp{border-color:var(--warn);color:var(--warn)} .ty.rec{border-color:var(--rec);color:var(--rec)}
.ty.on{border-color:var(--ok);color:var(--ok);background:#0e2a16;font-weight:600}
.grphdr{width:100%;font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);margin:6px 0 2px;border-top:1px solid var(--bd);padding-top:5px}
.fac label{font-size:11px;color:var(--mut)} .fac select{display:block;margin-top:2px}
.iss{font-size:10.5px;padding:3px 9px;border-radius:6px;font-family:var(--mono);cursor:pointer;border:1px solid var(--bd);background:var(--panel);color:var(--mut)}
.iss.on{border-color:var(--warn);color:var(--warn);background:#3a2d10;font-weight:600}
.search{font-size:12px;padding:4px 9px;border-radius:7px;border:1px solid var(--bd);background:var(--panel);color:var(--txt)}
.desc{font-size:11px;color:var(--mut);min-height:28px;margin-top:7px}
.note{width:100%;font-size:12px;padding:7px 9px;border-radius:8px;border:1px solid var(--bd);background:var(--panel);color:var(--txt);font-family:inherit;resize:vertical}
</style></head><body>
<div class="top">
  <b style="font-size:13px">gold labeler</b>
  <span id="tabs" style="display:flex;gap:6px;flex-wrap:wrap"></span>
  <button class="btn help" onclick="document.getElementById('guide').style.display=document.getElementById('guide').style.display==='block'?'none':'block'">❓ how to label</button>
  <span style="flex:1"></span>
  <span id="prog" class="muted"></span>
  <button class="btn save" onclick="download_()">⬇ download gold</button>
  <button class="btn" onclick="copy_()">copy</button>
</div>
<div class="guide" id="guide">__GUIDE__</div>
<div class="wrap">
  <div class="lcol">
    <div class="srcnav">
      <button class="btn" onclick="srcGo(-1)">◀</button>
      <select id="docsel" class="search" style="flex:1;min-width:120px" onchange="srcSet(this.value)"></select>
      <button class="btn" onclick="srcGo(1)">▶</button>
      <a class="btn" id="openlink" target="_blank">open ↗</a>
      <span class="muted" id="srcpos"></span>
    </div>
    <div class="abs" id="abspath" style="padding:4px 10px;border-bottom:1px solid var(--bd)"></div>
    <div class="src" id="src"></div>
  </div>
  <div class="lab">
    <div class="nav">
      <button class="btn" onclick="go(-1)">←</button>
      <input type="range" id="sl" min="1" value="1" oninput="jump(this.value)">
      <button class="btn" onclick="go(1)">→</button>
      <span class="muted" id="counter"></span>
    </div>
    <div class="card">
      <div style="display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-bottom:5px">
        <span class="muted">parser:</span><span class="pill" id="guess"></span>
        <span class="muted">my rec:</span><span class="pill" id="rec" style="color:var(--rec)"></span>
        <span class="pill w" id="fl" style="display:none"></span>
      </div>
      <div class="body" id="body"></div>
      <div class="muted" id="sec"></div>
    </div>
    <div id="ctxwrap" style="display:none;margin-bottom:9px"><div class="muted" style="margin-bottom:4px">↳ cross-document context — corroborations + ⚠ conflicts (👎 = bad link)</div><div id="ctx"></div></div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;gap:8px">
        <div style="font-size:12px;font-weight:600">label: <span id="sel" style="font-family:var(--mono);color:var(--ok)">— pick —</span></div>
        <span><button class="btn" onclick="useRec()" style="border-color:var(--rec);color:var(--rec)">use ★ rec</button>
        <input class="search" id="search" placeholder="filter 46…" style="width:120px" oninput="render()"></span>
      </div>
      <div class="grid" id="grid"></div>
      <div class="desc" id="desc"></div>
    </div>
    <div class="card fac">
      <div class="muted" style="margin-bottom:6px">facets — prefilled from rules; fix only if wrong</div>
      <div id="facets" style="display:flex;flex-wrap:wrap;gap:14px"></div>
    </div>
    <div class="card">
      <div class="muted" style="margin-bottom:6px">🔧 parser wrong? (flags the parser, not the type)</div>
      <div id="issues" style="display:flex;flex-wrap:wrap;gap:5px"></div>
    </div>
    <div class="card">
      <div class="muted" style="margin-bottom:5px">note — the WHY (single-source? conflicts with X? wrong split? ambiguous?). Trains the head too.</div>
      <textarea class="note" id="note" rows="2" oninput="setNote(this.value)" placeholder="optional note for this atom"></textarea>
    </div>
    <div class="card" style="border-color:var(--rec)">
      <div class="muted" style="margin-bottom:5px">💡 suggestions for <b id="sgdeal" style="color:var(--rec)"></b> — full creative freedom. Missing a type? Parser pattern you keep seeing? Better context idea? A rule we should add? Write anything; it shapes the next build.</div>
      <textarea class="note" id="suggest" rows="3" oninput="setSuggest(this.value)" placeholder="e.g. 'need a warranty_term type', 'BOM qty keeps getting split into a twin atom', 'context should link bare site codes like ATL-WEST', ..."></textarea>
    </div>
  </div>
</div>
<script id="ld" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('ld').textContent);
const T=DATA.T, D=DATA.D, SRC=DATA.SRC, DOCS=DATA.DOCS, TYPES=Object.keys(T), DEALS=Object.keys(D);
const C=DATA.C, FAC=DATA.F, FC=DATA.FC, AD=DATA.AD;
const SPECIAL={boilerplate:1,needs_extractor:1};
const ISSUES=["mis-typed","mis-split","dropped-neighbor","wrong-section","scrambled-fields","should-drop"];
const KEY="gold_labels_v4";
let store=JSON.parse(localStorage.getItem(KEY)||"{}");
let deal=DEALS[0], i=0, srcIdx=0, srcManual=false;
const $=id=>document.getElementById(id);
function E(dn,j){store[dn]=store[dn]||{}; store[dn][j]=store[dn][j]||{}; return store[dn][j];}
function rec(dn,j){return (store[dn]||{})[j]||{};}
function save(){localStorage.setItem(KEY,JSON.stringify(store));}
function setL(dn,j,v){E(dn,j).label=v; save();}
function setNote(v){E(deal,i).note=v; save();}
function setSuggest(v){store[deal]=store[deal]||{}; store[deal]._suggest=v; save();}
function collectSuggest(){const o={};DEALS.forEach(dn=>{if(store[dn]&&store[dn]._suggest)o[dn]=store[dn]._suggest;});return o;}
function toggleIssue(t){const e=E(deal,i);e.issues=e.issues||[];const k=e.issues.indexOf(t);if(k<0)e.issues.push(t);else e.issues.splice(k,1);save();render();}
function toggleCtx(k){const e=E(deal,i);e.ctxbad=e.ctxbad||[];const p=e.ctxbad.indexOf(k);if(p<0)e.ctxbad.push(k);else e.ctxbad.splice(p,1);save();render();}
function labeled(o){return o.label||(o.issues&&o.issues.length)||(o.ctxbad&&o.ctxbad.length);}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function tabs(){$('tabs').innerHTML='';DEALS.forEach(dn=>{const n=Object.values(store[dn]||{}).filter(labeled).length;const b=document.createElement('span');b.className='tab'+(dn===deal?' on':'');b.textContent=dn+' ('+n+'/'+D[dn].length+')';b.onclick=()=>{deal=dn;i=0;srcManual=false;render();};$('tabs').appendChild(b);});}
function chip(t){const cur=rec(deal,i).label,sel=cur===t,g=D[deal][i].g===t,r=D[deal][i].r===t,sp=SPECIAL[t];const b=document.createElement('span');b.className='ty'+(sel?' on':(r?' rec':(sp?' sp':'')));b.textContent=t+(r?' ★':'')+(g?' ●':'');b.onmouseover=()=>$('desc').textContent=t+' — '+(T[t]||'');b.onclick=()=>{setL(deal,i,t);render();};return b;}
function buildGrid(){
  const q=($('search').value||'').toLowerCase();const g=$('grid');g.innerHTML='';
  function grp(name,list){const sh=list.filter(t=>!q||t.includes(q)||(T[t]||'').toLowerCase().includes(q));if(!sh.length)return;const h=document.createElement('div');h.className='grphdr';h.textContent=name;g.appendChild(h);sh.forEach(t=>g.appendChild(chip(t)));}
  grp('admission',AD.filter(x=>x!=='keep'));
  Object.keys(C).forEach(fam=>grp(fam.replace(/_/g,' '),C[fam]));
}
function curFacet(dim){const e=rec(deal,i);if(e.facets&&e.facets[dim])return e.facets[dim];const fc=D[deal][i].fc||{};return fc[dim]||'unknown';}
function setFacet(dim,v){const e=E(deal,i);e.facets=e.facets||{};e.facets[dim]=v;save();}
function renderFacets(){const w=$('facets');w.innerHTML='';Object.keys(FAC).forEach(dim=>{const lab=document.createElement('label');lab.textContent=dim.replace(/_/g,' ');const sel=document.createElement('select');sel.className='search';FAC[dim].forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;o.selected=(v===curFacet(dim));sel.appendChild(o);});sel.onchange=()=>setFacet(dim,sel.value);lab.appendChild(sel);w.appendChild(lab);});}
function srcGo(d){const L=DOCS[deal];srcIdx=Math.max(0,Math.min(L.length-1,srcIdx+d));srcManual=true;render();}
function srcSet(j){srcIdx=parseInt(j);srcManual=true;render();}
function useRec(){setL(deal,i,D[deal][i].r);render();}
function srcPane(){
  const L=DOCS[deal],a=D[deal][i];
  if(!srcManual){const k=L.findIndex(x=>x.doc===a.doc);if(k>=0)srcIdx=k;}
  srcIdx=Math.max(0,Math.min(L.length-1,srcIdx));const dc=L[srcIdx],onAtom=dc.doc===a.doc;
  const sel=$('docsel');sel.innerHTML='';L.forEach((x,k)=>{const o=document.createElement('option');o.value=k;o.textContent=(x.isPdf?'📄 ':'▦ ')+x.doc+(x.n?' ('+x.n+')':'');o.selected=k===srcIdx;sel.appendChild(o);});
  $('srcpos').textContent=(srcIdx+1)+'/'+L.length; $('abspath').textContent=dc.path; $('openlink').href=dc.path;
  const e=$('src');
  if(dc.isPdf){const pg=(onAtom&&a.page)?'#page='+a.page:'';e.innerHTML='<iframe src="'+dc.path+pg+'"></iframe>';}
  else{const lines=SRC[deal][dc.doc]||[];e.innerHTML='<div class="srcdump"><div class="muted" style="margin-bottom:8px">'+dc.doc+' ('+(dc.path.split('.').pop())+') — '+lines.length+' parsed atoms'+(onAtom?'':' · browsing, not this atom\'s doc')+'</div>'+lines.map(l=>'<div class="srcline'+(onAtom&&l.slice(0,40)===a.b.slice(0,40)?' cur':'')+'">'+esc(l)+'</div>').join('')+'</div>';}
}
function render(){
  const A=D[deal],a=A[i],r=rec(deal,i);
  $('sl').max=A.length;$('sl').value=i+1;$('counter').textContent=(i+1)+' / '+A.length+' · '+a.doc;
  $('guess').textContent=a.g;$('rec').textContent=a.r;
  const fl=$('fl');if(a.f){fl.style.display='inline-block';fl.textContent=a.f;}else fl.style.display='none';
  $('body').textContent=a.b;$('sec').textContent=a.s||'(no section)';
  const bad=(r.ctxbad||[]);
  if(a.x&&a.x.length){$('ctxwrap').style.display='block';$('ctx').innerHTML=a.x.map((c,k)=>'<div class="ctx'+(bad.indexOf(k)>=0?' bad':'')+(c.cf?' conflict':'')+'"><span class="tdn" onclick="toggleCtx('+k+')">👎</span><div>'+(c.cf?'<span class="cflag">⚠ possible conflict — verify (may be a coincidental number/zip match → 👎 it): '+esc(c.cf)+'</span><br>':'')+'<b style="color:var(--info)">['+c.d+']</b> <span class="muted">('+c.sh+' · '+c.t+')</span><br><span style="font-family:var(--mono);color:var(--mut)">'+esc(c.b)+'</span></div></div>').join('');}else if(a.ss){$('ctxwrap').style.display='block';$('ctx').innerHTML='<div class="ctx" style="border-left-color:var(--warn)"><div><b style="color:var(--warn)">single-source</b> <span class="muted">no corroboration found in any other document — lower-trust; verify. (note it if that seems wrong)</span></div></div>';}else $('ctxwrap').style.display='none';
  $('sel').textContent=r.label||'— pick —';$('note').value=r.note||'';
  $('sgdeal').textContent=deal;$('suggest').value=(store[deal]&&store[deal]._suggest)||'';
  const iss=(r.issues||[]);$('issues').innerHTML='';ISSUES.forEach(t=>{const b=document.createElement('span');b.className='iss'+(iss.indexOf(t)>=0?' on':'');b.textContent=t;b.onclick=()=>toggleIssue(t);$('issues').appendChild(b);});
  buildGrid();renderFacets();
  let tot=0;DEALS.forEach(dn=>tot+=Object.values(store[dn]||{}).filter(labeled).length);$('prog').textContent=tot+' touched total';
  tabs();srcPane();
}
function go(d){i=Math.max(0,Math.min(D[deal].length-1,i+d));srcManual=false;$('search').value='';render();}
function jump(v){i=parseInt(v)-1;srcManual=false;render();}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1);});
function rows(){const r=[];DEALS.forEach(dn=>D[dn].forEach((a,j)=>{const e=rec(dn,j);if(!labeled(e))return;const badctx=(e.ctxbad||[]).map(k=>a.x[k]).filter(Boolean).map(c=>({doc:c.d,via:c.sh}));const lab=e.label||"";const fac={};Object.keys(FAC).forEach(dim=>{fac[dim]=(e.facets&&e.facets[dim])||(a.fc||{})[dim]||'unknown';});r.push({deal:dn,doc:a.doc,atom:a.b.slice(0,180),section:a.s,page:a.page,parser_guess:a.g,my_rec:a.r,label:lab,admission:(AD.indexOf(lab)>=0?lab:(lab?'keep':'')),coarse:(FC[lab]||null),facets:fac,note:e.note||"",parser_issues:e.issues||[],bad_context:badctx});}));return r;}
function payload_(){return {suggestions:collectSuggest(),atoms:rows()};}
function download_(){const b=new Blob([JSON.stringify(payload_(),null,1)],{type:'application/json'});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download='gold_labels.json';a.click();}
function copy_(){navigator.clipboard.writeText(JSON.stringify(payload_()));alert('copied '+rows().length+' graded atoms + suggestions');}
function applyHash(){const h=new URLSearchParams(location.hash.slice(1));if(h.get('deal')&&D[h.get('deal')])deal=h.get('deal');if(h.get('atom')){i=Math.max(0,Math.min(D[deal].length-1,parseInt(h.get('atom'))-1));srcManual=false;}if(h.get('doc')){srcIdx=parseInt(h.get('doc'))-1;srcManual=true;}render();}
window.addEventListener('hashchange',applyHash); if(location.hash)applyHash(); else render();
</script></body></html>"""

README = ("parser-os gold labeler\n======================\n\n"
"1. Double-click labeler.html (opens in your browser, works offline).\n"
"2. One tab per deal. Left = source doc (PDFs render inline; click 'open' if not).\n"
"   Right = the atom, my recommended type, the parser's guess, and the 46-type picker.\n"
"3. For each atom: click the correct type (or 'use rec'); toggle any parser-wrong\n"
"   flags; thumbs-down any bad cross-doc link; write the WHY in the note.\n"
"4. Hit 'download gold' (or 'copy') and send me gold_labels.json.\n\n"
"Your progress autosaves in the browser (localStorage) so you can stop/resume.\n")

def build(deal_paths, label_names=None, n_per=16, rare=None, out_dir="labeler_gold", out_zip="labeler_gold.zip", prebuilt=None):
    """deal_paths: {name: [file,...]}. prebuilt: optional {name:(corpus,srcdump,pathmap)}
    to skip parsing. label_names: optional {name: pretty tab label}."""
    OUT=Path(out_dir)
    if OUT.exists(): shutil.rmtree(OUT)
    (OUT/"sources").mkdir(parents=True)
    out_deals,out_src,out_docs={},{},{}
    for name,paths in deal_paths.items():
        if not paths: print(name,"- no docs"); continue
        if prebuilt and name in prebuilt: corpus,srcdump,pathmap=prebuilt[name]
        else: corpus,srcdump,pathmap=parse_deal(paths)
        if not corpus: print(name,"- 0 atoms, skipped"); continue
        ddir=OUT/"sources"/name; ddir.mkdir(parents=True,exist_ok=True)
        relmap={}
        for doc,p in pathmap.items():
            base=Path(p).name
            try: shutil.copy(p,ddir/base)
            except Exception as e: print("  ! copy",base,str(e)[:40])
            relmap[doc]=f"sources/{name}/{base}"
        # ---- powerful cross-document context (v2: entity + semantic + conflict) ----
        from _context_v2 import build_index as _bidx, context as _ctx
        ents, ctypes, _simfn, _kind = _bidx(corpus)
        tokdocs = defaultdict(set)
        for j, e in enumerate(ents):
            for key in ("ref", "site"):
                for tok in e[key]: tokdocs[(key, tok)].add(corpus[j]["doc"])
        for j, a in enumerate(corpus):
            hc = any((tokdocs[(key, tok)] - {a["doc"]}) for key in ("ref","site") for tok in ents[j][key])
            a["_hascross"] = hc; a["_hasconf"] = hc and bool(ents[j]["$"])
        rareset = rare or set()
        groups = defaultdict(list)
        for j, a in enumerate(corpus):
            leaf = a["section"].split(" > ")[-1] if a.get("section") else "(none)"
            groups[(a["doc"], leaf)].append(j)
        def _key(j):
            return (0 if corpus[j]["_hasconf"] else 1, 0 if ctypes[j] in rareset else 1,
                    0 if corpus[j]["_hascross"] else 1, len(corpus[j]["body"]))
        gl = [sorted(g, key=_key) for g in groups.values()]
        pick_idx, seen, tcount = [], set(), Counter()
        while len(pick_idx) < n_per and any(gl):
            prog = False
            for g in gl:
                if not g: continue
                j = g.pop(0); kb = corpus[j]["body"][:40].lower()
                if kb in seen or tcount[corpus[j]["type"]] >= 4: continue
                pick_idx.append(j); seen.add(kb); tcount[corpus[j]["type"]] += 1; prog = True
                if len(pick_idx) >= n_per: break
            if not prog: break
        out_deals[name] = []
        for j in pick_idx:
            a = corpus[j]; xc = _ctx(j, corpus, ents, ctypes, _simfn)
            r = recommend(a["body"], a.get("section",""), a["type"])
            out_deals[name].append({"g":to_v2(a["type"]),"r":r,"f":a["flag"],
                "b":a["body"],"s":a["section"].split(" > ")[-1] if a.get("section") else "",
                "doc":a["doc"],"path":relmap.get(a["doc"],""),"page":a["page"],
                "isPdf":relmap.get(a["doc"],"").lower().endswith(".pdf"),
                "x":xc,"ss":(0 if xc else 1),"fc":facet_prefill(a["body"], r)})
        out_src[name]=srcdump
        out_docs[name]=[{"doc":Path(p).stem,"path":relmap.get(Path(p).stem,""),
                         "isPdf":p.lower().endswith(".pdf"),"n":len(srcdump.get(Path(p).stem,[]))} for p in paths if Path(p).stem in relmap]
        D=out_deals[name]; rec_hit=sum(1 for a in D if a["r"]!=a["g"])
        print(f"{name:28s} -> {len(D):2d} atoms, {sum(1 for a in D if a['x'])} w/cross, {sum(1 for a in D if a.get('ss'))} single-src, {rec_hit} rec!=guess, {len(paths)} docs")
    payload={"T":TYPES,"D":out_deals,"SRC":out_src,"DOCS":out_docs,
             "C":COARSE,"F":FACETS,"FC":FINE2COARSE,"AD":ADMISSION}
    html=HTML.replace("__GUIDE__",GUIDE).replace("__DATA__",json.dumps(payload,ensure_ascii=False))
    (OUT/"labeler.html").write_text(html,encoding="utf-8")
    (OUT/"README.txt").write_text(README,encoding="utf-8")
    if os.path.exists(out_zip): os.remove(out_zip)
    with zipfile.ZipFile(out_zip,"w",zipfile.ZIP_DEFLATED) as z:
        for f in OUT.rglob("*"):
            if f.is_file(): z.write(f,f.relative_to(OUT.parent))
    mb=os.path.getsize(out_zip)/1e6
    natoms=sum(len(v) for v in out_deals.values())
    print(f"\nWROTE {out_zip}  ({mb:.1f} MB)  -  {len(out_deals)} deals, {natoms} atoms to grade")
    return out_deals
