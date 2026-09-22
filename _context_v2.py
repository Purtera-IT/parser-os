"""Powerful cross-document context for the gold labeler.

Three signals, combined and typed:
  1. ENTITY  — shared resolved deal-ref or site-id (NOT bare numbers/zips, NOT
     date-only). High precision.
  2. SEMANTIC — local TF-IDF cosine over words, so facts that share MEANING but
     no literal token still link (the questionnaire / RFQ / pricing docs that
     currently get 0 coverage). Offline; swappable for the real embedder later.
  3. CONFLICT — same entity anchor but a $ value differs -> flagged.

Typed edges: a weak cross-family link (e.g. physical_site <-> risk) is dropped
unless it's also a strong ref match or a high semantic match. Atoms with no link
are marked single_source (a real signal: lower-trust, no corroboration).
"""
import re, math, os
from collections import Counter
from _labeler_core import _ents, recommend

# local sentence-embedding fallback (bge-small), loaded once, lazily
global _LOCAL_MODEL, _LOCAL_MODEL_TRIED
_LOCAL_MODEL = None
_LOCAL_MODEL_TRIED = False
def _local_model():
    global _LOCAL_MODEL, _LOCAL_MODEL_TRIED
    if _LOCAL_MODEL_TRIED:
        return _LOCAL_MODEL
    _LOCAL_MODEL_TRIED = True
    name = os.environ.get("SOWSMITH_LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    try:
        from sentence_transformers import SentenceTransformer
        _LOCAL_MODEL = SentenceTransformer(name)
    except Exception:
        _LOCAL_MODEL = None
    return _LOCAL_MODEL

FAMILY = {
    "physical_site": "SITE", "site_attribute": "SITE", "site_access_window": "SITE",
    "site_access_restriction": "SITE", "site_infrastructure": "SITE", "site_room_mix": "SITE",
    "site_implementation_note": "SITE", "site_budget": "SITE", "site_allocation": "SITE",
    "commercial_total": "MONEY", "payment_term": "MONEY", "bom_line": "MONEY",
    "service_line": "MONEY", "pricing_assumption": "MONEY", "lead_time_constraint": "MONEY",
    "milestone_phase": "SCHED", "task": "SCHED", "deliverable": "SCHED", "cutover_step": "SCHED",
    "integration_checkpoint": "SCHED", "blackout_date_range": "SCHED", "deadline": "SCHED",
    "stakeholder": "PEOPLE", "approval_authority": "PEOPLE", "approval_decision": "PEOPLE", "signatory": "PEOPLE",
    "requirement": "REQ", "acceptance_criterion": "REQ", "electrical_acceptance_test": "REQ",
    "compliance_classification": "COMPLY", "compliance_rule": "COMPLY",
    "risk": "RISK", "mitigation": "RISK", "dependency": "RISK",
    "submission_req": "RFP", "eval_criterion": "RFP", "bonding_insurance": "RFP",
    "contract_term": "RFP", "addendum_qa": "RFP", "change_order_rule": "RFP",
    "data_flow_step": "INTEG", "system_mapping": "INTEG", "metadata_requirement": "INTEG",
}
# generic types link with anything
GENERIC = {"scope_item", "assumption", "exclusion", "constraint", "customer_instruction",
           "open_question", "deal_metadata", "vendor_line_item", "boilerplate",
           "dropped_sheet", "needs_extractor"}

# (sim_min, sim_strong) per similarity backend
THRESH = {"embed": (0.55, 0.70), "bge": (0.58, 0.68), "tfidf": (0.42, 0.60)}

def _fam(t): return FAMILY.get(t, "GEN")
def _compat(ta, tb):
    if ta in GENERIC or tb in GENERIC: return True
    return _fam(ta) == _fam(tb)

# site ids that differ only by a -NN suffix are the same site
_SITE_SUFFIX = re.compile(r"-\d+$")
def _site_canon(s): return _SITE_SUFFIX.sub("", s)
def _canon_set(ss): return {_site_canon(s) for s in ss}

# ---- local TF-IDF ------------------------------------------------------------
#
#
_WORD = re.compile(r"[a-z]{3,}")
def _toks(s): return _WORD.findall(s.lower())

def _cos(va, vb):
    if len(va) > len(vb): va, vb = vb, va
    return sum(x * vb.get(w, 0.0) for w, x in va.items())

def build_index(corpus):
    """Return (ents, types, simfn, kind). simfn(i,j)->cosine. Uses the REAL
    embedder when its endpoint is reachable (no train/serve skew with the graph
    head); falls back to local TF-IDF cosine when it's down."""
    ents = [_ents(a["body"] + " " + a.get("section", "")) for a in corpus]
    types = [recommend(a["body"], a.get("section", ""), a["type"]) for a in corpus]
    texts = [a["body"] + " " + a.get("section", "") for a in corpus]
    # 1) the real embedder, if reachable
    try:
        from app.core.embedding_retrieval import embedding_endpoint_reachable, embed_texts
        if embedding_endpoint_reachable():
            import numpy as np
            M = embed_texts(texts)
            if M is not None and len(M) == len(corpus):
                def simfn(i, j, _M=M): return float(_M[i] @ _M[j])
                return ents, types, simfn, "embed"
    except Exception:
        pass
    # 2) local bge-small
    if not os.environ.get("SOWSMITH_NO_LOCAL_EMBED"):
        try:
            import numpy as np
            mdl = _local_model()
            if mdl is not None:
                M = np.asarray(mdl.encode(texts, normalize_embeddings=True,
                                          batch_size=64, show_progress_bar=False),
                               dtype="float32")
                def simfn(i, j, _M=M): return float(_M[i] @ _M[j])
                return ents, types, simfn, "bge"
        except Exception:
            pass
    # 3) offline TF-IDF
    toks = [_toks(t) for t in texts]
    df = Counter()
    for d in toks:
        for w in set(d): df[w] += 1
    N = max(1, len(toks))
    idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}
    vecs = []
    for d in toks:
        tf = Counter(d); v = {w: c / max(1, len(d)) * idf.get(w, 0.0) for w, c in tf.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / nrm for w, x in v.items()})
    def simfn(i, j, _v=vecs): return _cos(_v[i], _v[j])
    return ents, types, simfn, "tfidf"

def context(i, corpus, ents, types, simfn, k=6, kind="tfidf"):
    a, ea, ta = corpus[i], ents[i], types[i]
    sim_min, sim_strong = THRESH.get(kind, THRESH["tfidf"])
    ca = _canon_set(ea["site"])
    cand = []
    for j, b in enumerate(corpus):
        if j == i or b["doc"] == a["doc"]: continue
        eb = ents[j]
        ref = ea["ref"] & eb["ref"]
        site = ea["site"] & eb["site"]
        cb = _canon_set(eb["site"])
        site_c = ca & cb
        sim = simfn(i, j)
        anchor = bool(ref or site or site_c)
        # both atoms name sites, and they are different sites: only a ref or a
        # near-duplicate text may link them
        if ca and cb and not site_c and not ref and sim < 0.85:
            continue
        if not anchor and sim < sim_min: continue
        # typed edge: cross-family links are weak unless anchored or near-identical
        #
        fam_ok = _compat(ta, types[j])
        # conflict only when both atoms are about the same thing
        same_thing = anchor or (_fam(ta) == _fam(types[j]) and sim >= 0.8) or sim >= 0.9
        cf = ""
        av = {x.rstrip(",").strip() for x in ea["$"]}
        bv = {x.rstrip(",").strip() for x in eb["$"]}
        da, db = sorted(av - bv), sorted(bv - av)
        if same_thing and av and bv and (da or db):
            cf = f"$ {', '.join(da[:2])} vs {', '.join(db[:2])}"
        # same canonical site but different suffixed ids
        #
        if not cf and site_c and not site:
            va = sorted(s for s in ea["site"] if _site_canon(s) in site_c)
            vb = sorted(s for s in eb["site"] if _site_canon(s) in site_c)
            if va != vb:
                cf = f"site id {','.join(va[:1])} vs {','.join(vb[:1])}"
        via = []
        if ref: via.append("ref " + ",".join(list(ref)[:1]))
        if site: via.append("site " + ",".join(list(site)[:1]))
        elif site_c: via.append("site~ " + ",".join(list(site_c)[:1]))
        if sim >= sim_min: via.append(f"~text {sim:.2f}")
        if not via: continue
        sc = (4 if ref else 0) + (2 if site or site_c else 0) + sim * 3 + (10 if cf else 0)
        if not fam_ok: sc -= 0.5
        cand.append({"d": b["doc"][:18], "t": b["type"], "b": b["body"][:90],
                     "sh": "; ".join(via), "cf": cf, "_sc": sc})
    cand.sort(key=lambda x: -x["_sc"])
    out = cand[:k]
    for o in out: o.pop("_sc", None)
    return out


if __name__ == "__main__":
    import json
    from collections import defaultdict
    for slug in ("optbot", "nyc_migration", "tsc_wireless_rfp"):
        corpus = json.load(open(f"_pool_cache/{slug}.json", encoding="utf-8"))["corpus"]
        ents, types, simfn, kind = build_index(corpus)
        tot = defaultdict(int); hit = defaultdict(int); conf = 0; single = 0; linked = 0
        for i, a in enumerate(corpus):
            xc = context(i, corpus, ents, types, simfn, kind=kind)
            tot[a["doc"]] += 1
            if xc:
                hit[a["doc"]] += 1; linked += 1
                conf += sum(1 for c in xc if c["cf"])
            else:
                single += 1
        print(f"\n=== {slug} [{kind}]: {linked}/{len(corpus)} ({100 * linked // max(1, len(corpus))}%) linked · "
              f"{single} single-source · {conf} conflict-links ===")
        for doc in sorted(tot):
            t, h = tot[doc], hit[doc]
            print(f"   {100 * h // max(1, t):3d}%  {h:4d}/{t:<5d} {doc[:46]}")
