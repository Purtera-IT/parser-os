"""Find the reproducible typing granularity: have two independent models assign a
CLEAN FACET to typed atoms, measure agreement, and show which original-41 labels
map cleanly vs which are fuzzy. This tells us the taxonomy that can hit >90%.

Output: facet-level agreement (the proposed clean taxonomy) + per-original-label
agreement, so we KNOW which classes to keep, collapse, or rubric.

Env: M1=qwen3:32b M2=qwen3:14b N=160
"""
import os, sqlite3, json, random, urllib.request, collections
import numpy as np

DB="_training_deepseek.db"
OLLAMA=os.environ.get("OLLAMA","http://localhost:11434")
M1=os.environ.get("M1","qwen3:32b"); M2=os.environ.get("M2","qwen3:14b")
N=int(os.environ.get("N","160"))

# proposed universal clean taxonomy (role-based facets, all managed services)
FACETS = ["SITE","COMMERCIAL","WORK","COMPLIANCE","PARTY","TIMING","META"]
RUBRIC = """You type a clause from a B2B managed-services deal (any trade) into ONE facet by ROLE:
- SITE: a place/location, site access, or physical site attribute.
- COMMERCIAL: a price, rate, quantity/BOM line, payment term, or total.
- WORK: an action/service/task/requirement/deliverable/milestone/acceptance the provider does.
- COMPLIANCE: a rule, certification, approval, insurance, or regulatory obligation.
- PARTY: a person/org that ACTS in the deal (PM, site contact, approver).
- TIMING: a deadline, blackout window, or lead-time constraint.
- META: deal-level metadata (project name, provider, document fields).
Reply with ONLY {"facet":"<ONE OF: SITE,COMMERCIAL,WORK,COMPLIANCE,PARTY,TIMING,META>"}."""

def ask(model,text):
    body=json.dumps({"model":model,"prompt":f"{RUBRIC}\n\nCLAUSE:\n{text[:600]}\n\nJSON:",
        "stream":False,"think":False,"options":{"temperature":0}}).encode()
    req=urllib.request.Request(f"{OLLAMA}/api/generate",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r:
        out=json.loads(r.read())["response"].upper()
    for f in FACETS:
        if f in out: return f
    return "WORK"

def judge_all(model,texts):
    print(f"  {model}: {len(texts)} atoms ...",flush=True)
    o=[]
    for j,t in enumerate(texts):
        o.append(ask(model,t))
        if j%40==0: print(f"    {model} {j+1}/{len(texts)}",flush=True)
    return o

def main():
    con=sqlite3.connect(DB)
    rows=con.execute("SELECT COALESCE(NULLIF(masked_text,''),raw_text),label FROM training_rows WHERE relation='atom_type' AND label IS NOT NULL AND label!='_keep' AND COALESCE(masked_text,raw_text,'')!=''").fetchall()
    con.close()
    by=collections.defaultdict(list)
    for t,l in rows: by[l].append(t)
    random.seed(0)
    samp=[]
    per=max(2,N//len(by))
    for l,ts in by.items():
        for t in random.sample(ts,min(per,len(ts))): samp.append((t,l))
    random.shuffle(samp); samp=samp[:N]
    texts=[t for t,_ in samp]; orig=[l for _,l in samp]
    print(f"typed-atom facet agreement: {len(samp)} atoms, {len(by)} original labels, {len(FACETS)} facets")
    r1=judge_all(M1,texts); r2=judge_all(M2,texts)

    fa=np.mean([a==b for a,b in zip(r1,r2)])
    print(f"\n================ FINE-TYPE FACET AGREEMENT ================")
    print(f"FACET-level agreement ({M1} vs {M2}) = {fa:.3f}   (>0.85 -> this taxonomy can hit 90)")
    # per original label: how often the two models agree on the facet
    perlbl=collections.defaultdict(lambda:[0,0])
    facet_of=collections.defaultdict(collections.Counter)
    for o,a,b in zip(orig,r1,r2):
        perlbl[o][1]+=1
        if a==b: perlbl[o][0]+=1
        if a==b: facet_of[o][a]+=1
    print("\nper ORIGINAL label -> agreement & dominant facet (low agreement = fuzzy/needs rubric):")
    for l in sorted(perlbl, key=lambda k:-perlbl[k][1]):
        ok,n=perlbl[l]; dom=facet_of[l].most_common(1)
        domf=dom[0][0] if dom else "?"
        flag="" if n<3 else (" <-- FUZZY" if ok/n<0.6 else "")
        print(f"  {l:26s} agree {ok}/{n} -> {domf}{flag}")
    json.dump([{"text":t[:160],"orig":o,"m1":a,"m2":b} for (t,_),o,a,b in zip(samp,orig,r1,r2)],
              open("_type_agreement.jsonl","w"))
    print(f"\nINTERPRETATION: facet agreement is the ceiling for a model trained on this taxonomy.")
    print("Labels with low agreement either collapse into their dominant facet or need a sub-rubric.")

if __name__=="__main__":
    main()
