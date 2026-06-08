"""Apply the universal keep-vs-typed RUBRIC and prove it raises reproducibility.

Modes:
  agree   : re-judge the same 120 held-out atoms with TWO independent models, BOTH
            given the rubric. If agreement jumps from the 59% no-rubric baseline,
            the rubric removed the ambiguity -> 90% becomes reachable.
  relabel : apply the rubric (qwen3:32b) to label keep-vs-typed for a set of atoms,
            writing rubric-consistent labels for gate train/eval.

Env: OLLAMA, M1=qwen3:32b, M2=qwen3:14b, N=120, HOLDOUT=0.25
"""
import os, sqlite3, hashlib, json, random, urllib.request, sys
import numpy as np

DB="_training_deepseek.db"; HOLDOUT=float(os.environ.get("HOLDOUT","0.25"))
OLLAMA=os.environ.get("OLLAMA","http://localhost:11434")
M1=os.environ.get("M1","qwen3:32b"); M2=os.environ.get("M2","qwen3:14b")
N=int(os.environ.get("N","120"))

RUBRIC = """You label clauses from B2B managed-services deal documents (SOWs/quotes) across ANY
trade (AV, cabling, electrical, fire-alarm, managed-IT). Decide ONE thing by ROLE, not topic:
is this a SUBSTANTIVE deal item (TYPED) or scaffolding/boilerplate (KEEP)?

Apply in order:
1. Document navigation (index, table of contents, "Section 3.2", sheet/page refs) -> KEEP.
2. Legend / notation key / definition table -> KEEP. Exception: a stated equipment QUANTITY
   ("48-PORT PATCH PANEL x 12") -> TYPED (BOM line).
3. Schema/field metadata ("ServiceNow Fields:", "Available Fields:", "Index: N | <field>") -> KEEP.
4. Form label/scaffold ("Project Name:", "Provider Name:", bare "Key |") -> KEEP the label;
   TYPED only if it carries a real fact value.
5. Contact/signature block (Name|Title|Email) -> TYPED only if the person ACTS in this deal
   (PM, site contact, approver). Sales rep / quote signatory -> KEEP.
6. Names an action/service the provider performs, or a requirement/site/price/milestone/
   condition -> TYPED, even if terse ("OS patching support", "test the HA pair").

Reply with ONLY {"decision":"typed"} or {"decision":"keep"}."""

def split(d):
    h=int(hashlib.sha256((d or "").encode()).hexdigest(),16)
    return "test" if (h%100)/100.0<HOLDOUT else "train"

def ask(model, text):
    body=json.dumps({"model":model,"prompt":f"{RUBRIC}\n\nCLAUSE:\n{text[:600]}\n\nJSON:",
        "stream":False,"think":False,"options":{"temperature":0}}).encode()
    req=urllib.request.Request(f"{OLLAMA}/api/generate",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r:
        out=json.loads(r.read())["response"].lower()
    if '"typed"' in out: return "typed"
    if '"keep"' in out: return "keep"
    return "typed" if out.rfind("typed")>out.rfind("keep") else "keep"

def load():
    con=sqlite3.connect(DB)
    rows=con.execute("SELECT COALESCE(NULLIF(masked_text,''),raw_text),label,deal_id FROM training_rows WHERE relation='atom_type' AND label IS NOT NULL AND COALESCE(masked_text,raw_text,'')!=''").fetchall()
    con.close()
    return rows

def gold_sample():
    rows=load(); te=[(t,l) for t,l,d in rows if split(d)=="test"]
    keep=[x for x in te if x[1]=="_keep"]; typed=[x for x in te if x[1]!="_keep"]
    random.seed(0)
    s=random.sample(keep,min(N//2,len(keep)))+random.sample(typed,min(N//2,len(typed)))
    random.shuffle(s); return s

def judge_all(model, texts):
    """All calls for ONE model in a single pass (so Ollama loads it once, no thrash)."""
    print(f"  judging {len(texts)} atoms with {model} ...", flush=True)
    out=[]
    for j,t in enumerate(texts):
        out.append(ask(model,t))
        if j%20==0: print(f"    {model} {j+1}/{len(texts)}",flush=True)
    return out

def agree():
    s=gold_sample()
    texts=[t for t,_ in s]
    teach=["keep" if l=="_keep" else "typed" for _,l in s]
    print(f"re-judging {len(s)} atoms with {M1} then {M2}, both under the RUBRIC ...")
    r1=judge_all(M1, texts)      # one full pass on M1
    r2=judge_all(M2, texts)      # then one full pass on M2
    a12=np.mean([a==b for a,b in zip(r1,r2)])
    a1t=np.mean([a==b for a,b in zip(r1,teach)])
    print("\n================ RUBRIC AGREEMENT TEST ================")
    print(f"NO-rubric baseline (teacher vs qwen3)         : 0.592")
    print(f"WITH rubric: {M1} vs {M2} agreement           : {a12:.3f}   <- did ambiguity drop?")
    print(f"WITH rubric: {M1} vs original teacher label   : {a1t:.3f}")
    print(f"\nINTERPRETATION: if {M1}/{M2} agreement >> 0.59, the rubric made the boundary")
    print("reproducible -> the label ceiling rose -> a model trained on rubric labels can chase 90%.")
    json.dump([{"text":t[:200],"m1":a,"m2":b,"teacher":c} for (t,_),a,b,c in zip(s,r1,r2,teach)],
              open("_rubric_agreement.jsonl","w"))

def relabel():
    """Relabel held-out (full) + a train sample with the rubric for gate train/eval."""
    rows=load()
    te=[(t,d) for t,l,d in rows if split(d)=="test"]
    trall=[(t,d) for t,l,d in rows if split(d)=="train"]
    random.seed(1); tr=random.sample(trall, min(int(os.environ.get("TRAIN_N","6000")), len(trall)))
    out={"train":[], "test":[]}
    for name,subset in (("test",te),("train",tr)):
        print(f"relabeling {name}: {len(subset)} atoms with {M1} (rubric) ...",flush=True)
        for j,(t,d) in enumerate(subset):
            out[name].append({"text":t,"deal":d,"y":1 if ask(M1,t)=="typed" else 0})
            if j%200==0: print(f"  {name} {j+1}/{len(subset)}",flush=True)
    json.dump(out, open("_rubric_gate_data.json","w"))
    nt=sum(r["y"] for r in out["train"]); print(f"\nwrote _rubric_gate_data.json (train typed frac {nt/len(out['train']):.2f})")

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "agree"
    (agree if mode=="agree" else relabel)()
