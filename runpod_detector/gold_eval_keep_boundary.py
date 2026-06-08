"""Build a GOLD yardstick for the keep-vs-typed boundary (88% of type-head error).

Our held-out 'truth' is itself noisy teacher labels, so a head that disagrees with
the teacher looks 'wrong' even when it's right. This samples held-out atoms and gets
an INDEPENDENT second opinion from qwen3:32b (a different model family than the
DeepSeek teacher) on the one decision that matters: is this atom a substantive
TYPED item, or boilerplate/metadata to set aside (_keep)?

Outputs a JSONL with, per atom: text, teacher-binary, gate(StageA)-binary,
qwen3-binary. Where qwen3 and teacher AGREE we treat it as gold; the DISAGREEMENTS
are surfaced for human/Claude adjudication. Then we report:
  - teacher-vs-gold agreement  (how noisy IS the teacher on this axis?)
  - gate-vs-teacher  vs  gate-vs-gold  (does the gate look better against clean labels?)

Run AFTER the GPU is free (qwen3:32b is heavy).  Env: N=120 ADJ_MODEL=qwen3:32b
"""
import os, sqlite3, hashlib, json, random, urllib.request, collections
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DB="_training_deepseek.db"; HOLDOUT=0.25
A="runs/type_head_v2/stageA/best"
OLLAMA=os.environ.get("OLLAMA","http://localhost:11434")
ADJ_MODEL=os.environ.get("ADJ_MODEL","qwen3:32b")
N=int(os.environ.get("N","120"))
OUT=os.environ.get("OUT","_gold_keep_boundary.jsonl")

RUBRIC = """You label clauses extracted from B2B services/construction deal documents (SOWs, quotes).
Decide ONE thing: is this clause a SUBSTANTIVE, discrete item worth capturing as structured data,
or is it KEEP (boilerplate, narrative filler, headers, contact/address metadata, section intros,
or generic legalese that is not a discrete actionable/site/commercial/work item)?

TYPED = a real requirement, task, deliverable, site detail, commercial/pricing line, milestone,
        compliance rule, stakeholder action, acceptance criterion, etc.
KEEP  = filler/boilerplate/metadata/headers/narrative that is not a discrete item.

Reply with ONLY a compact JSON object: {"decision":"typed"} or {"decision":"keep"}. No prose."""

def split(d):
    h=int(hashlib.sha256((d or "").encode()).hexdigest(),16)
    return "test" if (h%100)/100.0<HOLDOUT else "train"

def ask(text):
    body=json.dumps({"model":ADJ_MODEL,
        "prompt":f"{RUBRIC}\n\nCLAUSE:\n{text[:600]}\n\nJSON:",
        "stream":False,"think":False,"options":{"temperature":0}}).encode()
    req=urllib.request.Request(f"{OLLAMA}/api/generate",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r:
        out=json.loads(r.read())["response"]
    if '"typed"' in out.lower() or "typed" in out.lower(): return "typed"
    if '"keep"' in out.lower(): return "keep"
    return "typed" if "typed" in out.lower() else "keep"

def main():
    con=sqlite3.connect(DB)
    rows=con.execute("SELECT COALESCE(NULLIF(masked_text,''),raw_text),label,deal_id FROM training_rows WHERE relation='atom_type' AND label IS NOT NULL AND COALESCE(masked_text,raw_text,'')!=''").fetchall()
    con.close()
    te=[(t,l) for t,l,d in rows if split(d)=="test"]
    keep=[x for x in te if x[1]=="_keep"]; typed=[x for x in te if x[1]!="_keep"]
    random.seed(0)
    samp=random.sample(keep,min(N//2,len(keep)))+random.sample(typed,min(N//2,len(typed)))
    random.shuffle(samp)

    tok=AutoTokenizer.from_pretrained(A); mA=AutoModelForSequenceClassification.from_pretrained(A).eval()
    def gate(texts):
        out=[]
        with torch.no_grad():
            for i in range(0,len(texts),32):
                enc=tok(texts[i:i+32],truncation=True,max_length=128,padding=True,return_tensors="pt")
                out+= (torch.softmax(mA(**enc).logits,-1).argmax(-1)).tolist()  # 0=keep 1=typed
        return out
    g=gate([t for t,_ in samp])

    recs=[];
    for j,(t,l) in enumerate(samp):
        teacher = "keep" if l=="_keep" else "typed"
        q = ask(t)
        gate_b = "typed" if g[j]==1 else "keep"
        recs.append({"text":t[:200],"teacher":teacher,"qwen3":q,"gate":gate_b,"orig_label":l})
        if j%20==0: print(f"  adjudicated {j+1}/{len(samp)}",flush=True)

    json.dump(recs,open(OUT,"w"),indent=0)
    # metrics on the AGREED-gold subset
    agree=[r for r in recs if r["teacher"]==r["qwen3"]]
    disagree=[r for r in recs if r["teacher"]!=r["qwen3"]]
    def acc(field,ref,subset): return np.mean([r[field]==r[ref] for r in subset]) if subset else 0
    print(f"\n================ GOLD KEEP-BOUNDARY REPORT ================")
    print(f"sample={len(recs)}  teacher∩qwen3 AGREE={len(agree)} ({len(agree)/len(recs):.0%})  DISAGREE={len(disagree)}")
    print(f"teacher vs qwen3 agreement = {len(agree)/len(recs):.1%}  <- how noisy the teacher is on this axis")
    print(f"\nON THE AGREED-GOLD SUBSET ({len(agree)} atoms, trustworthy labels):")
    print(f"  gate accuracy vs gold = {acc('gate','teacher',agree):.3f}")
    print(f"\nON FULL SAMPLE:")
    print(f"  gate vs teacher = {acc('gate','teacher',recs):.3f}")
    print(f"  gate vs qwen3   = {acc('gate','qwen3',recs):.3f}")
    print(f"\n{len(disagree)} teacher/qwen3 DISAGREEMENTS written to {OUT} for adjudication. Samples:")
    for r in disagree[:12]:
        print(f"  teacher={r['teacher']:5s} qwen3={r['qwen3']:5s} gate={r['gate']:5s} | {r['text'][:80]!r}")

if __name__=="__main__":
    main()
