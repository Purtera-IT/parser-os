"""Expand the rubric-labeled gate TRAIN set (keep the SAME held-out for honest compare).
Relabels ADD_N more train-split atoms with the rubric (qwen3:14b) and appends them to
_rubric_gate_data.json. Test set untouched."""
import os, json, sqlite3, hashlib, random, urllib.request, shutil

DB="_training_deepseek.db"; OLLAMA="http://localhost:11434"
MODEL=os.environ.get("M","qwen3:14b"); ADD_N=int(os.environ.get("ADD_N","5000")); HOLDOUT=0.25
RUBRIC="""You label clauses from B2B managed-services deal documents (SOWs/quotes) across ANY
trade. Decide ONE thing by ROLE: is this a SUBSTANTIVE deal item (TYPED) or scaffolding/boilerplate (KEEP)?
1. Document navigation (index, table of contents, section/sheet/page refs) -> KEEP.
2. Legend / notation key / definition table -> KEEP. Exception: a stated equipment QUANTITY -> TYPED.
3. Schema/field metadata ("ServiceNow Fields:", "Available Fields:", "Index: N | <field>") -> KEEP.
4. Form label/scaffold ("Project Name:", "Provider Name:", bare "Key |") -> KEEP the label; TYPED only if real fact value.
5. Contact/signature block (Name|Title|Email) -> TYPED only if the person ACTS in this deal; sales rep/signatory -> KEEP.
6. Names an action/service/requirement/site/price/milestone/condition -> TYPED, even if terse.
Reply with ONLY {"decision":"typed"} or {"decision":"keep"}."""

def split(d):
    h=int(hashlib.sha256((d or "").encode()).hexdigest(),16)
    return "test" if (h%100)/100.0<HOLDOUT else "train"
def ask(text):
    body=json.dumps({"model":MODEL,"prompt":f"{RUBRIC}\n\nCLAUSE:\n{text[:600]}\n\nJSON:",
        "stream":False,"think":False,"options":{"temperature":0}}).encode()
    r=urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate",data=body,
        headers={"Content-Type":"application/json"}),timeout=120)
    o=json.loads(r.read())["response"].lower()
    return 1 if ('"typed"' in o or (o.rfind("typed")>o.rfind("keep"))) else 0

d=json.load(open("_rubric_gate_data.json"))
have=set(r["text"] for r in d["train"])
con=sqlite3.connect(DB)
rows=con.execute("SELECT COALESCE(NULLIF(masked_text,''),raw_text),deal_id FROM training_rows WHERE relation='atom_type' AND label IS NOT NULL AND COALESCE(masked_text,raw_text,'')!=''").fetchall()
con.close()
pool=[(t,dl) for t,dl in rows if split(dl)=="train" and t not in have]
random.seed(7); random.shuffle(pool); pool=pool[:ADD_N]
print(f"existing train={len(d['train'])} test={len(d['test'])} | relabeling {len(pool)} more with {MODEL}",flush=True)
for i,(t,dl) in enumerate(pool):
    d["train"].append({"text":t,"deal":dl,"y":ask(t)})
    if i%250==0: print(f"  {i+1}/{len(pool)}",flush=True)
shutil.copy("_rubric_gate_data.json","_rubric_gate_data.bak.json")
json.dump(d,open("_rubric_gate_data.json","w"))
import numpy as np
print(f"DONE: train now {len(d['train'])} (typed-frac {np.mean([r['y'] for r in d['train']]):.2f}), test {len(d['test'])}")
