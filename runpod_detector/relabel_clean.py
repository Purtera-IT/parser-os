"""Clean the gate labels by ENSEMBLE AGREEMENT (runs on the pod with local Ollama qwen3:32b).
Relabels every row with 32b, keeps ONLY rows where the stored 14b label AND the fresh 32b
label AGREE. Drops disagreements (guess-free). Writes the clean set back to
_rubric_gate_data.json. Diagnosis showed train 14b-vs-32b 0.82 / test 0.87 -> cleaning lifts
the measurable ceiling toward ~0.95 and removes train noise.

Run on the pod AFTER: ollama serve + `ollama pull qwen3:32b`.
"""
import json, urllib.request, os
OLLAMA=os.environ.get("OLLAMA","http://localhost:11434"); MODEL=os.environ.get("M","qwen3:32b")
RUBRIC="""You label clauses from B2B managed-services deal documents (SOWs/quotes) across ANY
trade. Decide ONE thing by ROLE: is this a SUBSTANTIVE deal item (TYPED) or scaffolding/boilerplate (KEEP)?
1. Document navigation (index, table of contents, section/sheet/page refs) -> KEEP.
2. Legend / notation key / definition table -> KEEP. Exception: a stated equipment QUANTITY -> TYPED.
3. Schema/field metadata ("ServiceNow Fields:", "Available Fields:", "Index: N | <field>") -> KEEP.
4. Form label/scaffold ("Project Name:", "Provider Name:", bare "Key |") -> KEEP the label; TYPED only if real fact value.
5. Contact/signature block (Name|Title|Email) -> TYPED only if the person ACTS in this deal; sales rep/signatory -> KEEP.
6. Names an action/service/requirement/site/price/milestone/condition -> TYPED, even if terse.
Reply with ONLY {"decision":"typed"} or {"decision":"keep"}."""
def ask(text):
    body=json.dumps({"model":MODEL,"prompt":f"{RUBRIC}\n\nCLAUSE:\n{text[:600]}\n\nJSON:",
        "stream":False,"think":False,"options":{"temperature":0}}).encode()
    r=urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate",data=body,
        headers={"Content-Type":"application/json"}),timeout=180)
    o=json.loads(r.read())["response"].lower()
    return 1 if ('"typed"' in o or o.rfind("typed")>o.rfind("keep")) else 0
d=json.load(open("_rubric_gate_data.json")); out={}
for split in ("train","test"):
    rows=d[split]; keep=[]; agree=0
    for i,r in enumerate(rows):
        v=ask(r["text"])
        if v==r["y"]: keep.append(r); agree+=1
        if i%250==0: print(f"  {split} {i+1}/{len(rows)} (kept {len(keep)})",flush=True)
    out[split]=keep
    print(f"[{split}] agreement {agree/len(rows):.3f} -> kept {len(keep)}/{len(rows)} clean rows",flush=True)
import shutil; shutil.copy("_rubric_gate_data.json","_rubric_gate_data.noisy.json")
json.dump(out,open("_rubric_gate_data.json","w"))
import numpy as np
print(f"CLEAN SET: train {len(out['train'])} (typed {np.mean([r['y'] for r in out['train']]):.2f}), "
      f"test {len(out['test'])} (typed {np.mean([r['y'] for r in out['test']]):.2f})")
print("Now train: BASE_MODEL=Qwen/Qwen3-Embedding-8B USE_LORA=1 BATCH=32 EPOCHS=6 python train_gate_rubric.py")
