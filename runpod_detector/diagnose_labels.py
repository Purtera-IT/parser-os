"""Diagnose WHY the gate is stuck: how noisy are the 14b labels (train + test)?
Relabel a sample with qwen3:32b (more reliable) and measure agreement with the
14b labels stored in _rubric_gate_data.json. Low agreement = label noise is the cap."""
import json, random, urllib.request, os
OLLAMA="http://localhost:11434"; MODEL="qwen3:32b"
N=int(os.environ.get("N","300"))
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
        headers={"Content-Type":"application/json"}),timeout=120)
    o=json.loads(r.read())["response"].lower()
    return 1 if ('"typed"' in o or o.rfind("typed")>o.rfind("keep")) else 0
d=json.load(open("_rubric_gate_data.json"))
def diag(split):
    rows=d[split]; random.seed(3); samp=random.sample(rows,min(N,len(rows)))
    agree=0; flip_kt=0; flip_tk=0
    for i,r in enumerate(samp):
        new=ask(r["text"]); old=r["y"]
        if new==old: agree+=1
        elif old==0 and new==1: flip_kt+=1   # data says keep, 32b says typed
        else: flip_tk+=1
        if i%50==0: print(f"  {split} {i+1}/{len(samp)}",flush=True)
    n=len(samp)
    print(f"\n[{split}] 14b-label vs fresh-32b agreement = {agree/n:.3f}  ({agree}/{n})")
    print(f"   disagreements: data=keep/32b=typed {flip_kt} | data=typed/32b=keep {flip_tk}")
    print(f"   stored typed-frac {sum(r['y'] for r in samp)/n:.2f}")
    return agree/n
print(f"=== diagnosing label noise with {MODEL} (N={N} each) ===")
tr=diag("train"); te=diag("test")
print(f"\n================ DIAGNOSIS ================")
print(f"TRAIN label self-consistency (14b vs 32b): {tr:.3f}")
print(f"TEST  label self-consistency (14b vs 32b): {te:.3f}  <- caps the measurable accuracy")
print("If TEST agreement ~0.8, no model can score above ~0.8 vs these labels (noisy yardstick).")
print("If agreement ~0.95, labels are clean -> the gap is the MODEL, not the labels.")
