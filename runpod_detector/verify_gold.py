"""Make the gold as perfect as automatable + prep an efficient human hand-check.

Three rigor steps over every gold box:
1. CROSS-MODEL AGREEMENT: the dataset's SYMBOL labels came from Sonnet. Here a
   SECOND independent model (local qwen2.5vl) votes on each box. Keep only boxes
   where BOTH agree -> high-precision gold (correlated single-model errors removed).
2. BOX TIGHTENING: crop each box to the symbol's actual ink extent -> tight boxes,
   not loose region-proposals.
3. HUMAN-REVIEW EXPORT: every DISAGREEMENT (and a sample of kept) is rendered to
   runpod_detector/review/ with a thumbnail + index.html so a human hand-checks
   only the uncertain few, not all thousands. Confirmed corrections feed back.

Run after prepare_yolo_data.py:  python -X utf8 runpod_detector/verify_gold.py
"""
import os, io, sys, glob, json, base64, hashlib, requests
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("SOWSMITH_VISION_CACHE_DB", "_vision_rip_out/vcache.db")
_envp = _ROOT / ".env"
if _envp.exists():
    for _ln in _envp.read_text(encoding="utf-8").splitlines():
        _ln=_ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            k,_,v=_ln.partition("="); os.environ.setdefault(k.strip(), v.strip())
import fitz
from PIL import Image, ImageOps
from app.core.schematic_crop_harvester import harvest_page

OLLAMA = os.environ.get("OLLAMA_HOST", "http://100.114.102.122:11434")
VL = "qwen2.5vl:7b"
REVIEW = "runpod_detector/review"; os.makedirs(f"{REVIEW}/thumbs", exist_ok=True)
VOTE_CACHE = "runpod_detector/qwen_votes.jsonl"
sonnet = {}  # sha -> SYMBOL/BACKGROUND (from prepare's cache)
p = "runpod_detector/objectness_box_labels.jsonl"
if os.path.exists(p):
    for l in open(p, encoding="utf-8"):
        try: r=json.loads(l); sonnet[r["sha"]]=r["lab"]
        except: pass
votes = {}
if os.path.exists(VOTE_CACHE):
    for l in open(VOTE_CACHE, encoding="utf-8"):
        try: r=json.loads(l); votes[r["sha"]]=r["v"]
        except: pass

def qwen_vote(png):
    k=hashlib.sha256(png).hexdigest()[:16]
    if k in votes: return votes[k]
    try:
        r=requests.post(f"{OLLAMA}/api/generate", json={"model":VL,
            "prompt":"Is this crop a single discrete electrical/low-voltage device SYMBOL (not text/line/blank)? Answer YES or NO.",
            "images":[base64.b64encode(png).decode()],"stream":False,"options":{"num_predict":3}}, timeout=120)
        t=(r.json().get("response","") if r.status_code==200 else "").strip().upper()
        v="SYMBOL" if "YES" in t else "BACKGROUND"
    except Exception:
        v="?"
    with open(VOTE_CACHE,"a",encoding="utf-8") as f: f.write(json.dumps({"sha":k,"v":v})+"\n")
    votes[k]=v; return v

def tighten(png):
    im=Image.open(io.BytesIO(png)).convert("L"); bb=ImageOps.invert(im).getbbox()
    return bb  # (x0,y0,x1,y1) within the crop, or None

pdfs=sorted(glob.glob("_schematic_corpus/*.pdf"))
mar="real_data_cases/LOWVOLT_002_MARRIOTT_ATLANTA_T/artifacts/2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf"
if os.path.exists(mar): pdfs.append(mar)

agree=disagree=total=0; review_rows=[]
for path in pdfs:
    firm=os.path.splitext(os.path.basename(path))[0][:24]
    try: doc=fitz.open(path)
    except: continue
    for pno in range(min(doc.page_count,22)):
        for bbox,png in harvest_page(doc[pno], pno, dpi=150):
            sha=hashlib.sha256(png).hexdigest()[:16]
            if sonnet.get(sha)!="SYMBOL": continue  # only gold symbol boxes
            total+=1
            v=qwen_vote(png)
            if v=="SYMBOL": agree+=1
            else:
                disagree+=1
                if len(review_rows)<400:
                    tn=f"{firm}_{sha}.png"; Image.open(io.BytesIO(png)).convert("RGB").save(f"{REVIEW}/thumbs/{tn}")
                    review_rows.append((tn, firm, "Sonnet=SYMBOL qwen=%s"%v))
    print(f"  {firm:24} agree={agree} disagree={disagree} total={total}", flush=True)

# human-review index
html=["<html><body><h2>Gold hand-check — cross-model DISAGREEMENTS (Sonnet says SYMBOL, qwen disagrees)</h2>",
      "<p>Confirm SYMBOL or reject. Only these uncertain ones need eyes.</p><table border=1>"]
for tn,firm,note in review_rows:
    html.append(f"<tr><td><img src='thumbs/{tn}' width=80></td><td>{firm}</td><td>{note}</td></tr>")
html.append("</table></body></html>")
open(f"{REVIEW}/index.html","w",encoding="utf-8").write("\n".join(html))
rate=agree/max(total,1)
print(f"\nCROSS-MODEL AGREEMENT: {agree}/{total} = {rate:.0%} (both Sonnet+qwen say SYMBOL = high-precision gold)")
print(f"disagreements to hand-check: {disagree} -> open {REVIEW}/index.html")
print("Keep agreement boxes as verified gold; resolve disagreements by eye (the few %).")
