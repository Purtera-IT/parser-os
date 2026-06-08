"""Build the UNIVERSAL symbol-detector dataset with SUPERVISED GOLD per page and a
HELD-OUT-BY-FIRM split so we learn EXACTLY whether it generalizes.

- Gold target per page: every harvested region is VLM-labeled SYMBOL/background
  (cached -> resumable, mostly free after first pass). SYMBOL boxes -> YOLO labels
  (class 0). One class = universal across firms; legend assigns meaning later.
- Split BY FIRM: whole firms are held out as TEST. Train/val come from the rest.
  So test mAP = cross-firm generalization (not memorization of seen drawings).

Output:
  dataset/images/{train,val,test}/*.png
  dataset/labels/{train,val,test}/*.txt
  dataset/data.yaml         (train+val for fitting)
  dataset/test.yaml         (held-out firms for the generalization verdict)
  dataset/MANIFEST.json     (counts, per-firm, per-split)

Run locally (resumable):  python -X utf8 runpod_detector/prepare_yolo_data.py
"""
import os, io, sys, glob, json, hashlib, base64, random
from pathlib import Path
# Self-contained: add project root to path + load .env (no _dbg dependency).
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("SOWSMITH_VISION_CACHE_DB", "_vision_rip_out/vcache.db")
_envp = _ROOT / ".env"
if _envp.exists():
    for _ln in _envp.read_text(encoding="utf-8").splitlines():
        _ln = _ln.strip()
        if _ln and not _ln.startswith("#") and "=" in _ln:
            _k, _, _v = _ln.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
import fitz
from PIL import Image
from app.core import llm_client as L
from app.core.schematic_crop_harvester import harvest_page

OUT = "runpod_detector/dataset"
for sp in ("train", "val", "test"):
    os.makedirs(f"{OUT}/images/{sp}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{sp}", exist_ok=True)
DPI, PT, MAX_PAGES = 150, 72.0, 22
OBJ = ("Small crop from an electrical/low-voltage drawing. Reply ONE word: SYMBOL "
       "if a discrete device/symbol icon; BACKGROUND if text/line/blank/partial.")

# Hold out whole firms for the generalization test (diverse disciplines).
HELD_OUT = {"va_electrical", "ildot_firealarm", "uri_telecom", "analytix_av"}

cache_path = "runpod_detector/objectness_box_labels.jsonl"
seen = {}
if os.path.exists(cache_path):
    for l in open(cache_path, encoding="utf-8"):
        try: r = json.loads(l); seen[r["sha"]] = r["lab"]
        except: pass

def is_symbol(png):
    k = hashlib.sha256(png).hexdigest()[:16]
    if k in seen: return seen[k] == "SYMBOL"
    r = (L.complete_vision(OBJ, base64.b64encode(png).decode(), max_tokens=4) or "").strip().upper()
    lab = "SYMBOL" if "SYMBOL" in r else "BACKGROUND"
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sha": k, "lab": lab}) + "\n")
    seen[k] = lab
    return lab == "SYMBOL"

pdfs = sorted(glob.glob("_schematic_corpus/*.pdf"))
mar = "real_data_cases/LOWVOLT_002_MARRIOTT_ATLANTA_T/artifacts/2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf"
if os.path.exists(mar): pdfs.append(mar)

random.seed(0)
manifest = {"firms": {}, "splits": {"train": 0, "val": 0, "test": 0}, "boxes": 0}
for path in pdfs:
    firm = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")[:30]
    held = any(h in firm for h in HELD_OUT)
    try: doc = fitz.open(path)
    except Exception: continue
    fimgs = fboxes = 0
    for pno in range(min(doc.page_count, MAX_PAGES)):
        page = doc[pno]
        regions = harvest_page(page, pno, dpi=DPI)
        if not regions: continue
        pix = page.get_pixmap(dpi=DPI, alpha=False); W, H = pix.width, pix.height
        scale = DPI / PT; lines = []
        for bbox_pt, png in regions:
            if not is_symbol(png): continue
            x0, y0, x1, y1 = [v*scale for v in bbox_pt]
            cx, cy, bw, bh = (x0+x1)/2/W, (y0+y1)/2/H, (x1-x0)/W, (y1-y0)/H
            if 0 < cx < 1 and 0 < cy < 1 and bw > 0 and bh > 0:
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines: continue
        sp = "test" if held else ("val" if random.random() < 0.12 else "train")
        name = f"{firm}_p{pno}"
        Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").save(f"{OUT}/images/{sp}/{name}.png")
        open(f"{OUT}/labels/{sp}/{name}.txt", "w").write("\n".join(lines))
        manifest["splits"][sp] += 1; manifest["boxes"] += len(lines)
        fimgs += 1; fboxes += len(lines)
    manifest["firms"][firm] = {"held_out": held, "imgs": fimgs, "boxes": fboxes}
    print(f"  {firm[:26]:26} {'[TEST]' if held else '[train]':8} imgs={fimgs:>3} boxes={fboxes:>4} | ${L.usage_snapshot()['cost_usd']:.2f}", flush=True)

base = os.path.abspath(OUT)
open(f"{OUT}/data.yaml","w").write(f"path: {base}\ntrain: images/train\nval: images/val\nnames:\n  0: symbol\n")
open(f"{OUT}/test.yaml","w").write(f"path: {base}\ntrain: images/test\nval: images/test\nnames:\n  0: symbol\n")
json.dump(manifest, open(f"{OUT}/MANIFEST.json","w"), indent=2)
print(f"\nDATASET: train={manifest['splits']['train']} val={manifest['splits']['val']} "
      f"test(held-out firms)={manifest['splits']['test']} | total boxes={manifest['boxes']}")
print(f"held-out firms: {sorted(HELD_OUT)}  | VLM cost ${L.usage_snapshot()['cost_usd']:.2f}")
