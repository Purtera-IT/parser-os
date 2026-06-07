"""Build a CLASS-AGNOSTIC symbol-detection YOLO dataset from the schematic corpus.

Universal by design: ONE class = "symbol". The detector learns to localize symbol
glyphs on any drawing (firm-independent); per-document LegendIndex assigns meaning
later. So a new firm's new symbols need ZERO retraining.

For each page: render full image, harvest candidate regions (with bbox), and use
the cached VLM objectness label (SYMBOL vs background). Every SYMBOL box becomes a
YOLO bbox (class 0). Output: images/, labels/, data.yaml -> upload to RunPod.

Run locally (uses the VLM only for crops not already cached -> mostly free):
    python -X utf8 runpod_detector/prepare_yolo_data.py
"""
import os, io, glob, json, hashlib, base64
os.environ.setdefault("SOWSMITH_VISION_CACHE_DB", "_vision_rip_out/vcache.db")
import _dbg_time_deepseek  # noqa  (loads .env)
import fitz
from PIL import Image
from app.core import llm_client as L
from app.core.schematic_crop_harvester import harvest_page

OUT = "runpod_detector/dataset"
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
DPI = 150
PT = 72.0
OBJ_PROMPT = ("Small crop from an electrical/low-voltage drawing. Reply ONE word: "
              "SYMBOL if it is a discrete device/symbol icon; BACKGROUND if text/line/"
              "blank/partial. One word only.")

cache_path = "runpod_detector/objectness_box_labels.jsonl"
seen = {}
if os.path.exists(cache_path):
    for l in open(cache_path, encoding="utf-8"):
        try: r = json.loads(l); seen[r["sha"]] = r["lab"]
        except: pass

def is_symbol(png):
    k = hashlib.sha256(png).hexdigest()[:16]
    if k in seen: return seen[k] == "SYMBOL"
    r = (L.complete_vision(OBJ_PROMPT, base64.b64encode(png).decode(), max_tokens=4) or "").strip().upper()
    lab = "SYMBOL" if "SYMBOL" in r else "BACKGROUND"
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sha": k, "lab": lab}) + "\n")
    seen[k] = lab
    return lab == "SYMBOL"

pdfs = sorted(glob.glob("_schematic_corpus/*.pdf"))
mar = "real_data_cases/LOWVOLT_002_MARRIOTT_ATLANTA_T/artifacts/2026-04-10 100% DD - MARRIOTT ATLANTA - T.pdf"
if os.path.exists(mar): pdfs.append(mar)

n_imgs = n_boxes = 0
for pi_path in pdfs:
    stem = os.path.splitext(os.path.basename(pi_path))[0].replace(" ", "_")[:30]
    try: doc = fitz.open(pi_path)
    except Exception: continue
    for pno in range(min(doc.page_count, 14)):
        page = doc[pno]
        regions = harvest_page(page, pno, dpi=DPI)
        if not regions: continue
        pix = page.get_pixmap(dpi=DPI, alpha=False)
        W, H = pix.width, pix.height
        scale = DPI / PT
        lines = []
        for bbox_pt, png in regions:
            if not is_symbol(png): continue
            x0, y0, x1, y1 = [v * scale for v in bbox_pt]
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            if 0 < cx < 1 and 0 < cy < 1 and bw > 0 and bh > 0:
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines: continue
        name = f"{stem}_p{pno}"
        Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB").save(f"{OUT}/images/{name}.png")
        open(f"{OUT}/labels/{name}.txt", "w").write("\n".join(lines))
        n_imgs += 1; n_boxes += len(lines)
    print(f"  {stem[:26]:26} imgs={n_imgs} boxes={n_boxes} | ${L.usage_snapshot()['cost_usd']:.2f}", flush=True)

open(f"{OUT}/data.yaml", "w").write(
    f"path: {os.path.abspath(OUT)}\ntrain: images\nval: images\nnames:\n  0: symbol\n")
print(f"\nDATASET READY: {n_imgs} images, {n_boxes} symbol boxes -> {OUT}")
print("Upload the dataset/ folder to RunPod and run train_detector.py")
