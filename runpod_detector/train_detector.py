"""Train the universal symbol detector AND report the generalization verdict.

Trains on train/val (seen firms), then evaluates on test = HELD-OUT FIRMS the
model never saw. The held-out mAP is the exact generalization number:
  mAP@50 >= 0.70 -> strong cross-firm generalization (universal works)
  0.50-0.70      -> usable, add more firms / boxes
  < 0.50         -> not generalizing yet -> more diverse data needed

Run on RunPod (1 GPU):  pip install ultralytics && python train_detector.py
"""
from pathlib import Path
import yaml
from ultralytics import YOLO

# Resolve the dataset relative to THIS script (works regardless of CWD or which
# machine made the data). The committed data.yaml/test.yaml carry a machine-
# specific absolute `path:` — we rewrite it to the real local dir at runtime.
HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"


def _portable_yaml(src_name, out_name):
    cfg = yaml.safe_load((DATASET / src_name).read_text())
    cfg["path"] = str(DATASET)
    out = DATASET / out_name
    out.write_text(yaml.safe_dump(cfg))
    return str(out)


def main():
    data_yaml = _portable_yaml("data.yaml", "_data.local.yaml")
    test_yaml = _portable_yaml("test.yaml", "_test.local.yaml")

    model = YOLO("yolov8m.pt")           # yolov8l/x for more capacity
    model.train(
        data=data_yaml, epochs=120, imgsz=1280, batch=8,
        mosaic=1.0, degrees=15, scale=0.5, translate=0.1, fliplr=0.0,
        patience=25, project="runs", name="symbol_detector",
    )
    best = "runs/symbol_detector/weights/best.pt"
    print(f"\nbest weights -> {best}")

    # GENERALIZATION VERDICT: evaluate on held-out firms (never trained on)
    print("\n=== GENERALIZATION TEST (held-out firms the model never saw) ===")
    m = YOLO(best)
    metrics = m.val(data=test_yaml, imgsz=1280, split="val")
    mAP50 = float(metrics.box.map50)
    mAP = float(metrics.box.map)
    verdict = ("STRONG cross-firm generalization" if mAP50 >= 0.70 else
               "USABLE — add firms/boxes" if mAP50 >= 0.50 else
               "NOT generalizing yet — need more diverse data")
    print(f"held-out mAP@50 = {mAP50:.3f} | mAP@50-95 = {mAP:.3f}")
    print(f"VERDICT: {verdict}")
    print("(this is the exact answer to 'does it generalize perfectly or not')")

if __name__ == "__main__":
    main()
