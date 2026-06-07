"""Train the UNIVERSAL class-agnostic symbol detector on RunPod (one GPU).

YOLOv8 fine-tune: 1 class ("symbol"). Pretrained COCO weights -> fast convergence.
After training, export best.pt -> download -> the local pipeline loads it as the
universal symbol detector (replaces region_proposals + the pixel objectness head).

On RunPod:
    pip install ultralytics
    python train_detector.py            # ~1-3 hrs on a 4090/A100
    # -> runs/detect/train/weights/best.pt  (download this)
"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")   # bump to yolov8l/x for more capacity if needed
    model.train(
        data="dataset/data.yaml",
        epochs=120,
        imgsz=1280,              # high res: symbols are small on E-size sheets
        batch=8,
        mosaic=1.0,             # heavy aug -> generalize across firms
        degrees=15, scale=0.5, translate=0.1, fliplr=0.0,  # symbols aren't mirror-symmetric
        patience=25,
        project="runs", name="symbol_detector",
    )
    print("done -> runs/symbol_detector/weights/best.pt")
