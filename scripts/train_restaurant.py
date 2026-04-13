"""
Fine-tune YOLOv8n on the restaurant dataset.

Usage:
    python scripts/train_restaurant.py
    python scripts/train_restaurant.py --epochs 50 --batch 8 --device cpu
    python scripts/train_restaurant.py --epochs 100 --device mps   # Apple Silicon

Outputs:
    runs/train/restaurant_ai/weights/best.pt   ← use this in /model/switch
    runs/train/restaurant_ai/weights/last.pt
"""
import sys
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_YAML = ROOT / "data" / "restaurant" / "data.yaml"
RUN_DIR   = ROOT / "runs" / "train"


def check_dataset():
    for split in ("train", "val"):
        img_dir   = ROOT / "data" / "restaurant" / "images" / split
        label_dir = ROOT / "data" / "restaurant" / "labels" / split
        imgs   = list(img_dir.glob("*.jpg")) if img_dir.exists() else []
        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []
        print(f"  {split:5s}: {len(imgs)} images | {len(labels)} labels")
        if not imgs:
            print(f"\n[ERROR] No images in {img_dir}")
            print("  Run: python scripts/collect_and_label.py --source <video_or_cam>")
            sys.exit(1)


def train(args):
    from ultralytics import YOLO

    print("=" * 60)
    print("Restaurant AI — Fine-tuning YOLOv8n")
    print("=" * 60)
    print(f"  data   : {DATA_YAML}")
    print(f"  epochs : {args.epochs}")
    print(f"  batch  : {args.batch}")
    print(f"  imgsz  : {args.imgsz}")
    print(f"  device : {args.device}")
    print(f"  freeze : {args.freeze} (backbone layers)")
    print()

    print("[dataset]")
    check_dataset()
    print()

    model = YOLO(args.weights)
    print(f"[model] Loaded base weights: {args.weights}\n")

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        freeze=args.freeze,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=30,
        project=str(RUN_DIR),
        name="restaurant_ai",
        exist_ok=True,
        save=True,
        verbose=True,
        workers=0 if args.device == "cpu" else 4,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        close_mosaic=10,
        augment=True,
    )

    best = RUN_DIR / "restaurant_ai" / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Best weights : {best}")
    if best.exists():
        print(f"\n[eval] Validating best.pt...")
        val_model = YOLO(str(best))
        metrics   = val_model.val(data=str(DATA_YAML), verbose=False)
        print(f"  mAP50    : {metrics.box.map50:.3f}")
        print(f"  mAP50-95 : {metrics.box.map:.3f}")
        print(f"  Precision: {metrics.box.mp:.3f}")
        print(f"  Recall   : {metrics.box.mr:.3f}")
        print(f"\nTo load in the dashboard:")
        print(f"  POST /model/switch   {{\"weights_path\": \"{best}\"}}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on restaurant data")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help="Base weights (default: yolov8n.pt)")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="0",
                        help="cuda device id, 'cpu', or 'mps' (Apple)")
    parser.add_argument("--freeze",  type=int,   default=10,
                        help="Freeze first N backbone layers (default 10)")
    args = parser.parse_args()
    train(args)
