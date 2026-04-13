"""
Pseudo-label data collection for Restaurant AI.

Usage:
    # From video file
    python scripts/collect_and_label.py --source path/to/video.mp4

    # From webcam (index or RTSP)
    python scripts/collect_and_label.py --source 0 --duration 120

    # Custom staff HSV range (default: dark/black uniform)
    python scripts/collect_and_label.py --source video.mp4 --hsv-upper 180 60 90

Pipeline:
    1. Extract frames at --interval (default every 15 frames)
    2. YOLO detect persons on each frame
    3. StaffDetector color heuristic → class 1 (staff) or 2 (customer)
    4. Write YOLO-format .txt labels
    5. Split 80/20 → data/restaurant/images/{train,val}
"""
import sys
import os
import argparse
import random
import shutil
from pathlib import Path

# Resolve project root (one level up from scripts/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import cv2
import numpy as np


# ── CONFIG ──────────────────────────────────────────────────────────────────

DATASET_DIR = ROOT / "data" / "restaurant"
CONF_THRESH  = 0.45        # YOLO detection confidence
FRAME_INTERVAL = 15        # sample every N frames
MAX_FRAMES   = 2000        # hard cap on extracted frames
TRAIN_SPLIT  = 0.80        # 80 % train, 20 % val
MIN_BOX_AREA = 800         # skip tiny detections (px²)

# Staff uniform: dark/black (HSV) — matches default in main.py
HSV_LOWER_DEFAULT = (0, 0, 0)
HSV_UPPER_DEFAULT = (180, 50, 80)


# ── HELPERS ──────────────────────────────────────────────────────────────────

def make_dirs():
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def box_to_yolo(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def is_staff(frame_hsv, box, hsv_lower, hsv_upper):
    x1, y1, x2, y2 = map(int, box)
    roi = frame_hsv[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    mask = cv2.inRange(roi, hsv_lower, hsv_upper)
    pixel_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])
    return pixel_ratio > 0.30


def open_source(source_str):
    src = int(source_str) if source_str.strip().isdigit() else source_str
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source_str}")
    return cap


# ── MAIN ─────────────────────────────────────────────────────────────────────

def collect(args):
    from ultralytics import YOLO

    hsv_lower = np.array(args.hsv_lower)
    hsv_upper = np.array(args.hsv_upper)

    make_dirs()
    model = YOLO("yolov8n.pt")

    cap = open_source(args.source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    duration_limit = args.duration * fps if args.duration else float("inf")

    print(f"[collect] Source: {args.source}")
    print(f"[collect] Interval: every {args.interval} frames | max {args.max_frames} samples")
    print(f"[collect] Staff HSV lower={tuple(hsv_lower)} upper={tuple(hsv_upper)}")

    samples   = []        # (frame_img, [(class_id, cx, cy, bw, bh), ...])
    frame_idx = 0
    skipped   = 0

    while len(samples) < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= duration_limit:
            break

        if frame_idx % args.interval == 0:
            h, w = frame.shape[:2]
            results = model(frame, conf=CONF_THRESH, classes=[0], verbose=False)[0]
            boxes   = results.boxes.xyxy.cpu().numpy() if results.boxes else []

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            annotations = []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BOX_AREA:
                    skipped += 1
                    continue
                cls = 1 if is_staff(frame_hsv, box, hsv_lower, hsv_upper) else 2
                cx, cy, bw, bh = box_to_yolo(box, w, h)
                annotations.append((cls, cx, cy, bw, bh))

            if annotations:
                samples.append((frame.copy(), annotations))

            if len(samples) % 50 == 0 and len(samples) > 0:
                print(f"  {len(samples)} labeled frames collected...")

        frame_idx += 1

    cap.release()
    print(f"[collect] {len(samples)} frames with detections | {skipped} tiny boxes skipped")

    # ── Split & save ──────────────────────────────────────────────────────────
    random.shuffle(samples)
    n_train = int(len(samples) * TRAIN_SPLIT)
    splits  = [("train", samples[:n_train]), ("val", samples[n_train:])]

    counts = {"train": {"staff": 0, "customer": 0}, "val": {"staff": 0, "customer": 0}}

    # Detect existing file count so we append rather than overwrite
    offsets = {}
    for split in ("train", "val"):
        existing = list((DATASET_DIR / "images" / split).glob("*.jpg"))
        offsets[split] = len(existing)

    for split, split_samples in splits:
        offset = offsets[split]
        for i, (img, anns) in enumerate(split_samples):
            stem = f"{split}_{offset + i:05d}"
            img_path   = DATASET_DIR / "images" / split / f"{stem}.jpg"
            label_path = DATASET_DIR / "labels" / split / f"{stem}.txt"

            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(label_path, "w") as f:
                for cls, cx, cy, bw, bh in anns:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    label = "staff" if cls == 1 else "customer"
                    counts[split][label] += 1

    print(f"\n[collect] Dataset written to {DATASET_DIR}")
    print(f"  train: {len(splits[0][1])} images | "
          f"staff={counts['train']['staff']} customer={counts['train']['customer']}")
    print(f"  val  : {len(splits[1][1])} images | "
          f"staff={counts['val']['staff']} customer={counts['val']['customer']}")
    print(f"\n  data.yaml → {DATASET_DIR / 'data.yaml'}")
    print("  Next: python scripts/train_restaurant.py")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect + pseudo-label restaurant frames")
    parser.add_argument("--source",    required=True,
                        help="Video file path, webcam index (0), or RTSP URL")
    parser.add_argument("--interval",  type=int,   default=FRAME_INTERVAL,
                        help=f"Sample every N frames (default {FRAME_INTERVAL})")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=MAX_FRAMES,
                        help=f"Max labeled frames to collect (default {MAX_FRAMES})")
    parser.add_argument("--duration",  type=int,   default=0,
                        help="Max seconds to read from camera (0=unlimited)")
    parser.add_argument("--hsv-lower", dest="hsv_lower", type=int, nargs=3,
                        default=list(HSV_LOWER_DEFAULT), metavar=("H", "S", "V"),
                        help="Staff uniform HSV lower bound (default 0 0 0)")
    parser.add_argument("--hsv-upper", dest="hsv_upper", type=int, nargs=3,
                        default=list(HSV_UPPER_DEFAULT), metavar=("H", "S", "V"),
                        help="Staff uniform HSV upper bound (default 180 50 80)")
    args = parser.parse_args()
    collect(args)
