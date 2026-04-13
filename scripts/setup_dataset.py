"""
Download coco128 (built into ultralytics) and convert to restaurant format.

What it does:
  1. Downloads coco128 (~7MB, 128 images with YOLO labels)
  2. Filters to images that contain persons (class 0)
  3. Applies HSV color heuristic on each person box → staff (1) or customer (2)
  4. Writes to data/restaurant/{images,labels}/{train,val}/
  5. Prints dataset stats

Usage:
    python scripts/setup_dataset.py
    python scripts/setup_dataset.py --hsv-upper 180 60 90   # light-blue uniforms
"""
import sys
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEST      = ROOT / "data" / "restaurant"
TRAIN_SPLIT = 0.80
MIN_BOX_AREA = 0.002   # minimum box area as fraction of image (filter tiny boxes)

# Default: dark/black uniform (matches main.py default)
HSV_LOWER_DEFAULT = (0,   0,   0)
HSV_UPPER_DEFAULT = (180, 50, 80)


def download_coco128():
    """Download coco128 via ultralytics (cached after first run)."""
    print("[1/4] Downloading coco128 dataset...")
    from ultralytics.data.utils import check_det_dataset
    info = check_det_dataset("coco128.yaml")
    src_root = Path(info["path"])
    print(f"      coco128 at: {src_root}")
    return src_root


def load_yolo_label(label_path: Path):
    """Return list of (class_id, cx, cy, bw, bh) from a YOLO .txt."""
    anns = []
    if not label_path.exists():
        return anns
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                anns.append(tuple(float(x) for x in parts[:5]))
    return anns


def is_staff(frame_hsv, cx, cy, bw, bh, h, w, hsv_lower, hsv_upper):
    import cv2, numpy as np
    x1 = max(0, int((cx - bw/2) * w))
    y1 = max(0, int((cy - bh/2) * h))
    x2 = min(w, int((cx + bw/2) * w))
    y2 = min(h, int((cy + bh/2) * h))
    roi = frame_hsv[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    mask = cv2.inRange(roi, hsv_lower, hsv_upper)
    return (np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])) > 0.30


def convert(src_root: Path, hsv_lower, hsv_upper):
    import cv2, numpy as np

    print("[2/4] Scanning images for persons...")

    img_dirs = [
        src_root / "images" / "train2017",
        src_root / "images" / "val2017",
        src_root / "images",
    ]
    img_dir = next((d for d in img_dirs if d.exists()), None)
    if img_dir is None:
        # walk to find any images
        img_dir = src_root

    lbl_dirs = [
        src_root / "labels" / "train2017",
        src_root / "labels" / "val2017",
        src_root / "labels",
    ]
    lbl_dir = next((d for d in lbl_dirs if d.exists()), img_dir)

    imgs = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
    imgs = sorted(set(imgs))
    print(f"      {len(imgs)} total images found")

    samples = []   # (img_path, [(new_cls, cx, cy, bw, bh), ...])

    for img_path in imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        # also try same parent but labels subdir
        if not lbl_path.exists():
            lbl_path = img_path.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")

        anns = load_yolo_label(lbl_path)
        person_anns = [a for a in anns if int(a[0]) == 0]  # COCO class 0 = person

        if not person_anns:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        new_anns = []
        for _, cx, cy, bw, bh in person_anns:
            if bw * bh < MIN_BOX_AREA:
                continue
            cls = 1 if is_staff(frame_hsv, cx, cy, bw, bh, h, w, hsv_lower, hsv_upper) else 2
            new_anns.append((cls, cx, cy, bw, bh))

        if new_anns:
            samples.append((img_path, new_anns))

    print(f"      {len(samples)} images with person annotations kept")
    return samples


def save_split(samples, hsv_lower, hsv_upper):
    print("[3/4] Writing dataset to data/restaurant/ ...")
    import cv2

    for split in ("train", "val"):
        (DEST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DEST / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.shuffle(samples)
    n_train = int(len(samples) * TRAIN_SPLIT)
    splits  = [("train", samples[:n_train]), ("val", samples[n_train:])]

    counts = {}
    for split, split_samples in splits:
        staff = customer = 0
        for i, (img_path, anns) in enumerate(split_samples):
            stem = f"{split}_{i:05d}"
            shutil.copy2(img_path, DEST / "images" / split / f"{stem}.jpg")
            with open(DEST / "labels" / split / f"{stem}.txt", "w") as f:
                for cls, cx, cy, bw, bh in anns:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    if cls == 1: staff += 1
                    else:        customer += 1
        counts[split] = {"images": len(split_samples), "staff": staff, "customer": customer}

    return counts


def main(args):
    import numpy as np
    hsv_lower = np.array(args.hsv_lower)
    hsv_upper = np.array(args.hsv_upper)

    src_root = download_coco128()
    samples  = convert(src_root, hsv_lower, hsv_upper)

    if not samples:
        print("[ERROR] No person images found. Check coco128 download.")
        sys.exit(1)

    counts = save_split(samples, hsv_lower, hsv_upper)

    print("\n[4/4] Done.")
    print(f"  train : {counts['train']['images']} images | "
          f"staff={counts['train']['staff']} customer={counts['train']['customer']}")
    print(f"  val   : {counts['val']['images']} images | "
          f"staff={counts['val']['staff']} customer={counts['val']['customer']}")
    print(f"\n  Staff HSV: lower={tuple(hsv_lower)} upper={tuple(hsv_upper)}")
    print(f"  NOTE: coco128 has no restaurant uniforms → most boxes will be 'customer'.")
    print(f"        This trains the model to detect people correctly.")
    print(f"        Re-run with your own footage later for real staff/customer split.\n")
    print("Next step:")
    print("  python scripts/train_restaurant.py --epochs 50 --device cpu")
    print("  (or --device mps on Apple Silicon / --device 0 for CUDA GPU)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsv-lower", dest="hsv_lower", type=int, nargs=3,
                        default=list(HSV_LOWER_DEFAULT), metavar=("H", "S", "V"))
    parser.add_argument("--hsv-upper", dest="hsv_upper", type=int, nargs=3,
                        default=list(HSV_UPPER_DEFAULT), metavar=("H", "S", "V"))
    main(parser.parse_args())
