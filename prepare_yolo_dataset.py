"""
prepare_yolo_dataset.py

Builds yolo_dataset/images and yolo_dataset/labels (train and val) from:
  - pillbox_production_images_full_202008.zip  -> class 0 "pill" (boxes estimated automatically)
  - val2017.zip (COCO)                          -> class 1 "not_pill" (empty label files)

Pill images are kept only if they match a row in the Pillbox CSV.
Both classes are sampled to the same count, then split TRAIN_FRACTION / (1 - TRAIN_FRACTION).

Run from the gullhacks folder:  python prepare_yolo_dataset.py
"""
import csv
import random
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ---------------- settings ----------------
ROOT = Path(__file__).resolve().parent
PILL_ZIP = ROOT / "pillbox_production_images_full_202008.zip"
COCO_ZIP = ROOT / "val2017.zip"
CSV_PATH = ROOT / "Pillbox_-_Archived_Data_20261107.csv"
OUT = ROOT / "yolo_dataset"

# First run: leave as None. The script prints the CSV columns and exits.
# Then set it to the column whose values match the image filenames.
CSV_KEY_COLUMN = "splimage"

# "prefix": filename before the first "_" must match the CSV value
# "stem":   whole filename (without extension) must match the CSV value
MATCH_MODE = "prefix"

TRAIN_FRACTION = 0.8
SEED = 42
CLEAR_EXISTING = False       # True deletes old .jpg/.png/.txt in the output folders first
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
PILL_CLASS = 0
NOT_PILL_CLASS = 1
# -----------------------------------------


def print_csv_columns():
    with open(CSV_PATH, newline="", encoding="utf-8-sig", errors="replace") as f:
        headers = next(csv.reader(f))
    print("CSV columns:")
    for i, h in enumerate(headers):
        print(f"  {i:3d}: {h}")
    print("\nSet CSV_KEY_COLUMN to the column whose values match your image filenames, then rerun.")


def load_csv_keys():
    with open(CSV_PATH, newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        if CSV_KEY_COLUMN not in reader.fieldnames:
            raise SystemExit(f"Column {CSV_KEY_COLUMN!r} not found in CSV.")
        return {row[CSV_KEY_COLUMN].strip() for row in reader if row[CSV_KEY_COLUMN]}


def image_key(member_name):
    stem = Path(member_name).stem
    return stem.split("_")[0] if MATCH_MODE == "prefix" else stem


def is_image(member_name):
    return member_name.lower().endswith(IMAGE_EXTS) and "__MACOSX" not in member_name


def decode(data_bytes):
    return cv2.imdecode(np.frombuffer(data_bytes, np.uint8), cv2.IMREAD_COLOR)


def pill_label(img):
    """Estimate one YOLO box around the pill; fall back to the whole image."""
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # Assumes pills are darker than the background (inverted Otsu)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, bw, bh = 0, 0, w, h
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area_frac = cv2.contourArea(c) / (w * h)
        if 0.005 < area_frac < 0.9:
            x, y, bw, bh = cv2.boundingRect(c)
    xc = (x + bw / 2) / w
    yc = (y + bh / 2) / h
    return f"{PILL_CLASS} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}\n"


def split(items, rng):
    items = items[:]
    rng.shuffle(items)
    cut = int(len(items) * TRAIN_FRACTION)
    return {"train": items[:cut], "val": items[cut:]}


def clear_outputs():
    for sub in ("images", "labels"):
        for split_name in ("train", "val"):
            folder = OUT / sub / split_name
            if folder.exists():
                for p in folder.iterdir():
                    if p.suffix.lower() in (".jpg", ".png", ".txt"):
                        p.unlink()


def main():
    if CSV_KEY_COLUMN is None:
        print_csv_columns()
        return

    keys = load_csv_keys()
    print(f"CSV has {len(keys)} unique values in {CSV_KEY_COLUMN!r}")

    with zipfile.ZipFile(PILL_ZIP) as pz:
        pill_names = sorted(n for n in pz.namelist()
                            if is_image(n) and image_key(n) in keys)
    with zipfile.ZipFile(COCO_ZIP) as cz:
        coco_all = sorted(n for n in cz.namelist() if is_image(n))

    if not pill_names:
        raise SystemExit("No pill images matched the CSV. Check CSV_KEY_COLUMN and MATCH_MODE.")

    print(f"Matched pill images: {len(pill_names)}")
    print(f"COCO images available: {len(coco_all)}")

    # Balance the classes: same number of COCO negatives as pill images
    rng = random.Random(SEED)
    coco_names = rng.sample(coco_all, min(len(pill_names), len(coco_all)))

    pill_split = split(pill_names, rng)
    coco_split = split(coco_names, rng)

    if CLEAR_EXISTING:
        clear_outputs()
    for sub in ("images", "labels"):
        for split_name in ("train", "val"):
            (OUT / sub / split_name).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    manifest_rows = []
    idx = 0

    with zipfile.ZipFile(PILL_ZIP) as pz:
        for split_name, names in pill_split.items():
            for name in names:
                img = decode(pz.read(name))
                if img is None:
                    counts["unreadable"] += 1
                    continue
                stem = f"pill_{idx:06d}"
                cv2.imwrite(str(OUT / "images" / split_name / f"{stem}.jpg"), img)
                (OUT / "labels" / split_name / f"{stem}.txt").write_text(pill_label(img))
                manifest_rows.append([stem, "pill", name, split_name, PILL_CLASS])
                counts[f"pill_{split_name}"] += 1
                idx += 1

    with zipfile.ZipFile(COCO_ZIP) as cz:
        for split_name, names in coco_split.items():
            for name in names:
                img = decode(cz.read(name))
                if img is None:
                    counts["unreadable"] += 1
                    continue
                stem = f"coco_{idx:06d}"
                cv2.imwrite(str(OUT / "images" / split_name / f"{stem}.jpg"), img)
                (OUT / "labels" / split_name / f"{stem}.txt").write_text("")  # empty = no pill
                manifest_rows.append([stem, "not_pill", name, split_name, NOT_PILL_CLASS])
                counts[f"not_pill_{split_name}"] += 1
                idx += 1

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["new_name", "class_name", "source_file", "split", "class_id"])
        w.writerows(manifest_rows)

    print("\nDone.")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    print(f"Manifest written to {OUT / 'manifest.csv'}")


if __name__ == "__main__":
    main()