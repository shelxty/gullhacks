"""
Live webcam pill detector: trains a YOLO model on your Roboflow dataset, then
finds pills in the webcam feed, names each one, and reports its color.
Click "Next" (or press N / Q) to see a summary of everything detected.

Install:
    pip install -U ultralytics roboflow opencv-python numpy python-dotenv

Usage:
    python web2pills.py            # trains if no model exists yet, then opens webcam
    python web2pills.py --train    # force (re)training, then opens webcam
    python web2pills.py --epochs 80

Your apikey.env (next to this script) should contain:
    API_KEY=your_roboflow_key
"""

import argparse
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

# ---------------- Settings ----------------
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / "apikey.env")          # looks next to this script, not the cwd
API_KEY = os.environ.get("API_KEY")

WORKSPACE = "shelley-ma"
PROJECT = "drugdetection-qscvy"           # the project ID (the part of the Roboflow URL)
VERSION = None                            # None = use the newest version; or set e.g. 1

DATASET_DIR = HERE / "dataset"
WEIGHTS_PATH = HERE / "pill_detector.pt"  # trained model is copied here
BASE_MODEL = "yolov8n.pt"                 # small + fast; try yolov8s.pt for more accuracy

CAMERA_INDEX = 0
CONFIDENCE = 0.50
MIN_DETECTIONS_TO_REPORT = 5              # ignore one-frame flickers in the summary
BUTTON_HEIGHT = 60
WINDOW = "Pill Detector"


# ---------------- Download dataset (fixed) ----------------
def latest_version(project):
    try:
        nums = [int(str(v.version).split("/")[-1]) for v in project.versions()]
        return max(nums)
    except Exception as e:
        raise RuntimeError(
            "Could not find any dataset versions. In Roboflow, open the project -> "
            "Versions -> Generate a version, then run again (or set VERSION above)."
        ) from e


def download_dataset():
    """Download in YOLO format. Returns the path to data.yaml."""
    yaml_path = DATASET_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"Dataset already in {DATASET_DIR}, skipping download.")
        return yaml_path

    if not API_KEY:
        raise RuntimeError("API_KEY not found. Check that apikey.env is next to this "
                           "script and contains API_KEY=...")
    from roboflow import Roboflow

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = VERSION or latest_version(project)
    print(f"Downloading {PROJECT} version {version} ...")
    dataset = project.version(version).download("yolov8", location=str(DATASET_DIR))
    return Path(dataset.location) / "data.yaml"


# ---------------- Fix up the dataset layout ----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(folder):
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def find_split_dir(split_names):
    """Find an 'images' folder whose parent folder is named like a split (train/valid/test)."""
    for d in sorted(DATASET_DIR.rglob("images")):
        if d.is_dir() and d.parent.name in split_names and count_images(d) > 0:
            return d
    return None


def make_val_split_from_train(train_images, fraction=0.15):
    """Move ~15% of training images (plus labels) into a new 'valid' split.

    Roboflow augmentation makes several copies of one photo (names contain '.rf.').
    We split by the ORIGINAL photo so copies of the same photo never land in both
    train and valid (that would inflate your validation scores).
    """
    root = train_images.parent.parent
    train_labels = train_images.parent / "labels"
    val_images = root / "valid" / "images"
    val_labels = root / "valid" / "labels"
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for p in train_images.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            groups[p.name.split(".rf.")[0]].append(p)

    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    n_val = max(1, int(len(keys) * fraction))
    for key in keys[:n_val]:
        for img in groups[key]:
            shutil.move(str(img), str(val_images / img.name))
            lbl = train_labels / (img.stem + ".txt")
            if lbl.exists():
                shutil.move(str(lbl), str(val_labels / lbl.name))
    return val_images


def prepare_data_yaml(original_yaml):
    """Write a data yaml with absolute paths based on the folders that really exist."""
    with open(original_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_dir = find_split_dir(["train"])
    if train_dir is None:
        folders = sorted(str(p.relative_to(DATASET_DIR)) for p in DATASET_DIR.rglob("*") if p.is_dir())
        raise RuntimeError("No training images found in the downloaded dataset. "
                           "Folders present:\n  " + "\n  ".join(folders[:40]))

    val_dir = find_split_dir(["valid", "val", "validation"])
    if val_dir is None:
        print("No validation images found in the download; holding out 15% of "
              "the training images as a validation set.")
        val_dir = make_val_split_from_train(train_dir)
    test_dir = find_split_dir(["test"])

    fixed = {
        "path": train_dir.parent.parent.as_posix(),
        "train": train_dir.as_posix(),
        "val": val_dir.as_posix(),
        "names": cfg["names"],
    }
    if test_dir is not None:
        fixed["test"] = test_dir.as_posix()

    out_path = train_dir.parent.parent / "data_fixed.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(fixed, f, sort_keys=False)

    print(f"Images -> train: {count_images(train_dir)}, valid: {count_images(val_dir)}, "
          f"test: {count_images(test_dir) if test_dir else 0}")
    return out_path


# ---------------- Train ----------------
def train_model(epochs):
    data_yaml = prepare_data_yaml(download_dataset())
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,            # lower this (e.g. 8) if you run out of memory
        patience=15,         # stop early if validation stops improving
        project=str(HERE / "runs"),
        name="pill_detector",
        exist_ok=True,
        workers=2,           # set to 0 if Windows complains about memory/paging file
    )
    shutil.copy(model.trainer.best, WEIGHTS_PATH)
    print(f"\nSaved trained model to {WEIGHTS_PATH}")


# ---------------- Color naming ----------------
def name_color(bgr_patch):
    """Basic color name for an image patch (median HSV, simple rules)."""
    if bgr_patch.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = np.median(hsv, axis=0)   # OpenCV: H 0-179, S 0-255, V 0-255

    if v < 50:
        return "black"
    if s < 45:
        return "white" if v > 190 else "gray"
    if h < 10 or h >= 170:
        return "red" if v > 90 else "brown"
    if h < 22:
        return "orange" if v > 150 else "brown"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"


def box_color(frame, x1, y1, x2, y2):
    """Use only the center of the box so background pixels don't dominate."""
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = x1 + w // 4, x2 - w // 4
    cy1, cy2 = y1 + h // 4, y2 - h // 4
    return name_color(frame[max(cy1, 0):cy2, max(cx1, 0):cx2])


# ---------------- Button handling ----------------
state = {"next_clicked": False, "button": (0, 0, 0, 0)}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1, bx2, by2 = state["button"]
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            state["next_clicked"] = True


def draw_button(frame):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = 0, h - BUTTON_HEIGHT, w, h
    state["button"] = (x1, y1, x2, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 160, 60), -1)
    cv2.putText(frame, "Next", (w // 2 - 40, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


# ---------------- Summary screen ----------------
def build_summary_lines(records, duration):
    lines = [f"Camera was open for {duration:.1f} seconds", ""]
    reported = {k: v for k, v in records.items()
                if v["detections"] >= MIN_DETECTIONS_TO_REPORT}
    if not reported:
        lines.append("No pills were detected.")
        return lines

    lines.append("Detected pills:")
    lines.append("")
    for label, info in sorted(reported.items(), key=lambda kv: -kv[1]["detections"]):
        colors = info["colors"]
        total = sum(colors.values())
        top_color = colors.most_common(1)[0][0]
        breakdown = ", ".join(f"{c} {100 * n / total:.0f}%" for c, n in colors.most_common(3))
        avg_conf = info["conf_sum"] / info["detections"]
        lines.append(f"- {label}  (avg confidence {avg_conf:.0%}, "
                     f"up to {info['max_at_once']} at once)")
        lines.append(f"     color: {top_color}   [{breakdown}]")
    return lines


def show_summary(lines):
    for line in lines:
        print(line)
    height = max(400, 60 + 32 * len(lines))
    canvas = np.zeros((height, 900, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (20, 40 + 32 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(canvas, "Press any key to close", (20, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 120), 1)
    cv2.imshow(WINDOW, canvas)
    cv2.waitKey(0)


# ---------------- Live webcam ----------------
def run_webcam():
    model = YOLO(str(WEIGHTS_PATH))
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY   # faster open on Windows
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different CAMERA_INDEX.")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    records = defaultdict(lambda: {"detections": 0, "conf_sum": 0.0,
                                   "max_at_once": 0, "colors": Counter()})
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=CONFIDENCE, verbose=False)[0]
        in_frame = Counter()

        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = box_color(frame, x1, y1, x2, y2)

            in_frame[label] += 1
            rec = records[label]
            rec["detections"] += 1
            rec["conf_sum"] += conf
            rec["colors"][color] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f} | {color}"
            cv2.rectangle(frame, (x1, y1 - 24), (x1 + 9 * len(text), y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        for label, n in in_frame.items():
            records[label]["max_at_once"] = max(records[label]["max_at_once"], n)

        draw_button(frame)
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if state["next_clicked"] or key in (ord("n"), ord("q")):
            break

    duration = time.time() - start
    cap.release()
    show_summary(build_summary_lines(records, duration))
    cv2.destroyAllWindows()


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="force (re)training")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.train or not WEIGHTS_PATH.exists():
        train_model(args.epochs)
    run_webcam()


if __name__ == "__main__":
    main()