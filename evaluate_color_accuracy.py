"""
Measure how accurate the webcam script's COLOR NAMING is on your Pillbox images (ground truth = CSV `splcolor_text`).

Put this file next to webcam_pill_detector.py, then run, for example:
    python evaluate_color_accuracy.py --csv Pillbox_-_Archived_Data_20261107.csv --images path/to/unzipped/images --n 1500

It prints
  * overall accuracy (is the measured main color one of the CSV's colors?)
  * recall for each color (of all pills the CSV calls RED, how many did we call red?)
  * which colors get confused with which
and writes color_eval_mistakes.csv listing the images it got wrong, so you can open them and look.

NOTE: this tests the color-naming rules on clean Pillbox photos (no webcam, no room-light problems), so it measures the
rules themselves. The white-balance step handles lighting for the live camera.
"""
import os
import argparse
import cv2
import numpy as np
import pandas as pd

import webcam_pill_detector as W

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Pillbox CSV file")
    ap.add_argument("--images", required=True, help="folder containing the Pillbox images (sub-folders are fine)")
    ap.add_argument("--n", type=int, default=1500, help="how many random images to test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str)
    df = df[df["has_image"].str.lower() == "true"].copy()
    df["stem"] = df["splimage"].str.strip().str.lower()
    df = df[df["stem"].notna() & (df["stem"] != "no_product_image") & df["splcolor_text"].notna()]

    disk = {}                                              # filename (no extension) -> full path
    for root, _, files in os.walk(args.images):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMAGE_EXTS:
                disk[stem.lower()] = os.path.join(root, f)
    df["path"] = df["stem"].map(disk)
    df = df[df["path"].notna()].drop_duplicates("stem")
    print(f"Images matched to CSV rows: {len(df)}")
    if len(df) == 0:
        print("No matches - check --images points at the folder with the pictures.")
        return
    df = df.sample(min(args.n, len(df)), random_state=args.seed)

    rows = []
    for _, r in df.iterrows():
        img = cv2.imread(r["path"])
        if img is None:
            continue
        sc = 400.0 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        blobs = W.find_objects(img, strict=False)           # no shape filters: Pillbox photos are just pills on a plain background
        if not blobs:
            rows.append({"path": r["path"], "truth": r["splcolor_text"], "measured": "NO PILL FOUND", "ok": False, "main_ok": False})
            continue
        best = max(blobs, key=lambda o: o["box"][2] * o["box"][3])
        truth = [t.strip().lower() for t in r["splcolor_text"].split(";") if t.strip()]
        measured = best["color_names"]
        rows.append({"path": r["path"], "truth": r["splcolor_text"], "measured": best["pixel_color"],
                     "truth_main": truth[0], "measured_main": measured[0],
                     "main_ok": measured[0] in truth,                      # main measured color is one of the CSV's colors
                     "ok": set(measured) == set(truth)})                  # exact same set of colors

    res = pd.DataFrame(rows)
    ok = res[res["measured"] != "NO PILL FOUND"]
    print(f"\nTested {len(res)} images ({len(res) - len(ok)} with no pill found)")
    print(f"Main color correct : {ok['main_ok'].mean() * 100:.1f}%")
    print(f"Exact color set    : {ok['ok'].mean() * 100:.1f}%")

    print("\nRecall per color (CSV's main color -> was our main color the same?)")
    per = ok.groupby("truth_main").agg(n=("main_ok", "size"), recall=("main_ok", "mean")).sort_values("n", ascending=False)
    per["recall"] = (per["recall"] * 100).round(1)
    print(per.to_string())

    print("\nMost common confusions (CSV main color -> our main color)")
    wrong = ok[~ok["main_ok"]]
    print(wrong.groupby(["truth_main", "measured_main"]).size().sort_values(ascending=False).head(15).to_string())

    bad = res[~res["main_ok"]]
    bad.to_csv("color_eval_mistakes.csv", index=False)
    print(f"\nWrote {len(bad)} mistakes to color_eval_mistakes.csv (open a few of those images and look at them).")


if __name__ == "__main__":
    main()
