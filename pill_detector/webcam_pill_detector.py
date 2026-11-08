import os
import time
from collections import Counter

import cv2
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

from pill_analysis import (
    WaterAnalyzer, find_objects, crop_roi,
)

CAMERA_INDEX = 0
PILL_WEIGHTS = "runs/detect/pills/weights/best.pt"   # your trained YOLO weights
DETECT_EVERY_N_FRAMES = 3                            # YOLO is slow; reuse results between runs
DEFAULT_WATER_ROI = (200, 120, 240, 240)             # x, y, w, h for a 640x480 feed

detector = None
if os.path.exists(PILL_WEIGHTS):
    try:
        from ultralytics import YOLO
        detector = YOLO(PILL_WEIGHTS)
        print(f"Loaded YOLO model: {PILL_WEIGHTS}")
    except ImportError:
        print("ultralytics not installed; using contour fallback.")
else:
    print("No YOLO weights found; using contour fallback for object detection.")


class App:
    def __init__(self, root):
        self.root = root
        root.title("Pill & Solution Detector")

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.video = tk.Label(root)
        self.video.pack()
        self.panel = tk.Label(root, justify="left", anchor="w",
                              font=("Consolas", 11), width=80)
        self.panel.pack(fill="x", padx=6)

        bar = tk.Frame(root)
        bar.pack(fill="x", pady=6)
        tk.Button(bar, text="Calibrate empty water", command=self.calibrate).pack(side="left", padx=4)
        tk.Button(bar, text="Set water box", command=self.set_roi).pack(side="left", padx=4)
        tk.Button(bar, text="Next", font=("Arial", 14, "bold"),
                  command=self.next_item).pack(side="right", padx=4)

        self.roi = DEFAULT_WATER_ROI
        self.water = WaterAnalyzer()
        self.frame = None
        self.dets = []
        self.water_state = {"calibrated": False}
        self.frame_idx = 0
        self.img_ref = None
        self.reset_session()

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_frame()

    # ---------- session log ----------
    def reset_session(self):
        self.session_start = time.time()
        self.max_counts = Counter()          # label -> most objects seen at once
        self.label_colors = {}               # label -> Counter of colors
        self.water_frames = Counter()        # flag -> frames where True
        self.tint_colors = Counter()
        self.water.reset_state()

    def record(self):
        frame_counts = Counter(d["label"] for d in self.dets)
        for label, n in frame_counts.items():
            self.max_counts[label] = max(self.max_counts[label], n)
            self.label_colors.setdefault(label, Counter())
        for d in self.dets:
            self.label_colors[d["label"]][d["color"]] += 1

        for key in ("dissolved", "tinted", "fizzing", "cloudy_suspended", "residue_settled"):
            if self.water_state.get(key):
                self.water_frames[key] += 1
        if self.water_state.get("tint_color"):
            self.tint_colors[self.water_state["tint_color"]] += 1

    # ---------- camera loop ----------
    def update_frame(self):
        try:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
                self.frame_idx += 1
                if self.frame_idx % DETECT_EVERY_N_FRAMES == 1 or not self.dets:
                    self.dets = find_objects(frame, detector)
                self.water_state = self.water.update(frame, self.roi)
                self.record()
                self.show(self.draw(frame.copy()))
                self.panel.config(text=self.live_text())
        except Exception as e:
            print(f"Frame error: {e}")
        finally:
            self.root.after(33, self.update_frame)

    def draw(self, vis):
        x, y, w, h = self.roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 200, 0), 2)
        for d in self.dets:
            x, y, w, h = d["box"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f'{d["label"]} ({d["color"]})'
            cv2.putText(vis, text, (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if self.water_state.get("calibrated"):
            s = self.water_state
            status = (f'dissolved:{s["dissolved"]} tint:{s["tint_color"] or "none"} '
                      f'fizz:{s["fizzing"]} cloudy:{s["cloudy_suspended"]} '
                      f'residue:{s["residue_settled"]}')
            cv2.putText(vis, status, (10, vis.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return vis

    def show(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.img_ref = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.config(image=self.img_ref)

    def live_text(self):
        objs = ", ".join(f'{d["label"]} [{d["color"]}]' for d in self.dets) or "no objects"
        s = self.water_state
        if not s.get("calibrated"):
            water = "Water: press 'Calibrate empty water' first"
        else:
            water = (f'Water: dissolved={s["dissolved"]}, tint={s["tint_color"] or "none"}, '
                     f'fizzing={s["fizzing"]}, cloudy={s["cloudy_suspended"]}, '
                     f'settled residue={s["residue_settled"]}')
        return f"Objects: {objs}\n{water}"

    # ---------- buttons ----------
    def calibrate(self):
        if self.frame is not None:
            self.water.calibrate(self.frame, self.roi)
            self.reset_session()

    # def set_roi(self):
    #     if self.frame is None:
    #         return
    #     roi = cv2.selectROI("Select water region, Enter to confirm", self.frame, showCrosshair=True)
    #     cv2.destroyWindow("Select water region, Enter to confirm")
    #     if roi[2] > 0 and roi[3] > 0:
    #         self.roi = tuple(int(v) for v in roi)
    def set_roi(self):
        if self.frame is None:
            return
        roi = cv2.selectROI("Select water region, Enter to confirm", self.frame, showCrosshair=True)
        cv2.destroyWindow("Select water region, Enter to confirm")
        if roi[2] > 0 and roi[3] > 0:
            self.roi = tuple(int(v) for v in roi)
            self.water.base_lab = None      # old baseline no longer matches the new box
            self.water.reset_state()
            self.water_state = {"calibrated": False}

    def next_item(self):
        duration = time.time() - self.session_start
        lines = [f"Session length: {duration:.1f} s", "", "OBJECTS DETECTED:"]
        if not self.max_counts:
            lines.append("  none")
        for label, n in self.max_counts.most_common():
            colors = ", ".join(f"{c} ({k})" for c, k in self.label_colors[label].most_common(3))
            lines.append(f"  {label}: max {n} in view, colors: {colors or 'unknown'}")

        lines += ["", "WATER:"]
        fps = 30.0
        for key, label in [("dissolved", "Fully dissolved"), ("tinted", "Tinted"),
                           ("fizzing", "Fizzing/bubbling"), ("cloudy_suspended", "Cloudy suspended residue"),
                           ("residue_settled", "Non-suspended residue")]:
            seconds = self.water_frames[key] / fps
            lines.append(f"  {label}: seen for about {seconds:.1f} s")
        if self.tint_colors:
            tint, _ = self.tint_colors.most_common(1)[0]
            lines.append(f"  Tint color: {tint}")

        win = tk.Toplevel(self.root)
        win.title("Session summary")
        txt = scrolledtext.ScrolledText(win, width=70, height=24, font=("Consolas", 11))
        txt.pack(padx=8, pady=8)
        txt.insert("1.0", "\n".join(lines))
        txt.config(state="disabled")

        self.reset_session()
        # self.water.calibrated = False
        self.water.base_lab = None  # require fresh calibration for the next item

    def close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()