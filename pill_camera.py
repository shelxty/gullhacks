import os
import time
from collections import Counter
import tkinter as tk

import cv2
from PIL import Image, ImageTk

from pill_analysis import WaterAnalyzer, find_objects

HERE = os.path.dirname(os.path.abspath(__file__))
CAMERA_INDEX = 0
PILL_WEIGHTS = os.path.join(HERE, "weights", "best.pt")
DETECT_EVERY_N_FRAMES = 3
DEFAULT_WATER_ROI = (200, 120, 240, 240)   # x, y, w, h for a 640x480 feed
FPS = 30.0
IGNORE_LABELS = {"not_pill"}               # from data.yaml class names

detector = None
if os.path.exists(PILL_WEIGHTS):
    try:
        from ultralytics import YOLO
        detector = YOLO(PILL_WEIGHTS)
        print(f"Loaded YOLO model: {PILL_WEIGHTS}")
    except ImportError:
        print("ultralytics not installed; using contour fallback.")
else:
    print(f"No weights at {PILL_WEIGHTS}; using contour fallback.")


class CameraPage(tk.Frame):
    def __init__(self, window, on_next, on_back):
        super().__init__(window.container, bg="#FFFFFF")
        self.on_next = on_next

        self.cap = None
        self.after_id = None
        self.running = False
        self.frame = None
        self.dets = []
        self.frame_idx = 0
        self.roi = DEFAULT_WATER_ROI
        self.water = WaterAnalyzer()
        self.water_state = {"calibrated": False}
        self.img_ref = None
        self.reset_session()

        tk.Button(self, text="Back", command=on_back).pack(pady=4)

        self.video = tk.Label(self)
        self.video.pack()
        self.panel = tk.Label(self, justify="left", anchor="w",
                              font=("Consolas", 10), width=60)
        self.panel.pack(fill="x", padx=6)

        bar = tk.Frame(self)
        bar.pack(fill="x", pady=6)
        tk.Button(bar, text="Calibrate empty water", command=self.calibrate).pack(side="left", padx=4)
        tk.Button(bar, text="Set water box", command=self.set_roi).pack(side="left", padx=4)
        tk.Button(bar, text="Next", font=("Arial", 12, "bold"),
                  command=self.finish).pack(side="right", padx=4)

    # ---------- lifecycle (App.show_page calls these) ----------
    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(CAMERA_INDEX)   # on Windows, try cv2.CAP_DSHOW if the feed fails
        self.running = True
        self.update_frame()

    def stop(self):
        self.running = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ---------- session log ----------
    def reset_session(self):
        self.session_start = time.time()
        self.max_counts = Counter()
        self.label_colors = {}
        self.water_frames = Counter()
        self.tint_colors = Counter()
        self.water.reset_state()

    def record(self):
        for label, n in Counter(d["label"] for d in self.dets).items():
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
        self.after_id = None
        if not self.running or self.cap is None:
            return
        try:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
                self.frame_idx += 1
                if self.frame_idx % DETECT_EVERY_N_FRAMES == 1 or not self.dets:
                    found = find_objects(frame, detector)
                    self.dets = [d for d in found if d["label"] not in IGNORE_LABELS]
                self.water_state = self.water.update(frame, self.roi)
                self.record()
                self.show(self.draw(frame.copy()))
                self.panel.config(text=self.live_text())
        except Exception as e:
            print(f"Frame error: {e}")
        finally:
            if self.running:
                self.after_id = self.after(33, self.update_frame)

    def draw(self, vis):
        x, y, w, h = self.roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 200, 0), 2)
        for d in self.dets:
            x, y, w, h = d["box"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f'{d["label"]} ({d["color"]})', (x, max(15, y - 6)),
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

    def set_roi(self):
        if self.frame is None:
            return
        roi = cv2.selectROI("Select water region, Enter to confirm", self.frame, showCrosshair=True)
        cv2.destroyWindow("Select water region, Enter to confirm")
        if roi[2] > 0 and roi[3] > 0:
            self.roi = tuple(int(v) for v in roi)
            self.water.base_lab = None
            self.water.reset_state()
            self.water_state = {"calibrated": False}

    def build_summary(self):
        lines = [f"Session length: {time.time() - self.session_start:.1f} s", "", "OBJECTS DETECTED:"]
        if not self.max_counts:
            lines.append("  none")
        for label, n in self.max_counts.most_common():
            colors = ", ".join(f"{c} ({k})" for c, k in self.label_colors[label].most_common(3))
            lines.append(f"  {label}: max {n} in view, colors: {colors or 'unknown'}")

        lines += ["", "WATER:"]
        for key, label in [("dissolved", "Fully dissolved"), ("tinted", "Tinted"),
                           ("fizzing", "Fizzing/bubbling"), ("cloudy_suspended", "Cloudy suspended residue"),
                           ("residue_settled", "Non-suspended residue")]:
            lines.append(f"  {label}: seen for about {self.water_frames[key] / FPS:.1f} s")
        if self.tint_colors:
            tint, _ = self.tint_colors.most_common(1)[0]
            lines.append(f"  Tint color: {tint}")
        return "\n".join(lines)

    def finish(self):
        summary = self.build_summary()
        self.reset_session()
        self.water.base_lab = None      # require fresh calibration for the next item
        self.water_state = {"calibrated": False}
        self.on_next(summary)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Camera test")
    container = tk.Frame(root)
    container.pack(fill="both", expand=True)

    class _Window:
        pass
    w = _Window()
    w.container = container

    page = CameraPage(w, on_next=lambda text: print(text), on_back=root.destroy)
    page.pack(fill="both", expand=True)
    page.start()
    root.protocol("WM_DELETE_WINDOW", lambda: (page.stop(), root.destroy()))
    root.mainloop()