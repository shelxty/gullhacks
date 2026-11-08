"""
Beginner starter: webcam property detector (rule-based, no training needed yet).

Install:   pip install opencv-python numpy
Optional:  pip install pytesseract   (+ install the Tesseract program) for label reading

How to use
  1. Put the item you want to look at in the LEFT green box (on a plain, light background).
  2. Put a clear glass of plain water in the RIGHT blue box. Put a striped/checkered
     card behind the glass (helps the cloudiness check).
  3. With the glass containing ONLY clear water, press  C  to calibrate.
  4. Add the item to the water and watch the labels update.
  5. Click "Next" (or press N) for the session summary. Press Q to quit.

These are simple image-processing rules to get you started. Later you will replace
each rule with a trained classifier (see the guide).
"""
import cv2
import numpy as np
import time
import json
from collections import deque, defaultdict, Counter

# ---------------- Layout ----------------
FRAME_W, FRAME_H = 640, 480
PANEL_W = 320
BAR_H = 60
OBJ_ROI = (20, 60, 300, 420)     # x1, y1, x2, y2  (object zone)
WATER_ROI = (340, 60, 620, 420)  # x1, y1, x2, y2  (water glass zone)
BTN = (FRAME_W + PANEL_W // 2 - 80, FRAME_H + 10, FRAME_W + PANEL_W // 2 + 80, FRAME_H + 50)

try:
    import pytesseract
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False


# ---------------- Helpers ----------------
def color_name(h, s, v):
    """Convert an OpenCV HSV value (H 0-179) into a simple color word."""
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 180 else "gray"
    if h < 8 or h >= 172:
        return "pink" if (s < 110 and v > 190) else "red"
    if h < 22:
        return "brown" if v < 140 else "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"


def mean_hsv(img_bgr, mask=None):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if mask is None:
        return [float(x) for x in hsv.reshape(-1, 3).mean(axis=0)]
    pts = hsv[mask > 0]
    if len(pts) == 0:
        return [0.0, 0.0, 0.0]
    return [float(x) for x in pts.mean(axis=0)]


def crop(frame, roi):
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]


def majority(buf):
    return sum(buf) > len(buf) / 2 if buf else False


# ---------------- Object detection (left box) ----------------
def detect_objects(obj_img):
    """Find blobs in the object zone. Returns a list of dicts."""
    gray = cv2.GaussianBlur(cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # Assume the background is the most common brightness along the border
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_is_light = np.median(border) > 127
    flag = cv2.THRESH_BINARY_INV if bg_is_light else cv2.THRESH_BINARY
    _, th = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    big = [c for c in contours if cv2.contourArea(c) > 400]
    tiny = [c for c in contours if 8 < cv2.contourArea(c) <= 400]

    results = []
    if len(tiny) >= 25 and len(big) <= 1:
        # Lots of tiny specks and no solid object -> powder
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, tiny, -1, 255, -1)
        x, y, w, h = cv2.boundingRect(np.vstack(tiny))
        hsv = mean_hsv(obj_img, mask)
        results.append({"form": "powder", "color": color_name(*hsv), "box": (x, y, w, h)})
        return results

    for c in big:
        x, y, w, h = cv2.boundingRect(c)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        # Heuristic: elongated shapes -> "pill" (capsule), round/flat -> "tablet"
        form = "pill" if aspect > 1.6 else "tablet"
        hsv = mean_hsv(obj_img, mask)
        results.append({"form": form, "color": color_name(*hsv), "box": (x, y, w, h)})
    return results


def read_label(obj_img):
    """Optional OCR. Returns cleaned text or ''."""
    if not HAVE_OCR:
        return ""
    try:
        g = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=2, fy=2)
        txt = pytesseract.image_to_string(g, config="--psm 6")
        return "".join(ch for ch in txt if ch.isalnum() or ch in " -").strip()[:30]
    except Exception:
        return ""


# ---------------- Water analysis (right box) ----------------
class WaterAnalyzer:
    def __init__(self):
        self.base_gray = None
        self.base_hsv = None
        self.base_lapvar = None
        self.prev_gray = None
        self.history = {k: deque(maxlen=15) for k in
                        ["dissolved", "tinted", "cloudy", "residue", "fizz"]}
        self.tint_color = "none"

    @property
    def calibrated(self):
        return self.base_gray is not None

    def calibrate(self, water_img):
        g = cv2.GaussianBlur(cv2.cvtColor(water_img, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        self.base_gray = g.astype(np.float32)
        self.base_hsv = mean_hsv(water_img)
        self.base_lapvar = cv2.Laplacian(g, cv2.CV_64F).var() + 1e-6
        self.prev_gray = g
        for d in self.history.values():
            d.clear()

    def update(self, water_img):
        if not self.calibrated:
            return None
        g = cv2.GaussianBlur(cv2.cvtColor(water_img, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        h, w = g.shape

        # --- Fizz: lots of small frame-to-frame changes ---
        fd = cv2.absdiff(g, self.prev_gray)
        fizz = (fd > 20).mean() > 0.015
        self.prev_gray = g

        # --- Particles: blobs that differ from the clear-water baseline ---
        diff = cv2.absdiff(g.astype(np.float32), self.base_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, blobs = cv2.threshold(diff.astype(np.uint8), 35, 255, cv2.THRESH_BINARY)
        blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        particles = [c for c in cnts if 30 < cv2.contourArea(c) < 0.25 * h * w]
        bottom_particles = [c for c in particles
                            if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2 > h * 0.65]
        # Settled residue: particles in the bottom third that are NOT jumping around (not fizz)
        residue = len(bottom_particles) > 0 and not fizz
        dissolved = len(particles) == 0

        # --- Cloudiness: background pattern gets blurry/low contrast ---
        lapvar = cv2.Laplacian(g, cv2.CV_64F).var()
        cloudy = (lapvar / self.base_lapvar) < 0.5

        # --- Tint: color shift of the water vs the baseline ---
        cur = mean_hsv(water_img)
        b = np.array(self.base_hsv)
        # compare in color space (Lab) which is closer to how we see color
        lab_cur = cv2.cvtColor(np.uint8([[water_img.reshape(-1, 3).mean(axis=0)]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        base_bgr = cv2.cvtColor(np.uint8([[[b[0], b[1], b[2]]]]), cv2.COLOR_HSV2BGR)
        lab_base = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        chroma_shift = np.hypot(lab_cur[1] - lab_base[1], lab_cur[2] - lab_base[2])
        tinted = chroma_shift > 8
        self.tint_color = color_name(cur[0], max(cur[1], 120), max(cur[2], 150)) if tinted else "none"

        for k, val in [("dissolved", dissolved), ("tinted", tinted), ("cloudy", cloudy),
                       ("residue", residue), ("fizz", fizz)]:
            self.history[k].append(bool(val))

        return self.current()

    def current(self):
        return {
            "fully_dissolved": majority(self.history["dissolved"]),
            "tint": self.tint_color if majority(self.history["tinted"]) else "none",
            "fizzing": majority(self.history["fizz"]),
            "cloudy_suspended_residue": majority(self.history["cloudy"]),
            "non_suspended_residue": majority(self.history["residue"]),
        }


# ---------------- Session log ----------------
class Session:
    def __init__(self):
        self.t0 = time.time()
        self.objects = {}                       # (form, color) -> info
        self.water_votes = defaultdict(Counter)  # property -> Counter of values
        self.fizz_ever = False

    def log_object(self, form, color, label):
        key = (form, color)
        now = time.time()
        e = self.objects.setdefault(key, {"form": form, "color": color, "labels": Counter(),
                                          "first_seen_s": round(now - self.t0, 1), "frames": 0})
        e["frames"] += 1
        e["last_seen_s"] = round(now - self.t0, 1)
        if label:
            e["labels"][label] += 1

    def log_water(self, props):
        for k, v in props.items():
            self.water_votes[k][v] += 1
        if props["fizzing"]:
            self.fizz_ever = True

    def report(self):
        objs = []
        for e in self.objects.values():
            if e["frames"] < 5:      # ignore one-frame flickers
                continue
            lab = e["labels"].most_common(1)[0][0] if e["labels"] else "(none read)"
            objs.append({"form": e["form"], "color": e["color"], "label": lab,
                         "first_seen_s": e["first_seen_s"], "last_seen_s": e["last_seen_s"]})
        water = {k: c.most_common(1)[0][0] for k, c in self.water_votes.items()}
        if water:
            water["fizzing_seen_at_any_time"] = self.fizz_ever
        return {"duration_s": round(time.time() - self.t0, 1), "objects": objs,
                "water_properties": water,
                "ethanol_soluble": "not measured (needs a separate ethanol test)"}


# ---------------- Drawing ----------------
def put(img, text, x, y, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_main(canvas, frame, objs, water_state, calibrated):
    canvas[:FRAME_H, :FRAME_W] = frame
    cv2.rectangle(canvas, OBJ_ROI[:2], OBJ_ROI[2:], (0, 200, 0), 2)
    cv2.rectangle(canvas, WATER_ROI[:2], WATER_ROI[2:], (255, 150, 0), 2)
    put(canvas, "OBJECT ZONE", OBJ_ROI[0], OBJ_ROI[1] - 8, (0, 200, 0))
    put(canvas, "WATER ZONE", WATER_ROI[0], WATER_ROI[1] - 8, (255, 150, 0))

    for o in objs:
        x, y, w, h = o["box"]
        x += OBJ_ROI[0]
        y += OBJ_ROI[1]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
        put(canvas, f'{o["form"]}, {o["color"]}', x, max(15, y - 6), (0, 255, 255))

    # Right-hand panel
    px = FRAME_W + 10
    canvas[:, FRAME_W:] = (30, 30, 30)
    put(canvas, "LIVE DETECTIONS", px, 25, (255, 255, 255), 0.6, 2)
    y = 55
    if objs:
        for o in objs[:4]:
            put(canvas, f'Object: {o["form"]} / {o["color"]}', px, y, (0, 255, 255))
            y += 22
    else:
        put(canvas, "Object: none", px, y, (150, 150, 150))
        y += 22
    y += 10
    put(canvas, "WATER", px, y, (255, 150, 0), 0.55, 2)
    y += 25
    if not calibrated:
        put(canvas, "Press C with clear water", px, y, (80, 80, 255))
        put(canvas, "in the glass to calibrate", px, y + 20, (80, 80, 255))
    elif water_state:
        rows = [("Fully dissolved", "YES" if water_state["fully_dissolved"] else "NO"),
                ("Tint", water_state["tint"]),
                ("Fizzing", "YES" if water_state["fizzing"] else "no"),
                ("Cloudy suspension", "YES" if water_state["cloudy_suspended_residue"] else "no"),
                ("Settled residue", "YES" if water_state["non_suspended_residue"] else "no")]
        for name, val in rows:
            put(canvas, f"{name}: {val}", px, y)
            y += 24
    put(canvas, "Label OCR: " + ("on" if HAVE_OCR else "off (install pytesseract)"),
        px, FRAME_H - 15, (150, 150, 150), 0.4)

    # Bottom bar + Next button
    canvas[FRAME_H:, :FRAME_W] = (20, 20, 20)
    put(canvas, "C = calibrate clear water   N = next   Q = quit", 10, FRAME_H + 35, (180, 180, 180))
    cv2.rectangle(canvas, BTN[:2], BTN[2:], (0, 160, 0), -1)
    put(canvas, "Next", BTN[0] + 55, BTN[1] + 26, (255, 255, 255), 0.7, 2)


def draw_summary(canvas, report):
    canvas[:] = (25, 25, 25)
    put(canvas, "SESSION SUMMARY", 20, 35, (255, 255, 255), 0.9, 2)
    put(canvas, f'Camera open for {report["duration_s"]} s', 20, 60, (180, 180, 180))
    y = 95
    put(canvas, "Objects detected:", 20, y, (0, 255, 255), 0.65, 2)
    y += 28
    if not report["objects"]:
        put(canvas, "  none", 20, y)
        y += 24
    for i, o in enumerate(report["objects"], 1):
        put(canvas, f'  {i}. form={o["form"]}  color={o["color"]}  label="{o["label"]}"  '
                    f'seen {o["first_seen_s"]}s-{o["last_seen_s"]}s', 20, y)
        y += 24
    y += 15
    put(canvas, "Water properties (most common reading):", 20, y, (255, 150, 0), 0.65, 2)
    y += 28
    if not report["water_properties"]:
        put(canvas, "  no water data (did you press C to calibrate?)", 20, y)
        y += 24
    for k, v in report["water_properties"].items():
        put(canvas, f"  {k}: {v}", 20, y)
        y += 24
    y += 6
    put(canvas, f'  ethanol_soluble: {report["ethanol_soluble"]}', 20, y)
    put(canvas, "Saved to session_report.json   |   B = back   Q = quit", 20, canvas.shape[0] - 20,
        (180, 180, 180))


# ---------------- Main ----------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    state = {"clicked": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and BTN[0] <= x <= BTN[2] and BTN[1] <= y <= BTN[3]:
            state["clicked"] = True

    win = "Property Detector"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    water = WaterAnalyzer()
    session = Session()
    canvas = np.zeros((FRAME_H + BAR_H, FRAME_W + PANEL_W, 3), np.uint8)
    show_summary = False
    report = None
    frame_i = 0
    last_label = ""

    while True:
        if not show_summary:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

            obj_img = crop(frame, OBJ_ROI)
            objs = detect_objects(obj_img)
            if objs and HAVE_OCR and frame_i % 20 == 0:
                last_label = read_label(obj_img)
            for o in objs:
                session.log_object(o["form"], o["color"], last_label)

            water_img = crop(frame, WATER_ROI)
            wstate = water.update(water_img)
            if wstate:
                session.log_water(wstate)

            draw_main(canvas, frame, objs, wstate, water.calibrated)
            frame_i += 1
        else:
            draw_summary(canvas, report)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("c") and not show_summary:
            water.calibrate(crop(frame, WATER_ROI))
        if (key == ord("n") or state["clicked"]) and not show_summary:
            state["clicked"] = False
            report = session.report()
            with open("session_report.json", "w") as f:
                json.dump(report, f, indent=2)
            show_summary = True
        if key == ord("b") and show_summary:
            show_summary = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()