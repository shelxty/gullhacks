"""
Live webcam pill detector (VS Code version).

Uses your trained CNN (pill_cnn.keras + labels.json) to classify
  - form  (tablet / pill)         <- from the CNN
  - color (one or more colors)    <- from the CNN, plus a pixel-based color check
and uses simple image rules for the water tests (dissolved, tint, fizzing,
cloudy suspension, settled residue). "Powder" is not in your CNN yet, so a
rule-based powder guess is used until you train that class.

Install (in a terminal, ideally inside a virtual environment):
    pip install tensorflow opencv-python numpy
    pip install pytesseract          # optional, for label (imprint) reading; also needs the Tesseract program

Put these 3 files in the same folder and run:  python webcam_pill_detector.py
    webcam_pill_detector.py   pill_cnn.keras   labels.json

How to use
  1. Place the pill(s) in the LEFT green box on a plain, light background.
  2. Put a clear glass of plain water in the RIGHT blue box (a striped card behind it helps).
  3. With only clear water in the glass, press  C  to calibrate the water checks.
  4. COLOR: put a plain WHITE sheet of paper under/around the pill so the green box edge is paper. The script uses it
     as a white reference, so colors stay correct under yellow/blue room light. W = freeze/unfreeze that reference.
  5. SHOT GLASS: press G and drag a box around the LIQUID inside the glass (saved for next time), then press C once
     with clear water to calibrate. Tint is measured as the change from that clear-water baseline.
  6. Click "Next" (or press N) for the full session summary. B = back, Q = quit.
"""
import os
import json
import time
from collections import deque, defaultdict, Counter

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's noisy startup messages
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

# ---------------- Settings you can change ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "pill_cnn.keras")
LABELS_PATH = os.path.join(HERE, "labels.json")
CAMERA_INDEX = 0            # 0 = default webcam; try 1 if you have several
COLOR_THRESHOLD = 0.5       # a CNN color counts as "present" above this probability
REJECT_CONF = 0.55          # CNN confidence below this -> object is thrown away (raise to 0.8 after you add a not_pill class)
UNCERTAIN_CONF = 0.90       # between REJECT_CONF and this -> shown in orange as "uncertain"
BG_CHECK = False            # True = refuse to detect unless the zone edge is plain (usually unnecessary; shape filters do the work)
BG_STD_MAX = 28             # how "plain" the zone's edge must be (gray-level spread); raise if it complains too often
MAX_AREA_FRAC = 0.25        # an object bigger than this fraction of the object zone is not a pill (shirt, hand, face)
MIN_SOLIDITY = 0.84         # pills are compact/convex; hands and clothes are ragged (lower value = more permissive)
MAX_ASPECT = 3.5            # longest side / shortest side; capsules are ~2-3, arms and sleeves are longer
MIN_EXTENT = 0.50           # blob area / bounding-box area; thin diagonal shapes score low
MIN_AREA = 250              # smallest blob (pixels) that can be a pill; lower it if your pills are tiny in the picture
MIN_DELTA = 5               # how different from the background a pixel must be (lower = more sensitive)
DEBUG = True                # show WHY blobs were rejected (press D to toggle)
NEG_DIR = os.path.join(HERE, "negatives")  # "not a pill" images are saved here (press X) for retraining
POS_DIR = os.path.join(HERE, "positives")  # webcam photos of REAL pills are saved here (press T or P) for retraining
USE_NOT_PILL = False        # False = ignore the model's "not_pill" answer. Keep False until you retrain with webcam photos of real pills.
AUTO_WB = True              # use the paper at the edge of the green box as a white reference (fixes color casts from room lighting)
LOCK_CAMERA_WB = True       # ask the webcam to stop auto-adjusting its own white balance (ignored by some webcams)
TINT_MIN_DC = 5.0           # liquid must shift this far in color (CIELAB a/b units) from clear water to count as "tinted"; lower = more sensitive
SETTINGS_PATH = os.path.join(HERE, "settings.json")  # remembers the shot-glass box you drew with G
MAX_OBJECTS = 4             # classify at most this many objects per frame (keeps it fast)
MIN_TRACK_FRAMES = 8        # an object must be seen this many frames to appear in the summary

# ---------------- Layout ----------------
FRAME_W, FRAME_H = 640, 480
PANEL_W = 340
BAR_H = 60
OBJ_ROI = (20, 60, 300, 420)     # x1, y1, x2, y2  (object zone)
WATER_ROI = (340, 60, 620, 420)  # x1, y1, x2, y2  (water glass zone)
BTN = (FRAME_W + PANEL_W // 2 - 80, FRAME_H + 10, FRAME_W + PANEL_W // 2 + 80, FRAME_H + 50)

try:
    import pytesseract
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False


# ---------------- Load the trained model ----------------
class PillModel:
    def __init__(self, model_path, labels_path):
        with open(labels_path) as f:
            lab = json.load(f)
        self.forms = lab["forms"]
        self.colors = lab["colors"]
        self.size = int(lab.get("img_size", 128))
        # compile=False: we only need predictions, not training settings
        self.model = keras.models.load_model(model_path, compile=False)
        self.model.predict(np.zeros((1, self.size, self.size, 3), np.float32), verbose=0)  # warm-up

    def prep(self, bgr_crop):
        """Same preprocessing as training: RGB, resize with padding (no squishing), 0-255 floats."""
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB).astype(np.float32)
        return tf.image.resize_with_pad(rgb, self.size, self.size).numpy()

    def predict(self, crops):
        """crops: list of BGR images -> list of dicts."""
        if not crops:
            return []
        batch = np.stack([self.prep(c) for c in crops])
        out = self.model.predict(batch, verbose=0)
        # which form classes may be chosen? (skip "not_pill" unless USE_NOT_PILL is on)
        allowed = [i for i, f in enumerate(self.forms) if f != "not_pill" or USE_NOT_PILL]
        results = []
        for pf, pc in zip(out["form"], out["color"]):
            sub = pf[allowed]
            fi = allowed[int(np.argmax(sub))]
            conf = float(pf[fi]) if USE_NOT_PILL else float(pf[fi] / max(1e-9, sub.sum()))   # re-normalise over the allowed classes
            on = [self.colors[i] for i, p in enumerate(pc) if p >= COLOR_THRESHOLD]
            if not on:                                   # nothing passed the threshold -> use the single best guess
                on = [self.colors[int(np.argmax(pc))] + "?"]
            results.append({"form": self.forms[fi], "form_conf": conf, "cnn_colors": on,
                            "probs": {f: float(p) for f, p in zip(self.forms, pf)}})
        return results


# ---------------- Color science (CIELAB) ----------------
# CIELAB separates lightness (L) from color (a = green<->red, b = blue<->yellow). Naming colors from a/b, not from RGB/HSV,
# keeps names stable when the lighting gets brighter or darker.
def bgr_to_lab(bgr):
    """(N,3) uint8 BGR pixels -> (N,3) float CIELAB (L 0-100, a and b roughly -128..127)."""
    px = np.asarray(bgr, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(-1, 3)


def name_lab(L, a, b):
    """CIELAB value -> one color word."""
    C = float(np.hypot(a, b))                       # chroma: how colorful (0 = gray)
    h = float(np.degrees(np.arctan2(b, a)) % 360)   # hue angle: which color
    if L <= 14:
        return "black"
    warm = 20 <= h < 112                            # reds..yellows: cream/beige whites often have a little of this
    if C < (12 if warm else 8):                     # (nearly) colorless; a bluish tint is noticed at a lower chroma than a creamy one
        return "white" if L >= 70 else ("black" if L <= 25 else "gray")
    if warm and L >= 80 and C < 16:
        return "white"                              # cream / off-white
    if h >= 345 or h < 20:                          # rose, hot pink, crimson
        return "pink" if L >= 62 else "red"
    if h < 48:                                      # red .. red-orange
        if L >= 70 and C < 50:
            return "pink"                           # salmon / light coral
        return "brown" if (L < 40 and C < 40) else "red"
    if h < 80:                                      # orange .. tan
        if L < 55:
            return "brown"
        return "beige" if C < 32 else "orange"
    if h < 112:
        return "yellow" if L >= 60 else "brown"     # dark yellow = olive/brown
    if h < 175:
        return "green"
    if h < 215:
        return "turquoise"
    if h < 308:
        return "blue"
    return "purple"


def pill_colors(img_bgr, mask, erode=True):
    """Measure a pill's color from the pixels inside `mask`. Handles two-tone capsules (e.g. 'red/white').
    Returns {"names": [...], "text": "red/white", "lab": (L, a, b)}."""
    m = mask
    if erode:                                       # shave the edge: shadows and anti-aliasing live there
        x, y, w, h = cv2.boundingRect(mask)
        k = max(3, int(0.15 * min(w, h)))
        e = cv2.erode(mask, np.ones((k, k), np.uint8))
        if cv2.countNonZero(e) >= 40:
            m = e
    pix = img_bgr[m > 0]
    if len(pix) == 0:
        return {"names": ["unknown"], "text": "unknown", "lab": (0.0, 0.0, 0.0)}
    if len(pix) > 3000:
        pix = pix[np.random.default_rng(0).choice(len(pix), 3000, replace=False)]
    lab = bgr_to_lab(pix)
    lab = lab[lab[:, 0] <= np.percentile(lab[:, 0], 92)]      # drop the brightest 8% (glare / specular highlights)
    med = np.median(lab, axis=0)                              # median = robust to imprints and leftover shadow
    names, main = [name_lab(*med)], med
    if len(lab) >= 60:                                        # is it two colors? (capsules, two-tone tablets)
        wgt = np.array([0.5, 1.0, 1.0], np.float32)           # count lightness only half as much as color
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centers = cv2.kmeans((lab * wgt).astype(np.float32), 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        share = np.bincount(labels.ravel(), minlength=2) / len(labels)
        c = centers / wgt
        n0, n1 = name_lab(*c[0]), name_lab(*c[1])
        if n0 != n1 and np.linalg.norm(centers[0] - centers[1]) > 30 and share.min() >= 0.2:
            order = np.argsort(-share)
            names, main = [(n0, n1)[i] for i in order], c[order[0]]
    return {"names": names, "text": "/".join(names), "lab": tuple(float(v) for v in main)}


def name_tint(dL, da, db):
    """Describe how liquid changed vs clear water, e.g. 'light yellow'. Only the a/b shift decides the hue."""
    dC = float(np.hypot(da, db))
    h = float(np.degrees(np.arctan2(db, da)) % 360)
    ref_L = 45.0 if dL < -25 else 65.0                        # a much darker shift reads as brown rather than orange
    word = name_lab(ref_L, 50 * np.cos(np.radians(h)), 50 * np.sin(np.radians(h)))
    if word == "pink" and dC >= 20:
        word = "red"                                          # a weak red dye looks pink, a strong one looks red
    strength = "faint" if dC < 10 else ("light" if dC < 25 else "strong")
    return f"{strength} {word}"


class WhiteBalance:
    """Uses the plain paper at the edge of the object zone as a white reference and rescales R/G/B so it looks neutral."""

    def __init__(self):
        self.gains = np.ones(3, np.float32)
        self.locked = False
        self.found = False

    def update(self, frame):
        if self.locked:
            return
        zone = frame[OBJ_ROI[1]:OBJ_ROI[3], OBJ_ROI[0]:OBJ_ROI[2]]
        t = 12
        border = np.concatenate([zone[:t].reshape(-1, 3), zone[-t:].reshape(-1, 3),
                                 zone[:, :t].reshape(-1, 3), zone[:, -t:].reshape(-1, 3)])
        lab = bgr_to_lab(border[::5])
        med_lab = np.median(lab, axis=0)
        # accept only a bright, evenly lit, nearly colorless border (= paper); skin, shirts, rooms fail this test
        self.found = bool(med_lab[0] > 45 and np.hypot(med_lab[1], med_lab[2]) < 30 and lab[:, 0].std() < 10)
        if not self.found:
            return
        med = np.median(border, axis=0).astype(np.float32)    # BGR
        g = np.clip(235.0 / np.maximum(med, 1.0), 0.5, 4.0)   # makes the paper neutral AND a fixed brightness (also cancels dim/bright rooms)
        self.gains = 0.7 * self.gains + 0.3 * g               # smooth, so it does not flicker

    def apply(self, frame):
        return np.clip(frame.astype(np.float32) * self.gains, 0, 255).astype(np.uint8)

    def status(self):
        if self.locked:
            return "WB: locked (W to unlock)"
        return "WB: paper reference OK" if self.found else "WB: NO paper at box edge (colors less reliable)"


def load_settings():
    global WATER_ROI
    try:
        with open(SETTINGS_PATH) as f:
            WATER_ROI = tuple(int(v) for v in json.load(f)["water_roi"])
    except Exception:
        pass


def save_settings():
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump({"water_roi": list(WATER_ROI)}, f)
    except Exception:
        pass


def crop(frame, roi):
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]


def majority(buf):
    return sum(buf) > len(buf) / 2 if buf else False


# ---------------- Finding objects (left box) ----------------
def background_ok(obj_img):
    """True if the edge of the object zone is a plain, even surface (paper, table). A shirt/face/room is not."""
    gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
    t = 12
    border = np.concatenate([gray[:t, :].ravel(), gray[-t:, :].ravel(), gray[:, :t].ravel(), gray[:, -t:].ravel()])
    return float(border.std()) <= BG_STD_MAX


def foreground_mask(obj_img):
    """Fit a smooth background to the zone's border (handles lighting gradients), keep pixels that differ from it."""
    small = cv2.resize(obj_img, (obj_img.shape[1] // 2, obj_img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    xn, yn = xs / w * 2 - 1, ys / h * 2 - 1
    A = np.stack([np.ones_like(xn), xn, yn, xn * yn, xn ** 2, yn ** 2], -1).reshape(-1, 6)
    band = np.zeros((h, w), bool)
    t = 6
    band[:t] = band[-t:] = True
    band[:, :t] = band[:, -t:] = True
    flat = lab.reshape(-1, 3)
    coef = np.linalg.lstsq(A[band.ravel()], flat[band.ravel()], rcond=None)[0]   # quadratic surface per color channel
    dE = np.linalg.norm(flat - A @ coef, axis=1).reshape(h, w)                    # distance from the fitted background
    thr = float(np.clip(4.0 * np.median(dE[band]), MIN_DELTA, 30))               # adapt to camera noise (median = robust to bright/clipped corners)
    m = (dE > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.resize(m, (obj_img.shape[1], obj_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return m, thr


def find_objects(obj_img, strict=True, info=None):
    """Locate blobs. strict=True keeps only pill-shaped ones; strict=False returns everything (for collecting negatives).
    info (a dict) is filled with debug details: threshold used, how many blobs, and why blobs were rejected."""
    H, W = obj_img.shape[:2]
    th, thr = foreground_mask(obj_img)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if info is not None:
        info.update({"thr": thr, "rejected": [], "n_blobs": 0})

    max_area = (MAX_AREA_FRAC if strict else 0.9) * H * W
    found, tiny = [], []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area <= 8:
            continue
        if area <= MIN_AREA:
            tiny.append(c)
            continue
        if info is not None:
            info["n_blobs"] += 1
        if strict:
            hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-6
            reason = None
            if area >= max_area:
                reason = "too big"
            elif x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
                reason = "touches edge"
            elif area / hull_area < MIN_SOLIDITY:
                reason = f"ragged {area / hull_area:.2f}"
            elif max(w, h) / max(1, min(w, h)) > MAX_ASPECT:
                reason = "too long"
            elif area / (w * h) < MIN_EXTENT:
                reason = "thin shape"
            if reason:
                if info is not None:
                    info["rejected"].append(((x, y, w, h), reason))
                continue
        elif area >= max_area:
            continue
        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        pc = pill_colors(obj_img, mask)
        found.append({"box": (x, y, w, h), "pixel_color": pc["text"], "color_names": pc["names"], "lab": pc["lab"], "is_powder": False})
        if len(found) >= MAX_OBJECTS:
            break

    if strict and not found and 25 <= len(tiny) <= 4000:   # many specks and no solid object -> powder (rule-based)
        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, tiny, -1, 255, -1)
        x, y, w, h = cv2.boundingRect(np.vstack(tiny))
        if w * h < MAX_AREA_FRAC * H * W:
            pc = pill_colors(obj_img, mask, erode=False)
            found.append({"box": (x, y, w, h), "pixel_color": pc["text"], "color_names": pc["names"], "lab": pc["lab"], "is_powder": True})
    return found


def save_negatives(obj_img, blobs):
    """Save 'this is NOT a pill' crops: every blob the detector finds + one random patch of the zone."""
    os.makedirs(NEG_DIR, exist_ok=True)
    stamp = time.time_ns()
    crops = [box_crop(obj_img, o["box"]) for o in blobs]
    H, W = obj_img.shape[:2]
    sz = np.random.randint(60, min(H, W))
    y0, x0 = np.random.randint(0, H - sz + 1), np.random.randint(0, W - sz + 1)
    crops.append(obj_img[y0:y0 + sz, x0:x0 + sz])
    for i, c in enumerate(crops):
        if c.size:
            cv2.imwrite(os.path.join(NEG_DIR, f"neg_{stamp}_{i}.jpg"), c)
    return len(crops)


def save_positives(obj_img, blobs, form):
    """Save webcam photos of REAL pills into positives/<form>/<pixel color>/ (you can fix wrong color folders by hand)."""
    n = 0
    stamp = time.time_ns()
    for i, o in enumerate(blobs):
        c = box_crop(obj_img, o["box"])
        if c.size:
            folder = os.path.join(POS_DIR, form, o["color_names"][0])   # main color only (two-tone pills go in their main color's folder)
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, f"pos_{stamp}_{i}.jpg"), c)
            n += 1
    return n


def box_crop(img, box, pad=0.15):
    """Crop a box with some margin around it (the CNN saw whole pills with a bit of background)."""
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(img.shape[1], x + w + px), min(img.shape[0], y + h + py)
    return img[y1:y2, x1:x2]


def read_label(obj_img):
    """Optional OCR of printed text/imprint. Unreliable on small embossed imprints."""
    if not HAVE_OCR:
        return ""
    try:
        g = cv2.resize(cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2)
        txt = pytesseract.image_to_string(g, config="--psm 6")
        return "".join(ch for ch in txt if ch.isalnum() or ch in " -").strip()[:30]
    except Exception:
        return ""


# ---------------- Tracking: remember each distinct object ----------------
class Track:
    _next_id = 1

    def __init__(self, cx, cy, t):
        self.id = Track._next_id
        Track._next_id += 1
        self.cx, self.cy = cx, cy
        self.frames = 0
        self.first, self.last = t, t
        self.forms = Counter()
        self.cnn_colors = Counter()
        self.pixel_colors = Counter()
        self.conf_sum = 0.0
        self.labels = Counter()

    def add(self, det, t):
        self.frames += 1
        self.last = t
        self.forms[det["form"]] += 1
        self.pixel_colors[det["pixel_color"]] += 1
        self.conf_sum += det["form_conf"]
        for c in det["cnn_colors"]:
            self.cnn_colors[c] += 1
        if det.get("label"):
            self.labels[det["label"]] += 1

    def summary(self, t0):
        form = self.forms.most_common(1)[0][0]
        # CNN colors seen in at least 40% of this object's frames
        cnn = [c for c, n in self.cnn_colors.most_common() if n >= 0.4 * self.frames] or \
              [self.cnn_colors.most_common(1)[0][0]]
        return {
            "object_id": self.id,
            "form": form,
            "form_confidence": round(self.conf_sum / self.frames, 2),
            "color_cnn_guess": cnn,
            "color_measured": self.pixel_colors.most_common(1)[0][0],
            "label_text": self.labels.most_common(1)[0][0] if self.labels else "(none read)",
            "first_seen_s": round(self.first - t0, 1),
            "last_seen_s": round(self.last - t0, 1),
        }


class Session:
    MATCH_DIST = 80  # pixels: how far an object may move between frames and still be "the same object"

    def __init__(self):
        self.t0 = time.time()
        self.tracks = []
        self.water_votes = defaultdict(Counter)
        self.fizz_ever = False

    def log_objects(self, dets):
        now = time.time()
        used = set()
        for d in dets:
            x, y, w, h = d["box"]
            cx, cy = x + w / 2, y + h / 2
            best, bd = None, self.MATCH_DIST
            for tr in self.tracks:
                if tr.id in used or now - tr.last > 3:     # skip tracks that vanished more than 3 s ago
                    continue
                dist = np.hypot(tr.cx - cx, tr.cy - cy)
                if dist < bd:
                    best, bd = tr, dist
            if best is None:
                best = Track(cx, cy, now)
                self.tracks.append(best)
            best.cx, best.cy = cx, cy
            best.add(d, now)
            used.add(best.id)
            d["track_id"] = best.id

    def log_water(self, props):
        for k, v in props.items():
            if k.startswith("_"):      # debug-only entries are not votes
                continue
            self.water_votes[k][v] += 1
        if props["fizzing"]:
            self.fizz_ever = True

    def report(self):
        objs = [t.summary(self.t0) for t in self.tracks if t.frames >= MIN_TRACK_FRAMES]
        water = {k: c.most_common(1)[0][0] for k, c in self.water_votes.items()}
        if water:
            water["fizzing_seen_at_any_time"] = self.fizz_ever
        return {"duration_s": round(time.time() - self.t0, 1), "objects": objs,
                "water_properties": water,
                "ethanol_soluble": "not measured (needs a separate ethanol test)"}


# ---------------- Water analysis (right box) ----------------
class WaterAnalyzer:
    def __init__(self):
        self.base_gray = None
        self.base_lab = None
        self.tint_hist = deque(maxlen=15)
        self.tint_dbg = ""
        self.base_lapvar = None
        self.prev_gray = None
        self.history = {k: deque(maxlen=15) for k in ["dissolved", "tinted", "cloudy", "residue", "fizz"]}
        self.tint_color = "none"

    @property
    def calibrated(self):
        return self.base_gray is not None

    @staticmethod
    def inner_lab(water_img):
        """Median CIELAB color of the middle 60% of the liquid box (skips glass edges, rim and reflections)."""
        h, w = water_img.shape[:2]
        inner = water_img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        return np.median(bgr_to_lab(inner.reshape(-1, 3)[::5]), axis=0)

    def calibrate(self, water_img):
        g = cv2.GaussianBlur(cv2.cvtColor(water_img, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        self.base_gray = g.astype(np.float32)
        self.base_lab = self.inner_lab(water_img)
        self.tint_hist.clear()
        self.base_lapvar = cv2.Laplacian(g, cv2.CV_64F).var() + 1e-6
        self.prev_gray = g
        for d in self.history.values():
            d.clear()

    def update(self, water_img):
        if not self.calibrated:
            return None
        g = cv2.GaussianBlur(cv2.cvtColor(water_img, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        h, w = g.shape

        fizz = (cv2.absdiff(g, self.prev_gray) > 20).mean() > 0.015   # many small frame-to-frame changes
        self.prev_gray = g

        diff = cv2.GaussianBlur(cv2.absdiff(g.astype(np.float32), self.base_gray), (5, 5), 0)
        _, blobs = cv2.threshold(diff.astype(np.uint8), 35, 255, cv2.THRESH_BINARY)
        blobs = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        particles = [c for c in cnts if 30 < cv2.contourArea(c) < 0.25 * h * w]
        bottom = [c for c in particles
                  if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2 > h * 0.65]
        residue = len(bottom) > 0 and not fizz
        dissolved = len(particles) == 0

        cloudy = (cv2.Laplacian(g, cv2.CV_64F).var() / self.base_lapvar) < 0.5   # background pattern gets blurry

        lab = self.inner_lab(water_img)                               # current liquid color
        dL, da, db = lab - self.base_lab                              # change vs the clear-water baseline
        dC = float(np.hypot(da, db))
        tinted = dC > TINT_MIN_DC
        self.tint_hist.append(name_tint(dL, da, db) if tinted else "none")
        self.tint_dbg = f"tint shift dC={dC:.1f} dL={dL:+.0f} hue={np.degrees(np.arctan2(db, da)) % 360:.0f}"

        for k, val in [("dissolved", dissolved), ("tinted", tinted), ("cloudy", cloudy),
                       ("residue", residue), ("fizz", fizz)]:
            self.history[k].append(bool(val))
        return self.current()

    def current(self):
        return {
            "fully_dissolved": majority(self.history["dissolved"]),
            "tint": (Counter(t for t in self.tint_hist if t != "none").most_common(1) or [("none", 0)])[0][0]
                    if majority(self.history["tinted"]) else "none",
            "fizzing": majority(self.history["fizz"]),
            "cloudy_suspended_residue": majority(self.history["cloudy"]),
            "non_suspended_residue": majority(self.history["residue"]),
            "_debug": self.tint_dbg,
        }


# ---------------- Drawing ----------------
def put(img, text, x, y, color=(255, 255, 255), scale=0.5, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_main(canvas, frame, dets, water_state, calibrated, bg_ok=True, mode=None, n_saved=0, info=None, status=""):
    canvas[:FRAME_H, :FRAME_W] = frame
    cv2.rectangle(canvas, OBJ_ROI[:2], OBJ_ROI[2:], (0, 200, 0), 2)
    cv2.rectangle(canvas, WATER_ROI[:2], WATER_ROI[2:], (255, 150, 0), 2)
    put(canvas, "OBJECT ZONE - hold pills INSIDE this box", OBJ_ROI[0], OBJ_ROI[1] - 8, (0, 200, 0), 0.45)
    put(canvas, "WATER ZONE", WATER_ROI[0], WATER_ROI[1] - 8, (255, 150, 0))
    if not bg_ok:
        put(canvas, "Plain background needed here", OBJ_ROI[0] + 10, OBJ_ROI[1] + 25, (0, 0, 255), 0.5, 2)
    if mode == "neg":
        put(canvas, f"SAVING 'NOT A PILL' IMAGES ({n_saved} saved) - remove all pills!", 10, 20, (0, 0, 255), 0.55, 2)
    elif mode in ("tablet", "pill"):
        put(canvas, f"SAVING REAL {mode.upper()} PHOTOS ({n_saved} saved) - only {mode}s in the green box!", 10, 20, (0, 160, 0), 0.55, 2)

    if info is not None:   # debug overlay: rejected blobs in red with the reason
        for (bx, by, bw, bh), why in info["rejected"]:
            cv2.rectangle(canvas, (bx + OBJ_ROI[0], by + OBJ_ROI[1]), (bx + bw + OBJ_ROI[0], by + bh + OBJ_ROI[1]), (0, 0, 255), 1)
            put(canvas, why, bx + OBJ_ROI[0], by + OBJ_ROI[1] + 12, (0, 0, 255), 0.4)
        put(canvas, f'DEBUG blobs={info["n_blobs"]} rejected={len(info["rejected"])} kept={len(dets)} thr={info["thr"]:.0f}',
            OBJ_ROI[0] + 4, OBJ_ROI[3] - 6, (0, 0, 255), 0.4)
    for d in dets:
        x, y, w, h = d["box"]
        x, y = x + OBJ_ROI[0], y + OBJ_ROI[1]
        shaky = d["form_conf"] < UNCERTAIN_CONF
        col = (0, 165, 255) if shaky else (0, 255, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 2)
        tag = f'#{d.get("track_id", "?")} {d["form"]} {d["form_conf"] * 100:.0f}%' + (" (uncertain)" if shaky else "")
        if info is not None and "not_pill" in d.get("probs", {}):          # debug: how sure is the model this is NOT a pill?
            tag += f' [not_pill {d["probs"]["not_pill"] * 100:.0f}%]'
        put(canvas, tag, x, max(15, y - 22), col, 0.45)
        put(canvas, "color: " + d["pixel_color"] + "  (CNN guess: " + "/".join(d["cnn_colors"]) + ")", x, max(30, y - 7), col, 0.42)

    px = FRAME_W + 10
    canvas[:, FRAME_W:] = (30, 30, 30)
    put(canvas, "LIVE DETECTIONS", px, 25, (255, 255, 255), 0.6, 2)
    y = 52
    if dets:
        for d in dets[:MAX_OBJECTS]:
            put(canvas, f'#{d.get("track_id", "?")} {d["form"]} ({d["form_conf"] * 100:.0f}%)', px, y, (0, 255, 255))
            put(canvas, "  color (measured): " + d["pixel_color"], px, y + 17, (0, 255, 255), 0.45)
            put(canvas, "  color (CNN guess): " + ", ".join(d["cnn_colors"]), px, y + 33, (0, 180, 180), 0.4)
            y += 46
    else:
        put(canvas, "Object: none", px, y, (150, 150, 150))
        y += 24

    y = max(y + 8, 250)
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
        if info is not None and water_state.get("_debug"):
            put(canvas, water_state["_debug"], px, y, (0, 0, 255), 0.42)
    put(canvas, status, px, FRAME_H - 32, (0, 200, 255), 0.42)
    put(canvas, "Label OCR: " + ("on" if HAVE_OCR else "off (pip install pytesseract)"),
        px, FRAME_H - 15, (150, 150, 150), 0.4)

    canvas[FRAME_H:, :FRAME_W] = (20, 20, 20)
    put(canvas, "C calib  G glass  W white-ref  T/P/X save  D debug  N next", 10, FRAME_H + 35, (180, 180, 180))
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
    for o in report["objects"]:
        put(canvas, f'  #{o["object_id"]}  form={o["form"]} ({o["form_confidence"] * 100:.0f}%)  '
                    f'label="{o["label_text"]}"', 20, y)
        put(canvas, f'       color (measured) = {o["color_measured"]}   color (CNN guess) = {", ".join(o["color_cnn_guess"])}   '
                    f'seen {o["first_seen_s"]}s-{o["last_seen_s"]}s', 20, y + 20, (200, 200, 200), 0.45)
        y += 48
    y += 10
    put(canvas, "Water properties (most common reading):", 20, y, (255, 150, 0), 0.65, 2)
    y += 28
    if not report["water_properties"]:
        put(canvas, "  no water data (did you press C to calibrate?)", 20, y)
        y += 24
    for k, v in report["water_properties"].items():
        put(canvas, f"  {k}: {v}", 20, y)
        y += 24
    put(canvas, f'  ethanol_soluble: {report["ethanol_soluble"]}', 20, y + 6)
    put(canvas, "Saved to session_report.json   |   B = back   Q = quit", 20, canvas.shape[0] - 20, (180, 180, 180))


# ---------------- Main ----------------
def main():
    global DEBUG, WATER_ROI
    load_settings()   # restores the shot-glass box you drew last time
    for p in (MODEL_PATH, LABELS_PATH):
        if not os.path.exists(p):
            print(f"Missing file: {p}\nPut pill_cnn.keras and labels.json next to this script.")
            return
    print("Loading model...")
    pill_model = PillModel(MODEL_PATH, LABELS_PATH)
    print("Model ready. Forms:", pill_model.forms, "| Colors:", pill_model.colors)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Try changing CAMERA_INDEX at the top of the script.")
        return

    if LOCK_CAMERA_WB:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)   # stop the webcam from silently "correcting" away the liquid's tint

    state = {"clicked": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and BTN[0] <= x <= BTN[2] and BTN[1] <= y <= BTN[3]:
            state["clicked"] = True

    win = "Pill Detector"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    water, session, wb = WaterAnalyzer(), Session(), WhiteBalance()
    canvas = np.zeros((FRAME_H + BAR_H, FRAME_W + PANEL_W, 3), np.uint8)
    show_summary, report, frame_i, last_label = False, None, 0, ""
    mode, n_saved = None, 0   # mode: None / 'neg' / 'tablet' / 'pill' (what the camera is currently saving)

    while True:
        if not show_summary:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            if AUTO_WB:
                wb.update(frame)
            frame_wb = wb.apply(frame)      # color-corrected copy: ALL color measurements use this one
            obj_img = crop(frame_wb, OBJ_ROI)

            bg_ok = background_ok(obj_img) or not BG_CHECK
            dets, raw_dets, info = [], [], {"thr": 0, "rejected": [], "n_blobs": 0}
            if bg_ok:
                for o in find_objects(obj_img, info=info):
                    if o["is_powder"]:   # CNN has no powder class yet -> rule-based guess
                        o.update({"form": "powder (rule-based)", "form_conf": 1.0, "cnn_colors": ["n/a"]})
                    dets.append(o)
                solid = [d for d in dets if not d["is_powder"]]
                for d, res in zip(solid, pill_model.predict([box_crop(obj_img, d["box"]) for d in solid])):
                    d.update(res)
                raw_dets = list(dets)   # everything the shape filters accepted (before the CNN gets a say)
                kept = []
                for d in dets:       # throw away anything the CNN calls "not_pill" or is not confident about
                    if d["is_powder"] or (d["form"] != "not_pill" and d["form_conf"] >= REJECT_CONF):
                        kept.append(d)
                    else:
                        info["rejected"].append((d["box"], f'CNN: {d["form"]} {d["form_conf"] * 100:.0f}%'))
                dets = kept

            if mode == "neg" and frame_i % 6 == 0:
                n_saved += save_negatives(obj_img, find_objects(obj_img, strict=False))
            elif mode in ("tablet", "pill") and frame_i % 6 == 0:
                n_saved += save_positives(obj_img, [d for d in raw_dets if not d["is_powder"]], mode)

            if dets and HAVE_OCR and frame_i % 20 == 0:
                last_label = read_label(obj_img)
            for d in dets:
                d["label"] = last_label
            session.log_objects(dets)

            wstate = water.update(crop(frame_wb, WATER_ROI))
            if wstate:
                session.log_water(wstate)

            draw_main(canvas, frame, dets, wstate, water.calibrated, bg_ok, mode, n_saved, info if DEBUG else None, wb.status())
            frame_i += 1
        else:
            draw_summary(canvas, report)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            DEBUG = not DEBUG
        for k, m in ((ord("x"), "neg"), (ord("t"), "tablet"), (ord("p"), "pill")):
            if key == k and not show_summary:
                mode = None if mode == m else m   # press the same key again to stop saving
                n_saved = 0
        if key == ord("c") and not show_summary:
            water.calibrate(crop(frame_wb, WATER_ROI))
        if key == ord("w") and not show_summary:
            wb.locked = not wb.locked       # freeze the white reference (e.g. before you move the paper)
        if key == ord("g") and not show_summary:
            title = "Drag a box around the LIQUID inside the shot glass, then press ENTER"
            sel = cv2.selectROI(title, frame, showCrosshair=False)
            cv2.destroyWindow(title)
            if sel[2] >= 20 and sel[3] >= 20:
                WATER_ROI = (int(sel[0]), int(sel[1]), int(sel[0] + sel[2]), int(sel[1] + sel[3]))
                water = WaterAnalyzer()     # needs a fresh clear-water calibration (press C)
                save_settings()
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