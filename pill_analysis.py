import cv2
import numpy as np
from collections import deque


def crop_roi(frame, roi):
    x, y, w, h = [int(v) for v in roi]
    H, W = frame.shape[:2]
    x, y = max(0, x), max(0, y)
    return frame[y:min(H, y + h), x:min(W, x + w)]


def classify_hsv(h, s, v):
    """h in degrees (0-360), s and v in 0-1."""
    if v < 0.2:
        return "black"
    if s < 0.12:
        return "white" if v > 0.8 else "gray"
    if h < 15 or h >= 340:
        return "pink" if (s < 0.4 and v > 0.7) else "red"
    if h < 40:
        if s < 0.35 and v > 0.6:
            return "tan"
        return "brown" if v < 0.55 else "orange"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 260:
        return "blue"
    return "purple"


def dominant_color_name(roi):
    """Median color of the object, ignoring the background where possible."""
    if roi is None or roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    s = hsv[:, 1] / 255.0
    v = hsv[:, 2] / 255.0
    valid = (v > 0.15) & (v < 0.97)
    colored = valid & (s > 0.12)
    if colored.sum() > 0.05 * len(hsv):
        pick = colored
    elif valid.any():
        pick = valid
    else:
        pick = np.ones(len(hsv), dtype=bool)

    h = float(np.median(hsv[pick, 0])) * 2.0
    sm = float(np.median(s[pick]))
    vm = float(np.median(v[pick]))
    name = classify_hsv(h, sm, vm)
    if name in ("white", "gray", "black", "pink", "tan"):
        return name
    if sm < 0.25 and vm > 0.8:
        return "pale " + name
    return name


def median_lab(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.median(lab, axis=0)


def sharpness(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_det(frame, box, label, conf):
    x, y, w, h = box
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    return {
        "label": label,
        "conf": conf,
        "box": (x, y, w, h),
        "color": dominant_color_name(crop),
    }


def contour_objects(frame):
    """Fallback when no YOLO weights exist: finds object-like blobs."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        x, y, w, h = cv2.boundingRect(c)
        if 0.01 * H * W < w * h < 0.6 * H * W:
            out.append(make_det(frame, (x, y, w, h), "object", None))
    return out


def find_objects(frame, model=None):
    if model is None:
        return contour_objects(frame)
    res = model.predict(frame, conf=0.4, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        label = model.names[int(b.cls[0])]
        out.append(make_det(frame, (x1, y1, x2 - x1, y2 - y1), label, float(b.conf[0])))
    return out


class WaterAnalyzer:
    """
    Calibrate once with the EMPTY (clear water or empty glass) region.
    Then update() each frame. Thresholds are tunable at the top.
    """
    TINT_DELTA = 6.0        # chroma shift in Lab needed to call it tinted
    HAZE_L_DELTA = 6.0      # brightness rise needed for cloudiness
    HAZE_SHARP_RATIO = 0.7  # sharpness must fall below this ratio
    SOLID_DIFF = 25         # per-pixel diff from baseline counted as "solid"
    SOLID_FRAC = 0.03       # fraction of region that means "pill is in water"
    FIZZ_BLOB_MIN, FIZZ_BLOB_MAX = 2, 30
    FIZZ_AVG_MIN = 4        # average tiny moving blobs per frame

    def __init__(self):
        self.base_lab = None
        self.base_gray = None
        self.base_sharp = None
        self.prev_gray = None
        self.fizz_hist = deque(maxlen=15)
        self.solid_hist = deque(maxlen=60)  # about 2 seconds at 30 fps
        self.solid_seen = False
        self.dissolved = False

    @property
    def calibrated(self):
        return self.base_lab is not None

    def calibrate(self, frame, roi):
        crop = crop_roi(frame, roi)
        self.base_lab = median_lab(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        self.base_gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        self.base_sharp = sharpness(crop)
        self.reset_state()

    def reset_state(self):
        self.prev_gray = None
        self.fizz_hist.clear()
        self.solid_hist.clear()
        self.solid_seen = False
        self.dissolved = False

    def update(self, frame, roi):
        if not self.calibrated:
            return {"calibrated": False}

        crop = crop_roi(frame, roi)
        gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        gray = gray.astype(np.float32)
                # If the water box changed size since calibration, the baseline is invalid
        if gray.shape != self.base_gray.shape:
            self.reset_state()
            self.base_lab = None
            return {"calibrated": False}
        now_lab = median_lab(crop)

        # --- Tint: chroma shift versus the empty baseline ---
        chroma_delta = float(np.linalg.norm(now_lab[1:] - self.base_lab[1:]))
        tinted = chroma_delta > self.TINT_DELTA
        tint_color = dominant_color_name(crop) if tinted else None

        # --- Fizz: many tiny moving bright specks frame to frame ---
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray).astype(np.uint8)
            mask = (diff > 12).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            small = sum(
                1 for i in range(1, n)
                if self.FIZZ_BLOB_MIN <= stats[i, cv2.CC_STAT_AREA] <= self.FIZZ_BLOB_MAX
            )
            self.fizz_hist.append(small)
        self.prev_gray = gray
        fizz = len(self.fizz_hist) >= 10 and np.mean(self.fizz_hist) > self.FIZZ_AVG_MIN

        # --- Dissolved: a solid object appears, then its footprint vanishes ---
        diff_base = np.abs(gray - self.base_gray)
        solid_frac = float((diff_base > self.SOLID_DIFF).mean())
        self.solid_hist.append(solid_frac)
        if solid_frac > self.SOLID_FRAC:
            self.solid_seen = True
        if (self.solid_seen and len(self.solid_hist) == self.solid_hist.maxlen
                and np.mean(list(self.solid_hist)[-30:]) < 0.005):
            self.dissolved = True

        # --- Cloudy suspended: brighter and softer, but no settled blobs ---
        sharp = sharpness(crop)
        haze = ((now_lab[0] - self.base_lab[0]) > self.HAZE_L_DELTA
                and sharp < self.HAZE_SHARP_RATIO * self.base_sharp)

        # --- Non-suspended residue: persistent blobs in the bottom 35% ---
        h = crop.shape[0]
        bottom = (diff_base[int(h * 0.65):] > 20).astype(np.uint8)
        n2, _, st2, _ = cv2.connectedComponentsWithStats(bottom)
        settled_blobs = sum(1 for i in range(1, n2) if st2[i, cv2.CC_STAT_AREA] >= 40)

        return {
            "calibrated": True,
            "dissolved": self.dissolved,
            "tinted": tinted,
            "tint_color": tint_color,
            "fizzing": fizz,
            "cloudy_suspended": bool(haze and self.dissolved and settled_blobs == 0),
            "residue_settled": bool(settled_blobs > 0 and self.dissolved),
        }