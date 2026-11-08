# uses pretrained YOLO model 

import time 
from collections import Counter, defaultdict 
import cv2 
import numpy as np 
from ultralytics import YOLO 

camera_index = 0
confidence = 0.4 
ignore_classes = {"person"}
button_height = 60 
window = "detector"


def name_color(bgr_patch):
    """Return a basic color name for an image patch (median HSV, simple rules)."""
    if bgr_patch.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = np.median(hsv, axis=0)   # OpenCV ranges: H 0-179, S 0-255, V 0-255
 
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
    x1, y1, x2, y2 = 0, h - button_height, w, h
    state["button"] = (x1, y1, x2, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 160, 60), -1)
    cv2.putText(frame, "Next", (w // 2 - 40, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
 
 
# ---------------- Summary screen ----------------
def build_summary_lines(records, duration):
    lines = [f"Camera was open for {duration:.1f} seconds", ""]
    if not records:
        lines.append("No objects were detected.")
        return lines
    lines.append("Detected objects:")
    lines.append("")
    # Most frequently seen objects first
    for label, info in sorted(records.items(), key=lambda kv: -kv[1]["frames"]):
        colors = info["colors"]
        total = sum(colors.values())
        top_color = colors.most_common(1)[0][0]
        breakdown = ", ".join(f"{c} {100 * n / total:.0f}%" for c, n in colors.most_common(3))
        lines.append(f"- {label}  (seen in {info['frames']} frames)")
        lines.append(f"     color: {top_color}   [{breakdown}]")
    return lines
 
 
def show_summary(lines):
    for line in lines:
        print(line)
    height = max(400, 40 + 32 * len(lines))
    canvas = np.zeros((height, 900, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (20, 40 + 32 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(canvas, "Press any key to close", (20, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 120), 1)
    cv2.imshow(window, canvas)
    cv2.waitKey(0)
 
 
# ---------------- Main ----------------
def main():
    model = YOLO("yolov8n.pt")   # small pretrained model; downloads automatically
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different camera_index.")
 
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
 
    records = defaultdict(lambda: {"frames": 0, "colors": Counter()})
    start = time.time()
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
 
        result = model(frame, conf=confidence, verbose=False)[0]
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            if label in ignore_classes:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = box_color(frame, x1, y1, x2, y2)
 
            records[label]["frames"] += 1
            records[label]["colors"][color] += 1
 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {float(box.conf[0]):.2f} | {color}"
            cv2.rectangle(frame, (x1, y1 - 24), (x1 + 9 * len(text), y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1 + 3, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
 
        draw_button(frame)
        cv2.imshow(window, frame)
 
        key = cv2.waitKey(1) & 0xFF
        if state["next_clicked"] or key in (ord("n"), ord("q")):
            break
 
    duration = time.time() - start
    cap.release()
    show_summary(build_summary_lines(records, duration))
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()