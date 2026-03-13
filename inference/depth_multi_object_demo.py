# inference/depth_multi_object_demo.py
"""
Multi-object detection + monocular depth -> per-object distance estimation.
Requirements:
 - torch, torchvision
 - OpenCV (cv2)
 - numpy
 - scikit-learn (only if you re-train calib; not needed here)
 - internet on first run for torch.hub models (YOLOv5, MiDaS) unless cached.
"""

import time
import json
from pathlib import Path

import cv2
import torch
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
CALIB_FILE = Path("calibration/calib_params.json")
OBSTACLE_THRESHOLD = 0.7  # meters: per-object threshold
YOLO_MODEL_NAME = "yolov5s"  # small and fast; change if you want (yolov5n,m,..)
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# ---------------------------
# LOAD CALIBRATION
# ---------------------------
if not CALIB_FILE.exists():
    raise FileNotFoundError(f"Calibration file not found: {CALIB_FILE} -- run calibrate_fit.py first")

with open(CALIB_FILE, "r") as f:
    calib = json.load(f)
a = float(calib.get("a", 0.0))
b = float(calib.get("b", 0.0))
print("Loaded calibration:", calib)

# ---------------------------
# LOAD MiDaS
# ---------------------------
print("Loading MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# ---------------------------
# LOAD YOLOv5 detector
# ---------------------------
print(f"Loading {YOLO_MODEL_NAME} (YOLOv5) detector via torch.hub ...")
yolo = torch.hub.load("ultralytics/yolov5", YOLO_MODEL_NAME, pretrained=True)
yolo.to(DEVICE)
yolo.eval()

# The yolo model from hub accepts numpy arrays (BGR) but returns results for original size.

# ---------------------------
# OPEN WEBCAM
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press 'q' to quit. Starting demo...")

# helper: safe bbox int clamp
def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W-1, int(x1)))
    x2 = max(0, min(W-1, int(x2)))
    y1 = max(0, min(H-1, int(y1)))
    y2 = max(0, min(H-1, int(y2)))
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
    return x1, y1, x2, y2

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]

    # -----------------------
    # DEPTH MAP (MiDaS)
    # -----------------------
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(DEVICE)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()  # float32 depth "stat"

    # Normalize depth for visualization
    depth_vis = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    # -----------------------
    # OBJECT DETECTION (YOLO)
    # -----------------------
    # YOLO expects RGB or BGR? ultralytics accepts numpy BGR images as input
    results = yolo(frame[..., ::-1] if False else frame)  # using frame (BGR) works too in many builds
    # parse results
    detections = results.xyxy[0].cpu().numpy()  # (N,6): x1,y1,x2,y2,conf,class

    # build per-object list
    objects = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = results.names[int(cls)]
        x1i, y1i, x2i, y2i = clamp_bbox(x1, y1, x2, y2, W, H)
        # extract depth roi inside bbox
        roi = prediction[y1i:y2i, x1i:x2i]
        if roi.size == 0:
            # safeguard
            obj_pred = float(np.median(prediction[max(0,y1i-5):min(H,y2i+5), max(0,x1i-5):min(W,x2i+5)]))
        else:
            # use percentile near closest surface for robustness:
            # p = 10 percentile for "closest" depth (since MiDaS larger values -> closer in our calibration)
            # adjust percentile direction based on behavior seen in calibration (we used median previously).
            # Here we'll compute both median and 10th percentile and prefer lower distance.
            median_val = float(np.median(roi))
            p10 = float(np.percentile(roi, 10))
            # choose robust statistic: use p10 (closer) to detect nearest surface in bbox
            obj_pred = p10

        # convert to meters via calibration
        distance_m = a * obj_pred + b

        objects.append({
            "bbox": (x1i, y1i, x2i, y2i),
            "label": label,
            "conf": float(conf),
            "pred_stat": float(obj_pred),
            "distance_m": float(distance_m)
        })

    # -----------------------
    # SORT / prioritize objects by distance (closest first)
    # -----------------------
    objects_sorted = sorted(objects, key=lambda o: o["distance_m"])

    # annotate frame
    for i, obj in enumerate(objects_sorted):
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["label"]
        conf = obj["conf"]
        dist = obj["distance_m"]
        is_obstacle = dist < OBSTACLE_THRESHOLD

        color = (0, 0, 255) if is_obstacle else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        txt = f"{label} {conf:.2f} {dist:.2f}m"
        cv2.putText(frame, txt, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # mark index for clarity
        cv2.putText(frame, f"{i+1}", (x1+2, y1+14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # display nearest object info on top-left
    if len(objects_sorted) > 0:
        nearest = objects_sorted[0]
        txt2 = f"Nearest: {nearest['label']} {nearest['distance_m']:.2f} m"
    else:
        txt2 = "Nearest: -"
    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"{txt2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # combine visuals
    # resize depth_vis to match frame height
    depth_h, depth_w = depth_vis.shape[:2]
    if depth_h != H:
        depth_vis_resized = cv2.resize(depth_vis, (int(depth_w * H / depth_h), H))
    else:
        depth_vis_resized = depth_vis
    combined = np.hstack((frame, depth_vis_resized))

    cv2.imshow("RGB | Depth | Multi-Object Distance", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()