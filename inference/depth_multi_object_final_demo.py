"""
TESTING CODE
Multi Object Detection + Stable Depth Distance Estimation

Improvements:
- Center crop depth sampling (reduce background noise)
- Median depth estimation
- Temporal smoothing for stable distance
- Nearest obstacle detection
"""

import time
import json
from pathlib import Path

import cv2
import torch
import numpy as np

 
# CONFIG
CALIB_FILE = Path("calibration/calib_params.json")
OBSTACLE_THRESHOLD = 0.7
YOLO_MODEL = "yolov5s"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD CALIBRATION
if not CALIB_FILE.exists():
    raise FileNotFoundError("Calibration file not found")

with open(CALIB_FILE, "r") as f:
    calib = json.load(f)

a = calib["a"]
b = calib["b"]

print("Calibration loaded:", calib)

 
# LOAD MIDAS
print("Loading MiDaS...")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(DEVICE)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

 
# LOAD YOLO
print("Loading YOLO detector...")

yolo = torch.hub.load("ultralytics/yolov5", YOLO_MODEL, pretrained=True)
yolo.to(DEVICE)
yolo.eval()

 
# WEBCAM
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press Q to quit")

 
# DISTANCE MEMORY
distance_memory = {}

prev_time = time.time()
 
# LOOP
while True:

    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

      
    # DEPTH ESTIMATION
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(img_rgb).to(DEVICE)

    with torch.no_grad():

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # visualization
    depth_vis = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

      
    # OBJECT DETECTION
    results = yolo(frame)

    detections = results.xyxy[0].cpu().numpy()

    objects = []

    for det in detections:

        x1, y1, x2, y2, conf, cls = det
        label = results.names[int(cls)]

        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(W - 1, x2))
        y2 = int(min(H - 1, y2))

          
        # CENTER DEPTH SAMPLING
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        bw = int((x2 - x1) * 0.4)
        bh = int((y2 - y1) * 0.4)

        sx1 = max(0, cx - bw // 2)
        sx2 = min(W, cx + bw // 2)
        sy1 = max(0, cy - bh // 2)
        sy2 = min(H, cy + bh // 2)

        roi = prediction[sy1:sy2, sx1:sx2]

        if roi.size == 0:
            obj_pred = float(np.median(prediction[y1:y2, x1:x2]))
        else:
            obj_pred = float(np.median(roi))

          
        # DISTANCE CONVERSION
        distance_m = a * obj_pred + b

          
        # TEMPORAL SMOOTHING
        key = f"{label}_{cx}_{cy}"

        if key in distance_memory:
            distance_m = 0.7 * distance_memory[key] + 0.3 * distance_m

        distance_memory[key] = distance_m

        objects.append({
            "bbox": (x1, y1, x2, y2),
            "label": label,
            "conf": conf,
            "distance": distance_m
        })

      
    # SORT NEAREST
    objects = sorted(objects, key=lambda x: x["distance"])

      
    # DRAW OBJECTS
    for i, obj in enumerate(objects):

        x1, y1, x2, y2 = obj["bbox"]
        dist = obj["distance"]

        is_obstacle = dist < OBSTACLE_THRESHOLD

        color = (0, 0, 255) if is_obstacle else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        txt = f"{obj['label']} {dist:.2f}m"

        cv2.putText(
            frame,
            txt,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        cv2.putText(
            frame,
            f"{i+1}",
            (x1 + 3, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

      
    # NEAREST OBJECT
    if len(objects) > 0:
        nearest = objects[0]
        info = f"Nearest: {nearest['label']} {nearest['distance']:.2f} m"
    else:
        info = "Nearest: -"

      
    # FPS
    curr = time.time()
    fps = 1.0 / (curr - prev_time)
    prev_time = curr

    cv2.putText(frame, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

      
    # COMBINE VIEW
    depth_vis = cv2.resize(depth_vis, (W, H))

    combined = np.hstack((frame, depth_vis))

    cv2.imshow("RGB | Depth | Multi Object Distance", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()