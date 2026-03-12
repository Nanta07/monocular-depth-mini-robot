# inference/depth_webcam_demo.py
import cv2
import torch
import numpy as np
import json
import time
import os
from utils.viz import depth_to_colormap, draw_text

from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load MiDaS small
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Load calibration params if exist
calib = None
if os.path.exists("calib_params.json"):
    with open("calib_params.json","r") as f:
        calib = json.load(f)
        print("Loaded calib:", calib)
else:
    print("No calib_params.json found; running in relative-depth mode.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

fps_smooth = None
frame_count = 0
t0 = time.perf_counter()

SAVE_PRED_DIR = None  # placeholder; set if needed to save predictions

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H,W = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()
        end = time.perf_counter()

        # compute center ROI median
        h,w = prediction.shape
        y1,y2 = h//3, 2*h//3
        x1,x2 = w//3, 2*w//3
        roi = prediction[y1:y2, x1:x2]
        p_center = float(np.median(roi))

        # apply calibration
        est_dist = None
        warn = False
        if calib:
            sel = calib["selected"]
            if sel == "linear":
                a = calib["models"]["linear"]["a"]
                b = calib["models"]["linear"]["b"]
                est_dist = a * p_center + b
            else:
                a = calib["models"]["reciprocal"]["a"]
                b = calib["models"]["reciprocal"]["b"]
                est_dist = a / (p_center + b)
            if est_dist < 0:
                est_dist = None
            else:
                warn = est_dist < 0.6  # threshold for obstacle
        else:
            # heuristic warn by high relative depth value (tuned visually)
            warn = p_center > np.percentile(prediction, 95)

        # FPS calc
        frame_count += 1
        frametime = end - start
        fps = 1.0 / frametime if frametime > 0 else 0.0
        fps_smooth = fps if fps_smooth is None else 0.8*fps_smooth + 0.2*fps

        # visualization
        depth_col = depth_to_colormap(prediction)
        left = cv2.resize(frame, (W, W))
        right = cv2.resize(depth_col, (W, W))
        combined = np.hstack((left, right))

        label = f"PredStat: {p_center:.4f}"
        if est_dist is not None:
            label += f"  |  EstDist: {est_dist:.2f} m"
        label += f"  |  FPS: {fps_smooth:.1f}"

        draw_text(combined, label, xy=(10,30))
        if warn:
            draw_text(combined, "OBSTACLE!", xy=(10,70), color=(0,0,255), scale=1.0, thickness=3)

        cv2.rectangle(combined, (W//3, W//3),(2*W//3, 2*W//3), (0,255,0), 2)
        cv2.imshow("RGB | Depth (MiDaS_small)", combined)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()