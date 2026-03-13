import cv2
import torch
import numpy as np
import json
import time
from pathlib import Path

# Load calibration parameters
CALIB_FILE = Path("calibration/calib_params.json")

with open(CALIB_FILE) as f:
    calib = json.load(f)

a = calib["a"]
b = calib["b"]

print("Loaded calibration:", calib)

# Load MiDaS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading MiDaS model...")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

prev_time = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # ROI box
    box_w = int(W * 0.25)
    box_h = int(H * 0.25)

    x1 = W//2 - box_w//2
    y1 = H//2 - box_h//2
    x2 = x1 + box_w
    y2 = y1 + box_h

    # Depth prediction
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():

        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    roi = prediction[y1:y2, x1:x2]

    pred_stat = float(np.median(roi))

    # Convert to distance using calibration
    distance = a * pred_stat + b

    # Depth visualization
    depth_vis = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    # Draw ROI
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # FPS
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    # Text overlay
    cv2.putText(frame,f"PredStat: {pred_stat:.1f}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,255),2)

    cv2.putText(frame,f"Distance: {distance:.2f} m",
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,0),2)

    cv2.putText(frame,f"FPS: {fps:.1f}",
                (10,90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(255,255,0),2)

    combined = np.hstack((frame, depth_vis))

    cv2.imshow("RGB | Depth | Distance", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()