import cv2
import torch
import numpy as np
import json
import time
from pathlib import Path

 
# CONFIGURATION
CALIB_FILE = Path("calibration/calib_params.json")
OBSTACLE_THRESHOLD = 0.7  # meters


 
# LOAD CALIBRATION PARAMETERS
with open(CALIB_FILE) as f:
    calib = json.load(f)

a = calib["a"]
b = calib["b"]

print("Loaded calibration parameters:")
print(calib)

 
# LOAD MIDAS MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading MiDaS model...")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform


 
# OPEN WEBCAM
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press Q to exit")

prev_time = time.time()

# MAIN LOOP
while True:

    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # DEFINE ROI (CENTER BOX)
    box_w = int(W * 0.25)
    box_h = int(H * 0.25)

    x1 = W//2 - box_w//2
    y1 = H//2 - box_h//2
    x2 = x1 + box_w
    y2 = y1 + box_h

    # DEPTH ESTIMATION
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

    # COMPUTE ROI DEPTH
    roi = prediction[y1:y2, x1:x2]

    pred_stat = float(np.median(roi))

    # Convert depth → distance
    distance = a * pred_stat + b

    # OBSTACLE DETECTION
    obstacle = distance < OBSTACLE_THRESHOLD

    # DEPTH VISUALIZATION
    depth_vis = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    # DRAW ROI
    if obstacle:
        color = (0,0,255)  # red
    else:
        color = (0,255,0)  # green

    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
     
    # FPS CALCULATION
    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    # TEXT OVERLAY
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

    # OBSTACLE WARNING
    if obstacle:

        cv2.putText(frame,
                    "WARNING: OBSTACLE",
                    (10,130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,0,255),
                    3)
     
    # COMBINE RGB + DEPTH
    combined = np.hstack((frame, depth_vis))

    cv2.imshow("RGB | Depth | Obstacle Detection", combined)


     
    # EXIT KEY
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()