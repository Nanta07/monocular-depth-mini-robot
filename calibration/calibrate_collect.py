# calibration/calibrate_collect.py
# Run this to collect calibration samples: predicted midas stat + manual distance (meters)
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import os

OUT_DIR = Path("calibration/samples")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_FILE = OUT_DIR / "pairs.jsonl"

# load midas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Instructions:\n - Place known object inside green box\n - Press 'c' to capture sample and input distance (meters)\n - Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    # Define ROI rectangle (center)
    box_w, box_h = int(W * 0.25), int(H * 0.25)
    x1 = W//2 - box_w//2
    y1 = H//2 - box_h//2
    x2 = x1 + box_w
    y2 = y1 + box_h

    vis = frame.copy()
    cv2.rectangle(vis, (x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(vis, "Place object in green box. Press 'c' to capture.", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

    cv2.imshow("Calib Capture", vis)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        d_str = input("Enter known distance (meters), e.g. 0.5: ").strip()
        try:
            d = float(d_str)
        except:
            print("Invalid input. Skip.")
            continue

        # Prepare image for midas
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
        p_med = float(np.median(roi))
        timestamp = datetime.utcnow().isoformat()
        rec = {"timestamp": timestamp, "distance_m": d, "pred_stat": p_med}
        with open(PAIRS_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        # also save a small image and predicted map for traceability
        idx = int(sum(1 for _ in open(PAIRS_FILE,'r')))
        cv2.imwrite(str(OUT_DIR / f"img_{idx:03d}.png"), frame)
        np.save(OUT_DIR / f"pred_{idx:03d}.npy", prediction.astype(np.float32))
        print("Saved sample:", rec)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()