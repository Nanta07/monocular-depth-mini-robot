import sys
import os

# Root direktori ke path 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import numpy as np
import json
import time
from pathlib import Path
from ultralytics import YOLO
from utils.viz import depth_to_colormap

# 1. KONFIGURASI & LOAD KALIBRASI [cite: 25, 53]
CALIB_FILE = Path("calibration/calib_params.json")
STOP_DIST = 0.8  # Meter (Threshold Berhenti)
GO_DIST = 1.5    # Meter (Threshold Aman)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_params():
    with open(CALIB_FILE, "r") as f:
        data = json.load(f)
    return data["a"], data["b"]

a, b = load_params()

# 2. LOAD MODELS (YOLO26n & MiDaS Small) 
print("Initializing Models...")
yolo = YOLO("yolo26n.pt").to(DEVICE)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    H, W = frame.shape[:2]

    # 3. DEPTH ESTIMATION [cite: 55, 56]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # 4. OBJECT DETECTION (NMS-Free YOLO26) [cite: 72]
    results = yolo(frame, verbose=False)[0]
    detections = results.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]

    # 5. ZONASI & LOGIKA JARAK [cite: 111, 112]
    w_third = W // 3
    zone_distances = {"LEFT": 10.0, "CENTER": 10.0, "RIGHT": 10.0}

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cx = int((x1 + x2) / 2)
        
        # Ambil Median Depth dari ROI Bounding Box [cite: 59, 60]
        roi = prediction[int(y1):int(y2), int(x1):int(x2)]
        if roi.size > 0:
            dist = a * np.median(roi) + b
            
            # Tentukan Zona Objek [cite: 84]
            if cx < w_third: zone = "LEFT"
            elif cx < 2 * w_third: zone = "CENTER"
            else: zone = "RIGHT"
            
            # Simpan jarak terdekat per zona
            if dist < zone_distances[zone]:
                zone_distances[zone] = dist

    # 6. DECISION ENGINE (NAVIGASI) 
    command = "GO STRAIGHT"
    color = (0, 255, 0)

    if zone_distances["CENTER"] < STOP_DIST:
        if zone_distances["LEFT"] > zone_distances["RIGHT"]:
            command = "STOP - TURN LEFT"
        else:
            command = "STOP - TURN RIGHT"
        color = (0, 0, 255)
    elif zone_distances["CENTER"] < GO_DIST:
        command = "CAUTION - SLOW DOWN"
        color = (0, 255, 255)

    # Logika evaluasi arah (Simplified)
    def evaluate_navigation(zones):
        # zones = {"LEFT": dist, "CENTER": dist, "RIGHT": dist}
        
        if zones["CENTER"] < 0.8: # Threshold STOP
            if zones["LEFT"] > zones["RIGHT"] and zones["LEFT"] > 1.5:
                return "STOP - AMBIL KIRI"
            elif zones["RIGHT"] > zones["LEFT"] and zones["RIGHT"] > 1.5:
                return "STOP - AMBIL KANAN"
            else:
                return "STOP - JALAN TERBLOCK"
        elif zones["CENTER"] < 1.5:
            return "HATI-HATI - KURANGI KECEPATAN"
        else:
            return "JALAN AMAN"

    # 7. VISUALISASI [cite: 63, 67]
    # Garis Zona
    cv2.line(frame, (w_third, 0), (w_third, H), (255, 255, 255), 1)
    cv2.line(frame, (2 * w_third, 0), (2 * w_third, H), (255, 255, 255), 1)
    
    # Label Jarak
    cv2.putText(frame, f"L: {zone_distances['LEFT']:.1f}m", (10, H-20), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"C: {zone_distances['CENTER']:.1f}m", (w_third+10, H-20), 0, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"R: {zone_distances['RIGHT']:.1f}m", (2*w_third+10, H-20), 0, 0.7, (255,255,255), 2)
    
    # Command Text
    cv2.rectangle(frame, (0, 0), (W, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"COMMAND: {command}", (20, 40), 0, 1.0, color, 3)

    cv2.imshow("Navigation Demo (YOLO26 + MiDaS)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()