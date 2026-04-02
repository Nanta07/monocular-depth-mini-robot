import sys
import os
import cv2
import torch
import numpy as np
import json
import time
from pathlib import Path
from ultralytics import YOLO

# --- CONFIG & CALIBRATION ---
CALIB_FILE = Path("calibration/calib_params.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_THRESH = 0.8  
SAFE_THRESH = 1.5  
ALPHA = 0.2  # Faktor kehalusan (0.1 = sangat halus, 0.9 = responsif tapi bergetar)

def load_calibration():
    if not CALIB_FILE.exists():
        return 300.0, 1.0 
    with open(CALIB_FILE, "r") as f:
        data = json.load(f)
    return float(data["a"]), float(data["b"])

a, b = load_calibration()

# --- INITIALIZE SMOOTHING ---
# Menyimpan memori jarak agar tidak melompat
smooth_dists = {"LEFT": 2.0, "CENTER": 2.0, "RIGHT": 2.0}

def evaluate_navigation(zones):
    """Logika navigasi yang lebih cerdas"""
    c_dist = zones["CENTER"]
    l_dist = zones["LEFT"]
    r_dist = zones["RIGHT"]

    if c_dist < STOP_THRESH:
        if l_dist > r_dist and l_dist > SAFE_THRESH:
            return "STOP - GESER KIRI", (0, 0, 255)
        elif r_dist > l_dist and r_dist > SAFE_THRESH:
            return "STOP - GESER KANAN", (0, 0, 255)
        else:
            return "STOP - JALAN TERBLOKIR", (0, 0, 255)
    elif c_dist < SAFE_THRESH:
        return "HATI-HATI - KURANGI KECEPATAN", (0, 255, 255)
    else:
        return "JALAN AMAN - MAJU", (0, 255, 0)

# --- LOAD MODELS ---
yolo = YOLO("yolo26n.pt").to(DEVICE)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

cap = cv2.VideoCapture(0)
cv2.namedWindow("Final Demo", cv2.WINDOW_NORMAL)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret: break
    H, W = frame.shape[:2]

    # 1. Depth Estimation
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # 2. Raw Zone Calculation
    w_third = W // 3
    raw_dists = {
        "LEFT": a * (1.0 / (np.median(prediction[:, :w_third]) + 1e-6)) + b,
        "CENTER": a * (1.0 / (np.median(prediction[:, w_third:2*w_third]) + 1e-6)) + b,
        "RIGHT": a * (1.0 / (np.median(prediction[:, 2*w_third:]) + 1e-6)) + b
    }

    # 3. YOLO Object Validation
    results = yolo(frame, verbose=False)[0]
    detections = results.boxes.data.cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cx = int((x1 + x2) / 2)
        # Ambil area tengah objek saja (70% center) agar lebih akurat
        bw, bh = x2 - x1, y2 - y1
        roi = prediction[int(y1+bh*0.15):int(y2-bh*0.15), int(x1+bw*0.15):int(x2-bw*0.15)]
        
        if roi.size > 0:
            obj_dist = a * (1.0 / (np.median(roi) + 1e-6)) + b
            zone = "LEFT" if cx < w_third else "CENTER" if cx < 2*w_third else "RIGHT"
            if obj_dist < raw_dists[zone]:
                raw_dists[zone] = obj_dist
            
            # Draw YOLO info
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{results.names[int(cls)]} {obj_dist:.1f}m", (int(x1), int(y1)-10), 0, 0.5, (0, 255, 0), 2)

    # 4. TEMPORAL SMOOTHING (Kunci agar demo bagus)
    for z in smooth_dists:
        smooth_dists[z] = (ALPHA * raw_dists[z]) + ((1 - ALPHA) * smooth_dists[z])

    # 5. UI & Visualization
    command, color = evaluate_navigation(smooth_dists)
    fps = 1 / (time.time() - start_time)

    # Overlay
    cv2.line(frame, (w_third, 0), (w_third, H), (255, 255, 255), 1)
    cv2.line(frame, (2*w_third, 0), (2*w_third, H), (255, 255, 255), 1)
    
    # Perintah Atas
    cv2.rectangle(frame, (0, 0), (W, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"COMMAND: {command}", (20, 45), 0, 0.9, color, 3)
    cv2.putText(frame, f"FPS: {fps:.1f}", (W-110, 45), 0, 0.6, (0, 255, 255), 2)

    # Status Bar Bawah
    cv2.rectangle(frame, (0, H-40), (W, H), (30, 30, 30), -1)
    cv2.putText(frame, f"L: {smooth_dists['LEFT']:.2f}m", (20, H-12), 0, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"C: {smooth_dists['CENTER']:.2f}m", (w_third+20, H-12), 0, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"R: {smooth_dists['RIGHT']:.2f}m", (2*w_third+20, H-12), 0, 0.6, (255, 255, 255), 2)

    cv2.imshow("Final Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()