import sys
import os
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO

# Fix Path agar bisa memanggil folder 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.viz import depth_to_colormap

# --- CONFIG & CALIBRATION ---
CALIB_FILE = Path("calibration/calib_params.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_THRESH = 0.8  # Meter
SAFE_THRESH = 1.5  # Meter

def load_calibration():
    if not CALIB_FILE.exists():
        raise FileNotFoundError("File kalibrasi tidak ditemukan! Jalankan auto_calibrate_nyu.py dulu.")
    with open(CALIB_FILE, "r") as f:
        data = json.load(f)
    print(f"Loaded Calibration: {data['model']} (R2: {data['r2']:.4f})")
    return float(data["a"]), float(data["b"])

a, b = load_calibration()

# --- NAVIGATION DECISION ENGINE ---
def evaluate_navigation(zones):
    """Logika navigasi berdasarkan jarak di 3 zona vertikal"""
    if zones["CENTER"] < STOP_THRESH:
        # Cari jalan keluar ke samping jika depan terblokir
        if zones["LEFT"] > zones["RIGHT"] and zones["LEFT"] > SAFE_THRESH:
            return "STOP - AMBIL KIRI", (0, 0, 255)
        elif zones["RIGHT"] > zones["LEFT"] and zones["RIGHT"] > SAFE_THRESH:
            return "STOP - AMBIL KANAN", (0, 0, 255)
        else:
            return "STOP - JALAN TERBLOKIR", (0, 0, 255)
    elif zones["CENTER"] < SAFE_THRESH:
        return "HATI-HATI - KURANGI KECEPATAN", (0, 255, 255)
    else:
        return "JALAN AMAN - MAJU", (0, 255, 0)

# --- INITIALIZE MODELS ---
print(f"Initializing YOLO26n & MiDaS on {DEVICE}...")
yolo = YOLO("yolo26n.pt").to(DEVICE)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    H, W = frame.shape[:2]

    # 1. DEPTH ESTIMATION
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # 2. OBJECT DETECTION (YOLO26 NMS-Free)
    results = yolo(frame, verbose=False)[0]
    detections = results.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]

    # 3. ZONASI 3 BAGIAN (Menggunakan Formula Reciprocal)
    w_third = W // 3
    # Inisialisasi jarak berdasarkan median area (Background Safety)
    zone_dists = {
        "LEFT": a * (1.0 / (np.median(prediction[:, :w_third]) + 1e-6)) + b,
        "CENTER": a * (1.0 / (np.median(prediction[:, w_third:2*w_third]) + 1e-6)) + b,
        "RIGHT": a * (1.0 / (np.median(prediction[:, 2*w_third:]) + 1e-6)) + b
    }

    # 4. OVERWRITE DENGAN JARAK OBJEK TERDEKAT
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = results.names[int(cls)]
        cx = int((x1 + x2) / 2)
        
        # Ambil median depth dari bounding box objek
        roi = prediction[int(y1):int(y2), int(x1):int(x2)]
        if roi.size > 0:
            # Hitung jarak objek dengan rumus reciprocal
            obj_dist = a * (1.0 / (np.median(roi) + 1e-6)) + b
            
            # Tentukan zona berdasarkan titik tengah objek
            zone = "LEFT" if cx < w_third else "CENTER" if cx < 2*w_third else "RIGHT"
            
            # Jika objek lebih dekat dari background, gunakan jarak objek
            if obj_dist < zone_dists[zone]:
                zone_dists[zone] = obj_dist

            # Visualisasi Bounding Box Objek
            color_box = (0, 0, 255) if obj_dist < STOP_THRESH else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
            cv2.putText(frame, f"{label} {obj_dist:.1f}m", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

    # 5. NAVIGATION DECISION
    command, color = evaluate_navigation(zone_dists)

    # 6. VISUALISASI DEMO
    # Garis pembatas zona
    cv2.line(frame, (w_third, 0), (w_third, H), (255, 255, 255), 1)
    cv2.line(frame, (2*w_third, 0), (2*w_third, H), (255, 255, 255), 1)
    
    # Overlay Perintah (Latar Hitam)
    cv2.rectangle(frame, (0, 0), (W, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"COMMAND: {command}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Label Jarak per Zona di bagian bawah
    cv2.putText(frame, f"L:{zone_dists['LEFT']:.1f}m", (10, H-20), 0, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"C:{zone_dists['CENTER']:.1f}m", (w_third+10, H-20), 0, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"R:{zone_dists['RIGHT']:.1f}m", (2*w_third+10, H-20), 0, 0.6, (255,255,255), 2)

    cv2.imshow("Final Prototype Navigation (YOLO26 + MiDaS)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()