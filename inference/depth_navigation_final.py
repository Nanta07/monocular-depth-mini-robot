import cv2
import torch
import numpy as np
import json
import time
from pathlib import Path
from ultralytics import YOLO

# --- CONFIG & LOCAL PARAMS ---
A_LOCAL = 281.39
B_LOCAL = 0.16
STOP_THRESH = 0.8  # Berhenti jika objek < 0.8 meter
SAFE_THRESH = 1.5  # Hati-hati jika objek < 1.5 meter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOAD MODELS ---
print(f"Mengaktifkan Sistem Navigasi pada {DEVICE}...")
yolo = YOLO("yolo26n.pt").to(DEVICE)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret: break
    H, W = frame.shape[:2]

    # 1. ESTIMASI KEDALAMAN
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()

    # 2. OPTIMASI ROI (Tahap 2: Fokus 60% Bawah Layar)
    # Kita mengabaikan 40% bagian atas karena biasanya berisi langit-langit/lampu
    roi_top = int(H * 0.4)
    w_third = W // 3
    
    def calculate_dist(depth_area):
        # Menggunakan rumus Reciprocal hasil kalibrasi lokalmu
        return A_LOCAL * (1.0 / (np.median(depth_area) + 1e-6)) + B_LOCAL

    # Jarak dasar per zona (Hanya area lantai)
    dist_l = calculate_dist(prediction[roi_top:, :w_third])
    dist_c = calculate_dist(prediction[roi_top:, w_third:2*w_third])
    dist_r = calculate_dist(prediction[roi_top:, 2*w_third:])

    # 3. DETEKSI OBJEK (YOLO)
    results = yolo(frame, verbose=False)[0]
    detections = results.boxes.data.cpu().numpy()
    
    # Simpan jarak terdekat yang ditemukan (baik dari zona atau objek)
    final_dists = {"LEFT": dist_l, "CENTER": dist_c, "RIGHT": dist_r}

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cx = int((x1 + x2) / 2)
        label = results.names[int(cls)]
        
        # Ambil median depth dari kotak objek
        obj_roi = prediction[int(y1):int(y2), int(x1):int(x2)]
        if obj_roi.size > 0:
            obj_m = calculate_dist(obj_roi)
            zone = "LEFT" if cx < w_third else "CENTER" if cx < 2*w_third else "RIGHT"
            # Jika objek lebih dekat dari background lantai, gunakan jarak objek
            if obj_m < final_dists[zone]:
                final_dists[zone] = obj_m

            # Visualisasi Bounding Box
            color_box = (0, 0, 255) if obj_m < STOP_THRESH else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
            cv2.putText(frame, f"{label} {obj_m:.1f}m", (int(x1), int(y1)-10), 0, 0.5, color_box, 2)

    # 4. LOGIKA INSTRUKSI (Decision Engine)
    if final_dists["CENTER"] < STOP_THRESH:
        if final_dists["LEFT"] > final_dists["RIGHT"] and final_dists["LEFT"] > SAFE_THRESH:
            command, color = "STOP - AMBIL KIRI", (0, 0, 255)
        elif final_dists["RIGHT"] > final_dists["LEFT"] and final_dists["RIGHT"] > SAFE_THRESH:
            command, color = "STOP - AMBIL KANAN", (0, 0, 255)
        else:
            command, color = "STOP - JALAN TERBLOKIR", (0, 0, 255)
    elif final_dists["CENTER"] < SAFE_THRESH:
        command, color = "HATI-HATI - KURANGI KECEPATAN", (0, 255, 255)
    else:
        command, color = "JALAN AMAN - MAJU", (0, 255, 0)

    # 5. VISUALISASI DEMO
    fps = 1 / (time.time() - start_time)
    # Header Info
    cv2.rectangle(frame, (0, 0), (W, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"COMMAND: {command}", (20, 40), 0, 0.7, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (W-110, 40), 0, 0.6, (255, 255, 255), 1)
    
    # Boundary Line ROI (Menunjukkan area yang diproses)
    cv2.line(frame, (0, roi_top), (W, roi_top), (255, 255, 255), 1)
    cv2.line(frame, (w_third, roi_top), (w_third, H), (255, 255, 255), 1)
    cv2.line(frame, (2*w_third, roi_top), (2*w_third, H), (255, 255, 255), 1)
    
    # Footer Distances
    cv2.putText(frame, f"L:{final_dists['LEFT']:.1f}m", (10, H-20), 0, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"C:{final_dists['CENTER']:.1f}m", (w_third+10, H-20), 0, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"R:{final_dists['RIGHT']:.1f}m", (2*w_third+10, H-20), 0, 0.6, (255,255,255), 2)

    cv2.imshow("Final Demo - Assistant for Visually Impaired", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()