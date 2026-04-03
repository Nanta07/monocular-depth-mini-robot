import cv2
import torch
import numpy as np
import json
import time
import csv
import os
from pathlib import Path
from ultralytics import YOLO

# --- CONFIG & LOCAL PARAMS ---
A_LOCAL = 281.39
B_LOCAL = 0.16
STOP_THRESH = 0.8  
SAFE_THRESH = 1.5  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- INITIALIZE LOGGING ---
# Membuat folder logs jika belum ada
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
session_id = time.strftime("%Y%m%d_%H%M%S")
csv_file = log_dir / f"session_log_{session_id}.csv"

# Inisialisasi data untuk statistik akhir
history_data = []
start_session_time = time.time()

# Menulis header CSV
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "fps", "command", "dist_l", "dist_c", "dist_r", "object_detected"])

# --- LOAD MODELS ---
print(f"Mengaktifkan Sistem Navigasi pada {DEVICE}...")
yolo = YOLO("yolo26n.pt").to(DEVICE)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

cap = cv2.VideoCapture(0)

print(f"⏺️ Sesi dimulai. Data direkam ke: {csv_file}")

try:
    while True:
        frame_start_time = time.time()
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
        roi_top = int(H * 0.4)
        w_third = W // 3
        
        def calculate_dist(depth_area):
            return A_LOCAL * (1.0 / (np.median(depth_area) + 1e-6)) + B_LOCAL

        dist_l = calculate_dist(prediction[roi_top:, :w_third])
        dist_c = calculate_dist(prediction[roi_top:, w_third:2*w_third])
        dist_r = calculate_dist(prediction[roi_top:, 2*w_third:])

        # 3. DETEKSI OBJEK (YOLO)
        results = yolo(frame, verbose=False)[0]
        detections = results.boxes.data.cpu().numpy()
        
        final_dists = {"LEFT": dist_l, "CENTER": dist_c, "RIGHT": dist_r}
        objects_in_frame = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cx = int((x1 + x2) / 2)
            label = results.names[int(cls)]
            objects_in_frame.append(label)
            
            obj_roi = prediction[int(y1):int(y2), int(x1):int(x2)]
            if obj_roi.size > 0:
                obj_m = calculate_dist(obj_roi)
                zone = "LEFT" if cx < w_third else "CENTER" if cx < 2*w_third else "RIGHT"
                if obj_m < final_dists[zone]:
                    final_dists[zone] = obj_m

                color_box = (0, 0, 255) if obj_m < STOP_THRESH else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
                cv2.putText(frame, f"{label} {obj_m:.1f}m", (int(x1), int(y1)-10), 0, 0.5, color_box, 2)

        # 4. LOGIKA INSTRUKSI
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

        # 5. VISUALISASI & RECORDING DATA
        fps = 1 / (time.time() - frame_start_time)
        
        # Simpan ke memori untuk laporan akhir
        current_log = [
            time.strftime("%H:%M:%S"), 
            round(fps, 2), 
            command, 
            round(final_dists["LEFT"], 2), 
            round(final_dists["CENTER"], 2), 
            round(final_dists["RIGHT"], 2),
            ", ".join(list(set(objects_in_frame))) if objects_in_frame else "None"
        ]
        history_data.append(current_log)

        # Append ke file CSV secara real-time (agar aman jika crash)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(current_log)

        # UI Visual
        cv2.rectangle(frame, (0, 0), (W, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"COMMAND: {command}", (20, 40), 0, 0.7, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (W-110, 40), 0, 0.6, (255, 255, 255), 1)
        
        cv2.line(frame, (0, roi_top), (W, roi_top), (255, 255, 255), 1)
        cv2.line(frame, (w_third, roi_top), (w_third, H), (255, 255, 255), 1)
        cv2.line(frame, (2*w_third, roi_top), (2*w_third, H), (255, 255, 255), 1)
        
        cv2.putText(frame, f"L:{final_dists['LEFT']:.1f}m", (10, H-20), 0, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"C:{final_dists['CENTER']:.1f}m", (w_third+10, H-20), 0, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"R:{final_dists['RIGHT']:.1f}m", (2*w_third+10, H-20), 0, 0.6, (255,255,255), 2)

        cv2.imshow("Final Demo - Analytics Enabled", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    # 6. GENERATE ANALYTICS SUMMARY (Laporan Akhir untuk Dosen)
    cap.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_session_time
    if len(history_data) > 0:
        avg_fps = np.mean([d[1] for d in history_data])
        stop_count = len([d for d in history_data if "STOP" in d[2]])
        total_frames = len(history_data)
        
        print("\n" + "="*40)
        print("📊 LAPORAN EVALUASI PERFORMA SISTEM")
        print("="*40)
        print(f"Sesi ID          : {session_id}")
        print(f"Total Durasi     : {total_time:.2f} detik")
        print(f"Total Frame      : {total_frames}")
        print(f"Rata-rata FPS    : {avg_fps:.2f}")
        print(f"Respon STOP      : {stop_count} kali ({(stop_count/total_frames)*100:.1f}%)")
        print(f"Log CSV tersimpan: {csv_file}")
        print("="*40)