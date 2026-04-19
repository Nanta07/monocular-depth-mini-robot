#inference/nav_main.py
import cv2
import torch
import numpy as np
import pyttsx3
import threading
import time
import csv
from pathlib import Path
from ultralytics import YOLO

try:
    import pythoncom
except ImportError:
    pythoncom = None

# --- CONFIG & CALIBRATION PARAMETERS ---
A_LOCAL = 281.39
B_LOCAL = 0.16
STOP_THRESH = 0.8  # Meter (Zona Bahaya)
SAFE_THRESH = 1.5  # Meter (Zona Waspada)
ALPHA = 0.6        # EMA Smoothing
TTS_COOLDOWN = 3.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VoiceAssistant:
    def __init__(self):
        self.last_say_time = 0
        self.is_speaking = False

    def _speak(self, text):
        self.is_speaking = True
        if pythoncom is not None: pythoncom.CoInitialize()
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
        except: pass
        finally:
            if pythoncom is not None: pythoncom.CoUninitialize()
            self.is_speaking = False

    def say(self, text):
        current_time = time.time()
        if not self.is_speaking and (current_time - self.last_say_time > TTS_COOLDOWN):
            self.last_say_time = current_time
            t = threading.Thread(target=self._speak, args=(text,), daemon=True)
            t.start()

print(f"[INFO] Initializing on {DEVICE}...")
yolo = YOLO("yolo26n.pt").to(DEVICE)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

prev_dists = {"L": None, "C": None, "R": None}
def apply_ema(current, key):
    global prev_dists
    if prev_dists[key] is None: prev_dists[key] = current
    smoothed = (ALPHA * current) + ((1 - ALPHA) * prev_dists[key])
    prev_dists[key] = smoothed
    return smoothed

log_path = Path("logs")
log_path.mkdir(exist_ok=True)
log_filename = log_path / f"session_log_{int(time.time())}.csv"
f = open(log_filename, mode='w', newline='')
writer = csv.writer(f)
writer.writerow(["Timestamp", "FPS", "Dist_L", "Dist_C", "Dist_R", "Command", "Objects"])

cap = cv2.VideoCapture(0)
voice = VoiceAssistant()
voice.say("Sistem siap, mode portrait aktif")

try:
    while cap.isOpened():
        start_time = time.time()
        ret, frame_raw = cap.read()
        if not ret: break
        
        # --- SIMULASI PORTRAIT (Crop tengah kamera laptop menjadi vertikal) ---
        hr, wr = frame_raw.shape[:2]
        target_w = int(hr * 0.75) # Rasio 3:4 portrait
        start_x = (wr - target_w) // 2
        frame = frame_raw[:, start_x:start_x+target_w]
        H, W = frame.shape[:2]

        # 1. DEPTH ESTIMATION
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(DEVICE)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

        # 2. HORIZON BAND ZONING (SAFETY NET)
        # Mengambil area pandang lurus (30% ke 75% tinggi layar), abaikan langit dan lantai langsung di bawah kaki
        nav_y1, nav_y2 = int(H * 0.3), int(H * 0.75)
        w_third = W // 3
        
        def calc_dist(area):
            if area.size == 0: return 9.9
            return A_LOCAL * (1.0 / (np.median(area) + 1e-6)) + B_LOCAL

        raw_l = calc_dist(prediction[nav_y1:nav_y2, :w_third])
        raw_c = calc_dist(prediction[nav_y1:nav_y2, w_third:2*w_third])
        raw_r = calc_dist(prediction[nav_y1:nav_y2, 2*w_third:])

        dist_l = apply_ema(raw_l, "L")
        dist_c = apply_ema(raw_c, "C")
        dist_r = apply_ema(raw_r, "R")

        # 3. OBJECT DETECTION (YOLO) INDEPENDENT
        results = yolo(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        
        detected_labels = []
        danger_yolo_objs = [] # Objek yang benar-benar ada di depan & dekat

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = yolo.names[int(clss[i])]
            detected_labels.append(label)

            # Ekstrak jarak spesifik hanya 50% area bawah kotak objek
            crop_y1 = y1 + int((y2 - y1) * 0.5)
            obj_depth_area = prediction[crop_y1:y2, x1:x2]
            obj_dist = calc_dist(obj_depth_area)
            
            # Cek apakah objek ini berada di zona bahaya DAN di tengah jalan
            cx = (x1 + x2) // 2
            is_center = (w_third < cx < 2*w_third)
            
            if obj_dist < STOP_THRESH and is_center:
                danger_yolo_objs.append(label)

            # Gambar Bounding Box
            color_box = (0, 0, 255) if obj_dist < STOP_THRESH else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
            cv2.putText(frame, f"{label} {obj_dist:.1f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

        # 4. DECISION ENGINE (SMART LOGIC)
        # Prioritas 1: Ada objek YOLO berbahaya di tengah
        if danger_yolo_objs:
            obj_name = danger_yolo_objs[0]
            command = f"STOP - ADA {obj_name.upper()}"
            color = (0, 0, 255)
            voice.say(f"Berhenti, ada {obj_name}")
            
        # Prioritas 2: Tidak ada objek YOLO, tapi horizon tengah tertutup (Tembok/Kaca)
        elif dist_c < STOP_THRESH:
            # Cek opsi menghindar
            if dist_l > dist_r and dist_l > SAFE_THRESH:
                command, color = "STOP - HINDARI KE KIRI", (0, 165, 255) # Orange
                voice.say("Rintangan di depan, hindari ke kiri")
            elif dist_r > dist_l and dist_r > SAFE_THRESH:
                command, color = "STOP - HINDARI KE KANAN", (0, 165, 255)
                voice.say("Rintangan di depan, hindari ke kanan")
            else:
                command, color = "STOP - JALAN BUNTU", (0, 0, 255)
                voice.say("Jalan buntu, berhenti")
                
        # Prioritas 3: Zona waspada
        elif dist_c < SAFE_THRESH:
            command, color = "HATI-HATI", (0, 255, 255) # Kuning
            
        # Prioritas 4: Aman
        else:
            command, color = "JALAN AMAN", (0, 255, 0)

        # 5. UI SENSOR PARKIR & VISUALISASI
        # Menggambar area Horizon Band yang diproses
        cv2.rectangle(frame, (0, nav_y1), (W, nav_y2), (255, 255, 255), 1)
        cv2.putText(frame, "Horizon Sensor Area", (10, nav_y1+20), 0, 0.5, (255,255,255), 1)
        
        # UI Garis Parkir (Di bagian bawah layar)
        park_y_base = H - 80
        # Warna garis tergantung jarak tengah (Merah, Kuning, Hijau)
        park_color = (0, 0, 255) if dist_c < STOP_THRESH else (0, 255, 255) if dist_c < SAFE_THRESH else (0, 255, 0)
        
        # Gambar garis trapesium ala sensor parkir
        pt1, pt2 = (int(W*0.2), park_y_base), (int(W*0.8), park_y_base)
        pt3, pt4 = (int(W*0.1), park_y_base+60), (int(W*0.9), park_y_base+60)
        cv2.line(frame, pt1, pt3, park_color, 3)
        cv2.line(frame, pt2, pt4, park_color, 3)
        for i in range(1, 4):
            y_line = park_y_base + (i * 20)
            x_offset = int(W*0.1) - int((W*0.1) * (i/3))
            cv2.line(frame, (pt1[0]-x_offset, y_line), (pt2[0]+x_offset, y_line), park_color, max(1, 4-i))

        # Header
        fps = 1 / (time.time() - start_time)
        cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"CMD: {command}", (10, 25), 0, 0.5, color, 2)
        
        # Footer
        cv2.rectangle(frame, (0, H-30), (W, H), (0, 0, 0), -1)
        cv2.putText(frame, f"L:{dist_l:.1f}m", (10, H-10), 0, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"C:{dist_c:.1f}m", (W//2 - 30, H-10), 0, 0.5, park_color, 2)
        cv2.putText(frame, f"R:{dist_r:.1f}m", (W - 80, H-10), 0, 0.5, (255,255,255), 1)

        # 6. LOGGING
        objs_str = " | ".join(detected_labels) if detected_labels else "None"
        writer.writerow([time.time(), round(fps, 2), round(dist_l, 2), round(dist_c, 2), round(dist_r, 2), command, objs_str])
        f.flush()

        cv2.imshow("Mobile Portrait Navigation Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    f.close()