#inference/nav_main.py
import cv2
import torch
import numpy as np
import pyttsx3
import threading
import csv
import time
from pathlib import Path
from ultralytics import YOLO

try:
    import pythoncom
except ImportError:
    pythoncom = None
    print("[WARNING] pywin32 is not installed. Audio might not work. Run: pip install pywin32")

# --- CONFIG & CALIBRATION PARAMETERS ---
A_LOCAL = 281.39
B_LOCAL = 0.16
STOP_THRESH = 0.8  # Meter
SAFE_THRESH = 1.5  # Meter
ALPHA = 0.6        # EMA Smoothing factor
TTS_COOLDOWN = 3.5 # Detik jeda suara

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VoiceAssistant:
    def __init__(self):
        self.last_say_time = 0
        self.is_speaking = False

    def _speak(self, text):
        self.is_speaking = True
        if pythoncom is not None:
            pythoncom.CoInitialize()
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[AUDIO ERROR]: {e}")
        finally:
            if pythoncom is not None:
                pythoncom.CoUninitialize()
            self.is_speaking = False

    def say(self, text):
        current_time = time.time()
        if not self.is_speaking and (current_time - self.last_say_time > TTS_COOLDOWN):
            self.last_say_time = current_time
            t = threading.Thread(target=self._speak, args=(text,), daemon=True)
            t.start()

# --- INITIALIZE MODELS ---
print(f"[INFO] Initializing Navigation System on {DEVICE}...")
yolo = YOLO("yolo26n.pt").to(DEVICE)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

prev_dists = {"LEFT": None, "CENTER": None, "RIGHT": None}

def apply_ema(current, key):
    global prev_dists
    if prev_dists[key] is None: prev_dists[key] = current
    smoothed = (ALPHA * current) + ((1 - ALPHA) * prev_dists[key])
    prev_dists[key] = smoothed
    return smoothed

log_path = Path("logs")
log_path.mkdir(exist_ok=True)
log_filename = log_path / f"session_log_{int(time.time())}.csv"

cap = cv2.VideoCapture(0)
voice = VoiceAssistant()

voice.say("System is ready")
print(f"[INFO] Saving logs to: {log_filename}")

f = open(log_filename, mode='w', newline='')
writer = csv.writer(f)
writer.writerow(["Timestamp", "FPS", "Effective_Dist", "Command", "Objects"])

try:
    while cap.isOpened():
        start_time = time.time()
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

        # 2. BACKGROUND/FLOOR ZONING (Safety Net)
        roi_top = int(H * 0.4)
        w_third = W // 3
        
        raw_l = A_LOCAL * (1.0 / (np.median(prediction[roi_top:, :w_third]) + 1e-6)) + B_LOCAL
        raw_c = A_LOCAL * (1.0 / (np.median(prediction[roi_top:, w_third:2*w_third]) + 1e-6)) + B_LOCAL
        raw_r = A_LOCAL * (1.0 / (np.median(prediction[roi_top:, 2*w_third:]) + 1e-6)) + B_LOCAL

        dist_l = apply_ema(raw_l, "LEFT")
        dist_c = apply_ema(raw_c, "CENTER")
        dist_r = apply_ema(raw_r, "RIGHT")

        # 3. OBJECT DETECTION & OBJECT DEPTH EXTRACTION
        results = yolo(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        clss = results.boxes.cls.cpu().numpy()
        
        detected_labels = []
        closest_obj_dist = 999.0
        closest_obj_name = None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            label = yolo.names[int(clss[i])]
            detected_labels.append(label)

            # Ekstrak peta kedalaman KHUSUS untuk area objek ini
            obj_depth_area = prediction[y1:y2, x1:x2]
            
            if obj_depth_area.size > 0:
                # Dapatkan median kedalaman spesifik objek
                obj_median = np.median(obj_depth_area)
                obj_dist = A_LOCAL * (1.0 / (obj_median + 1e-6)) + B_LOCAL
                
                # Visualisasi Bounding Box Objek di Layar
                color_box = (0, 0, 255) if obj_dist < STOP_THRESH else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                cv2.putText(frame, f"{label} {obj_dist:.1f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2)

                # Simpan data objek terdekat
                if obj_dist < closest_obj_dist:
                    closest_obj_dist = obj_dist
                    closest_obj_name = label

        # 4. DECISION ENGINE (Menggunakan data Objek DAN Lantai)
        # Sistem akan mengambil mana yang lebih dekat: Zona Tengah ATAU Objek Terdekat
        effective_dist = min(dist_c, closest_obj_dist)

        if effective_dist < STOP_THRESH:
            # Jika objek terdekat memicu STOP, sebutkan objeknya. Jika tidak, sebut "Obstacle"
            alert_name = closest_obj_name if (closest_obj_dist < STOP_THRESH and closest_obj_name) else "obstacle"
            command, color = f"STOP - {alert_name.upper()} DETECTED", (0, 0, 255)
            voice.say(f"Stop, {alert_name} ahead")
            
        elif effective_dist < SAFE_THRESH:
            command, color = "WARNING - SLOW DOWN", (0, 255, 255)
            voice.say("Caution, slow down")
            
        else:
            command, color = "CLEAR - MOVE FORWARD", (0, 255, 0)

        # 5. VISUALIZATION (STYLING)
        fps = 1 / (time.time() - start_time)
        
        cv2.rectangle(frame, (0, 0), (W, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"CMD: {command}", (20, 40), 0, 0.7, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (W-110, 40), 0, 0.6, (255, 255, 255), 1)
        
        cv2.line(frame, (0, roi_top), (W, roi_top), (255, 255, 255), 1)
        cv2.line(frame, (w_third, roi_top), (w_third, H), (255, 255, 255), 1)
        cv2.line(frame, (2*w_third, roi_top), (2*w_third, H), (255, 255, 255), 1)
        
        cv2.rectangle(frame, (0, H-40), (W, H), (0, 0, 0), -1)
        cv2.putText(frame, f"L: {dist_l:.1f}m", (20, H-12), 0, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"C: {dist_c:.1f}m", (w_third+20, H-12), 0, 0.6, (0,255,0) if dist_c > SAFE_THRESH else color, 2)
        cv2.putText(frame, f"R: {dist_r:.1f}m", (2*w_third+20, H-12), 0, 0.6, (255,255,255), 1)

        # 6. LOGGING
        objs_str = " | ".join(detected_labels) if detected_labels else "None"
        writer.writerow([time.time(), round(fps, 2), round(effective_dist, 2), command, objs_str])
        f.flush() 

        cv2.imshow("Nav-Assist Prototype", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    f.close()
    print("\n" + "="*40)
    print(f"[SUCCESS] Session ended. Data saved to: {log_filename}")
    print("="*40)