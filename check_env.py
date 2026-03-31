import torch
from ultralytics import YOLO

# Cek CUDA (jika ada GPU di Lenovo LOQ kamu)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}")

try:
    # Inisialisasi YOLO26n (akan otomatis download jika belum ada)
    model = YOLO("yolo26n.pt") 
    print("YOLO26n successfully initialized!") 
    
    # Inisialisasi MiDaS Small
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device).eval()
    print("MiDaS Small successfully initialized!") 
except Exception as e:
    print(f"Error during initialization: {e}")