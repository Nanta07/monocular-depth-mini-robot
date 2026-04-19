# calibration/process_local_calib.py

import json
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

# --- PERBAIKAN PATH DISINI ---
# Menggunakan Pathlib agar script bisa berjalan di Windows/Linux tanpa masalah
BASE_DIR = Path(__file__).resolve().parent.parent # Root folder project
PAIRS_FILE = BASE_DIR / "calibration" / "samples" / "pairs.jsonl"
OUTPUT_FILE = BASE_DIR / "calibration" / "calib_params.json"

# 1. Load data dari pairs.jsonl
distances = []
preds = []

if not PAIRS_FILE.exists():
    print(f"❌ Error: File tidak ditemukan di {PAIRS_FILE}")
    exit()

with open(PAIRS_FILE, 'r') as f:
    for line in f:
        data = json.loads(line)
        # Filter: Fokus pada jarak yang stabil (0.4m - 2.5m)
        if 0.3 <= data['distance_m'] <= 2.5:
            distances.append(data['distance_m'])
            preds.append(data['pred_stat'])

# 2. Hitung Regresi: distance = a * (1/pred) + b
X = 1.0 / (np.array(preds).reshape(-1, 1))
y = np.array(distances)

model = LinearRegression()
model.fit(X, y)

a_val = model.coef_[0]
b_val = model.intercept_
r2 = model.score(X, y)

# 3. Simpan hasil ke calib_params.json
params = {
    "model": "local_webcam_calibration",
    "a": float(a_val),
    "b": float(b_val),
    "r2": float(r2)
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(params, f, indent=4)

print(f"✅ Kalibrasi Selesai!")
print(f"📊 Data yang diproses: {len(distances)} baris")
print(f"📊 R2 Score: {r2:.4f}")
print(f"🔢 Parameter Baru Disimpan di: {OUTPUT_FILE}")
print(f"   a: {a_val:.2f}, b: {b_val:.2f}")