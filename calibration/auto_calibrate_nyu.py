import h5py
import torch
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- CONFIG ---
DATASET_PATH = 'dataset/nyu_depth_v2/nyu_depth_v2_labeled.mat'
OUT_FILE = Path("calibration/calib_params.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 500  # Gunakan 500-800 sampel untuk hasil optimal

# --- LOAD MODELS ---
print(f"Loading MiDaS Small on {DEVICE}...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(DEVICE).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

distances = []
preds = []

print(f"Processing {NUM_SAMPLES} samples from NYU Dataset...")

try:
    with h5py.File(DATASET_PATH, 'r') as f:
        images = f['images'] # (1449, 3, 640, 480)
        depths = f['depths'] # (1449, 640, 480)
        
        # Mengambil sampel secara berurutan (0 sampai NUM_SAMPLES)
        for i in range(min(NUM_SAMPLES, 1449)):
            # 1. Ekstraksi & Transpose (NYU format is weird)
            img = images[i].transpose(2, 1, 0) # Jadi (480, 640, 3)
            true_depth = depths[i].transpose(1, 0) # Jadi (480, 640)

            # Skip jika data ground truth banyak yang kosong/nol
            if np.mean(true_depth == 0) > 0.15:
                continue

            # 2. Prediksi MiDaS
            input_batch = transform(img).to(DEVICE)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), 
                    size=img.shape[:2], 
                    mode="bicubic", 
                    align_corners=False
                ).squeeze().cpu().numpy()

            # 3. Statistik ROI (Center Area)
            h, w = prediction.shape
            roi_p = prediction[h//4:3*h//4, w//4:3*w//4]
            roi_d = true_depth[h//4:3*h//4, w//4:3*w//4]

            # Hanya ambil piksel yang memiliki nilai jarak valid (> 0)
            mask = roi_d > 0
            if np.any(mask):
                preds.append(np.median(roi_p[mask]))
                distances.append(np.median(roi_d[mask]))

            if (i+1) % 50 == 0:
                print(f"Processed {i+1} images...")

except Exception as e:
    print(f"Error reading dataset: {e}")
    sys.exit()

# --- RECIPROCAL REGRESSION ---
if len(preds) > 10:
    X_inv = 1.0 / (np.array(preds).reshape(-1, 1) + 1e-6)
    y = np.array(distances)

    model = LinearRegression()
    model.fit(X_inv, y)

    y_pred = model.predict(X_inv)
    r2 = r2_score(y, y_pred)

    params = {
        "model": "reciprocal_nyu_final",
        "a": float(model.coef_[0]),
        "b": float(model.intercept_),
        "r2": float(r2),
        "samples_used": len(preds)
    }

    with open(OUT_FILE, "w") as f:
        json.dump(params, f, indent=4)

    print(f"\n--- SUCCESS ---")
    print(f"File Saved: {OUT_FILE}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Formula: distance = {params['a']:.4f} * (1/pred) + {params['b']:.4f}")
else:
    print("Not enough samples collected for regression.")