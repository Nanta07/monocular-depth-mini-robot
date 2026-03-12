# calibration/calibrate_fit.py
# Fit calibration: try linear d = a * p + b and reciprocal d = a / (p + b)
# Save best-fit params to calib_params.json
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import math

PAIRS_FILE = Path("calibration/samples/pairs.jsonl")
OUT_FILE = Path("calib_params.json")
if not PAIRS_FILE.exists():
    raise RuntimeError("No calibration samples found. Run calibrate_collect.py first.")

X = []
y = []
for line in PAIRS_FILE.read_text().strip().splitlines():
    rec = json.loads(line)
    X.append([rec["pred_stat"]])
    y.append(rec["distance_m"])
X = np.array(X)
y = np.array(y)

# Linear fit
lr = LinearRegression()
lr.fit(X, y)
pred_lin = lr.predict(X)
r2_lin = r2_score(y, pred_lin)

# Reciprocal fit: d = a / (p + b)
# Fit by non-linear least squares via simple transform (use np.linalg.lstsq on 1/d)
# We'll solve for a and b minimizing || d - a/(p+b) ||^2 using simple optimization
from scipy.optimize import least_squares

def residuals_recip(params, p, d):
    a, b = params
    pred = a / (p + b)
    return pred - d

p = X.flatten()
init = [1.0, 0.0]
res = least_squares(residuals_recip, init, args=(p, y))
a_rec, b_rec = res.x
pred_rec = a_rec / (p + b_rec)
r2_rec = r2_score(y, pred_rec)

# Choose best by R2
best = {
    "linear": {"a": float(lr.coef_[0]), "b": float(lr.intercept_), "r2": float(r2_lin)},
    "reciprocal": {"a": float(a_rec), "b": float(b_rec), "r2": float(r2_rec)}
}
best_model = "linear" if r2_lin >= r2_rec else "reciprocal"
print("Linear R2:", r2_lin, "Reciprocal R2:", r2_rec, "Selected:", best_model)

calib = {"selected": best_model, "models": best}
OUT_FILE.write_text(json.dumps(calib, indent=2))
print("Saved calib params to", OUT_FILE)