import numpy as np
from utils.metrics import rmse, abs_rel, sq_rel, log_rmse, delta_thresholds
import glob
import cv2
import json

# Suppose you have paired GT depth maps and predicted maps in folders:
# gt/*.png (float32 saved as 16-bit PNG scaled by 1000; or npy)
# pred/*.npy

gt_files = sorted(glob.glob("eval/gt/*.npy"))
pred_files = sorted(glob.glob("eval/pred/*.npy"))

all_metrics = []
for gfp, pfp in zip(gt_files, pred_files):
    gt = np.load(gfp)
    pred = np.load(pfp)
    mask = (gt>0)
    gt_m = gt[mask]
    pred_m = pred[mask]
    # maybe apply calibration here first if needed
    m = {
        "rmse": float(rmse(gt_m, pred_m)),
        "abs_rel": float(abs_rel(gt_m, pred_m)),
        "sq_rel": float(sq_rel(gt_m, pred_m)),
        "log_rmse": float(log_rmse(gt_m, pred_m))
    }
    m.update(delta_thresholds(gt_m, pred_m))
    all_metrics.append(m)

# print mean
import pandas as pd
df = pd.DataFrame(all_metrics)
print(df.mean())