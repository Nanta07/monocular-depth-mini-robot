# utils/metrics.py
import numpy as np

def rmse(gt, pred):
    return np.sqrt(np.mean((gt - pred) ** 2))

def abs_rel(gt, pred):
    return np.mean(np.abs(gt - pred) / gt)

def sq_rel(gt, pred):
    return np.mean(((gt - pred) ** 2) / gt)

def log_rmse(gt, pred):
    return np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

def delta_thresholds(gt, pred, threshs=(1.25, 1.25**2, 1.25**3)):
    ratios = np.maximum(gt / pred, pred / gt)
    r = {}
    for t in threshs:
        r[f"delta_{t:.3f}"] = np.mean(ratios < t)
    return r