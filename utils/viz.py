# utils/viz.py
import cv2
import numpy as np

def depth_to_colormap(depth, normalize=True, dtype=np.uint8):
    """
    Convert depth (2D float) to color image (BGR) for display.
    """
    if normalize:
        dmin, dmax = np.nanmin(depth), np.nanmax(depth)
        if np.isfinite(dmin) and np.isfinite(dmax) and (dmax - dmin) > 1e-6:
            dn = (depth - dmin) / (dmax - dmin)
        else:
            dn = np.zeros_like(depth)
        depth8 = (dn * 255.0).astype(dtype)
    else:
        depth8 = np.clip(depth, 0, 255).astype(dtype)
    cmap = cv2.applyColorMap(depth8, cv2.COLORMAP_INFERNO)
    return cmap

def draw_text(img, text, xy=(10,30), color=(255,255,255), scale=0.7, thickness=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)