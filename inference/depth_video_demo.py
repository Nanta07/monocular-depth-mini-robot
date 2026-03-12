# inference/depth_video_demo.py
import cv2, torch, numpy as np, time
from utils.viz import depth_to_colormap, draw_text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cap = cv2.VideoCapture("test_navigation.mp4")  # ganti file video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    t1 = time.perf_counter()
    fps = 1.0/(t1-t0)
    depth_col = depth_to_colormap(prediction)
    combined = np.hstack((cv2.resize(frame,(frame.shape[1], frame.shape[0])),
                          cv2.resize(depth_col,(frame.shape[1], frame.shape[0]))))
    draw_text(combined, f"FPS:{fps:.1f}", (10,30))
    cv2.imshow("Video Depth Demo", combined)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()