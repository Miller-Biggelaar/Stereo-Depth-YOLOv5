# MAINFILE_detect_and_depth_timed.py
import cv2
import numpy as np
import pandas as pd
import time
import math
import pickle
from pathlib import Path
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

# --- User settings ---
IMAGE_DIR = Path("realworld_images/test/images")
CALIB_PKL = Path("stereo_calibration.pkl")
OUT_DIR = Path("realworld_images/test/depthmaps")
OUT_DIR_CSV = Path("CSVrepo")
MODEL_PATH = Path("yolov5/runs/train/synth_plus_real_ft13/weights/best.pt")

MAX_VIZ_DEPTH_M = 6.0
CONF_THRESH = 0.5  # minimum YOLO confidence
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load calibration ---
with open(CALIB_PKL, "rb") as f:
    calib = pickle.load(f)

# --- Initialize YOLOv5 model ---
device = select_device("cpu")  # Raspberry Pi CPU only
model = DetectMultiBackend(MODEL_PATH, device=device)
stride, names, pt = model.stride, model.names, model.pt

# --- Stereo parameters ---
baseline = 0.06  # meters
focal_length_mm = 4.74  # mm
pixel_size_mm = 0.0014  # mm/px
focal_length_px = focal_length_mm / pixel_size_mm

# --- Data collectors ---
summary_rows = []
timing_rows = []

# --- Match stereo pairs ---
left_images = sorted(IMAGE_DIR.glob("left_*.jpg"))
for left_path in left_images:
    base_name = left_path.name.replace("left_", "")
    right_path = left_path.with_name("right_" + base_name)
    if not right_path.exists():
        print(f"⚠️ Skipping {left_path.name} (no matching right image)")
        continue

    print(f"🟢 Processing: {left_path.name}")

    # Start total timer
    t_total_start = time.perf_counter()

    # --- Load left/right images ---
    imgL = cv2.imread(str(left_path))
    imgR = cv2.imread(str(right_path))
    if imgL is None or imgR is None:
        print(f"❌ Could not load {left_path.name}, skipping.")
        continue

    # --- 1️⃣ YOLOv5 Inference ---
    t_detect_start = time.perf_counter()
    img = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    im = torch.from_numpy(img).to(device)
    im = im.permute(2, 0, 1).float() / 255.0
    im = im.unsqueeze(0)
    pred = model(im)
    pred = non_max_suppression(pred, conf_thres=CONF_THRESH)
    t_detect_end = time.perf_counter()

    # Extract detections
    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], imgL.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                x_center = (x1 + x2) / 2 / imgL.shape[1]
                y_center = (y1 + y2) / 2 / imgL.shape[0]
                detections.append((x_center, y_center, float(conf), int(cls)))

    # --- 2️⃣ Stereo Depth Computation ---
    t_depth_start = time.perf_counter()
    rectL = cv2.remap(imgL, calib["stereo_map_left_x"], calib["stereo_map_left_y"], cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, calib["stereo_map_right_x"], calib["stereo_map_right_y"], cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=7,
        P1=8 * 3 * 7**2,
        P2=32 * 3 * 7**2,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=15,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_HH4
    )

    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    points_3d = cv2.reprojectImageTo3D(disp, calib["Q"])
    X, Y, Z = points_3d[:, :, 0], points_3d[:, :, 1], points_3d[:, :, 2]
    depth_clipped = np.clip(Z, 0, MAX_VIZ_DEPTH_M)
    t_depth_end = time.perf_counter()

    # --- 3️⃣ Distance Computation + Annotation ---
    t_distance_start = time.perf_counter()
    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_clipped, alpha=255.0 / MAX_VIZ_DEPTH_M),
        cv2.COLORMAP_INFERNO
    )

    for (x_center, y_center, conf, cls) in detections:
        h, w = depth_clipped.shape
        px, py = int(x_center * w), int(y_center * h)
        if not (0 <= px < w and 0 <= py < h):
            continue

        z = Z[py, px]
        if np.isnan(z) or z <= 0 or z > MAX_VIZ_DEPTH_M:
            continue

        x3d, y3d, z3d = X[py, px], Y[py, px], Z[py, px]
        euclidean = math.sqrt(x3d**2 + y3d**2 + z3d**2)
        ground_distance = math.sqrt(x3d**2 + z3d**2)

        label = f"{names[cls]} {euclidean:.2f}m (ground: {ground_distance:.2f}m)"
        cv2.circle(depth_vis, (px, py), 6, (255, 255, 255), -1)
        cv2.putText(depth_vis, label, (px - 25, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        summary_rows.append([
            left_path.name, cls, conf, x_center, y_center, round(float(z), 3), round(float(euclidean), 3), round(float(ground_distance), 3)
        ])

    base_out = OUT_DIR / f"depthmap_{left_path.stem}_HH4.jpg"
    cv2.imwrite(str(base_out), depth_vis)
    t_distance_end = time.perf_counter()

    # --- 4️⃣ Record timing ---
    t_detect = t_detect_end - t_detect_start
    t_depth = t_depth_end - t_depth_start
    t_distance = t_distance_end - t_distance_start
    t_total = time.perf_counter() - t_total_start

    timing_rows.append([
        left_path.name, len(detections),
        round(t_detect, 3), round(t_depth, 3),
        round(t_distance, 3), round(t_total, 3)
    ])

# --- Save detection and timing data ---
pd.DataFrame(summary_rows, columns=["filename", "class", "confidence", "x_center", "y_center", "depth_m", "euclidean_m", "ground_distance_m"]) \
    .to_csv(OUT_DIR_CSV / "detections_with_depth_HH4.csv", index=False)

pd.DataFrame(timing_rows, columns=["filename", "num_detections", "t_detect", "t_depth", "t_distance", "t_total"]) \
    .to_csv(OUT_DIR_CSV / "processing_times_HH4.csv", index=False)

# --- Print summary ---
df_time = pd.DataFrame(timing_rows, columns=["filename", "num_detections", "t_detect", "t_depth", "t_distance", "t_total"])
print("\n✅ Processing complete.")
print(f"Average detection time: {df_time['t_detect'].mean():.3f}s")
print(f"Average depth computation: {df_time['t_depth'].mean():.3f}s")
print(f"Average total time per pair: {df_time['t_total'].mean():.3f}s")
print(f"≈ {1 / df_time['t_total'].mean():.2f} FPS")
print(f"Results saved to: {OUT_DIR}")