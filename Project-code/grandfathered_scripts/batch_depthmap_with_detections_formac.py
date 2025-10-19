# batch_depthmap_with_detections.py
import cv2
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import math
import re

# --- User settings ---
IMAGE_DIR = Path("realworld_images/test/images")
CALIB_PKL = Path("stereo_calibration.pkl")
DETECTIONS_CSV = Path("realworld_images/test/detections_fromtest.csv")
OUT_DIR = Path("realworld_images/test/depthmaps")
MAX_VIZ_DEPTH_M = 6.0
CONF_THRESH = 0.5  # minimum confidence for detections

OUT_DIR.mkdir(exist_ok=True, parents=True)

# --- Load calibration ---
with open(CALIB_PKL, "rb") as f:
    calib = pickle.load(f)

# --- Load detections CSV ---
detections = pd.read_csv(DETECTIONS_CSV)

# --- Prepare output summary list ---
summary_rows = []

# --- Stereo parameters ---
baseline = 0.06  # meters
focal_length_mm = 4.74  # mm
pixel_size_mm = 0.0014  # mm/px
focal_length_px = focal_length_mm / pixel_size_mm  # px

# --- Match left/right image pairs ---
left_images = sorted(IMAGE_DIR.glob("left_*.jpg"))
for left_img_path in left_images:
    base_name = left_img_path.name.replace("left_", "")
    right_img_path = left_img_path.with_name("right_" + base_name)
    if not right_img_path.exists():
        print(f"⚠️ Skipping {left_img_path.name} — no matching right image found.")
        continue

    print(f"🟢 Processing pair: {left_img_path.name} / {right_img_path.name}")

    # --- Read images ---
    imgR = cv2.imread(str(left_img_path))
    imgL = cv2.imread(str(right_img_path))
    if imgL is None or imgR is None:
        print(f"❌ Could not read one or both images for {base_name}, skipping.")
        continue

    # --- Rectify images ---
    rectL = cv2.remap(imgL, calib["stereo_map_left_x"], calib["stereo_map_left_y"], cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, calib["stereo_map_right_x"], calib["stereo_map_right_y"], cv2.INTER_LINEAR)

    # --- StereoSGBM setup ---
    min_disp = 0
    num_disp = 64  # tuned for 2–6m range
    block_size = 7
    uniqueness_ratio = 15
    speckle_windowsize = 50
    speckle_range = 2
    pre_filtercap = 63
    P1 = 8 * 3 * block_size**2
    P2 = 32 * 3 * block_size**2

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        preFilterCap=pre_filtercap,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_windowsize,
        speckleRange=speckle_range,
        mode=cv2.STEREO_SGBM_MODE_HH4
    )

    # --- Compute disparity ---
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    # --- Reproject to 3D ---
    points_3d = cv2.reprojectImageTo3D(disp, calib["Q"])
    X = points_3d[:, :, 0]
    Y = points_3d[:, :, 1]
    depth = points_3d[:, :, 2]
    depth_clipped = np.clip(depth, 0, MAX_VIZ_DEPTH_M)

    # --- Visualize ---
    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_clipped, alpha=255.0 / MAX_VIZ_DEPTH_M),
        cv2.COLORMAP_INFERNO
    )

    # --- Annotate detections ---
    h, w = depth.shape
    subset = detections[detections["filename"].str.contains(left_img_path.stem)]
    for _, row in subset.iterrows():
        conf = float(row["confidence"])
        if conf < CONF_THRESH:
            continue

        x_norm, y_norm = float(row["x_center"]), float(row["y_center"])
        px, py = int(x_norm * w), int(y_norm * h)

        if not (0 <= px < w and 0 <= py < h):
            continue

        z = depth[py, px]
        if np.isnan(z) or z <= 0 or z > MAX_VIZ_DEPTH_M:
            continue

        x3d, y3d, z3d = X[py, px], Y[py, px], depth[py, px]
        euclidean = math.sqrt(x3d**2 + y3d**2 + z3d**2)

        label = f"Z={z:.2f}m"
        cv2.circle(depth_vis, (px, py), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(depth_vis, (px, py), 4, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.putText(depth_vis, label, (px - 20, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        summary_rows.append([
            left_img_path.name, row["class"], x_norm, y_norm,
            row["width"], row["height"], conf, round(float(z), 3), round(float(euclidean), 3)
        ])

    # --- Save results ---
    base_out = OUT_DIR / f"depthmap_{left_img_path.stem}.jpg"
    np.save(OUT_DIR / f"depth_{left_img_path.stem}.npy", depth)
    cv2.imwrite(str(base_out), depth_vis)

# --- Save summary CSV ---
summary_csv = OUT_DIR / "detections_with_depth.csv"
summary_df = pd.DataFrame(summary_rows, columns=[
    "filename", "class", "x_center", "y_center", "width", "height", "confidence", "depth_m", "euclidean_m"
])
summary_df.to_csv(summary_csv, index=False)

print(f"\n✅ Completed processing {len(left_images)} stereo pairs.")
print(f"✅ Results saved to: {OUT_DIR}")
print(f"✅ Detection-depth summary: {summary_csv}")