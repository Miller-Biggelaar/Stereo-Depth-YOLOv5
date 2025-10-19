# single_pair_depthmap.py
import cv2
import numpy as np
import pickle
from pathlib import Path
import math

# --- User settings ---
CALIB_PKL   = "stereo_calibration.pkl"   # path to your calibration pickle
OUT_DIR     = "depth_only_results"        # where to save results
MAX_VIZ_DEPTH_M = 6.0                  # for visualizing depth map (meters)

Path(OUT_DIR).mkdir(exist_ok=True)

# --- Load calibration ---
with open(CALIB_PKL, "rb") as f:
    calib = pickle.load(f)

from glob import glob
import os

# --- Gather all left/right pairs ---
left_dir = Path("realworld_images/left_images")
right_dir = Path("realworld_images/right_images")
left_imgs = sorted(left_dir.glob("left_*.jpg"))
right_imgs = sorted(right_dir.glob("right_*.jpg"))

# Build a dict for quick lookup of right images by suffix
right_suffix_to_path = {p.name[len("right_"):]: p for p in right_imgs}

# Find all matching pairs
pairs = []
for left_path in left_imgs:
    suffix = left_path.name[len("left_"):]
    if suffix in right_suffix_to_path:
        pairs.append((left_path, right_suffix_to_path[suffix]))

if not pairs:
    raise RuntimeError("❌ No matching left/right image pairs found.")

print(f"Found {len(pairs)} matching left/right image pairs.")

baseline = 0.06  # meters
focal_length_mm = 4.74  # mm
pixel_size_mm = 0.0014  # mm/px
distances = [2, 3, 4, 5, 6]  # meters
# Will print table for each image shape in loop
# --- Process each pair ---
for idx, (left_path, right_path) in enumerate(pairs, 1):
    base_name = left_path.name[len("left_"):-4]  # Remove prefix and .jpg
    print(f"\n[{idx}/{len(pairs)}] Processing pair: {left_path.name} & {right_path.name}")
    imgR = cv2.imread(str(left_path))
    imgL = cv2.imread(str(right_path))
    if imgL is None or imgR is None:
        print(f"❌ Could not read images: {left_path}, {right_path}. Skipping.")
        continue
    # --- Rectify using precomputed stereo maps ---
    rectL = cv2.remap(imgL, calib["stereo_map_left_x"], calib["stereo_map_left_y"], cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, calib["stereo_map_right_x"], calib["stereo_map_right_y"], cv2.INTER_LINEAR)

    # --- Compute and print expected disparity range table (once per shape) ---
    img_width_px = rectL.shape[1]
    focal_length_px = focal_length_mm / pixel_size_mm  # effective focal length in pixels
    disparities = []
    print("=== Expected Disparities ===")
    print(f"Effective focal length (px): {focal_length_px:.1f}")
    print("Distance (m) | Disparity (px)")
    print("-----------------------------")
    for Z in distances:
        disp_val = (baseline * focal_length_px) / Z
        disparities.append(disp_val)
        print(f"{Z:.1f}         | {disp_val:.1f}")
    max_disparity = disparities[0]
    suggested_num_disp = 16 * math.ceil(1.2 * max_disparity / 16)
    print(f"Suggested num_disp: {suggested_num_disp}")
    print("============================")

    # --- StereoSGBM parameters ---
    min_disp            = 0
    num_disp            = 64   # TRY 128 - must be divisible by 16 - 16, 32, or 64 are good! was 32, GPT said higher for realistic scenes    
    block_size          = 7     # TRY 11-13 - smaller block size = more local detail and more sensitive to noise
    uniqueness_ratio    = 15    # Higher ratio = requires stronger confidence in a match.
    speckle_windowsize  = 50    # Sets the minimum region size (in pixels) that the stereo algorithm considers valid
    speckle_range       = 2     # TRY 4 - Max. disparity difference within a connected region to still be considered “consistent.”
    pre_filtercap       = 63    # Pixel intensities are normalized and clipped to +-value, Default is usually 31; currently using 63 (maximum allowed).

    P1 = 8 * 3 * block_size * block_size
    P2 = 32 * 3 * block_size * block_size
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        preFilterCap=pre_filtercap, # WAS 31
        uniquenessRatio=uniqueness_ratio, # WAS 10
        speckleWindowSize=speckle_windowsize, # WAS 50
        speckleRange=speckle_range, # WAS 2
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # --- Compute disparity map ---
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    # --- Convert disparity to depth ---
    points_3d = cv2.reprojectImageTo3D(disp, calib["Q"])
    depth = points_3d[:, :, 2]  # Z coordinate in meters (assuming Q was built that way)

    # --- Visualize disparity/depth ---
    disp_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(np.nan_to_num(disp, nan=0.0), alpha=255 / num_disp),
        cv2.COLORMAP_TURBO
    )
    depth_clipped = np.clip(depth, 0, MAX_VIZ_DEPTH_M)
    depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_clipped, alpha=255.0 / MAX_VIZ_DEPTH_M), cv2.COLORMAP_INFERNO)

    disp_fname = f"{OUT_DIR}/disparity_{base_name}.jpg"
    depthmap_fname = f"{OUT_DIR}/depthmap_{base_name}.jpg"
    depthnpy_fname = f"{OUT_DIR}/depth_{base_name}.npy"

    cv2.imwrite(disp_fname, disp_vis)
    cv2.imwrite(depthmap_fname, depth_vis)
    np.save(depthnpy_fname, depth)

    print(f"   Disparity/depth saved: {disp_fname}, {depthmap_fname}, {depthnpy_fname}")

    # Optional: comment out all visualization for batch
    # cv2.imshow("Left Rectified", rectL)
    # cv2.imshow("Right Rectified", rectR)
    # cv2.imshow("Disparity", disp_vis)
    # cv2.imshow("Depth Map", depth_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()