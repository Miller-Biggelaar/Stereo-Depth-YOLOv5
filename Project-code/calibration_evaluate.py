# evaluate_calibration.py
import pickle
import numpy as np
import math

CALIB_PKL = "stereo_calibration_full.pkl"

def rotation_to_euler(R):
    """Convert 3x3 rotation matrix to Euler angles (degrees)."""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x, y, z])

print("🔍 Evaluating stereo calibration:", CALIB_PKL)
with open(CALIB_PKL, "rb") as f:
    calib = pickle.load(f)

keys = calib.keys()
print("Available keys:", keys)

# --- Check for expected entries ---
required_keys = ["M1", "M2", "D1", "D2", "R", "T"]
missing = [k for k in required_keys if k not in calib]
if missing:
    print(f"⚠️ Missing expected keys: {missing}")
else:
    M1, M2, D1, D2, R, T = [calib[k] for k in required_keys]

    print("\n--- Intrinsic Parameters ---")
    fx1, fy1, cx1, cy1 = M1[0,0], M1[1,1], M1[0,2], M1[1,2]
    fx2, fy2, cx2, cy2 = M2[0,0], M2[1,1], M2[0,2], M2[1,2]
    print(f"Left Camera fx={fx1:.1f}, fy={fy1:.1f}, cx={cx1:.1f}, cy={cy1:.1f}")
    print(f"Right Camera fx={fx2:.1f}, fy={fy2:.1f}, cx={cx2:.1f}, cy={cy2:.1f}")

    f_mean = (fx1 + fx2) / 2
    print(f"Average focal length ≈ {f_mean:.1f} px")

    print("\n--- Distortion Coefficients ---")
    print("Left:", np.round(D1.flatten(), 5))
    print("Right:", np.round(D2.flatten(), 5))

    print("\n--- Extrinsic Parameters ---")
    baseline = np.linalg.norm(T)
    print(f"Translation vector (T): {T.flatten()}")
    print(f"Estimated baseline: {baseline*1000:.1f} mm")

    euler = rotation_to_euler(R)
    print(f"Rotation between cameras (°): roll={euler[0]:.3f}, pitch={euler[1]:.3f}, yaw={euler[2]:.3f}")

    # Theoretical disparity range
    baseline_m = baseline
    distances = [1, 2, 3, 4, 5, 6]
    print("\n--- Theoretical Disparities ---")
    print("Distance (m) | Expected Disparity (px)")
    print("--------------------------------------")
    disparities = []
    for Z in distances:
        disp = (baseline_m * f_mean) / Z
        disparities.append(disp)
        print(f"{Z:>10.2f} | {disp:>10.1f}")

    # --- Save summary to CSV ---
    import os
    import csv
    csv_dir = "CSVrepo"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "calibration_summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header
        writer.writerow([
            "Average focal length (px)",
            "Baseline (m)",
            "Baseline (mm)",
            "Rotation roll (deg)",
            "Rotation pitch (deg)",
            "Rotation yaw (deg)",
            "Disparity @1m (px)",
            "Disparity @2m (px)",
            "Disparity @3m (px)",
            "Disparity @4m (px)",
            "Disparity @5m (px)",
            "Disparity @6m (px)"
        ])
        # Data row
        writer.writerow([
            round(f_mean, 2),
            round(baseline_m, 5),
            round(baseline_m * 1000, 2),
            round(euler[0], 3),
            round(euler[1], 3),
            round(euler[2], 3),
            *[round(d, 2) for d in disparities]
        ])
    print(f"\n📄 Calibration summary saved to {csv_path}")

# --- Optional: check for rectification maps ---
if "stereo_map_left_x" in calib and "stereo_map_right_x" in calib:
    print("\n✅ Rectification maps found (stereo_map_left/right_x/y).")
else:
    print("\n⚠️ No rectification maps found in calibration file.")