# --- analyse_depth_performance_detailed.py ---
# Computes both summary statistics and per-measurement error CSVs

import pandas as pd
from pathlib import Path
import re

# --- File paths ---
measured_path = Path("measured_distance.csv")
detection_files = {
    "SGBM": Path("detections_with_depth_SGBM.csv"),
    "SGBM_3WAY": Path("detections_with_depth_SGBM_3WAY.csv"),
    "HH": Path("detections_with_depth_HH.csv"),
    "HH4": Path("detections_with_depth_HH4.csv"),
}

# --- Helper function ---
def extract_angle(filename):
    match = re.search(r"_(\d{2})deg", filename)
    return int(match.group(1)) if match else None

# --- Load measured distances ---
measured_df = pd.read_csv(measured_path)
measured_df["filename_base"] = measured_df["filename"].apply(lambda x: Path(x).name)

# --- Storage for summary ---
summary = []

for mode, det_file in detection_files.items():
    print(f"Processing mode: {mode}")

    det_df = pd.read_csv(det_file)
    # Clean any NaN or non-string filenames before applying Path
    det_df = det_df[det_df["filename"].notna()].copy()
    det_df["filename"] = det_df["filename"].astype(str)
    det_df["filename_base"] = det_df["filename"].apply(lambda x: Path(x).name if isinstance(x, str) else None)  
    det_df["angle_deg"] = det_df["filename"].apply(extract_angle)

    # Merge detection and measured data
    merged = pd.merge(
        det_df,
        measured_df,
        how="inner",
        on=["filename_base", "grid_row", "grid_col"],
        suffixes=("_det", "_meas")
    )

    # Compute absolute errors and percent errors
    merged["ground_abs_error"] = abs(merged["ground_distance_m"] - merged["expected_ground_distance_m"])
    merged["ground_abs_error_percent"] = (merged["ground_abs_error"] / merged["expected_ground_distance_m"]) * 100
    merged["euclid_abs_error"] = abs(merged["euclidean_m"] - merged["expected_euclidean_m"])
    merged["euclid_abs_error_percent"] = (merged["euclid_abs_error"] / merged["expected_euclidean_m"]) * 100

    # --- Save detailed per-measurement results ---
    detailed_output = merged[
        [
            "filename_base",
            "grid_row",
            "grid_col",
            "ground_distance_m",
            "expected_ground_distance_m",
            "ground_abs_error",
            "ground_abs_error_percent",
            "euclidean_m",
            "expected_euclidean_m",
            "euclid_abs_error",
            "euclid_abs_error_percent",
        ]
    ]
    detailed_output.to_csv(f"detailed_errors_{mode}.csv", index=False)
    print(f"✅ Saved detailed per-measurement errors to detailed_errors_{mode}.csv")

    # --- Compute summary ---
    def summarize(df, label="All"):
        return {
            "Mode": mode,
            "Angle": label,
            "Ground_MAE_m": df["ground_abs_error"].mean(),
            "Ground_MPE_%": df["ground_abs_error_percent"].mean(),
            "Ground_STD_m": df["ground_abs_error"].std(),
            "Euclid_MAE_m": df["euclid_abs_error"].mean(),
            "Euclid_MPE_%": df["euclid_abs_error_percent"].mean(),
            "Euclid_STD_m": df["euclid_abs_error"].std(),
        }

    # Global summary
    summary.append(summarize(merged, "All"))

    # Per-angle summary
    for angle, sub in merged.groupby("angle_deg"):
        summary.append(summarize(sub, angle))

# --- Save summary ---
summary_df = pd.DataFrame(summary)
summary_df.to_csv("depth_performance_summary.csv", index=False)
print("📊 Summary saved as depth_performance_summary.csv")