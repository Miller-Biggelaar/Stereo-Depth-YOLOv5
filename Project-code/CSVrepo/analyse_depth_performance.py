import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import os

# Define file paths
base_dir = Path(".")
measured_file = base_dir / "measured_distance.csv"
detection_files = {
    "SGBM": base_dir / "detections_with_depth_SGBM.csv",
    "SGBM_3WAY": base_dir / "detections_with_depth_SGBM_3WAY.csv",
    "HH": base_dir / "detections_with_depth_HH.csv",
    "HH4": base_dir / "detections_with_depth_HH4.csv",
}

# Load measured distances
measured_df = pd.read_csv(measured_file)
measured_df["filename"] = measured_df["filename"].astype(str)
measured_df["filename_base"] = measured_df["filename"].apply(lambda x: Path(x).name)

# Extract angle info (e.g., left_20deg_...)
def extract_angle(filename):
    match = re.search(r"_(\d{2})deg", filename)
    return int(match.group(1)) if match else None

measured_df["angle_deg"] = measured_df["filename"].apply(extract_angle)

# Prepare results list
results = []

# Process each stereo mode
for mode, det_file in detection_files.items():
    det_df = pd.read_csv(det_file)
    det_df["filename"] = det_df["filename"].astype(str)
    det_df["filename_base"] = det_df["filename"].apply(lambda x: Path(x).name)
    det_df["angle_deg"] = det_df["filename"].apply(extract_angle)

    # Merge measured and detected data
    merged = pd.merge(
        det_df,
        measured_df,
        how="inner",
        on=["filename_base", "grid_row", "grid_col"],
        suffixes=("_det", "_meas")
    )

    # Compute absolute and percent errors
    merged["ground_abs_error"] = abs(merged["ground_distance_m"] - merged["expected_ground_distance_m"])
    merged["ground_percent_error"] = (merged["ground_abs_error"] / merged["expected_ground_distance_m"]) * 100
    merged["euclid_abs_error"] = abs(merged["euclidean_m"] - merged["expected_euclidean_m"])
    merged["euclid_percent_error"] = (merged["euclid_abs_error"] / merged["expected_euclidean_m"]) * 100

    # --- TOTAL METRICS ---
    total_ground_mae = merged["ground_abs_error"].mean()
    total_ground_mpe = merged["ground_percent_error"].mean()
    total_ground_std = merged["ground_abs_error"].std()

    total_euclid_mae = merged["euclid_abs_error"].mean()
    total_euclid_mpe = merged["euclid_percent_error"].mean()
    total_euclid_std = merged["euclid_abs_error"].std()

    results.append({
        "Mode": mode,
        "Angle": "All",
        "Ground_MAE_m": total_ground_mae,
        "Ground_MPE_%": total_ground_mpe,
        "Ground_STD_m": total_ground_std,
        "Euclid_MAE_m": total_euclid_mae,
        "Euclid_MPE_%": total_euclid_mpe,
        "Euclid_STD_m": total_euclid_std
    })

    # Ensure angle_deg exists after merge
    if "angle_deg" not in merged.columns:
        merged["angle_deg"] = merged["filename_base"].apply(extract_angle)

    # --- PER-ANGLE METRICS ---
    for angle, sub in merged.groupby("angle_deg"):
        results.append({
            "Mode": mode,
            "Angle": angle,
            "Ground_MAE_m": sub["ground_abs_error"].mean(),
            "Ground_MPE_%": sub["ground_percent_error"].mean(),
            "Ground_STD_m": sub["ground_abs_error"].std(),
            "Euclid_MAE_m": sub["euclid_abs_error"].mean(),
            "Euclid_MPE_%": sub["euclid_percent_error"].mean(),
            "Euclid_STD_m": sub["euclid_abs_error"].std()
        })

# Save results
results_df = pd.DataFrame(results)
output_file = base_dir / "depth_performance_summary.csv"
results_df.to_csv(output_file, index=False)

print(f"✅ Depth performance analysis complete.")
print(f"Results saved to: {output_file}")
print(results_df)

# --- Plotting ---

# Create output folder for plots
plots_dir = base_dir / "Analysis_Plots"
os.makedirs(plots_dir, exist_ok=True)

# 1. Bar chart of overall ground MAE per stereo mode
overall_df = results_df[results_df["Angle"] == "All"]
plt.figure(figsize=(8,6))
plt.bar(overall_df["Mode"], overall_df["Ground_MAE_m"], color='skyblue')
plt.xlabel("Stereo Mode")
plt.ylabel("Ground MAE (m)")
plt.title("Overall Ground MAE per Stereo Mode")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "overall_ground_mae_per_mode.png")
plt.close()

# 1.1 Bar chart of overall euclid MAE per stereo mode
overall_df = results_df[results_df["Angle"] == "All"]
plt.figure(figsize=(8,6))
plt.bar(overall_df["Mode"], overall_df["Euclid_MAE_m"], color='skyblue')
plt.xlabel("Stereo Mode")
plt.ylabel("Euclid MAE (m)")
plt.title("Overall Euclid MAE per Stereo Mode")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "overall_euclid_mae_per_mode.png")
plt.close()

# 2. Grouped bar chart showing ground MAE by angle for each stereo mode
angle_df = results_df[results_df["Angle"] != "All"].copy()
angle_df["Angle"] = angle_df["Angle"].astype(int)
angle_df = angle_df.sort_values("Angle")
modes = angle_df["Mode"].unique()
angles = sorted(angle_df["Angle"].unique())
bar_width = 0.15
x = range(len(angles))

plt.figure(figsize=(12,6))
for i, mode in enumerate(modes):
    mode_data = angle_df[angle_df["Mode"] == mode]
    # Align bars for each mode
    positions = [pos + i*bar_width for pos in x]
    plt.bar(positions, mode_data["Ground_MAE_m"], width=bar_width, label=mode)

plt.xlabel("Angle (deg)")
plt.ylabel("Ground MAE (m)")
plt.title("Ground MAE by Angle for Each Stereo Mode")
plt.xticks([pos + bar_width*(len(modes)/2 - 0.5) for pos in x], angles)
plt.legend(title="Stereo Mode")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "ground_mae_by_angle_per_mode.png")
plt.close()

# 3. Error vs angle line plot with standard deviation bars (Ground MAE ± STD)
plt.figure(figsize=(12,6))
for mode in modes:
    mode_data = angle_df[angle_df["Mode"] == mode]
    plt.errorbar(
        mode_data["Angle"],
        mode_data["Ground_MAE_m"],
        yerr=mode_data["Ground_STD_m"],
        label=mode,
        capsize=3,
        marker='o',
        linestyle="None"
    )
plt.xlabel("Angle (deg)")
plt.ylabel("Ground MAE (m)")
plt.title("Ground MAE vs Angle with Std Dev Bars")
plt.legend(title="Stereo Mode")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "ground_mae_vs_angle_stddev.png")
plt.close()

# 4. Scatter plot comparing ground and Euclidean MAE (overall)
plt.figure(figsize=(8,6))
plt.scatter(overall_df["Ground_MAE_m"], overall_df["Euclid_MAE_m"], color='purple')
for i, row in overall_df.iterrows():
    plt.text(row["Ground_MAE_m"], row["Euclid_MAE_m"], row["Mode"], fontsize=9, ha='right', va='bottom')
plt.xlabel("Ground MAE (m)")
plt.ylabel("Euclidean MAE (m)")
plt.title("Scatter Plot: Ground MAE vs Euclidean MAE (Overall)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "scatter_ground_vs_euclid_mae.png")
plt.close()

# 5. Error vs angle line plot with standard deviation bars (Euclid MAE ± STD)
plt.figure(figsize=(12,6))
for mode in modes:
    mode_data = angle_df[angle_df["Mode"] == mode]
    plt.errorbar(
        mode_data["Angle"],
        mode_data["Euclid_MAE_m"],
        yerr=mode_data["Euclid_STD_m"],
        label=mode,
        capsize=3,
        marker='o',
        linestyle="None"
    )

plt.xlabel("Angle (deg)")
plt.ylabel("Euclid_MAE_m (m)")
plt.title("Euclid_MAE_m vs Angle with Std Dev Bars")
plt.legend(title="Stereo Mode")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "euclid_mae_m_vs_angle_stddev.png")
plt.close()

# 6. Individual Euclid_MAE_m vs Angle plots per stereo mode
for mode in modes:
    mode_data = angle_df[angle_df["Mode"] == mode]
    plt.figure(figsize=(12,6))
    plt.errorbar(
        mode_data["Angle"],
        mode_data["Euclid_MAE_m"],
        yerr=mode_data["Euclid_STD_m"],
        capsize=3,
        marker='o',
        linestyle="None"
    )
    plt.xlabel("Angle (deg)")
    plt.ylabel("Euclid_MAE_m (m)")
    plt.title(f"Euclid_MAE_m vs Angle for {mode}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / f"euclid_mae_m_vs_angle_{mode}.png")
    plt.close()

    # 7. Individual Ground_MAE_m vs Angle plots per stereo mode
for mode in modes:
    mode_data = angle_df[angle_df["Mode"] == mode]
    plt.figure(figsize=(12,6))
    plt.errorbar(
        mode_data["Angle"],
        mode_data["Ground_MAE_m"],
        yerr=mode_data["Ground_STD_m"],
        capsize=3,
        marker='o',
        linestyle="None"
    )
    plt.xlabel("Angle (deg)")
    plt.ylabel("Ground_MAE_m (m)")
    plt.title(f"Ground_MAE_m vs Angle for {mode}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / f"ground_mae_m_vs_angle_{mode}.png")
    plt.close()