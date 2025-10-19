# analyse_stereo.py
# Compare SGBM variants (SGBM, SGBM_3WAY, HH, HH4) against ground-truth distances

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- File paths ---
DATA_DIR = Path("CSVrepo")
OUTPUT_DIR = Path("Analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

measured_file = DATA_DIR / "measured_distance.csv"
modes = {
    "SGBM_3WAY": DATA_DIR / "detections_with_depth_SGBM_3WAY.csv",
    "SGBM": DATA_DIR / "detections_with_depth_SGBM.csv",
    "HH": DATA_DIR / "detections_with_depth_HH.csv",
    "HH4": DATA_DIR / "detections_with_depth_HH4.csv",
}

# --- Load ground-truth distances ---
measured = pd.read_csv(measured_file)

# --- Container for per-mode summary stats ---
summary_rows = []

# --- Container for all merged data across modes for combined angle analysis ---
all_merged = []

# --- Loop through each mode and compute errors ---
for mode, path in modes.items():
    print(f"\n🔍 Analyzing {mode} ...")
    det = pd.read_csv(path)

    # Merge with expected distances
    merged = det.merge(
        measured,
        on=["filename", "grid_row", "grid_col"],
        how="inner",
        validate="many_to_one"
    )

    n_matches = len(merged)
    print(f"  • {n_matches} matched detections")

    # Compute errors
    merged["abs_err_ground"] = (merged["ground_distance_m"] - merged["expected_ground_distance_m"]).abs()
    merged["pct_err_ground"] = merged["abs_err_ground"] / merged["expected_ground_distance_m"] * 100
    merged["abs_err_euclid"] = (merged["euclidean_m"] - merged["expected_euclidean_m"]).abs()
    merged["pct_err_euclid"] = merged["abs_err_euclid"] / merged["expected_euclidean_m"] * 100

    # Summaries
    summary = {
        "Mode": mode,
        "Count": n_matches,
        "Mean Abs Error (ground, m)": merged["abs_err_ground"].mean(),
        "Mean % Error (ground)": merged["pct_err_ground"].mean(),
        "Mean Abs Error (euclid, m)": merged["abs_err_euclid"].mean(),
        "Mean % Error (euclid)": merged["pct_err_euclid"].mean(),
        "Std Dev (ground)": merged["abs_err_ground"].std(),
        "Median Abs Error (ground)": merged["abs_err_ground"].median(),
        "Outlier Rate (>20%)": (merged["pct_err_ground"] > 20).mean() * 100,
    }
    summary_rows.append(summary)

    # --- Plots per mode ---
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=merged,
        x="expected_ground_distance_m", y="ground_distance_m",
        s=20, alpha=0.6, edgecolor=None
    )
    plt.plot([0, 6], [0, 6], "r--", lw=1)
    plt.title(f"{mode}: Measured vs Expected Ground Distance")
    plt.xlabel("Expected ground distance (m)")
    plt.ylabel("Measured ground distance (m)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{mode}_scatter_measured_vs_expected.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.histplot(merged["pct_err_ground"], bins=25, kde=True)
    plt.title(f"{mode}: Percent Error Distribution (Ground)")
    plt.xlabel("Percent error (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{mode}_percent_error_hist.png", dpi=200)
    plt.close()

    # --- Error vs Distance plot with SD ---
    bins = pd.interval_range(start=0, end=6, freq=0.25)
    merged['distance_bin'] = pd.cut(merged['expected_ground_distance_m'], bins=bins, include_lowest=True)
    error_vs_distance = merged.groupby('distance_bin')['abs_err_ground'].agg(['mean', 'std']).reset_index()
    # Use bin midpoints for x axis
    error_vs_distance['bin_mid'] = error_vs_distance['distance_bin'].apply(lambda x: x.mid)

    plt.figure(figsize=(6, 4))
    plt.plot(error_vs_distance['bin_mid'], error_vs_distance['mean'], marker='o', linestyle='-', label='Mean abs error')
    plt.title(f"{mode}: Mean Absolute Ground Error vs Expected Distance")
    plt.xlabel("Expected ground distance (m)")
    plt.ylabel("Mean absolute ground error (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{mode}_error_vs_distance_with_SD.png", dpi=200)
    plt.close()

    # --- Error vs Angle analysis ---
    import re
    # Extract angle from filename (e.g., '15deg')
    def extract_angle_deg(filename):
        m = re.search(r'(\d{2})deg', str(filename))
        return int(m.group(1)) if m else None
    merged['angle_deg'] = merged['filename'].apply(extract_angle_deg)
    angle_stats = merged.groupby('angle_deg')['abs_err_ground'].agg(['mean', 'std']).reset_index()
    angle_stats = angle_stats.dropna(subset=['angle_deg'])

    plt.figure(figsize=(6, 4))
    plt.errorbar(angle_stats['angle_deg'], angle_stats['mean'], yerr=angle_stats['std'], fmt='o-', capsize=4)
    plt.title(f"{mode}: Mean Absolute Ground Error vs Angle")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Mean absolute ground error (m)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{mode}_error_vs_angle.png", dpi=200)
    plt.close()

    # Append mode info to merged for combined angle analysis
    merged['Mode'] = mode
    all_merged.append(merged)

# --- Combine and export summary ---
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "stereo_mode_summary.csv", index=False)
summary_df.to_csv(DATA_DIR / "stereo_mode_summary.csv", index=False)
print("\n✅ Summary written to stereo_mode_summary.csv\n")
print(summary_df)

# --- Combined comparison plots ---
plt.figure(figsize=(6, 4))
sns.barplot(data=summary_df, x="Mode", y="Mean Abs Error (ground, m)", palette="viridis")
plt.title("Mean Absolute Ground Distance Error per SGBM Mode")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_mean_abs_error.png", dpi=200)
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(data=summary_df.melt(id_vars="Mode", 
                                 value_vars=["Mean % Error (ground)", "Mean % Error (euclid)"]),
            x="Mode", y="value", hue="variable", palette="coolwarm")
plt.title("Comparison of Percent Errors Across Modes")
plt.ylabel("Mean Percent Error (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_percent_error.png", dpi=200)
plt.close()

print("📊 Saved comparison plots:")
print("  - Analysis/*_scatter_measured_vs_expected.png")
print("  - Analysis/*_percent_error_hist.png")
print("  - Analysis/comparison_mean_abs_error.png")
print("  - Analysis/comparison_percent_error.png")

# --- Combined plots per angle ---
all_merged_df = pd.concat(all_merged, ignore_index=True)

# Extract angle_deg again in case missing
import re
def extract_angle_deg(filename):
    m = re.search(r'(\d{2})deg', str(filename))
    return int(m.group(1)) if m else None
all_merged_df['angle_deg'] = all_merged_df['filename'].apply(extract_angle_deg)

# Filter to angles of interest
angles_of_interest = [20, 35, 40, 45]
angle_mode_stats = all_merged_df.groupby(['angle_deg', 'Mode'])['pct_err_ground'].agg(['mean', 'std']).reset_index()
angle_mode_stats = angle_mode_stats[angle_mode_stats['angle_deg'].isin(angles_of_interest)]

for angle in angles_of_interest:
    data_angle = angle_mode_stats[angle_mode_stats['angle_deg'] == angle]
    plt.figure(figsize=(6,4))
    plt.bar(data_angle['Mode'], data_angle['mean'], yerr=data_angle['std'], capsize=5, color=sns.color_palette("viridis", len(data_angle)))
    plt.title(f"Percent Error Comparison at {angle}° Angle")
    plt.xlabel("SGBM Mode")
    plt.ylabel("Mean Percent Error (%)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"angle_{angle}_error_comparison.png", dpi=200)
    plt.close()


# --- Grid-based error matrix ---
# For each mode, compute mean percent ground error grouped by grid_row and grid_col.
grid_errs = []
for mode, path in modes.items():
    det = pd.read_csv(path)
    merged = det.merge(
        measured,
        on=["filename", "grid_row", "grid_col"],
        how="inner",
        validate="many_to_one"
    )
    merged["abs_err_ground"] = (merged["ground_distance_m"] - merged["expected_ground_distance_m"]).abs()
    merged["pct_err_ground"] = merged["abs_err_ground"] / merged["expected_ground_distance_m"] * 100
    # Group by grid row/col and calculate mean percent error
    grid_group = merged.groupby(['grid_row', 'grid_col'])['pct_err_ground'].mean().reset_index()
    grid_group = grid_group.rename(columns={'pct_err_ground': f'mean_err_{mode}'})
    grid_errs.append(grid_group)

# Merge all four modes into one DataFrame
from functools import reduce
grid_merged = reduce(
    lambda left, right: pd.merge(left, right, on=['grid_row', 'grid_col'], how='outer'),
    grid_errs
)

# Sort for grid (descending grid_row, ascending grid_col)
grid_merged = grid_merged.sort_values(['grid_row', 'grid_col'], ascending=[False, True])

# Prepare a pivot-like 4x4 matrix with formatted error strings
rows = sorted(grid_merged['grid_row'].unique(), reverse=True)
cols = sorted(grid_merged['grid_col'].unique())
matrix = []
for r in rows:
    row_cells = []
    for c in cols:
        cell = grid_merged[(grid_merged['grid_row'] == r) & (grid_merged['grid_col'] == c)]
        if not cell.empty:
            v = cell.iloc[0]
            sgbm = v.get('mean_err_SGBM', float('nan'))
            s3w = v.get('mean_err_SGBM_3WAY', float('nan'))
            hh = v.get('mean_err_HH', float('nan'))
            hh4 = v.get('mean_err_HH4', float('nan'))
            cell_str = f"(SGBM: {sgbm:.1f}%, 3WAY: {s3w:.1f}%, HH: {hh:.1f}%, HH4: {hh4:.1f}%)"
        else:
            cell_str = ""
        row_cells.append(cell_str)
    matrix.append(row_cells)

# Save the table as grid_error_matrix.csv
import csv
grid_error_matrix_path = OUTPUT_DIR / "grid_error_matrix.csv"
with open(grid_error_matrix_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["grid_row\\grid_col"] + cols
    writer.writerow(header)
    for i, r in enumerate(rows):
        writer.writerow([r] + matrix[i])

print(f"🗺️  Grid error matrix saved to {grid_error_matrix_path}")