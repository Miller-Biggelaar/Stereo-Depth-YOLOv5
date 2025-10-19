import pandas as pd
from pathlib import Path

# --- File paths ---
DETECTIONS_CSV = Path("realworld_images/test/depthmaps/detections_with_depth.csv")
MEASURED_CSV = Path("realworld_images/test/measured_distance.csv")
OUT_CSV = Path("realworld_images/test/depth_comparison.csv")

# --- Load data ---
det = pd.read_csv(DETECTIONS_CSV)
meas = pd.read_csv(MEASURED_CSV)

# --- Keep only relevant columns ---
det = det[["filename", "grid_row", "grid_col", "euclidean_m"]]
meas = meas[["filename", "grid_row", "grid_col", "camera_to_square_m"]]

# --- Merge on filename + grid_row + grid_col ---
merged = pd.merge(
    det,
    meas,
    on=["filename", "grid_row", "grid_col"],
    how="inner",  # only keep matches
    suffixes=("_det", "_meas")
)

# --- Compute error metrics ---
merged["abs_error_m"] = (merged["euclidean_m"] - merged["camera_to_square_m"]).abs()
merged["percent_error"] = (merged["abs_error_m"] / merged["camera_to_square_m"]) * 100

# --- Round for readability ---
merged = merged.round({
    "euclidean_m": 3,
    "camera_to_square_m": 3,
    "abs_error_m": 3,
    "percent_error": 2
})

# --- Save results ---
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_CSV, index=False)

# --- Print summary ---
print(f"✅ Compared {len(merged)} detections against measured distances.")
print(f"✅ Results saved to: {OUT_CSV}")

if len(merged) > 0:
    # Overall statistics
    avg_error = merged["abs_error_m"].mean()
    std_error = merged["abs_error_m"].std()
    var_error = merged["abs_error_m"].var()
    avg_percent = merged["percent_error"].mean()
    std_percent = merged["percent_error"].std()
    var_percent = merged["percent_error"].var()

    print(f"📊 Mean absolute error:      {avg_error:.3f} m")
    print(f"📊 Std dev absolute error:   {std_error:.3f} m")
    print(f"📊 Variance absolute error:  {var_error:.3f} m^2")
    print(f"📊 Mean percent error:       {avg_percent:.2f}%")
    print(f"📊 Std dev percent error:    {std_percent:.2f}%")
    print(f"📊 Variance percent error:   {var_percent:.2f} %^2")

    # Per-image grouped statistics
    per_image = merged.groupby("filename").agg(
        mean_abs_error_m = ("abs_error_m", "mean"),
        std_abs_error_m = ("abs_error_m", "std"),
        var_abs_error_m = ("abs_error_m", "var"),
        mean_percent_error = ("percent_error", "mean"),
        std_percent_error = ("percent_error", "std"),
        var_percent_error = ("percent_error", "var"),
        n = ("abs_error_m", "count")
    ).reset_index()
    # Round for readability
    per_image = per_image.round({
        "mean_abs_error_m": 3,
        "std_abs_error_m": 3,
        "var_abs_error_m": 3,
        "mean_percent_error": 2,
        "std_percent_error": 2,
        "var_percent_error": 2,
        "n": 0
    })
    # Save per-image stats
    per_image_csv = OUT_CSV.parent / "depth_comparison_per_image.csv"
    per_image.to_csv(per_image_csv, index=False)
    print(f"📄 Per-image statistics saved to: {per_image_csv}")