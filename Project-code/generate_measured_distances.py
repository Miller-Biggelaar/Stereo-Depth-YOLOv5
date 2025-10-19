import csv
from pathlib import Path
import re
import math

# --- Settings ---
OUT_CSV = Path("CSVrepomeasured_distance.csv")
IMG_DIR = Path("realworld_images/test/images")

# Grid geometry (in meters)
GRID_WIDTH = 2.00     # 200 cm
GRID_HEIGHT = 2.25    # 225 cm
X_MARGIN = 0.25       # 25 cm from side edge to first square center
Y_MARGIN = 0.30       # 30 cm from top edge to first square center
X_SPACING = 0.50      # 50 cm between centers horizontally
Y_SPACING = 0.55      # 55 cm between centers vertically
GRID_ROWS = 4
GRID_COLS = 4

# --- Helper to parse metadata from filename ---
def parse_filename(fname):
    # Example: left_20deg_154cmgnd_250cmgrid_20250926_111529.jpg
    pattern = r"left_(\d+)deg_(\d+)cmgnd_(\d+)cmgrid"
    m = re.search(pattern, fname)
    if not m:
        return None
    angle_deg = float(m.group(1))
    cam_height_m = float(m.group(2)) / 100.0
    cam_dist_m = float(m.group(3)) / 100.0
    return angle_deg, cam_height_m, cam_dist_m

# --- Compute grid point distances ---
def compute_distances(cam_height_m, cam_dist_m):
    results = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            # Horizontal offset from grid center (X)
            x_offset = (X_MARGIN + c * X_SPACING) - (GRID_WIDTH / 2.0)
            # Depth offset from camera (Z)
            z_offset = cam_dist_m + Y_MARGIN + r * Y_SPACING
            # Ground distance (plan view)
            ground_dist = math.sqrt(x_offset**2 + z_offset**2)
            # Total 3D distance including camera height
            total_dist = math.sqrt(ground_dist**2 + cam_height_m**2)
            results.append((r+1, c+1, x_offset, z_offset, ground_dist, total_dist))
    return results

# --- Main ---
rows = []
for img_path in sorted(IMG_DIR.glob("left_*.jpg")):
    parsed = parse_filename(img_path.name)
    if not parsed:
        print(f"⚠️ Skipping {img_path.name}: could not parse metadata.")
        continue
    angle_deg, cam_height_m, cam_dist_m = parsed
    distances = compute_distances(cam_height_m, cam_dist_m)
    for r, c, x_off, z_off, ground, total in distances:
        rows.append({
            "filename": img_path.name,
            "grid_row": r,
            "grid_col": c,
            "horizontal_offset_m": round(x_off, 3),
            "depth_offset_m": round(z_off, 3),
            "expected_ground_distance_m": round(ground, 3),
            "expected_euclidean_m": round(total, 3)
        })

# --- Write CSV ---
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "filename", "grid_row", "grid_col",
        "horizontal_offset_m", "depth_offset_m",
        "ground_distance_m", "camera_to_square_m"
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Saved measured distances for {len(rows)} grid points across {len(set(r['filename'] for r in rows))} images.")
print(f"📄 Output file: {OUT_CSV}")