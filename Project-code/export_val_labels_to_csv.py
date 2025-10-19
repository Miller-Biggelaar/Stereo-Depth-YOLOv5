import csv
from pathlib import Path
import re

# === User settings ===
LABELS_DIR = Path("yolov5/runs/val/synth_plus_real_ft13_eval3/labels")
CONF_THRESH = 0.25  # only include detections >= this confidence
OUTPUT_CSV = LABELS_DIR.parent / "detections_long_format.csv"

# === Prepare CSV output ===
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "class", "x_center", "y_center", "width", "height", "confidence"])

    # Loop over all label files
    for txt_file in sorted(LABELS_DIR.glob("*.txt")):
        base_name = re.sub(r"_jpg\.rf\..*\.txt$", ".txt", txt_file.name)
        with open(txt_file, "r") as infile:
            for line in infile:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue  # skip malformed lines
                class_id, x, y, w, h, conf = parts
                if float(conf) >= CONF_THRESH:
                    writer.writerow([base_name, class_id, x, y, w, h, conf])

print(f"✅ Exported detections (one per row, conf ≥ {CONF_THRESH}) to:\n{OUTPUT_CSV}")