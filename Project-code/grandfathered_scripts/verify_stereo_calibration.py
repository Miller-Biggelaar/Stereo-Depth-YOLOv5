# verify_stereo_calibration.py
# Visual check to verify stereo calibration and rectification using saved images

import cv2
import numpy as np
import pickle
from pathlib import Path

# --- User paths ---
CALIBRATION_FILE = 'stereo_calibration.pkl'
LEFT_IMG_PATH  = 'calib_images/left/left_02.jpg'
RIGHT_IMG_PATH = 'calib_images/right/right_02.jpg'

def load_calibration(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def draw_horizontal_lines(img, spacing=40, color=(0, 255, 0)):
    h = img.shape[0]
    for y in range(spacing, h, spacing):
        cv2.line(img, (0, y), (img.shape[1], y), color, 1)
    return img

def verify_rectification():
    # Load calibration
    calib = load_calibration(CALIBRATION_FILE)

    # Load images from disk (cv2.imread returns BGR)
    imgL = cv2.imread(LEFT_IMG_PATH, cv2.IMREAD_COLOR)
    imgR = cv2.imread(RIGHT_IMG_PATH, cv2.IMREAD_COLOR)

    if imgL is None or imgR is None:
        raise FileNotFoundError(
            f"Could not load one or both images.\n  Left : {LEFT_IMG_PATH}\n  Right: {RIGHT_IMG_PATH}"
        )

    # The rectification maps define the expected image size
    mapLx, mapLy = calib['stereo_map_left_x'], calib['stereo_map_left_y']
    mapRx, mapRy = calib['stereo_map_right_x'], calib['stereo_map_right_y']
    H, W = mapLx.shape[:2]

    # Ensure inputs match calibration resolution (resize if needed, with a warning)
    if imgL.shape[:2] != (H, W):
        print(f"[WARN] Left image size {imgL.shape[1]}x{imgL.shape[0]} "
              f"does not match calibration {W}x{H}. Resizing.")
        imgL = cv2.resize(imgL, (W, H), interpolation=cv2.INTER_LINEAR)
    if imgR.shape[:2] != (H, W):
        print(f"[WARN] Right image size {imgR.shape[1]}x{imgR.shape[0]} "
              f"does not match calibration {W}x{H}. Resizing.")
        imgR = cv2.resize(imgR, (W, H), interpolation=cv2.INTER_LINEAR)

    # Apply rectification maps (keep BGR)
    rectL = cv2.remap(imgL, mapLx, mapLy, interpolation=cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, interpolation=cv2.INTER_LINEAR)

    # Add epipolar lines
    rectL_lines = draw_horizontal_lines(rectL.copy(), spacing=40, color=(0, 255, 0))
    rectR_lines = draw_horizontal_lines(rectR.copy(), spacing=40, color=(0, 255, 0))

    # Combine and display
    combined = cv2.hconcat([rectL_lines, rectR_lines])
    cv2.imshow("Rectified Stereo Pair (lines should align horizontally)", combined)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: save the visual for your report
    out_path = Path("calib_images/rectified_check_left02_right02.jpg")
    cv2.imwrite(str(out_path), combined)
    print(f"Saved: {out_path.resolve()}")

if __name__ == "__main__":
    verify_rectification()