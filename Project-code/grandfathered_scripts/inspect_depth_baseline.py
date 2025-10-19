# inspect_depth_baseline.py
# Computes and visualizes the disparity/depth map using StereoSGBM from a stereo pair

import cv2
import numpy as np
import pickle

# --- Configuration ---
LEFT_IMAGE_PATH = "left_23.jpg"
RIGHT_IMAGE_PATH = "right_23.jpg"
CALIBRATION_FILE = "stereo_calibration.pkl"

# --- Toggle for swapping left/right images ---
SWAP_LEFT_RIGHT = False  # set to False if already correct

if SWAP_LEFT_RIGHT:
    img_left = cv2.imread(RIGHT_IMAGE_PATH)
    img_right = cv2.imread(LEFT_IMAGE_PATH)
else:
    img_left = cv2.imread(LEFT_IMAGE_PATH)
    img_right = cv2.imread(RIGHT_IMAGE_PATH)

# # --- Load stereo images ---
# img_left = cv2.imread(LEFT_IMAGE_PATH)
# img_right = cv2.imread(RIGHT_IMAGE_PATH)

if img_left is None or img_right is None:
    raise FileNotFoundError("One or both input images could not be loaded.")

# --- Load calibration data ---
with open(CALIBRATION_FILE, 'rb') as f:
    calib = pickle.load(f)

# Rectify images
rectified_left = cv2.remap(img_left, calib['stereo_map_left_x'], calib['stereo_map_left_y'], cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, calib['stereo_map_right_x'], calib['stereo_map_right_y'], cv2.INTER_LINEAR)

# Convert to grayscale
gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

# --- StereoSGBM parameters ---
min_disp = 0
num_disp = 64  # must be divisible by 16 - 16, 32, or 64 are good!
block_size = 7  # WAS 5

matcher = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15, # WAS 10
    speckleWindowSize=50,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Compute disparity
raw_disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0

# Mask invalid disparities
disparity_masked = np.ma.masked_less_equal(raw_disparity, 0)

# Normalize for display
disp_display = cv2.normalize(disparity_masked, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disp_display = np.uint8(disp_display)

# --- Reproject to 3D (optional depth map) ---
depth_map = cv2.reprojectImageTo3D(raw_disparity, calib['Q'])

# --- Display ---
cv2.imshow("Rectified Left", rectified_left)
cv2.imshow("Rectified Right", rectified_right)
cv2.imshow("Disparity Map", disp_display)
print("Click on the Disparity Map to get depth.")

print("Disparity stats: min =", np.min(raw_disparity), "max =", np.max(raw_disparity))
print("Q matrix:\n", calib['Q'])

# Mouse callback for inspecting depth
clicked_points = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        d = raw_disparity[y, x]
        if d > 0:
            point_3d = depth_map[y, x]
            print(f"Clicked at ({x}, {y}): Disparity = {d:.2f}, Depth (Z) = {point_3d[2]:.2f} meters")
        else:
            print(f"Clicked at ({x}, {y}): Invalid disparity")

cv2.setMouseCallback("Disparity Map", on_mouse)

cv2.waitKey(0)
cv2.destroyAllWindows()