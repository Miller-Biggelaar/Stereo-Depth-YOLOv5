# stereo_depth_map_inspector.py

import cv2
import numpy as np
import pickle
import os
import time
from picamera2 import Picamera2

# === Config ===
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CALIBRATION_FILE = 'stereo_calibration.pkl'
SAVE_DIR = './depth_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# Global depth map for mouse callback
clicked_depth_map = None

def load_calibration(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Calibration data loaded from {filename}")
    return data

def capture_stereo_images():
    picam2_left = Picamera2(1)
    picam2_right = Picamera2(0)

    config_left = picam2_left.create_still_configuration(main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT)})
    config_right = picam2_right.create_still_configuration(main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT)})

    picam2_left.configure(config_left)
    picam2_right.configure(config_right)

    picam2_left.start()
    picam2_right.start()

    time.sleep(1)  # let auto-exposure settle

    captured = False
    frame_left = None
    frame_right = None
    window_name = "Stereo Preview (ESC: quit, SPACE: capture)"
    try:
        while True:
            img_left = picam2_left.capture_array()
            img_right = picam2_right.capture_array()

            # Rotate images 90 degrees clockwise
            img_left = cv2.rotate(img_left, cv2.ROTATE_90_CLOCKWISE)
            img_right = cv2.rotate(img_right, cv2.ROTATE_90_CLOCKWISE)

            # Convert from RGB to BGR for OpenCV display
            img_left_disp = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)
            img_right_disp = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)

            # Concatenate side by side
            preview = cv2.hconcat([img_left_disp, img_right_disp])
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1)
            if key == 27:  # ESC - quit without capturing
                frame_left = None
                frame_right = None
                break
            elif key == 32:  # SPACE - capture and return
                frame_left = img_left
                frame_right = img_right
                captured = True
                break
        cv2.destroyWindow(window_name)
    finally:
        picam2_left.close()
        picam2_right.close()
        cv2.destroyAllWindows()

    if frame_left is None or frame_right is None:
        print("Capture cancelled.")
        exit(0)
    return frame_left, frame_right

def rectify_images(img_left, img_right, calib):
    rect_left = cv2.remap(img_left,
                          calib['stereo_map_left_x'], calib['stereo_map_left_y'],
                          interpolation=cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right,
                           calib['stereo_map_right_x'], calib['stereo_map_right_y'],
                           interpolation=cv2.INTER_LINEAR)
    
    # Convert RGB → BGR before display
    rect_left = cv2.cvtColor(rect_left, cv2.COLOR_RGB2BGR)
    rect_right = cv2.cvtColor(rect_right, cv2.COLOR_RGB2BGR)

    return rect_left, rect_right

def compute_disparity(rect_left, rect_right):
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    # --- StereoSGBM parameters ---
    min_disp            = 0
    num_disp            = 64   # must be divisible by 16 - 16, 32, or 64 are good! was 32, GPT said higher for realistic scenes    
    block_size          = 9     # smaller block size = more local detail and more sensitive to noise
    uniqueness_ratio    = 15    # Higher ratio = requires stronger confidence in a match.
    speckle_windowsize  = 50    # Sets the minimum region size (in pixels) that the stereo algorithm considers valid
    speckle_range       = 2     # Max. disparity difference within a connected region to still be considered “consistent.”
    pre_filtercap       = 63    # Pixel intensities are normalized and clipped to +-value, Default is usually 31; currently using 63 (maximum allowed).

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio, # WAS 5
        speckleWindowSize=speckle_windowsize,  # Sets the minimum region size (in pixels) that the stereo algorithm considers valid - Was 100
        speckleRange=speckle_range,        # The max disparity difference within a connected region to still be considered “consistent.” - Was 32
        preFilterCap=pre_filtercap,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
    return disparity

def compute_depth(disparity, Q):
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3D[:, :, 2]  # Z values (depth)
    return depth_map

def on_mouse_click(event, x, y, flags, param):
    global clicked_depth_map
    if event == cv2.EVENT_LBUTTONDOWN and clicked_depth_map is not None:
        depth_val = clicked_depth_map[y, x]
        if np.isfinite(depth_val) and depth_val > 0:
            print(f"Depth at ({x}, {y}) = {depth_val:.3f} meters")
        else:
            print(f"Depth at ({x}, {y}) is invalid (probably occlusion or out of range).")

def save_and_display(rect_left, disparity, depth_map):
    global clicked_depth_map
    clicked_depth_map = depth_map.copy()

    disp_vis = cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=2), cv2.COLORMAP_JET)

    max_depth = 2.0 
    depth_vis = np.clip(depth_map, 0, max_depth)
    depth_vis = cv2.convertScaleAbs(depth_vis, alpha=255.0 / max_depth)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    # Save outputs
    cv2.imwrite(os.path.join(SAVE_DIR, 'rectified_leftv7.jpg'), rect_left)
    cv2.imwrite(os.path.join(SAVE_DIR, 'disparity_mapv7.jpg'), disp_vis)
    cv2.imwrite(os.path.join(SAVE_DIR, 'depth_mapv7.jpg'), depth_vis)
    np.save(os.path.join(SAVE_DIR, 'depth_mapv7.npy'), depth_map)
    print(f"Saved outputs to {SAVE_DIR}")

    # Show windows
    cv2.namedWindow("Depth (click to inspect)")
    cv2.setMouseCallback("Depth (click to inspect)", on_mouse_click)

    while True:
        cv2.imshow("Rectified Left", rect_left)
        cv2.imshow("Disparity", disp_vis)
        cv2.imshow("Depth (click to inspect)", depth_vis)

        if cv2.waitKey(1) == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()

def main():
    calib = load_calibration(CALIBRATION_FILE)
    img_left, img_right = capture_stereo_images()
    rect_left, rect_right = rectify_images(img_left, img_right, calib)
    disparity = compute_disparity(rect_left, rect_right)
    depth_map = compute_depth(disparity, calib['Q'])
    save_and_display(rect_left, disparity, depth_map)

if __name__ == "__main__":
    main()