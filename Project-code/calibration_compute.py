# --- Stereo Camera Calibration Script ---
# This script performs stereo camera calibration using captured images of a checkerboard pattern.
# It calculates intrinsic parameters, distortion coefficients, and rectification maps.

import cv2
import numpy as np
import glob

# Checkerboard dimensions (number of internal corners per row and column)
CHECKERBOARD = (9, 6)

# Paths to the calibration images for the left and right cameras (Had to swap them for Q matricx to work properly :p)
CALIBRATION_IMAGES_RIGHT = './calib_images/left/*.jpg'
CALIBRATION_IMAGES_LEFT = './calib_images/right/*.jpg'

def calibrate_stereo_cameras():
    # Prepare object points based on the checkerboard pattern
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    square_size = 0.020  # 20 mm = 0.020 m
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

    objpoints = []        # 3D points in real world space
    imgpoints_left = []   # 2D points in left camera images
    imgpoints_right = []  # 2D points in right camera images

    # Load calibration images
    images_left = sorted(glob.glob(CALIBRATION_IMAGES_LEFT))
    images_right = sorted(glob.glob(CALIBRATION_IMAGES_RIGHT))

    # Detect chessboard corners in image pairs
    for left_img_path, right_img_path in zip(images_left, images_right):
        img_left = cv2.imread(left_img_path)
        img_right = cv2.imread(right_img_path)

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

        if ret_left and ret_right:
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

    # Calibrate each camera individually
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

    # Perform stereo calibration to find rotation and translation between cameras
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_l, dist_l, mtx_r, dist_r, gray_left.shape[::-1],
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"Estimated baseline (camera separation): {abs(T[0][0]):.4f} meters")

    # Compute rectification transforms for both cameras
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1,
        cameraMatrix2, distCoeffs2,
        gray_left.shape[::-1], R, T, alpha=0
    )

    # Compute rectification maps for remapping the images
    stereo_map_left_x, stereo_map_left_y = cv2.initUndistortRectifyMap(
        cameraMatrix1, distCoeffs1, R1, P1, gray_left.shape[::-1], cv2.CV_16SC2
    )
    stereo_map_right_x, stereo_map_right_y = cv2.initUndistortRectifyMap(
        cameraMatrix2, distCoeffs2, R2, P2, gray_right.shape[::-1], cv2.CV_16SC2
    )

    # Package the calibration data into a dictionary
    calibration_data = {
        'M1': mtx_l,
        'D1': dist_l,
        'M2': mtx_r,
        'D2': dist_r,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'stereo_map_left_x': stereo_map_left_x,
        'stereo_map_left_y': stereo_map_left_y,
        'stereo_map_right_x': stereo_map_right_x,
        'stereo_map_right_y': stereo_map_right_y,
        'Q': Q
    }

    return calibration_data

if __name__ == '__main__':
    calibration_result = calibrate_stereo_cameras()
    print("Stereo calibration complete.")
    
    # --- Save full calibration parameters ---
    import pickle
    with open("stereo_calibration_full.pkl", "wb") as f:
        pickle.dump(calibration_result, f)
    print("✅ Full calibration parameters saved to stereo_calibration_full.pkl")
    # Save the calibration_result using the calibration_save.py script