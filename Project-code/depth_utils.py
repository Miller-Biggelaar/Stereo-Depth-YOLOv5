import cv2
import numpy as np

def compute_disparity_map(left_rect, right_rect, num_disparities=64, block_size=15):
    gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    return disparity

def get_depth_from_disparity(disparity_map, x, y, Q_matrix):
    disparity_value = disparity_map[y, x]
    if disparity_value <= 0:
        return None

    points_3D = cv2.reprojectImageTo3D(disparity_map, Q_matrix)
    return points_3D[y, x][2]  # Z = depth