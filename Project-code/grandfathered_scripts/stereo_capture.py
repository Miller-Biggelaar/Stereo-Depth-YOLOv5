import cv2

def capture_stereo_images(left_cam_id=0, right_cam_id=1, width=640, height=480):
    # Open cameras
    left_cam = cv2.VideoCapture(left_cam_id)
    right_cam = cv2.VideoCapture(right_cam_id)

    left_cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    left_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    right_cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    right_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret_left, left_frame = left_cam.read()
    ret_right, right_frame = right_cam.read()

    left_cam.release()
    right_cam.release()

    if not ret_left or not ret_right:
        raise RuntimeError("Failed to capture from both cameras.")

    return left_frame, right_frame