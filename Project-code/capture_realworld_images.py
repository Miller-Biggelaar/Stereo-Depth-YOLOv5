# capture_realworld_images.py
# Save stereo image pairs of real-world weeds using PiCamera2 with rotation + color fix
# Organised by angle folders, filenames include distance and timestamp

from picamera2 import Picamera2
import time
import cv2
import os
from datetime import datetime

# === Config ===
BASE_DIR_LEFT = "./realworld_images/left"
BASE_DIR_RIGHT = "./realworld_images/right"
CAPTURE_INTERVAL = 1  # seconds between captures
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Ensure base folders exist
os.makedirs(BASE_DIR_LEFT, exist_ok=True)
os.makedirs(BASE_DIR_RIGHT, exist_ok=True)

def make_capture_dirs(angle):
    # Create subfolders for the given angle if they don't exist
    left_dir = os.path.join(BASE_DIR_LEFT, f"angle_{angle}_degrees")
    right_dir = os.path.join(BASE_DIR_RIGHT, f"angle_{angle}_degrees")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    return left_dir, right_dir

def main():
    # Ask the user which angle they are capturing at
    current_angle = input("Enter the desired angle (in degrees): ").strip()
    current_height = input("Enter the camera height from the ground (in cm): ").strip()
    current_grid_distance = input("Enter the distance to the bottom of the grid (in cm): ").strip()
    left_dir, right_dir = make_capture_dirs(current_angle)

    # Init both cameras
    picam2_left = Picamera2(0)
    picam2_right = Picamera2(1)

    config_left = picam2_left.create_still_configuration(main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT)})
    config_right = picam2_right.create_still_configuration(main={"size": (IMAGE_WIDTH, IMAGE_HEIGHT)})

    picam2_left.configure(config_left)
    picam2_right.configure(config_right)

    picam2_left.start()
    picam2_right.start()

    print("Press SPACE to capture a stereo pair. Press ESC to quit.")

    img_counter = 0

    while True:
        # Capture from both cameras
        frame_left = picam2_left.capture_array()
        frame_right = picam2_right.capture_array()

        # === Fix orientation and color ===
        # 1. Rotate 90 degrees clockwise
        frame_left = cv2.rotate(frame_left, cv2.ROTATE_90_CLOCKWISE)
        frame_right = cv2.rotate(frame_right, cv2.ROTATE_90_CLOCKWISE)

        # 2. Convert RGB (Picamera2) to BGR (OpenCV)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)

        # Display preview side-by-side
        preview = cv2.hconcat([frame_left, frame_right])
        cv2.imshow("Stereo Preview (Left | Right)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == 32:  # SPACE to save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            left_path = os.path.join(left_dir, f"left_{current_angle}deg_{current_height}cmgnd_{current_grid_distance}cmgrid_{timestamp}.jpg")
            right_path = os.path.join(right_dir, f"right_{current_angle}deg_{current_height}cmgnd_{current_grid_distance}cmgrid_{timestamp}.jpg")

            cv2.imwrite(left_path, frame_left)
            cv2.imwrite(right_path, frame_right)

            print(f"[OK] Saved stereo pair #{img_counter} at angle {current_angle}°, height {current_height}cm, grid distance {current_grid_distance}cm")
            img_counter += 1
            time.sleep(CAPTURE_INTERVAL)

    picam2_left.close()
    picam2_right.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()