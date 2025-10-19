# capture_calibration_images_picamera2.py
# Save stereo image pairs using PiCamera2 with rotation + color fix

from picamera2 import Picamera2
import time
import cv2
import os

# === Config ===
CAPTURE_DIR_LEFT = "./calib_images/left"
CAPTURE_DIR_RIGHT = "./calib_images/right"
CAPTURE_INTERVAL = 1  # seconds between captures
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Ensure folders exist
os.makedirs(CAPTURE_DIR_LEFT, exist_ok=True)
os.makedirs(CAPTURE_DIR_RIGHT, exist_ok=True)

def main():
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
            left_path = os.path.join(CAPTURE_DIR_LEFT, f"left_{img_counter:02d}.jpg")
            right_path = os.path.join(CAPTURE_DIR_RIGHT, f"right_{img_counter:02d}.jpg")

            cv2.imwrite(left_path, frame_left)
            cv2.imwrite(right_path, frame_right)

            print(f"✅ Saved stereo pair #{img_counter}")
            img_counter += 1
            time.sleep(CAPTURE_INTERVAL)

    picam2_left.close()
    picam2_right.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()