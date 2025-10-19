# --- Stereo Calibration Image Capture Script ---
# Press SPACE to save stereo image pairs
# Press 'q' to quit

import os
import cv2
from datetime import datetime
from stereo_capture import capture_stereo_images

# Directory to save calibration images
LEFT_DIR = './calib_images/left'
RIGHT_DIR = './calib_images/right'

# Ensure output directories exist
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

print("📷 Starting stereo camera preview. Press SPACE to capture, 'q' to quit.")

image_count = 0

while True:
    try:
        # Capture stereo frames
        left_frame, right_frame = capture_stereo_images()

        # Stack images side-by-side for preview
        preview = cv2.hconcat([left_frame, right_frame])
        cv2.imshow("Stereo Preview (Left | Right)", preview)

        key = cv2.waitKey(1)

        if key == ord(' '):  # Spacebar pressed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            left_path = os.path.join(LEFT_DIR, f"left_{timestamp}.jpg")
            right_path = os.path.join(RIGHT_DIR, f"right_{timestamp}.jpg")
            cv2.imwrite(left_path, left_frame)
            cv2.imwrite(right_path, right_frame)
            image_count += 1
            print(f"✅ Captured image pair #{image_count}")

        elif key == ord('q'):  # Quit
            print("👋 Quitting preview.")
            break

    except RuntimeError as e:
        print(f"⚠️ Capture error: {e}")
        break

cv2.destroyAllWindows()