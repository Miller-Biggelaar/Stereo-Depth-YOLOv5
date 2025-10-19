# stereo_capture_picamera2.py
# Captures a single stereo image pair using PiCamera2

from picamera2 import Picamera2
import cv2
import time

def capture_stereo_images(width=640, height=480):
    # Initialize both cameras
    picam2_left = Picamera2(0)
    picam2_right = Picamera2(1)

    # Set camera configurations
    config_left = picam2_left.create_still_configuration(main={"size": (width, height)})
    config_right = picam2_right.create_still_configuration(main={"size": (width, height)})

    # Apply configurations and start both cameras
    picam2_left.configure(config_left)
    picam2_right.configure(config_right)

    picam2_left.start()
    picam2_right.start()

    # Small delay to allow auto-adjustments to settle
    time.sleep(1)

    # Capture frames
    frame_left = picam2_left.capture_array()
    frame_right = picam2_right.capture_array()

    # Stop cameras
    picam2_left.close()
    picam2_right.close()

    return frame_left, frame_right

if __name__ == "__main__":
    # Capture and preview the images
    left_img, right_img = capture_stereo_images()

    # Show both images
    cv2.imshow("Left Camera", left_img)
    cv2.imshow("Right Camera", right_img)

    print("Press any key to close previews...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()