# --- Synthetic Dataset Generator for YOLOv5 ---
# This script generates synthetic images of green shapes (representing weeds)
# on a brown background, with YOLOv5-format bounding box annotations.

import cv2
import numpy as np
import os
import random

# Settings
OUTPUT_DIR = './synthetic_dataset/'
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
LABELS_DIR = os.path.join(OUTPUT_DIR, 'labels')
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_IMAGES = 1000
MIN_SHAPES = 4
MAX_SHAPES = 16

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

def generate_random_shape(img, label_file):
    shape_type = random.choice(['circle', 'ellipse', 'rectangle'])
    center_x = random.randint(50, IMAGE_WIDTH - 50)
    center_y = random.randint(50, IMAGE_HEIGHT - 50)
    color = (0, random.randint(50, 255), 0)  # Random green shades

    if shape_type == 'circle':
        radius = random.randint(20, 50)
        cv2.circle(img, (center_x, center_y), radius, color, -1)
        x_min = center_x - radius
        y_min = center_y - radius
        x_max = center_x + radius
        y_max = center_y + radius

    elif shape_type == 'ellipse':
        axes = (random.randint(20, 50), random.randint(10, 40))
        angle = random.randint(0, 360)
        cv2.ellipse(img, (center_x, center_y), axes, angle, 0, 360, color, -1)
        x_min = center_x - axes[0]
        y_min = center_y - axes[1]
        x_max = center_x + axes[0]
        y_max = center_y + axes[1]

    elif shape_type == 'rectangle':
        width = random.randint(30, 60)
        height = random.randint(30, 60)
        cv2.rectangle(img, (center_x, center_y), (center_x + width, center_y + height), color, -1)
        x_min = center_x
        y_min = center_y
        x_max = center_x + width
        y_max = center_y + height

    # Convert bounding box to YOLO format (class x_center y_center width height), normalized
    class_id = 0  # Single class for weed
    x_center = (x_min + x_max) / 2 / IMAGE_WIDTH
    y_center = (y_min + y_max) / 2 / IMAGE_HEIGHT
    bbox_width = (x_max - x_min) / IMAGE_WIDTH
    bbox_height = (y_max - y_min) / IMAGE_HEIGHT

    label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

def generate_dataset():
    for i in range(NUM_IMAGES):
        # Create background
        background_color = (random.randint(60, 90), random.randint(60, 90), random.randint(60, 90)) # Blue, Green, Red
        img = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), background_color, dtype=np.uint8)

        label_path = os.path.join(LABELS_DIR, f'image_{i:04}.txt')
        with open(label_path, 'w') as label_file:
            num_shapes = random.randint(MIN_SHAPES, MAX_SHAPES)
            for _ in range(num_shapes):
                generate_random_shape(img, label_file)

        img_path = os.path.join(IMAGES_DIR, f'image_{i:04}.jpg')
        cv2.imwrite(img_path, img)

        if i % 20 == 0:
            print(f"Generated {i}/{NUM_IMAGES} images...")

    print("Synthetic dataset generation complete.")

if __name__ == '__main__':
    generate_dataset()