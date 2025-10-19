# --- Dataset Splitter Script ---
# Splits synthetic dataset into 'train' and 'val' folders in both images/ and labels/

import os
import random
import shutil

# Settings
DATASET_DIR = './synthetic_dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels')
TRAIN_RATIO = 0.7  # 70% training, 30% validation

def split_dataset():
    images = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for split, split_images in [('train', train_images), ('val', val_images)]:
        os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

        for img_name in split_images:
            label_name = img_name.replace('.jpg', '.txt')

            shutil.move(os.path.join(IMAGES_DIR, img_name), os.path.join(IMAGES_DIR, split, img_name))
            shutil.move(os.path.join(LABELS_DIR, label_name), os.path.join(LABELS_DIR, split, label_name))

        print(f"{split.capitalize()} set: {len(split_images)} images")

if __name__ == '__main__':
    split_dataset()