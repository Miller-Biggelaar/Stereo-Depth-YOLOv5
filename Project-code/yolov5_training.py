# --- YOLOv5 Training Launcher Script ---
# Automates training using YOLOv5's train.py with custom dataset and chosen model weights.

import subprocess

# Settings — adjust as needed
YOLOV5_DIR = './yolov5'  # Path to cloned YOLOv5 repo
DATASET_YAML = './data_real_test.yaml'  # Path to dataset.yml
WEIGHTS = 'yolov5n.pt'  # Could use 'yolov5s.pt' for better accuracy during training (maybe, bit more intensice for pi5)
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 1
EXPERIMENT_NAME = 'real_test_v1'

def launch_training():
    command = [
        'python', 'val.py',
        '--img', str(IMG_SIZE),
        '--batch', str(BATCH_SIZE),
        '--epochs', str(EPOCHS),
        '--data', DATASET_YAML,
        '--weights', WEIGHTS,
        '--name', EXPERIMENT_NAME
    ]

    subprocess.run(command, cwd=YOLOV5_DIR)

if __name__ == '__main__':
    launch_training()