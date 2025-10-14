Stereo-Weed-Detection

Stereo-based Weed Detection and Depth Estimation using YOLOv5 on Raspberry Pi 5

⸻

🌱 Overview

This repository contains the full codebase, data structure, and calibration files developed for a research project conducted at The University of Western Australia (UWA).
The system integrates YOLOv5 object detection with stereo-based depth estimation to estimate weed positions in three-dimensional space using Raspberry Pi Camera Module 3 sensors and a Raspberry Pi 5.

The project demonstrates a low-cost, embedded vision solution for precision agriculture — capable of identifying and localising weeds through 2D detection and stereo disparity mapping.

⸻

🧩 Key Features
	•	YOLOv5-based object detection (trained on synthetic and real weed datasets)
	•	Stereo image rectification and disparity computation using OpenCV SGBM
	•	Depth and Euclidean distance estimation with multiple SGBM modes:
	•	SGBM
	•	SGBM_3WAY
	•	HH
	•	HH4
	•	 Performance evaluation scripts for depth accuracy and processing speed
	•	 Stereo calibration utilities (capture_calibration_images_picamera2.py, calibration_compute.py, verify_stereo_calibration.py)
	•	 Designed for reproducibility and open research
  
🧱 Repository Structure
  Stereo-Weed-Detection/
│
├── yolov5/                          # YOLOv5 repository (custom trained model)
├── realworld_images/                # Captured stereo image pairs
│   ├── left_images/
│   └── right_images/
│
├── CSVrepo/                         # Processed data and analysis scripts
│   ├── analyse_depth_performance.py
│   ├── measured_distance.csv
│   └── detections_with_depth_*.csv
│
├── calibration/                     # Calibration and rectification utilities
│   ├── capture_calibration_images_picamera2.py
│   ├── calibration_compute.py
│   ├── calibration_save.py
│   └── verify_stereo_calibration.py
│
├── batch_depthmap_with_yolo_timed.py  # Combined YOLO + depth timing evaluation
├── singlepair_depthmap.py              # Depth mapping for single stereo pair
├── detect_only_on_folder.py            # Object detection without depth mapping
├── other misc. files
└── README.md

⚙️ Setup and Installation

Requirements:
	•	Raspberry Pi 5 (8 GB recommended)
	•	Python 3.11
	•	OpenCV 4.x
	•	PyTorch (CPU build)
	•	Pandas, NumPy, Matplotlib

Installation (on Raspberry Pi 5):
sudo apt update
sudo apt install python3-opencv python3-numpy python3-pandas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Clone the repository:
git clone https://github.com/yourusername/Stereo-Weed-Detection.git
cd Stereo-Weed-Detection

🧪 Usage Examples

1. Capture calibration images
   python calibration/capture_calibration_images_picamera2.py

2. Compute stereo calibration
   python calibration/calibration_compute.py

3. Verify rectification
   python calibration/verify_stereo_calibration.py

4. Run detection and depth mapping
   python batch_depthmap_with_yolo_timed.py

5. Analyse performance
   python CSVrepo/analyse_depth_performance.py

🚀 Future Work
	•	Real-world field testing under natural lighting and occlusion
	•	Integration with GPU/NPU accelerators (Jetson Nano, Coral TPU)
	•	Evaluation of lightweight stereo-matching networks (Fast-ACVNet, MobileStereoNet)
	•	Real-time video inference and adaptive weed mapping

⸻

🏫 Citation

If you use this project in your research, please cite:

Biggelaar, M. J. (2025). Advancing Weed Detection Technology: Exploring the Use of Stereoscopic Imaging for Depth-Aware Weed Detection. University of Western Australia.
