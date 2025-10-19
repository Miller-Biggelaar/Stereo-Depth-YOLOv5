# detect_and_depth_from_pairs.py
import os, sys, json, csv, pickle, time
from pathlib import Path
import numpy as np
import cv2

# ------------------ USER SETTINGS ------------------
LEFT_DIR  = './realworld_images/left_images'
RIGHT_DIR = './realworld_images/right_images'
CALIB_PKL = './stereo_calibration.pkl'          # your saved calibration
WEIGHTS   = './runs/train/exp/weights/best.pt'  # path to your YOLOv5 weights
OUT_DIR   = './depth_results_yolo'
MAX_VIZ_DEPTH_M = 6.0                           # cap for nice looking depth images
DEVICE = ''                                     # ''=auto, 'cpu', '0' for GPU:0, etc.
CONF_THRES = 0.25
IOU_THRES  = 0.45
# ---------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# --- YOLOv5 loader (local repo) ---
# Tell this script where the YOLOv5 repo is. By default it tries:
#  1) $YOLOV5_DIR env var
#  2) a sibling folder named "yolov5" next to this script
#  3) the parent directory (useful if the script sits inside the repo)
YOLOV5_DIR_ENV = os.getenv('YOLOV5_DIR')
_here = Path(__file__).resolve().parent
candidates = [
    Path(YOLOV5_DIR_ENV) if YOLOV5_DIR_ENV else None,
    _here / 'yolov5',
    _here,
    _here.parent,
]

yolo_path = None
for d in [p for p in candidates if p is not None]:
    if (d / 'models').exists() and (d / 'utils').exists():
        yolo_path = d
        break

if yolo_path is None:
    raise ImportError(
        "Could not locate YOLOv5 repo. Set the YOLOV5_DIR environment variable "
        "to the path containing 'models' and 'utils', or place this script next "
        "to a 'yolov5' folder."
    )

# Prepend the located repo path so imports like 'models' and 'utils' resolve.
if str(yolo_path) not in sys.path:
    sys.path.insert(0, str(yolo_path))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device
import torch

# --- Calibration ---
def load_calib(pkl_path):
    with open(pkl_path, 'rb') as f:
        calib = pickle.load(f)
    if 'Q' not in calib:
        # Build Q from rectification if needed
        # Expect R1,R2,P1,P2,Q in calib; if not, compute with stereoRectify
        raise ValueError("Calibration pickle must contain 'Q' or rectification params.")
    return calib

def rectify_pair(imgL, imgR, calib):
    # If you stored rectification maps, prefer them (fast). Otherwise build on the fly.
    if 'map1x' in calib:
        L = cv2.remap(imgL, calib['map1x'], calib['map1y'], cv2.INTER_LINEAR)
        R = cv2.remap(imgR, calib['map2x'], calib['map2y'], cv2.INTER_LINEAR)
        return L, R
    # Fallback: compute from intrinsics/extrinsics
    M1, D1, M2, D2 = calib['M1'], calib['D1'], calib['M2'], calib['D2']
    R, T = calib['R'], calib['T']
    h, w = imgL.shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(M1, D1, M2, D2, (w, h), R, T, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (w,h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(M2, D2, R2, P2, (w,h), cv2.CV_32FC1)
    calib['Q'] = Q
    L = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    R = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
    return L, R

# --- Disparity / depth ---
def build_sgbm():
    # sensible starting params for outdoor scenes; adjust to taste
    min_disp = 0
    num_disp = 160  # multiple of 16 (e.g., 128,160,192,256)
    block = 5
    P1 = 8 * 3 * block * block
    P2 = 32 * 3 * block * block
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return sgbm, min_disp, num_disp

def disparity_and_depth(rectL, rectR, Q):
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    sgbm, min_disp, num_disp = build_sgbm()
    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0.0] = np.nan  # invalid disparities
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # (H,W,3), Z in same length units as baseline * focal
    depth = points_3d[:, :, 2]  # Z
    return disp, depth

# --- YOLO inference ---
def load_model(weights, device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    return model, stride, names

@torch.no_grad()
def detect(model, img_bgr, conf=0.25, iou=0.45, imgsz=640):
    im = img_bgr.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # letterbox inside model; just forward raw image
    results = model(im, augment=False, visualize=False)
    pred = non_max_suppression(results, conf_thres=conf, iou_thres=iou, classes=None, max_det=300)[0]
    if pred is None or len(pred) == 0:
        return np.zeros((0,6)), []  # xyxy, conf, cls
    # scale boxes already match input since we passed raw image (no resize in this wrapper)
    return pred.cpu().numpy(), model.names

def median_depth_in_box(depth_map, box, valid_mask=None):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(depth_map.shape[1]-1, x2); y2 = min(depth_map.shape[0]-1, y2)
    roi = depth_map[y1:y2+1, x1:x2+1]
    if valid_mask is not None:
        roi = roi[valid_mask[y1:y2+1, x1:x2+1]]
    roi = roi[np.isfinite(roi)]
    if roi.size == 0:
        return { 'median_m': float('nan'), 'mean_m': float('nan'), 'min_m': float('nan'), 'max_m': float('nan')}
    return {
        'median_m': float(np.nanmedian(roi)),
        'mean_m':   float(np.nanmean(roi)),
        'min_m':    float(np.nanmin(roi)),
        'max_m':    float(np.nanmax(roi)),
    }

def colorize_depth(depth, cap_m=6.0):
    d = np.clip(depth, 0, cap_m)
    vis = cv2.convertScaleAbs(d, alpha=255.0/cap_m)
    return cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

def main():
    calib = load_calib(CALIB_PKL)
    model, stride, names = load_model(WEIGHTS, DEVICE)

    left_files  = sorted([f for f in os.listdir(LEFT_DIR)  if f.lower().endswith(('.jpg','.jpeg','.png'))])
    right_files = sorted([f for f in os.listdir(RIGHT_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))])

    csv_path = Path(OUT_DIR, 'detections_with_depth.csv')
    with open(csv_path, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['image','class','conf','x1','y1','x2','y2','median_m','mean_m','min_m','max_m'])

        for lf in left_files:
            rf = lf.replace('left_', 'right_')
            if rf not in right_files:
                print(f"[skip] no right match for {lf}")
                continue

            left_path  = str(Path(LEFT_DIR, lf))
            right_path = str(Path(RIGHT_DIR, rf))
            imgL = cv2.imread(left_path)
            imgR = cv2.imread(right_path)
            if imgL is None or imgR is None:
                print(f"[warn] failed to read pair for {lf}")
                continue

            # 1) rectify
            rectL, rectR = rectify_pair(imgL, imgR, calib)

            # 2) run detection on left rectified image
            dets, class_names = detect(model, rectL, CONF_THRES, IOU_THRES)

            # If nothing detected, optionally skip depth computation
            if dets.shape[0] == 0:
                # still save an overlay for traceability
                out_img = rectL.copy()
                cv2.imwrite(str(Path(OUT_DIR, f'overlay_{lf.replace("left_","")}')), out_img)
                continue

            # 3) compute disparity + depth once
            disp, depth = disparity_and_depth(rectL, rectR, calib['Q'])

            # 4) per-bbox depth stats + nice overlay
            overlay = rectL.copy()
            depth_vis = colorize_depth(depth, MAX_VIZ_DEPTH_M)
            disp_vis  = cv2.applyColorMap(cv2.convertScaleAbs(np.nan_to_num(disp, nan=0.0), alpha=2), cv2.COLORMAP_JET)

            for *xyxy, conf, cls in dets:
                stats = median_depth_in_box(depth, xyxy)
                c = int(cls)
                label = f"{class_names[c] if c < len(class_names) else c} {conf:.2f} | z~{stats['median_m']:.2f} m"
                x1,y1,x2,y2 = map(int, xyxy)
                cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(overlay, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                writer.writerow([lf, class_names[c] if c < len(class_names) else c, f"{conf:.3f}", x1,y1,x2,y2,
                                 stats['median_m'], stats['mean_m'], stats['min_m'], stats['max_m']])

            base = Path(lf).stem.replace('left_', '')
            cv2.imwrite(str(Path(OUT_DIR, f'overlay_{base}.jpg')), overlay)
            cv2.imwrite(str(Path(OUT_DIR, f'depth_{base}.jpg')), depth_vis)
            cv2.imwrite(str(Path(OUT_DIR, f'disp_{base}.jpg')),  disp_vis)
            np.save(str(Path(OUT_DIR, f'depth_{base}.npy')), depth)

            print(f"[ok] {lf} -> {OUT_DIR}")

    print(f"\nCSV written to {csv_path}")

if __name__ == '__main__':
    main()