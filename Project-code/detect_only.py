# detect_only_on_folder.py
import os, sys, csv
from pathlib import Path
import cv2
import numpy as np
import torch

# ------------------ USER SETTINGS ------------------
#IM_DIR     = './synthetic_dataset/images/val'   # Folder to scan
IM_DIR     = './realworld_images/test/images'   # Folder to scan
WEIGHTS    = './yolov5/runs/train/synth_plus_real_ft13/weights/best.pt'
OUT_DIR    = './detect_only_results'
OUT_DIR_CSV= './CSVrepo'                        #Change any instances of OUT_DIR to this to save csv to repo
CONF_THRES = 0.5
IOU_THRES  = 0.45
IMGSZ      = 640          # YOLOv5 inference size
DEVICE     = ''           # '' auto, 'cpu', or '0' for GPU:0
CLASS_FILTER = None       # e.g., [0] to keep only class 0; or None for all
# ---------------------------------------------------

# add YOLOv5 repo path (absolute)
yolo_path = Path(__file__).resolve().parent / 'yolov5'
if not (yolo_path.exists() and (yolo_path / 'models').exists() and (yolo_path / 'utils').exists()):
    raise ImportError(f"YOLOv5 repo not found at: {yolo_path}. Expected 'models/' and 'utils/' inside it.")
sys.path.insert(0, str(yolo_path))
if not Path(WEIGHTS).exists():
    print(f"[warn] Weights not found at {WEIGHTS}. Update WEIGHTS to your best.pt path.")

os.makedirs(OUT_DIR, exist_ok=True)

# --- YOLOv5 internals ---
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device

def load_model(weights, device, imgsz):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)
    return model, stride, names, imgsz, device

@torch.no_grad()
def infer_one(model, bgr, imgsz, conf, iou, stride, class_filter=None, device=''):
    # 1) letterbox + to tensor (as in detect.py)
    img0 = bgr
    im = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(model.device)
    im = im.float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    # 2) forward + NMS
    pred = model(im, augment=False, visualize=False)
    det = non_max_suppression(pred, conf_thres=conf, iou_thres=iou,
                              classes=class_filter, max_det=300)[0]

    # 3) scale boxes back to original image size
    results = []
    if det is not None and len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
        results = det.cpu().numpy()
    return results  # array of [x1,y1,x2,y2, conf, cls]

def draw_and_save(img, dets, names, out_path):
    img = img.copy()
    for *xyxy, conf, cls in dets:
        x1,y1,x2,y2 = map(int, xyxy)
        label = f"{conf:.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(img, label, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imwrite(out_path, img)

def main():
    model, stride, names, imgsz, device = load_model(WEIGHTS, DEVICE, IMGSZ)

    images = sorted([p for p in Path(IM_DIR).glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    if not images:
        print(f"No images found in {IM_DIR}")
        return

    csv_path = Path(OUT_DIR, 'detections_test.csv')
    with open(csv_path, 'w', newline='') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['image','conf','x1','y1','x2','y2'])

        for p in images:
            img = cv2.imread(str(p))
            if img is None:
                print(f"[warn] cannot read {p}")
                continue

            dets = infer_one(model, img, imgsz, CONF_THRES, IOU_THRES, stride, CLASS_FILTER, DEVICE)

            # write CSV rows
            if dets is not None and len(dets):
                for *xyxy, conf, cls in dets:
                    cname = names[int(cls)] if int(cls) < len(names) else int(cls)
                    w.writerow([p.name, f"{float(conf):.3f}"] + list(map(int, xyxy)))
            else:
                # Optionally write a 'no detections' row
                pass

            # save overlay image
            out_img = Path(OUT_DIR, f"det_{p.stem}.jpg")
            draw_and_save(img, dets, names, str(out_img))
            print(f"[ok] {p.name} -> {out_img.name}")

    print(f"\nCSV written to {csv_path}")

if __name__ == '__main__':
    main()