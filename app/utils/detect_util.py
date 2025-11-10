from ultralytics import YOLO
import cv2
import numpy as np
from config.base_config import BASE_MODEL_VALUE

MODEL_PATH = "utils/model/best.pt"
try:
    model = YOLO(MODEL_PATH)
    CLASS_NAMES = model.names
except Exception as e:
    print(f"[ERROR] Failed to load model from {MODEL_PATH}. Error: {e}")
    model = None
    CLASS_NAMES = {}


ALLOWED_CLASSES = set(BASE_MODEL_VALUE["ALLOWED_CLASSES"])

def analyze_frame(frame: np.ndarray):
    """
    Phân tích một frame để detect xe và keypoints (hướng).
    """
    if model is None:
        print("[ERROR] Model is not loaded. Cannot analyze frame.")
        return []

    detections = []
    
    try:
        results = model(frame, verbose=False)
        
        if results[0].keypoints is None:
            print("[ERROR] Model is not a POSE model or failed to detect keypoints.")
            return []

        # detect results 
        boxes = results[0].boxes.cpu().numpy()
        keypoints = results[0].keypoints.cpu().numpy()

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = CLASS_NAMES.get(cls_id, "unknown")

            if cls_name not in ALLOWED_CLASSES:
                continue
            
            # Bounding box (x1, y1, x2, y2)
            bbox = boxes.xyxy[i].astype(int)
            
            # keypoint 
            # kpt : [x, y, conf]
            kpt = keypoints.xy[i][0].astype(int) 

            if kpt[0] > 0 and kpt[1] > 0:
                detections.append({
                    "class": cls_name,
                    "bbox": bbox,
                    "keypoint": kpt #[x, y]
                })

        return detections

    except Exception as e:
        print(f"[ERROR] in analyze_frame: {e}")
        return []