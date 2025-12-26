import cv2
from flask import Blueprint, request, jsonify
import numpy as np
import pickle
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# import utility functions
from utils.call_frame_api import get_camera_frame
from utils.detect_util import analyze_frame
from utils.vectors_chao_score import compute_chaos_score
from utils.segment_util import detect_road_area

realtime_model_apply_bp = Blueprint("realtime_model_apply", __name__)

# load model and scaler at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
MODEL_DIR = os.path.join(APP_DIR, 'utils', 'model', 'logicstic')
APPLIED_DIR = os.path.join(APP_DIR, 'data', 'applied')

# ensure applied directory exists
os.makedirs(APPLIED_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, 'logistic_model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

# try to load model and scaler
try:
    with open(model_path, 'rb') as f:
        logistic_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"[INFO] loaded logistic model from {model_path}")
    print(f"[INFO] loaded scaler from {scaler_path}")
except Exception as e:
    print(f"[ERROR] failed to load model or scaler: {e}")
    logistic_model = None
    scaler = None

@realtime_model_apply_bp.route("/realtime-score", methods=["GET"])
def realtime_score():
    full_camera_url = request.args.get("url")
    if not full_camera_url:
        return jsonify({
            "status": "error",
            "message": "missing parameter 'url'"
        }), 400
    
    # check if model is loaded
    if logistic_model is None or scaler is None:
        return jsonify({
            "status": "error",
            "message": "logistic model not loaded. please train model first at /model/train-logistic",
            "detection_data": {},
            "model_prediction": {}
        }), 500
    
    try:
        # extract camera_id from url
        parsed = urlparse(full_camera_url)
        captured_id = parse_qs(parsed.query).get('id')
        if captured_id:
            camera_id = captured_id[0]
        elif "id=" in full_camera_url:
            camera_id = full_camera_url.split("id=")[-1]
        else:
            camera_id = full_camera_url
    except:
        camera_id = full_camera_url
    
    try:
        current_time = datetime.now()
        
        # 1. get camera frame
        frame = get_camera_frame(full_camera_url, timestamp=current_time)
        
        if frame is None:
            return jsonify({
                "status": "error",
                "message": "failed to get frame from camera",
                "detection_data": {},
                "model_prediction": {}
            }), 500
        
        # 2. detect vehicles using yolo model
        detections = analyze_frame(frame)
        
        # 3. calculate vectors and count vehicles by type
        vectors = []
        counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
        
        for det in detections:
            bbox = det["bbox"]
            kpt = det["keypoint"]
            cls_name = det["class"]
            
            # calculate vector from center to keypoint
            x1, y1, x2, y2 = map(int, bbox)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            tip_x = int(kpt[0])
            tip_y = int(kpt[1])
            tail_x = center_x * 2 - tip_x
            tail_y = center_y * 2 - tip_y
            
            vectors.append((tail_x, tail_y, tip_x, tip_y))
            
            if cls_name in counts:
                counts[cls_name] += 1
            
            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Draw vector (arrow)
            cv2.arrowedLine(frame, (tail_x, tail_y), (tip_x, tip_y), (0, 0, 255), 2, tipLength=0.3)
            
        # Save image with vectors
        file_name = current_time.strftime("%Y%m%d_%H%M%S.jpg")
        save_path = os.path.join(APPLIED_DIR, file_name)
        cv2.imwrite(save_path, frame)
        print(f"[INFO] Saved applied frame to {save_path}")
        
        # 4. compute chaos score and road area
        raw_chaos = compute_chaos_score(vectors)
        chao_score = 0.0
        if isinstance(raw_chaos, dict):
            chao_score = raw_chaos.get('final_score', raw_chaos.get('score', 0.0))
        elif isinstance(raw_chaos, (int, float)):
            chao_score = float(raw_chaos)
        
        road_pixels = detect_road_area(frame)
        
        # 5. prepare detection data with all features
        detection_data = {
            "camera_id": camera_id,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "car_count": counts["car"],
            "truck_count": counts["truck"],
            "bus_count": counts["bus"],
            "motorcycle_count": counts["motorcycle"],
            "road_area_pixels": int(road_pixels),
            "vectors_chao_score": float(chao_score)
        }
        
        # 6. apply prediction model
        # prepare features in same order as training
        X = np.array([[
            detection_data['car_count'],
            detection_data['truck_count'],
            detection_data['bus_count'],
            detection_data['motorcycle_count'],
            detection_data['road_area_pixels'],
            detection_data['vectors_chao_score']
        ]])
        
        # scale features
        X_scaled = scaler.transform(X)
        
        # predict congestion probability
        congestion_score = float(logistic_model.predict_proba(X_scaled)[0, 1])
        
        # 7. prepare response with required format
        response_data = {
            "detection_data": detection_data,
            "model_prediction": {
                "congestion_score": congestion_score,
                "threshold": 0.45
            },
            "status": "success"
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"[ERROR] realtime_score: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "detection_data": {},
            "model_prediction": {}
        }), 500
