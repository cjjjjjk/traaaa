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
    """
    api to detect vehicles and apply logistic regression model
    returns detection data + model prediction score
    """
    full_camera_url = request.args.get("url")
    if not full_camera_url:
        return jsonify({"error": "missing parameter 'url'"}), 400
    
    # check if model is loaded
    if logistic_model is None or scaler is None:
        return jsonify({
            "status": "error",
            "message": "logistic model not loaded. please train model first at /model/train-logistic"
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
                "message": "failed to get frame from camera"
            }), 500
        
        # 2. detect vehicles (yolo)
        detections = analyze_frame(frame)
        
        # 3. calculate vectors & count vehicles
        vectors = []
        counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
        
        for det in detections:
            bbox = det["bbox"]
            kpt = det["keypoint"]
            
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            tip_x = int(kpt[0])
            tip_y = int(kpt[1])
            tail_x = center_x * 2 - tip_x
            tail_y = center_y * 2 - tip_y
            
            vectors.append((tail_x, tail_y, tip_x, tip_y))
            
            cls_name = det["class"]
            if cls_name in counts:
                counts[cls_name] += 1
        
        # 4. calculate chaos score & road area
        raw_chaos = compute_chaos_score(vectors)
        chao_score = 0.0
        if isinstance(raw_chaos, dict):
            chao_score = raw_chaos.get('final_score', raw_chaos.get('score', 0.0))
        elif isinstance(raw_chaos, (int, float)):
            chao_score = float(raw_chaos)
        
        road_pixels = detect_road_area(frame)
        
        # 5. prepare detection data
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
        
        # 6. apply logistic regression model
        # prepare features in same order as training
        feature_cols = ['car_count', 'truck_count', 'bus_count', 'motorcycle_count', 'road_area_pixels', 'vectors_chao_score']
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
        
        # predict probability
        predicted_prob = logistic_model.predict_proba(X_scaled)[0, 1]
        predicted_class = logistic_model.predict(X_scaled)[0]
        
        # 7. prepare response
        response_data = {
            "status": "success",
            "detection_data": detection_data,
            "model_prediction": {
                "congestion_probability": float(predicted_prob),
                # "congestion_class": int(predicted_class),
                "threshold": 0.38,
                # "interpretation": "congested" if predicted_class == 1 else "not congested"
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"[ERROR] realtime_score: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
