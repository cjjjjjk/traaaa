from flask import Blueprint, Response
import json
import cv2
import os
import glob
from datetime import datetime

from utils.detect_util import analyze_frame
from utils.vectors_chao_score import compute_chaos_score
from utils.segment_util import detect_road_area
from routes.data.crud_frame_data import insert_tracking_data

local_crawler_bp = Blueprint("local_crawler", __name__)


@local_crawler_bp.route("/local-crawler", methods=["GET"])
def local_crawler():
    """
    API crawl data từ các frame trong thư mục data/frame/
    Đọc tất cả file jpg, xử lý và gửi vào Sheet
    """
    
    # default camera id
    camera_id = "587ee2aeb807da0011e33d52"
    
    # define input directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
    input_dir = os.path.join(APP_DIR, 'data', 'frames')
    
    # check if directory exists
    if not os.path.exists(input_dir):
        return json.dumps({
            "status": "error",
            "message": f"input directory not found: {input_dir}"
        }), 404
    
    # get all jpg files
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(input_dir, "*.JPG")))
    
    # remove duplicates (Windows is case-insensitive)
    image_files = list(set(image_files))
    
    if not image_files:
        return json.dumps({
            "status": "error",
            "message": f"no jpg files found in {input_dir}"
        }), 404
    
    # sort files by name
    image_files.sort()
    
    def generate():
        for idx, image_path in enumerate(image_files):
            try:
                # extract timestamp from filename
                # format: 20251222_145301.jpg -> 2025-12-22 14:53:01
                filename = os.path.basename(image_path)
                filename_no_ext = os.path.splitext(filename)[0]
                
                try:
                    # parse timestamp from filename
                    timestamp_str = filename_no_ext.replace('_', '')
                    year = int(timestamp_str[0:4])
                    month = int(timestamp_str[4:6])
                    day = int(timestamp_str[6:8])
                    hour = int(timestamp_str[8:10])
                    minute = int(timestamp_str[10:12])
                    second = int(timestamp_str[12:14])
                    current_time = datetime(year, month, day, hour, minute, second)
                except:
                    # fallback to current time if parsing fails
                    current_time = datetime.now()
                    print(f"[WARNING] failed to parse timestamp from {filename}, using current time")
                
                # 1. read frame from file
                frame = cv2.imread(image_path)
                
                if frame is None:
                    error_msg = json.dumps({
                        "status": "error", 
                        "message": f"failed to read image: {filename}"
                    }) + "\n"
                    yield error_msg
                    continue
                
                # 2. Detect (YOLO)
                detections = analyze_frame(frame)
                
                # 3. Tính toán Vectors & Đếm xe
                vectors = []
                counts = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
                
                for det in detections:
                    bbox = det["bbox"]
                    kpt = det["keypoint"]
                    
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    tip_x = int(kpt[0])
                    tip_y = int(kpt[1])
                    tail_x = center_x*2 - tip_x
                    tail_y = center_y*2 - tip_y
                    
                    vectors.append((tail_x, tail_y, tip_x, tip_y))
                    
                    cls_name = det["class"]
                    if cls_name in counts:
                        counts[cls_name] += 1
                
                # 4. Tính chỉ số (Chaos Score & Road Area)
                raw_chaos = compute_chaos_score(vectors)
                chao_score = 0.0
                if isinstance(raw_chaos, dict):
                    chao_score = raw_chaos.get('final_score', raw_chaos.get('score', 0.0))
                elif isinstance(raw_chaos, (int, float)):
                    chao_score = float(raw_chaos)
                
                road_pixels = detect_road_area(frame)
                
                # 5. Gửi dữ liệu lên Google Sheet
                log_data = {
                    "camera_id": camera_id,
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "car_count": counts["car"],
                    "truck_count": counts["truck"],
                    "bus_count": counts["bus"],
                    "motorcycle_count": counts["motorcycle"],
                    "road_area_pixels": int(road_pixels),
                    "vectors_chao_score": float(chao_score),
                    "congestion_score": 0
                }
                
                print(f"[LOCAL_CRAWLER] Processing file {idx+1}/{len(image_files)}: {filename}")
                sheet_status = insert_tracking_data(log_data)
                
                # 6. Yield JSON về Client (Streaming)
                response_data = {
                    "status": "success",
                    "saved_to_sheet": sheet_status,
                    "processed_file": filename,
                    "progress": f"{idx + 1}/{len(image_files)}",
                    "data": log_data
                }
                
                yield json.dumps(response_data) + "\n"
                
            except Exception as e:
                err_data = json.dumps({
                    "status": "error", 
                    "message": str(e),
                    "file": os.path.basename(image_path) if 'image_path' in locals() else "unknown"
                }) + "\n"
                yield err_data
                print(f"[ERROR] processing {image_path}: {e}")
        
        # send completion message
        completion_msg = json.dumps({
            "status": "completed",
            "message": f"processed all {len(image_files)} frames",
            "total_frames": len(image_files)
        }) + "\n"
        yield completion_msg
    
    return Response(generate(), mimetype='application/json')
