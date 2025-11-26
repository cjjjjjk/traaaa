from flask import Blueprint, request, Response
import time
import json
import numpy as np
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# Import các hàm tiện ích
from utils.call_frame_api import get_camera_frame
from utils.detect_util import analyze_frame
from utils.vectors_chao_score import compute_chaos_score
from utils.segment_util import detect_road_area
from routes.data.crud_frame_data import insert_tracking_data 
# from config.base_config import VEHICLE_CLASS_COLORS, BASE_APICALL_VALUE

realtime_crawler_bp = Blueprint("realtime_crawler", __name__)

@realtime_crawler_bp.route("/realtime-crawler", methods=["GET"])
def realtime_crawler():
    """
    API chạy liên tục (Streaming JSON).
    Cứ 15 giây sẽ xử lý 1 frame -> Gửi vào Sheet -> Trả về 1 dòng JSON cho client.
    """
    full_camera_url = request.args.get("url")
    if not full_camera_url:
        return json.dumps({"error": "Missing parameter 'url'"}), 400

    try:
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

    def generate():
        while True:
            try:
                # 1. Gọi API lấy frame
                frame = get_camera_frame(full_camera_url)
                
                if frame is None:
                    # Nếu lỗi lấy ảnh, báo lỗi json rồi thử lại sau
                    error_msg = json.dumps({"status": "error", "message": "No frame"}) + "\n"
                    yield error_msg
                    time.sleep(5)
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
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "car_count": counts["car"],
                    "truck_count": counts["truck"],
                    "bus_count": counts["bus"],
                    "motorcycle_count": counts["motorcycle"],
                    "road_area_pixels": int(road_pixels),
                    "vectors_chao_score": float(chao_score),
                    "congestion_score": 0
                }
                
                sheet_status = insert_tracking_data(log_data)

                # 6. Yield JSON về Client (Streaming)
                response_data = {
                    "status": "success",
                    "saved_to_sheet": sheet_status,
                    "data": log_data
                }
                
                # Quan trọng: Thêm \n để client phân biệt từng lần gửi
                yield json.dumps(response_data) + "\n"

                # 7. Nghỉ 15 giây
                time.sleep(15)

            except Exception as e:
                # Nếu có lỗi trong vòng lặp, log ra và tiếp tục chạy
                err_data = json.dumps({"status": "error", "message": str(e)}) + "\n"
                yield err_data
                print(f"[ERROR] Loop exception: {e}")
                time.sleep(5) # Nghỉ ngắn nếu lỗi

    # Trả về response dạng stream text
    return Response(generate(), mimetype='application/json')