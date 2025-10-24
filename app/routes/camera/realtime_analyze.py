from flask import Blueprint, Response, request, jsonify
import cv2
import time
import numpy as np
import requests
from datetime import datetime
from utils.yolo_cam_util import analyze_frame_cam
import warnings
warnings.filterwarnings("ignore")

realtime_analyze_bp = Blueprint("realtime_analyze", __name__)

# --- Cookie & headers y như realtime_detect.py ---
COOKIE_STR = """ASP.NET_SessionId=2pd400jxoiokdowndnju2zdr; .VDMS=E6D97CFABE16A23E4774230850C06957F36B5B3EA1FF8360A13AFBE6CB680692EF54AF6792F4900295E280829B09309FC1C42F3998D647739B7CDA12A3BBEC339828A6DCD87CE1E69EDE5FED4D54D3E641C6DE126BC213E650AF93CA7D237E3F0BBF1FF20C50762E6AE7BB049B7EE780F66A625A"""
def parse_cookie_string(s):
    return dict(part.split("=", 1) for part in s.split("; ") if "=" in part)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "vi,en;q=0.9",
    "Referer": "https://giaothong.hochiminhcity.gov.vn/",
}
COOKIES = parse_cookie_string(COOKIE_STR)

session = requests.Session()
session.headers.update(HEADERS)
session.cookies.update(COOKIES)

# --- Vùng ROI cố định ---
PTS_IMAGE = np.float32([[129, 91], [299, 96], [401, 216], [30, 200]])

@realtime_analyze_bp.route("/realtime-analyze", methods=["GET"])
def realtime_analyze():
    """
    Stream dữ liệu JSON phân tích theo thời gian thực từ camera.
    Trả về JSON liên tục (Server-Sent Events hoặc JSON stream).
    Ví dụ:
        /realtime-analyze?url=https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx?id=587ee2aeb807da0011e33d52
    """
    camera_base = request.args.get("url")
    if not camera_base:
        return jsonify({"error": "Thiếu tham số 'url'"}), 400

    def generate():
        while True:
            try:
                ts = int(time.time() * 1000)
                full_url = f"{camera_base}&bg=black&w=400&h=260&t={ts}"

                r = session.get(full_url, timeout=10, verify=False)
                if r.status_code != 200 or r.content[:4] == b"<!DO":
                    print(f"[WARNING] Không lấy được ảnh ({r.status_code})")
                    time.sleep(1)
                    continue

                img = np.frombuffer(r.content, np.uint8)
                frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
                if frame is None:
                    print("[WARNING] Không decode được frame.")
                    time.sleep(1)
                    continue

                detections, _ = analyze_frame_cam(frame)

                # --- Tính toán các chỉ số ---
                total_vehicles = len(detections)

                # Đếm theo loại
                type_counts = {}
                total_area = 0

                for det in detections:
                    cls_name = det.get("class", "unknown")
                    x1 = det.get("x1", 0)
                    y1 = det.get("y1", 0)
                    x2 = det.get("x2", 0)
                    y2 = det.get("y2", 0)

                    area = abs((x2 - x1) * (y2 - y1))
                    total_area += area
                    type_counts[cls_name] = type_counts.get(cls_name, 0) + 1

                roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(roi_mask, [PTS_IMAGE.astype(np.int32)], 255)
                roi_area = cv2.countNonZero(roi_mask)

                occupancy_ratio = (total_area / roi_area * 100) if roi_area > 0 else 0

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "total_vehicles": total_vehicles,
                    "type_counts": type_counts,
                    "bounding_box_area": int(total_area),
                    "roi_area": int(roi_area),
                    "occupancy_percent": round(occupancy_ratio, 2)
                }

                # Gửi JSON như stream
                yield f"data: {result}\n\n"
                time.sleep(15)

            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(2)
                continue

    return Response(generate(), mimetype="text/event-stream")
